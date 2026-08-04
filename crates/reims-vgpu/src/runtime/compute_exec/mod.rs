//! Product-path compute bind/dispatch for `SEGMENT_TYPE_COMPUTE`.
//!
//! Executable surface:
//! - `0xd0` set compute pipeline (type-7 → kernel function MTLB + optional stage-input)
//! - `0xcb` / `0xd9` set buffers (+ optional attribute stride for dynamic stage-input layouts)
//! - `0xcf` / `0xda` set buffer offset (+ optional attribute stride)
//! - `0xce` set textures (type-2/3 GVA + type-11; sample vs storage via reflection)
//! - `0xcc` / `0xcd` set samplers (+ optional LOD clamp)
//! - `0xd1` direct stage-in region / `0xd2` indirect stage-in region (guest buffer args)
//! - `0xd3` threadgroup memory length
//! - `0xd8` imageblock dimensions
//! - `0xc8`/`0xca` direct dispatch; `0xc9`/`0xe6` indirect (guest args → direct encode)
//! - `0xdb` dispatch type (serial/concurrent)
//!
//! Fences: stream walk (`fence_exec`). Control-flow (`0xdc`–`0xe2`) encodes
//! host Metal SPI on a multi-record [`crate::runtime::compute_session`] (same
//! encoder for the segment). ICB (`0xe4`/`0xe5`) materializes type-7 `0x36` and
//! executes filled host command slots (CPU fill via [`crate::runtime::icb`];
//! stream fill opcodes remain unknown). Nested dispatches on an open session
//! encode onto that encoder (inside SPI); writeback runs after session commit.
//! Barriers and compressed-texture flush are ordered no-ops.
//!
//! One-shot encode uses [`crate::backend::metal::compute::compute_core`]; nested
//! encode uses `compute_encode_on_encoder`. Buffer and storage-image writeback
//! is GVA / type-11 staged.

use crate::contract::endian::ld32;
use crate::contract::pixel_format;
use crate::model::DeviceState;
use crate::runtime::decode::compute::{
    BufferBinding, Command as ComputeCommand, Kind, RefBinding, SamplerBinding,
};
use crate::runtime::decode::resource::{
    decode_function_descriptor, decode_heap_texture, decode_texture_descriptor,
    decode_type7_descriptor, texture_type8_opcode, ComputeStageInputDescriptor,
    Descriptor as ResourceDescriptor, HEAP_TEXTURE_OPCODE, HEAP_TEXTURE_WIDE_OPCODE,
    OBJECT_TYPE_BUFFER, OBJECT_TYPE_FUNCTION, OBJECT_TYPE_TEXTURE, OBJECT_TYPE_TEXTURE_VARIANT,
    OBJECT_TYPE_TEXTURE_VIEW, OBJECT_TYPE_TYPE7, TEXTURE_VIEW_OPCODE_BUFFER_TEXTURE,
    TEXTURE_VIEW_OPCODE_BUFFER_TEXTURE_WIDE,
};
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
use crate::runtime::decode::resource::{decode_sampler_descriptor, TYPE7_OBJECT_SAMPLER};
use crate::runtime::gva_mem;
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::mapper;
use crate::runtime::mapping_write;
use crate::runtime::metal_draw::host_alloc_len;
use crate::runtime::objects;

/// Cap on Metal compute buffer slots (matches backend `REIMS_VGPU_METAL_MAX_BUFFERS`).
pub const MAX_COMPUTE_BUFFER_SLOTS: u32 = 31;
/// Cap on compute texture stream indices (Metal bind = 32 + index).
pub const MAX_COMPUTE_TEXTURE_SLOTS: u32 = 31;
/// Cap on compute sampler stream indices (Metal bind = 64 + index).
pub const MAX_COMPUTE_SAMPLER_SLOTS: u32 = 16;
/// Cap on threadgroup-memory indices (plan `REIMS_VGPU_COMPUTE_PLAN_MAX_THREADGROUP_MEMORY`).
pub const MAX_THREADGROUP_MEMORY_SLOTS: u32 = 16;
/// `MTLDispatchThreadgroupsIndirectArguments` = three `uint32_t` (12 bytes).
pub const INDIRECT_THREADGROUPS_ARGS_LEN: usize = 12;
/// `MTLDispatchThreadsIndirectArguments` = six `uint32_t` (24 bytes).
pub const INDIRECT_THREADS_ARGS_LEN: usize = 24;
/// `MTLStageInRegionIndirectArguments` = six `uint32_t` (24 bytes).
pub const STAGE_IN_INDIRECT_ARGS_LEN: usize = 24;

/// Fail-visible, deduped record of a compute resource bind dropped because its
/// slot index exceeds the argument-table cap. The guest bound a real resource
/// (`ref != 0`, or a non-empty threadgroup allocation) at a slot we cannot
/// represent, so the dispatch runs *missing that bind* — wrong compute output
/// with no other symptom, previously silent. Runs on the drain worker (off the
/// QEMU main core). Deduped per `(table, index)` so a repeating dispatch cannot
/// flood, and a healthy guest — which binds within the Metal argument-table caps —
/// never fires it. The cap comparison is exclusive (`index >= MAX_*`) to match the
/// backend, which sizes its argument-table arrays to exactly these counts
/// (`[false; REIMS_VGPU_METAL_MAX_BUFFERS]`) and guards `idx >= REIMS_VGPU_METAL_MAX_*` before
/// indexing — so slot `MAX` is out of range and a bind there is a genuine drop, not
/// a boundary the accum should have accepted.
fn note_compute_bind_overflow(table: &'static str, index: u32, resource_ref: u32, cap: u32) {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<(&'static str, u32)>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    if seen
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert((table, index))
    {
        crate::observe::fail(format!(
            "compute_bind_overflow reason={table}_index_overflow index={index} \
             arg={resource_ref} cap={cap} (bind dropped; dispatch runs without it)"
        ));
    }
}

#[derive(Clone, Debug, Default)]
pub struct ComputeBufferBind {
    pub index: u32,
    pub buffer_ref: u32,
    pub offset: u64,
    pub attribute_stride: u64,
    pub has_attribute_stride: bool,
}

#[derive(Clone, Debug, Default)]
pub struct ComputeTextureBind {
    /// Stream texture index (`0xce first + i`); Metal bind = 32 + index.
    pub index: u32,
    pub texture_ref: u32,
}

#[derive(Clone, Debug, Default)]
pub struct ComputeSamplerBind {
    /// Stream sampler index; Metal bind = 64 + index.
    pub index: u32,
    pub sampler_ref: u32,
    pub lod_min_bits: u32,
    pub lod_max_bits: u32,
    pub has_lod_clamp: bool,
}

#[derive(Clone, Debug, Default)]
pub struct ThreadgroupMemoryBind {
    pub index: u32,
    pub length: u64,
}

#[derive(Clone, Debug, Default)]
pub struct StageInRegion {
    pub origin_x: u64,
    pub origin_y: u64,
    pub origin_z: u64,
    pub size_x: u64,
    pub size_y: u64,
    pub size_z: u64,
}

#[derive(Clone, Debug, Default)]
pub struct StageInRegionIndirect {
    pub buffer_ref: u32,
    pub buffer_offset: u64,
}

#[derive(Clone, Debug, Default)]
pub struct ImageblockDimensions {
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug, Default)]
pub struct ComputeAccum {
    pub pipeline_ref: u32,
    pub buffers: Vec<ComputeBufferBind>,
    pub textures: Vec<ComputeTextureBind>,
    pub samplers: Vec<ComputeSamplerBind>,
    pub threadgroup_memory: Vec<ThreadgroupMemoryBind>,
    /// Last direct `0xd1` stage-in region (cleared by `0xd2`).
    pub stage_in_region: Option<StageInRegion>,
    /// Last `0xd2` indirect stage-in (clears direct region).
    pub stage_in_region_indirect: Option<StageInRegionIndirect>,
    /// Last `0xd8` imageblock dimensions.
    pub imageblock: Option<ImageblockDimensions>,
    /// Last decoded `0xdb` dispatch type (Metal serial/concurrent); 0 = serial.
    pub dispatch_type: u32,
}

impl ComputeAccum {
    pub fn set_pipeline(&mut self, pipeline_ref: u32) {
        if pipeline_ref != 0 {
            self.pipeline_ref = pipeline_ref;
        }
    }

    pub fn bind_buffers(&mut self, first: u32, entries: &[BufferBinding]) {
        for (i, e) in entries.iter().enumerate() {
            let index = first.saturating_add(i as u32);
            if e.ref_ == 0 {
                // A nil entry clears the slot. Retaining the previous bind
                // instead is not a stale read but a write: the retained buffer
                // is staged again on the next dispatch, and reflection calling
                // it writable sends the dispatch's output back into a guest
                // resource the guest explicitly unbound. Same rule the render
                // rail states on `ExecResult::buffer_unbinds` and applies in
                // `exec::apply_binds`, over the same wire form.
                self.buffers.retain(|b| b.index != index);
                crate::runtime::drain::note_store_route("compute_unbind_buffer");
                continue;
            }
            if index >= MAX_COMPUTE_BUFFER_SLOTS {
                note_compute_bind_overflow("buffer", index, e.ref_, MAX_COMPUTE_BUFFER_SLOTS);
                continue;
            }
            let bind = ComputeBufferBind {
                index,
                buffer_ref: e.ref_,
                offset: e.offset,
                attribute_stride: e.attribute_stride,
                has_attribute_stride: e.has_attribute_stride,
            };
            if let Some(slot) = self.buffers.iter_mut().find(|b| b.index == index) {
                *slot = bind;
            } else {
                self.buffers.push(bind);
            }
        }
    }

    pub fn set_buffer_offset(&mut self, index: u32, offset: u64, attribute_stride: Option<u64>) {
        if let Some(slot) = self.buffers.iter_mut().find(|b| b.index == index) {
            slot.offset = offset;
            if let Some(s) = attribute_stride {
                slot.attribute_stride = s;
                slot.has_attribute_stride = true;
            }
        }
    }

    pub fn bind_textures(&mut self, first: u32, entries: &[RefBinding]) {
        for (i, e) in entries.iter().enumerate() {
            let index = first.saturating_add(i as u32);
            if e.ref_ == 0 {
                // Clears the slot; see `bind_buffers`. A retained texture is
                // the sharper case of the two, because `writeback_texture`
                // lands the dispatch's result in the guest surface behind it.
                self.textures.retain(|t| t.index != index);
                crate::runtime::drain::note_store_route("compute_unbind_texture");
                continue;
            }
            if index >= MAX_COMPUTE_TEXTURE_SLOTS {
                note_compute_bind_overflow("texture", index, e.ref_, MAX_COMPUTE_TEXTURE_SLOTS);
                continue;
            }
            let bind = ComputeTextureBind {
                index,
                texture_ref: e.ref_,
            };
            if let Some(slot) = self.textures.iter_mut().find(|t| t.index == index) {
                *slot = bind;
            } else {
                self.textures.push(bind);
            }
        }
    }

    pub fn bind_samplers(&mut self, first: u32, entries: &[SamplerBinding]) {
        for (i, e) in entries.iter().enumerate() {
            let index = first.saturating_add(i as u32);
            if e.ref_ == 0 {
                // Clears the slot; see `bind_buffers`.
                self.samplers.retain(|s| s.index != index);
                crate::runtime::drain::note_store_route("compute_unbind_sampler");
                continue;
            }
            if index >= MAX_COMPUTE_SAMPLER_SLOTS {
                note_compute_bind_overflow("sampler", index, e.ref_, MAX_COMPUTE_SAMPLER_SLOTS);
                continue;
            }
            let bind = ComputeSamplerBind {
                index,
                sampler_ref: e.ref_,
                lod_min_bits: e.lod_min_bits,
                lod_max_bits: e.lod_max_bits,
                has_lod_clamp: e.has_lod_clamp,
            };
            if let Some(slot) = self.samplers.iter_mut().find(|s| s.index == index) {
                *slot = bind;
            } else {
                self.samplers.push(bind);
            }
        }
    }

    pub fn set_threadgroup_memory(&mut self, index: u32, length: u64) {
        if index >= MAX_THREADGROUP_MEMORY_SLOTS {
            // A non-empty allocation at an over-cap slot is a genuine dropped bind
            // (the kernel expects threadgroup memory here); a zero length is an
            // unbind, expected control flow. `arg` carries the requested length.
            if length != 0 {
                note_compute_bind_overflow(
                    "threadgroup",
                    index,
                    length.min(u32::MAX as u64) as u32,
                    MAX_THREADGROUP_MEMORY_SLOTS,
                );
            }
            return;
        }
        let bind = ThreadgroupMemoryBind { index, length };
        if let Some(slot) = self
            .threadgroup_memory
            .iter_mut()
            .find(|t| t.index == index)
        {
            *slot = bind;
        } else {
            self.threadgroup_memory.push(bind);
        }
    }

    pub fn set_stage_in_region(&mut self, region: StageInRegion) {
        self.stage_in_region_indirect = None;
        self.stage_in_region = Some(region);
    }

    pub fn set_stage_in_region_indirect(&mut self, buffer_ref: u32, buffer_offset: u64) {
        if buffer_ref == 0 {
            return;
        }
        self.stage_in_region = None;
        self.stage_in_region_indirect = Some(StageInRegionIndirect {
            buffer_ref,
            buffer_offset,
        });
    }

    pub fn set_imageblock(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.imageblock = Some(ImageblockDimensions { width, height });
    }
}

/// The compute rail's refusal vocabulary.
///
/// Every refusing variant carries the **registered slug of the check that
/// refused**, not just its class. Before that payload existed, nine of these
/// variants were payload-free and 129 construction sites collapsed into them —
/// `MetalFailed` alone spoke for 38 checks, `MissingTexture` for 25 — so a live
/// `compute_dispatches_fail` counter told you a dispatch died and nothing else.
/// The slug is what makes the class greppable; the class is what decides the
/// caller's recovery.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComputeStatus {
    Ok,
    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    MetalBackend(crate::backend::metal::error::Status),
    MissingPipeline(&'static str),
    MissingMtlb(&'static str),
    MissingBuffer(&'static str),
    MissingTexture(&'static str),
    MissingSampler(&'static str),
    BadGrid(&'static str),
    GuestIo(&'static str),
    MetalFailed(&'static str),
    NoMetal(&'static str),
    Unsupported(&'static str),
}

impl crate::observe::Refusal for ComputeStatus {
    fn refusal(&self) -> Option<&'static str> {
        match self {
            // The only non-refusal. Keeping it in the same enum is what makes
            // `Emit::refusal` unable to log a success by accident.
            Self::Ok => None,
            #[cfg(all(feature = "backend-metal", target_os = "macos"))]
            Self::MetalBackend(status) => status.refusal(),
            Self::MissingPipeline(slug)
            | Self::MissingMtlb(slug)
            | Self::MissingBuffer(slug)
            | Self::MissingTexture(slug)
            | Self::MissingSampler(slug)
            | Self::BadGrid(slug)
            | Self::GuestIo(slug)
            | Self::MetalFailed(slug)
            | Self::NoMetal(slug)
            | Self::Unsupported(slug) => Some(slug),
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        // The class next to the reason: `MissingTexture` vs `MetalFailed` is
        // what the caller acted on, and a reader correlating a log line with a
        // recovery path needs both.
        #[cfg(all(feature = "backend-metal", target_os = "macos"))]
        if let Self::MetalBackend(status) = self {
            let mut fields = crate::observe::Refusal::fields(status);
            fields.push(("recovery", "metal_failed".to_string()));
            return fields;
        }
        vec![("class", self.class().to_string())]
    }
}

impl ComputeStatus {
    /// The variant name, for the `class=` field and for call sites that render
    /// their own line.
    pub fn class(&self) -> &'static str {
        match self {
            Self::Ok => "ok",
            #[cfg(all(feature = "backend-metal", target_os = "macos"))]
            Self::MetalBackend(status) => {
                if status.is_args() {
                    "metal_args"
                } else {
                    "metal_execute"
                }
            }
            Self::MissingPipeline(_) => "missing_pipeline",
            Self::MissingMtlb(_) => "missing_mtlb",
            Self::MissingBuffer(_) => "missing_buffer",
            Self::MissingTexture(_) => "missing_texture",
            Self::MissingSampler(_) => "missing_sampler",
            Self::BadGrid(_) => "bad_grid",
            Self::GuestIo(_) => "guest_io",
            Self::MetalFailed(_) => "metal_failed",
            Self::NoMetal(_) => "no_metal",
            Self::Unsupported(_) => "unsupported",
        }
    }

    /// The registered slug this status carries, or `"ok"` when it is not a
    /// refusal. For sites that render a `reason=` into a longer line of their
    /// own rather than building one with [`crate::observe::Emit`].
    pub fn reason(&self) -> &'static str {
        use crate::observe::Refusal as _;
        self.refusal().unwrap_or("ok")
    }
}

/// A malformed translated kernel module before descriptor reflection/execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComputeSpirvDecline {
    HeaderTooShort { len: usize, minimum: usize },
    LengthMisaligned { len: usize, alignment: usize },
}

impl crate::observe::Decline for ComputeSpirvDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::HeaderTooShort { .. } => "compute_spirv_header_too_short",
            Self::LengthMisaligned { .. } => "compute_spirv_length_misaligned",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::HeaderTooShort { len, minimum } => {
                vec![("len", len.to_string()), ("minimum", minimum.to_string())]
            }
            Self::LengthMisaligned { len, alignment } => vec![
                ("len", len.to_string()),
                ("alignment", alignment.to_string()),
            ],
        }
    }
}

crate::observe::decline_display!(ComputeSpirvDecline);

impl std::error::Error for ComputeSpirvDecline {}

/// Apply one decoded compute command to accum, or run a dispatch / sequencing op.
///
/// `seg` carries the whole segment's mutable state: the accum this record
/// updates, the multi-record encoder a dispatch encodes onto when one is open,
/// and the latched sequencing failure (ICB / control encode error) that refuses
/// later dispatches.
pub fn apply_record<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    cmd: &ComputeCommand,
    seg: &mut crate::runtime::compute_session::ComputeSegment,
) -> Option<ComputeStatus> {
    let started = std::time::Instant::now();
    let out = apply_record_inner(state, host, task_id, cmd, seg);
    crate::runtime::drain::note_drain_phase(crate::runtime::drain::DrainPhase::Compute, started);
    out
}

fn apply_record_inner<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    cmd: &ComputeCommand,
    seg: &mut crate::runtime::compute_session::ComputeSegment,
) -> Option<ComputeStatus> {
    match cmd.kind {
        Kind::Pipeline => {
            seg.acc.set_pipeline(cmd.pipeline_ref);
            None
        }
        Kind::BufferBind | Kind::BufferBindAttributeStride => {
            seg.acc.bind_buffers(cmd.first, &cmd.buffers);
            None
        }
        Kind::BufferOffset => {
            seg.acc
                .set_buffer_offset(cmd.first, cmd.buffer_offset, None);
            None
        }
        Kind::BufferOffsetAttributeStride => {
            seg.acc
                .set_buffer_offset(cmd.first, cmd.buffer_offset, Some(cmd.attribute_stride));
            None
        }
        Kind::TextureBind => {
            seg.acc.bind_textures(cmd.first, &cmd.textures);
            None
        }
        Kind::SamplerBind | Kind::SamplerLod => {
            seg.acc.bind_samplers(cmd.first, &cmd.samplers);
            None
        }
        Kind::DispatchType => {
            seg.acc.dispatch_type = cmd.dispatch_type;
            None
        }
        Kind::StageInRegion => {
            seg.acc.set_stage_in_region(StageInRegion {
                origin_x: cmd.stage_in_region.origin.x,
                origin_y: cmd.stage_in_region.origin.y,
                origin_z: cmd.stage_in_region.origin.z,
                size_x: cmd.stage_in_region.size.x,
                size_y: cmd.stage_in_region.size.y,
                size_z: cmd.stage_in_region.size.z,
            });
            None
        }
        Kind::StageInRegionIndirect => {
            seg.acc.set_stage_in_region_indirect(
                cmd.stage_in_indirect_buffer_ref,
                cmd.stage_in_indirect_buffer_offset,
            );
            None
        }
        Kind::ThreadgroupMemory => {
            seg.acc.set_threadgroup_memory(
                cmd.threadgroup_memory_index,
                cmd.threadgroup_memory_length,
            );
            None
        }
        Kind::ImageblockDimensions => {
            seg.acc
                .set_imageblock(cmd.imageblock_width, cmd.imageblock_height);
            None
        }
        Kind::DispatchThreadgroups
        | Kind::DispatchThreads
        | Kind::DispatchThreadgroupsIndirect
        | Kind::DispatchThreadsIndirect => {
            if seg.block.is_some() {
                return Some(ComputeStatus::Unsupported("dispatch_in_sequencing_block"));
            }
            // Open multi-record session (control-flow SPI): encode on that encoder.
            if let Some(sess) = seg.session.as_mut() {
                return Some(execute_dispatch_nested(
                    state, host, task_id, &seg.acc, cmd, sess,
                ));
            }
            Some(execute_dispatch(state, host, task_id, &seg.acc, cmd))
        }
        // Five kinds the product answers by doing nothing, each counted
        // separately. `None` here is also what every state-accumulating record
        // above returns, so a no-op and a drop are the same silence — and these
        // are the records where the difference matters, because unlike a
        // `BufferBind` they carry ordering the guest expects us to honour.
        //
        // The barrier group is a deliberate no-op and the reason is structural:
        // the product submits one dispatch at a time and waits, so every
        // resource and scope barrier the guest asks for is already implied by
        // the boundary between two records. `UseHeaps`/`UseResources` are
        // residency hints for a driver that pages resources; we resolve every
        // binding per dispatch, so there is nothing for them to keep resident.
        // These counters exist to price that argument, not to doubt it — if
        // they are large, the per-record submit is what they are the cost of.
        //
        // That argument is load-bearing in a way it did not look, and the
        // capture now says how much traffic rests on it. Under
        // `-setSupportsComputePassDescriptorDispatchType:` Apple's serializer
        // emits a scope barrier — `0xd7`, `Buffers|Textures` — after **every**
        // dispatch and every ICB execution of a serial pass, measured on all six
        // selectors (`reims_vgpu_wire::ops::compute::OPCODE_MEMORY_BARRIER_SCOPE`, and
        // `reims_vgpu_wire::ops::compute::MemoryBarrierScope` carries the
        // derivation). So a guest that negotiates that flag doubles this rail's
        // record count and every second record lands here. The no-op stays
        // right, and on the Vulkan arm it is stronger than "pass granularity":
        // `backend::vulkan::engine::exec_compute::execute_compute_inner` begins,
        // ends and submits one command buffer per dispatch, so consecutive
        // dispatches are separated by a queue submission rather than by a
        // barrier inside one. `compute_noop_barrier` reading high is that
        // capability being on, not a defect.
        //
        // The fence pair has no such argument and never had one; it sat in the
        // barrier group's arm without sharing its comment. An `MTLFence` update
        // or wait inside a compute encoder is ordering the guest stated
        // explicitly, and nothing else in the crate handles these two kinds —
        // `fence_exec` serves the event rail, not this one. If either counter
        // is non-zero, that is guest-stated ordering we are discarding, and it
        // wants a contract answer rather than another counter.
        Kind::UpdateFence => {
            crate::runtime::drain::note_store_route("compute_noop_update_fence");
            None
        }
        Kind::WaitFence => {
            crate::runtime::drain::note_store_route("compute_noop_wait_fence");
            None
        }
        Kind::BarrierResources | Kind::BarrierScope => {
            crate::runtime::drain::note_store_route("compute_noop_barrier");
            None
        }
        Kind::UseHeaps | Kind::UseResources => {
            crate::runtime::drain::note_store_route("compute_noop_residency_hint");
            None
        }
        Kind::CompressedTextureFlush => {
            crate::runtime::drain::note_store_route("compute_noop_compressed_flush");
            None
        }
        Kind::ControlStartDoWhile
        | Kind::ControlEndDoWhile
        | Kind::ControlStartWhile
        | Kind::ControlEndWhile
        | Kind::ControlStartIf
        | Kind::ControlStartElse
        | Kind::ControlEndIf
        | Kind::ExecuteCommandsInBuffer
        | Kind::ExecuteCommandsInBufferIndirect => Some(
            crate::runtime::compute_session::apply_sequencing(state, host, task_id, cmd, seg),
        ),
        Kind::Unknown => None,
    }
}

pub(crate) fn load_mtlb<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    func_ref: u32,
) -> Option<Vec<u8>> {
    // ref==0 is "no function bound" (legitimate, e.g. no fragment stage) — stay
    // silent. Every other None is a bound function that failed to materialize,
    // collapsing into the caller's coarse MissingMtlb; log the reason (audit).
    if func_ref == 0 {
        return None;
    }
    let miss = |reason: &str, detail: String| -> Option<Vec<u8>> {
        crate::observe::fail(format!(
            "compute_load_mtlb fail reason={reason} task={task_id} func_ref={func_ref} {detail}"
        ));
        None
    };
    let Some(entry) = objects::lookup_list_entry(state, host, task_id, func_ref) else {
        return miss("no_entry", String::new());
    };
    if entry.object_type != OBJECT_TYPE_FUNCTION {
        return miss("wrong_type", format!("ot={}", entry.object_type));
    }
    let Some(desc) = objects::read_descriptor(state, host, task_id, &entry) else {
        return miss("no_desc", String::new());
    };
    let Ok(f) = decode_function_descriptor(&desc) else {
        return miss("decode", format!("desc_len={}", desc.len()));
    };
    if f.blob_gva == 0 || f.blob_size < 4 {
        return miss(
            "bad_blob",
            format!("blob_gva={:#x} blob_size={}", f.blob_gva, f.blob_size),
        );
    }
    // Guest blob_size is authoritative — no product 1 MiB MTLB ceiling.
    let Some(len) = host_alloc_len(f.blob_size as u64) else {
        return miss(
            "host_len",
            format!("blob_gva={:#x} blob_size={}", f.blob_gva, f.blob_size),
        );
    };
    let mut mtlb = vec![0u8; len];
    if gva_mem::read_task_gva_by_id(
        host,
        &state.tasks,
        task_id,
        f.blob_gva,
        &mut mtlb,
        state.page_shift,
    )
    .is_err()
    {
        return miss(
            "gva_read",
            format!("blob_gva={:#x} blob_size={}", f.blob_gva, f.blob_size),
        );
    }
    Some(mtlb)
}

pub(crate) struct LoadedComputePipeline {
    pub kernel_func_ref: u32,
    /// Product-ready stage-input (None if absent, dropped caps, or incomplete).
    pub stage_input: Option<ComputeStageInputDescriptor>,
}

pub(crate) fn load_compute_pipeline<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    pipeline_ref: u32,
) -> Option<LoadedComputePipeline> {
    // ref==0 is "no pipeline bound" (legitimate) — silent. Other None = a bound
    // pipeline that failed to materialize → caller's coarse MissingPipeline; log
    // the reason (audit).
    if pipeline_ref == 0 {
        return None;
    }
    let miss = |reason: &str, detail: String| -> Option<LoadedComputePipeline> {
        crate::observe::fail(format!(
            "compute_load_pipeline fail reason={reason} task={task_id} pipe_ref={pipeline_ref} {detail}"
        ));
        None
    };
    let Some(entry) = objects::lookup_list_entry(state, host, task_id, pipeline_ref) else {
        return miss("no_entry", String::new());
    };
    if entry.object_type != OBJECT_TYPE_TYPE7 {
        return miss("wrong_type", format!("ot={}", entry.object_type));
    }
    let Some(desc) = objects::read_descriptor(state, host, task_id, &entry) else {
        return miss("no_desc", String::new());
    };
    let Ok(decoded) = decode_type7_descriptor(&desc) else {
        return miss("decode", format!("desc_len={}", desc.len()));
    };
    match decoded {
        ResourceDescriptor::ComputePipeline(cp) if cp.kernel_func_ref != 0 => {
            let stage_input = cp.stage_input.and_then(|si| {
                // Dropped entries mean the wire exceeded product/backend caps — fail closed
                // by omitting stage-input rather than silently truncating.
                if si.dropped_attributes != 0 || si.dropped_layouts != 0 {
                    return None;
                }
                if si.attributes.is_empty() && si.layouts.is_empty() {
                    return None;
                }
                Some(si)
            });
            Some(LoadedComputePipeline {
                kernel_func_ref: cp.kernel_func_ref,
                stage_input,
            })
        }
        ResourceDescriptor::ComputePipeline(_) => miss("kernel_func_zero", String::new()),
        _ => miss("not_compute_pipeline", String::new()),
    }
}

/// Resolve a type-1 buffer GVA base + size for task-local reads.
fn buffer_gva_size<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    buffer_ref: u32,
) -> Option<(u64, u64)> {
    if buffer_ref == 0 {
        return None;
    }
    let entry = objects::lookup_list_entry(state, host, task_id, buffer_ref)?;
    if entry.object_type != OBJECT_TYPE_BUFFER {
        return None;
    }
    let desc_bytes = objects::read_descriptor(state, host, task_id, &entry)?;
    let desc = crate::runtime::decode::resource::decode_buffer_descriptor(&desc_bytes).ok()?;
    // Product x86 page_shift=12; arm64e=14. Never use arm-only RESOURCE_PAGE_SHIFT
    // default on the live path (compute GuestIo Unmapped class serial-234118).
    desc.backing_gva_size(state.page_shift)
}

/// Read `len` bytes from a type-1 buffer at `offset` (product + session helpers).
pub(crate) fn read_buffer_window<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    buffer_ref: u32,
    offset: u64,
    len: usize,
) -> Result<Vec<u8>, ComputeStatus> {
    let (base, size) = buffer_gva_size(state, host, task_id, buffer_ref)
        .ok_or(ComputeStatus::MissingBuffer("compute_buf_win_no_backing"))?;
    if offset
        .checked_add(len as u64)
        .map(|e| e > size)
        .unwrap_or(true)
    {
        return Err(ComputeStatus::MissingBuffer("compute_buf_win_oob"));
    }
    let gva = base
        .checked_add(offset)
        .ok_or(ComputeStatus::MissingBuffer("compute_buf_win_gva_overflow"))?;
    let mut bytes = vec![0u8; len];
    gva_mem::read_task_gva_by_id(
        host,
        &state.tasks,
        task_id,
        gva,
        &mut bytes,
        state.page_shift,
    )
    .map_err(|_| ComputeStatus::GuestIo("compute_buf_win_read"))?;
    Ok(bytes)
}

pub(crate) struct StagedBuffer {
    pub bind: ComputeBufferBind,
    pub gva: u64,
    pub bytes: Vec<u8>,
    /// Guest pages this buffer resolved to when it was staged — before the
    /// dispatch, and before a nested session accumulated however many more
    /// jobs before flushing. `writeback_buffer` runs at the far end of that
    /// gap, so a walk taken there answers where the address points now rather
    /// than whether it is still this buffer's memory. Empty when the
    /// stage-time walk resolved nothing, which leaves the write unbounded as
    /// it was; the writer's own walk then fails closed on its own terms.
    pub pages: std::collections::HashSet<u64>,
}

pub(crate) fn stage_buffer<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    bind: &ComputeBufferBind,
) -> Result<StagedBuffer, ComputeStatus> {
    // Eight distinct checks answer with `MissingBuffer`; the status carries
    // which one, so the caller's line and this one name the same slug.
    let miss = |st: ComputeStatus, detail: String| -> Result<StagedBuffer, ComputeStatus> {
        crate::observe::fail(format!(
            "compute_stage_buf fail reason={} ref={} off={:#x} {detail}",
            st.reason(),
            bind.buffer_ref,
            bind.offset
        ));
        Err(st)
    };
    let Some(entry) = objects::lookup_list_entry(state, host, task_id, bind.buffer_ref) else {
        return miss(
            ComputeStatus::MissingBuffer("compute_stage_buf_no_entry"),
            String::new(),
        );
    };
    if entry.object_type != OBJECT_TYPE_BUFFER {
        return miss(
            ComputeStatus::MissingBuffer("compute_stage_buf_wrong_type"),
            format!("ot={}", entry.object_type),
        );
    }
    let Some(desc_bytes) = objects::read_descriptor(state, host, task_id, &entry) else {
        return miss(
            ComputeStatus::MissingBuffer("compute_stage_buf_no_desc"),
            String::new(),
        );
    };
    let Ok(desc) = crate::runtime::decode::resource::decode_buffer_descriptor(&desc_bytes) else {
        return miss(
            ComputeStatus::MissingBuffer("compute_stage_buf_decode"),
            format!("desc_len={}", desc_bytes.len()),
        );
    };
    // Device page_shift (x86=12): handle<<shift is the guest VA. Using the arm
    // default (14) mis-places buffers → walker Unmapped (live compute GuestIo).
    let Some((base_gva, size)) = desc.backing_gva_size(state.page_shift) else {
        return miss(
            ComputeStatus::MissingBuffer("compute_stage_buf_no_backing"),
            format!("handle={:#x}", desc.handle),
        );
    };
    if bind.offset >= size {
        return miss(
            ComputeStatus::MissingBuffer("compute_stage_buf_off_oob"),
            format!("size={size:#x}"),
        );
    }
    let avail = size - bind.offset;
    let Some(want) = host_alloc_len(avail).filter(|&n| n > 0) else {
        return miss(
            ComputeStatus::MissingBuffer("compute_stage_buf_want_bad"),
            format!("size={size:#x} avail={avail:#x}"),
        );
    };
    let Some(gva) = base_gva.checked_add(bind.offset) else {
        return miss(
            ComputeStatus::MissingBuffer("compute_stage_buf_gva_overflow"),
            format!("base={base_gva:#x} size={size:#x}"),
        );
    };
    let mut bytes = vec![0u8; want];
    if let Err(e) = gva_mem::read_task_gva_by_id(
        host,
        &state.tasks,
        task_id,
        gva,
        &mut bytes,
        state.page_shift,
    ) {
        // Full walk diagnosis on one line — max learn from a single product boot.
        let walk = gva_mem::diagnose_gva_walk(host, &state.tasks, task_id, gva, state.page_shift);
        // Also probe object base (no offset) in case only the offset page fails.
        let base_walk = if gva != base_gva {
            gva_mem::diagnose_gva_walk(host, &state.tasks, task_id, base_gva, state.page_shift)
        } else {
            String::new()
        };
        crate::observe::fail(format!(
            "compute_stage_buf_gva task={task_id} ref={} base={base_gva:#x} off={:#x} gva={gva:#x} want={want} size={size:#x} page_shift={} err={e:?} | {walk}{}",
            bind.buffer_ref,
            bind.offset,
            state.page_shift,
            if base_walk.is_empty() {
                String::new()
            } else {
                format!(" | base_walk {base_walk}")
            }
        ));
        return Err(ComputeStatus::GuestIo("compute_stage_buf_gva_read"));
    }
    let pages = staged_span_pages(state, host, task_id, gva, bytes.len() as u64);
    Ok(StagedBuffer {
        bind: bind.clone(),
        gva,
        bytes,
        pages,
    })
}

enum TextureWriteback {
    None,
    Linear {
        texture_ref: u32,
        gva: u64,
        pixel_format: u16,
        row_stride: u64,
        width: u32,
        height: u32,
        bpp: u32,
        /// Guest pages this window resolved to when the texture was staged,
        /// i.e. **before** the dispatch that produces the bytes.
        ///
        /// `writeback_texture` runs after the GPU has finished, and the guest
        /// runs on its own vCPUs across that gap; a walk taken then answers
        /// where the address points *now*, which is a different question from
        /// whether it is still this texture's memory. Empty when the stage-time
        /// walk resolved nothing, which leaves the write unbounded exactly as
        /// it was — the writer's own walk fails closed on its own terms.
        pages: std::collections::HashSet<u64>,
    },
    Type11 {
        mapping_id: u32,
        surface_offset: u64,
        surface_bpr: u32,
        span_end: u64,
        width: u32,
        height: u32,
        bpp: u32,
    },
}

/// Guest pages a linear storage window resolves to at stage time.
///
/// Taken before the dispatch so the set names the memory the *command* was
/// issued against, not whatever the address points at once the GPU is done.
/// An empty set means the walk resolved nothing and the writeback stays
/// unbounded, which is what it was before this existed.
fn staged_window_pages<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    gva: u64,
    row_stride: u64,
    height: u32,
) -> std::collections::HashSet<u64> {
    let Some(span) = row_stride.checked_mul(height as u64) else {
        return std::collections::HashSet::new();
    };
    staged_span_pages(state, host, task_id, gva, span)
}

/// [`staged_window_pages`] for a flat span — the buffer rail's shape.
fn staged_span_pages<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    gva: u64,
    span: u64,
) -> std::collections::HashSet<u64> {
    let mut pages = std::collections::HashSet::new();
    if gva == 0 || span == 0 {
        return pages;
    }
    pages.extend(gva_mem::task_gva_page_gpa_set(
        host,
        &state.tasks,
        task_id,
        gva,
        span,
        state.page_shift,
    ));
    pages
}

pub(crate) struct StagedTexture {
    pub binding: u32,
    /// Raw Metal pixel format from the exact texture/view descriptor.
    pub pixel_format: u16,
    /// Product storage-selector ABI when this Metal format is storage-capable.
    /// Sample-only formats such as RGB9E5Float intentionally have no selector.
    pub storage_selector: Option<u32>,
    pub width: u32,
    pub height: u32,
    pub bytes: Vec<u8>,
    pub is_storage: bool,
    #[cfg(feature = "backend-vulkan")]
    residency: Option<ComputeStorageResidencyCandidate>,
    /// Stage-time guest read skipped (resident generation verified); `bytes`
    /// is a zero placeholder the engine must never seed.
    #[cfg(feature = "backend-vulkan")]
    seed_skipped: bool,
    /// Sampled input whose window the engine already holds GPU-resident (a
    /// prior dispatch's storage output at this generation): the guest read was
    /// skipped, `bytes` is a zero placeholder, and the engine must seed the
    /// sampled image by copy-on-sample from the resident (never the bytes).
    #[cfg(feature = "backend-vulkan")]
    sample_resident: Option<(crate::model::ComputeStorageResidencyKey, u32)>,
    writeback: TextureWriteback,
}

#[cfg(feature = "backend-vulkan")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ComputeStorageResidencyCandidate {
    key: crate::model::ComputeStorageResidencyKey,
    seed_generation: u32,
}

/// Deferred-readback policy for a compute storage output. Deferring keeps the
/// dispatch result GPU-resident-only (guest pages stale until a flush choke
/// point), which is only a safe authority when the device grants
/// `deferred_gpu_only_content` — portability-subset (MoltenVK) devices must
/// write guest pages synchronously instead.
#[cfg(feature = "backend-vulkan")]
fn compute_defer_readback_allowed(
    deferred_gpu_only_content: bool,
    has_residency: bool,
    writeback_deferrable: bool,
) -> bool {
    deferred_gpu_only_content && has_residency && writeback_deferrable
}

/// Bound on mirror entries per mapping: a ping-pong canvas needs 2, planar
/// layouts a few more; anything beyond is assumed to be stale-key debris.
///
/// **The 8 is not derived, and the eviction below is the only thing standing
/// between this map and unbounded growth.** If every stale key were already
/// invalidated — `invalidate_storage_residency_window` runs on every overlap,
/// and every guest-page writer calls it — no cap would be needed at all, and
/// this would be a mechanism covering for an incomplete invalidation rather
/// than a bound. Which of those it is has never been measured, because the
/// eviction was silent.
///
/// `compute_mirror_evicted` is that measurement, and its **first reading is
/// zero**: one driven x86/Vulkan boot — Chess, Maps, the WebGL aquarium,
/// Wikipedia and apple.com, with page-downs and title-bar drags — evicted
/// nothing. So the cap does not bind on this workload and is a runaway guard,
/// not a working policy.
///
/// That is one boot and one workload, which is not enough to delete a guard
/// that is the only bound on this map. What would be: the same zero across a
/// boot that drives multiplanar video and several ping-pong canvases at once,
/// which is the case the "planar layouts a few more" guess was aimed at.
///
/// How close the population came was measured separately, and reads **2**
/// across every boot in a 72 MB accumulated log. That is exactly the ping-pong
/// canvas this doc predicted needs 2, so the shape of the guess is confirmed
/// while the number is not: 8 is 4x the observed high-water mark, and nothing
/// has yet produced the "planar layouts a few more" case that chose it.
#[cfg(feature = "backend-vulkan")]
const STORAGE_RESIDENCY_WINDOWS_PER_MAPPING: usize = 8;

#[cfg(feature = "backend-vulkan")]
fn note_storage_residency_writeback(state: &mut DeviceState, texture: &StagedTexture) {
    let Some(candidate) = texture.residency else {
        return;
    };
    // Linear windows keep their authority in the host_linear_textures entry
    // (resident_gen), never in the mapping-keyed mirror.
    if candidate.key.is_linear() {
        return;
    }
    if candidate.key.is_heap() {
        state.compute_storage_residency.insert(
            candidate.key,
            next_mapping_content_generation(candidate.seed_generation),
        );
        return;
    }
    // The engine registered the resident at exactly next(seed_generation)
    // (ComputeStorageResidency::output_generation). The mirror must store the
    // same currency — not the mapping-level content generation — so disjoint
    // sibling-window writebacks (ping-pong canvases) cannot desync the pair.
    let generation = next_mapping_content_generation(candidate.seed_generation);
    // Drop intersecting windows (normally already gone, because the writeback
    // wrote guest pages and every guest-page writer calls the same overlap
    // invalidation — kept here as defense in depth); keep disjoint siblings
    // (ping-pong canvases) but bound the count.
    let mapping_id = candidate.key.mapping_id;
    state.invalidate_storage_residency_window(
        mapping_id,
        candidate.key.surface_offset,
        candidate.key.span_end,
    );
    let siblings: Vec<crate::model::ComputeStorageResidencyKey> = state
        .compute_storage_residency
        .keys()
        .filter(|key| key.mapping_id == mapping_id && **key != candidate.key)
        .cloned()
        .collect();
    // Counting the window inserted below, so the cap bounds the population this
    // mapping actually holds.
    let over_cap = (siblings.len() + 1).saturating_sub(STORAGE_RESIDENCY_WINDOWS_PER_MAPPING);
    for victim in siblings.iter().take(over_cap) {
        state.compute_storage_residency.remove(victim);
        // Dropping a mirror entry costs the next read of that window its
        // resident and sends it back to guest pages. That is safe, but it is
        // not free and it must not be invisible.
        crate::observe::off(format!(
            "compute_mirror_evicted mid={mapping_id} off={} end={} siblings={} cap={}",
            victim.surface_offset,
            victim.span_end,
            siblings.len(),
            STORAGE_RESIDENCY_WINDOWS_PER_MAPPING
        ));
    }
    state
        .compute_storage_residency
        .insert(candidate.key, generation);
}

#[cfg(feature = "backend-vulkan")]
fn next_mapping_content_generation(current: u32) -> u32 {
    let next = current.wrapping_add(1);
    if next == 0 {
        1
    } else {
        next
    }
}

/// Measure storage-image seed traffic by structurally reflected content access.
///
/// `write_only` is intentionally still seeded: access alone does not prove a
/// dispatch overwrites every texel. The proxy makes that retained transfer
/// cost visible while preserving partial-write semantics.
#[cfg(feature = "backend-vulkan")]
fn log_storage_image_access(pipe: u32, binding: u32, access: &str, bytes: u64) {
    crate::observe::off(format!(
        "compute_linux storage_access pipe={pipe} bind={binding} access={access} seed=1 bytes={bytes}"
    ));
}

/// What an engine-resident copy of a window can serve one staged binding.
///
/// `Seed` means a storage binding's output is already GPU-resident at this
/// generation, so the guest read that would seed it is unnecessary. `Sample`
/// names the resident key a sampled binding reads directly instead.
#[cfg(feature = "backend-vulkan")]
#[derive(Clone, Copy)]
pub(crate) enum ResidentServe {
    Seed(u32),
    Sample(crate::model::ComputeStorageResidencyKey, u32),
}

/// The gate every staging rail applies before falling back to a guest read.
///
/// `mirror_generation` is the runtime's residency mirror for `key`; the engine
/// must agree with it, because the mirror can outlive an evicted resident. A
/// sampled binding additionally needs the resident's vk format to equal the one
/// the view will bind — the engine's resident-bind path guards that equality
/// and would fail the whole request on mismatch.
///
/// `None` means the resident cannot serve this binding: the caller either reads
/// the guest window or, where the resident is the only copy, names the loss.
#[cfg(feature = "backend-vulkan")]
pub(crate) fn resident_serve(
    key: crate::model::ComputeStorageResidencyKey,
    mirror_generation: u32,
    is_storage: bool,
    pixel_format: u16,
) -> Option<ResidentServe> {
    if is_storage {
        return (crate::backend::vulkan::engine::compute_resident_storage_generation(&key)
            == Some(mirror_generation))
        .then_some(ResidentServe::Seed(mirror_generation));
    }
    let (engine_generation, engine_format) =
        crate::backend::vulkan::engine::compute_resident_sample_source(&key)?;
    (engine_generation == mirror_generation
        && mtl_to_engine_sampled(pixel_format)
            .is_some_and(|f| f.vk_format() == engine_format.vk_format()))
    .then_some(ResidentServe::Sample(key, mirror_generation))
}

/// Load tight raw texels for a compute texture binding (type-2/3, type-5→surface, or type-11).
///
/// Type-5 (`RefTextureHandle`) is the live CI wallpaper path (`compute_stage_tex … ot=5`).
/// RE (type-5 wire + metal_draw sample path): surfaceID@0 is a type-4 object id (= mapping
/// mid). Product draw samples call [`objects::ensure_surface_for_present`] on that id and
/// stage from the **mapping registry**, never re-resolving the surface id through the
/// compute task's object list (that list uses a separate texture-ref namespace — live
/// ensure=1 then MissingTexture/GuestIo class when `resolve_type11_ref(task, sid)` hit a
/// different type-11 slot).
pub(crate) fn stage_texture_raw<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    binding: u32,
    is_storage: bool,
) -> Result<StagedTexture, ComputeStatus> {
    // Type-5 RefTextureHandle → surface_id (live CI binds ot5).
    let mut stage_ref = texture_ref;
    let mut from_type5 = false;
    let mut from_type4_direct = false;
    let mut type5_record: Option<objects::Type5TextureView> = None;
    let mut view_level = 0;
    let mut view_pixel_format = None;
    let mut heap_texture = None;
    // A linear texture object (type-2/3) must resolve through its own
    // descriptor, never through the mapping registry: its numeric ref shares
    // the id space with type-4 surface mids, so the `mappings.contains(ref)`
    // fallback below would wrongly grab a same-numbered surface (live class:
    // `ref=N ot=2` dragged into the type-11 path and failing silently against
    // the biplanar wallpaper mid). Same collision the type-5 path documents.
    // Resolve the object-list entry once: `ref_is_linear` and the type5/type4
    // classification below both read it for the same ref, and the guest object
    // list is immutable for the life of the dispatch (the device never writes
    // those pages). `ListObjectEntry` is `Copy`, so one guest-DMA read+decode
    // serves both instead of two.
    let ref_entry = objects::lookup_list_entry(state, host, task_id, texture_ref);
    if let Some(entry) = ref_entry {
        if entry.object_type == OBJECT_TYPE_TEXTURE_VIEW {
            let Some(desc) = objects::read_descriptor(state, host, task_id, &entry) else {
                crate::observe::fail(format!(
                    "compute_stage_tex view_fail reason=no_desc ref={texture_ref} desc_len={}",
                    entry.descriptor_length
                ));
                return Err(ComputeStatus::MissingTexture(
                    "compute_stage_tex_view_no_desc",
                ));
            };
            let opcode = texture_type8_opcode(&desc).unwrap_or(0);
            // Both opcodes are the same record: the wide one is what the guest's
            // serializer emits with `TextureDescriptor2` on. The length each
            // implies is `decode_heap_texture`'s to check — this site used to
            // check it too, against the narrow constant alone, which would have
            // rejected every wide record before its decoder saw the opcode.
            if opcode == HEAP_TEXTURE_OPCODE || opcode == HEAP_TEXTURE_WIDE_OPCODE {
                let record = match decode_heap_texture(&desc) {
                    Ok(record) => record,
                    Err(error) => {
                        crate::observe::Emit::decline("compute_stage_tex_heap", &error)
                            .field("ref", texture_ref)
                            .field("len", desc.len())
                            .fail();
                        return Err(ComputeStatus::MissingTexture(
                            "compute_stage_tex_heap_bad_record",
                        ));
                    }
                };
                let (heap_ref, use_offset, offset) =
                    (record.heap_ref, record.use_offset, record.offset);
                if heap_ref == 0 {
                    crate::observe::fail(format!(
                        "compute_stage_tex heap_fail reason=zero_heap ref={texture_ref}"
                    ));
                    return Err(ComputeStatus::MissingTexture(
                        "compute_stage_tex_heap_zero_ref",
                    ));
                }
                let body = if record.wide {
                    crate::runtime::heap_query::decode_wide_serialized_texture_descriptor(
                        record.descriptor,
                    )
                } else {
                    crate::runtime::heap_query::decode_serialized_texture_descriptor(
                        record.descriptor,
                    )
                };
                let descriptor = match body {
                    Ok(descriptor) => descriptor,
                    Err(error) => {
                        crate::observe::Emit::decline("compute_stage_tex_heap", &error)
                            .field("ref", texture_ref)
                            .field("heap", heap_ref)
                            .field("use_offset", use_offset)
                            .field("offset", format!("{offset:#x}"))
                            .fail();
                        return Err(ComputeStatus::MissingTexture(
                            "compute_stage_tex_heap_desc_decode",
                        ));
                    }
                };
                heap_texture = Some((heap_ref, use_offset, offset, descriptor));
            }
            if heap_texture.is_some() {
                // Heap textures are complete resource objects, not texture
                // views. Their backing is a host GPU residency identity.
            } else if opcode == TEXTURE_VIEW_OPCODE_BUFFER_TEXTURE
                || opcode == TEXTURE_VIEW_OPCODE_BUFFER_TEXTURE_WIDE
            {
                crate::observe::fail(format!(
                    "compute_stage_tex view_fail reason=buffer_texture_unsupported ref={texture_ref} opcode={opcode} desc_len={}",
                    desc.len()
                ));
                return Err(ComputeStatus::Unsupported(
                    "compute_buffer_texture_unsupported",
                ));
            } else {
                let view = match crate::runtime::metal_draw::resolve_texture_view_reasoned(
                    state,
                    host,
                    task_id,
                    texture_ref,
                ) {
                    Ok(view) => view,
                    Err(reason) => {
                        crate::observe::Emit::decline("compute_stage_tex_view_resolve", &reason)
                            .field("ref", texture_ref)
                            .field("opcode", format!("{opcode:#x}"))
                            .fail_once(texture_ref as u64);
                        return Err(ComputeStatus::MissingTexture(
                            "compute_stage_tex_view_resolve",
                        ));
                    }
                };
                if view
                    .swizzle
                    .as_ref()
                    .is_some_and(|plan| !pixel_format::swizzle_is_identity(plan))
                {
                    crate::observe::fail(format!(
                        "compute_stage_tex view_fail reason=swizzle_unsupported ref={texture_ref} base={} opcode={opcode} storage={}",
                        view.base_texture_ref, is_storage as u8
                    ));
                    return Err(ComputeStatus::Unsupported(
                        "compute_view_swizzle_unsupported",
                    ));
                }
                stage_ref = view.base_texture_ref;
                view_level = view.level;
                view_pixel_format = view.pixel_format;
            }
        }
    }
    if let Some((heap_ref, use_offset, offset, descriptor)) = heap_texture {
        if descriptor.texture_type != 2
            || descriptor.depth != 1
            || descriptor.mipmap_level_count != 1
            || descriptor.sample_count != 1
            || descriptor.array_length != 1
        {
            crate::observe::fail(format!(
                "compute_stage_tex heap_fail reason=shape ref={texture_ref} heap={heap_ref} type={} dims={}x{}x{} mips={} samples={} array={} use_offset={} offset={offset:#x}",
                descriptor.texture_type,
                descriptor.width,
                descriptor.height,
                descriptor.depth,
                descriptor.mipmap_level_count,
                descriptor.sample_count,
                descriptor.array_length,
                use_offset as u8
            ));
            return Err(ComputeStatus::Unsupported("compute_heap_shape"));
        }
        let (width, height, format) =
            (descriptor.width, descriptor.height, descriptor.pixel_format);
        let Some(bpp) = pixel_format::bytes_per_pixel(format) else {
            crate::observe::fail(format!(
                "compute_stage_tex heap_fail reason=fmt_bytes ref={texture_ref} heap={heap_ref} fmt={format:#x} {width}x{height}"
            ));
            return Err(ComputeStatus::Unsupported("compute_heap_fmt_bytes"));
        };
        let storage_selector = pixel_format::storage_selector(format).map(|s| s as u32);
        if is_storage && storage_selector.is_none() {
            crate::observe::fail(format!(
                "compute_stage_tex heap_fail reason=fmt_storage ref={texture_ref} heap={heap_ref} fmt={format:#x} {width}x{height}"
            ));
            return Err(ComputeStatus::Unsupported("compute_heap_fmt_storage"));
        }
        let Some(need) = (width as usize)
            .checked_mul(height as usize)
            .and_then(|texels| texels.checked_mul(bpp as usize))
        else {
            crate::observe::fail(format!(
                "compute_stage_tex heap_fail reason=host_len ref={texture_ref} heap={heap_ref} fmt={format:#x} {width}x{height} bpp={bpp}"
            ));
            return Err(ComputeStatus::Unsupported("compute_heap_host_len"));
        };
        #[cfg(feature = "backend-vulkan")]
        let key = crate::model::ComputeStorageResidencyKey::heap(
            task_id,
            texture_ref,
            width,
            height,
            format,
        );
        // A heap texture has no guest window to re-read: once the mirror claims
        // a resident, the engine's copy is the only content, so a resident the
        // engine can no longer serve is a loss, not a fallback.
        #[cfg(feature = "backend-vulkan")]
        let (seed_generation, seed_skipped, sample_resident) = match state
            .compute_storage_residency
            .get(&key)
            .copied()
        {
            None => (0, false, None),
            Some(generation) => match resident_serve(key, generation, is_storage, format) {
                Some(ResidentServe::Seed(generation)) => (generation, true, None),
                Some(ResidentServe::Sample(key, generation)) => (0, false, Some((key, generation))),
                None => {
                    crate::observe::fail(format!(
                            "compute_stage_tex heap_fail reason=resident_lost ref={texture_ref} heap={heap_ref} fmt={format:#x} {width}x{height} gen={generation} use_offset={} offset={offset:#x}",
                            use_offset as u8
                        ));
                    return Err(ComputeStatus::MissingTexture(
                        "compute_stage_tex_heap_resident_lost",
                    ));
                }
            },
        };
        #[cfg(not(feature = "backend-vulkan"))]
        let (seed_generation, sample_resident): (
            u32,
            Option<(crate::model::ComputeStorageResidencyKey, u32)>,
        ) = (0, None);
        crate::observe::off(format!(
            "compute_stage_tex heap_ok ref={texture_ref} heap={heap_ref} fmt={format:#x} {width}x{height} storage={} seed_gen={seed_generation} resident_sample={} use_offset={} offset={offset:#x}",
            is_storage as u8,
            sample_resident.is_some() as u8,
            use_offset as u8
        ));
        return Ok(StagedTexture {
            binding,
            pixel_format: format,
            storage_selector,
            width,
            height,
            bytes: vec![0; need],
            is_storage,
            #[cfg(feature = "backend-vulkan")]
            residency: is_storage.then_some(ComputeStorageResidencyCandidate {
                key,
                seed_generation,
            }),
            #[cfg(feature = "backend-vulkan")]
            seed_skipped,
            #[cfg(feature = "backend-vulkan")]
            sample_resident,
            writeback: TextureWriteback::None,
        });
    }
    let stage_entry = objects::lookup_list_entry(state, host, task_id, stage_ref);
    let ref_is_linear = stage_entry
        .map(|e| {
            e.object_type == OBJECT_TYPE_TEXTURE || e.object_type == OBJECT_TYPE_TEXTURE_VARIANT
        })
        .unwrap_or(false);
    if let Some(entry) = stage_entry {
        if entry.object_type == objects::OBJECT_TYPE_REF_TEXTURE {
            if let Some(desc) = objects::read_descriptor(state, host, task_id, &entry) {
                if desc.len() >= objects::TYPE5_MIN_LEN {
                    let sid = ld32(&desc[objects::TYPE5_SURFACE_ID..]);
                    if sid != 0 {
                        stage_ref = sid;
                        from_type5 = true;
                        type5_record = objects::decode_type5_texture_view(&desc);
                        let ok = objects::ensure_surface_for_present(state, host, sid);
                        // Per-bind type-5 descriptor RE census (args@+8 holds the
                        // serialized plane texture; product stage uses mapping geom
                        // only today). This is measurement, not a failure — it fired
                        // ~600×/boot on the always-on sink (same descriptor re-dumped
                        // per bind, no dedup), drowning genuine failures. Verbose-gated;
                        // build the head-hex only when REIMS_VGPU_DRAW_LOG is on. A genuine
                        // ensure failure surfaces downstream as `MissingTexture` (the
                        // mapping lookup below misses), so no always-on line is lost.
                        if crate::observe::draw_log_enabled() {
                            // The owner task the view names. `note_type5_owner_task`
                            // is the always-on check on its value; this echo carries
                            // it beside the descriptor it came out of.
                            let owner_task = desc
                                .get(objects::TYPE5_OWNER_TASK..objects::TYPE5_OWNER_TASK + 4)
                                .map(ld32)
                                .unwrap_or(0);
                            let args_n = desc.len().saturating_sub(objects::TYPE5_ARGS);
                            let mut args_hex = String::new();
                            if args_n > 0 {
                                let n = args_n.min(48);
                                args_hex.reserve(n * 2);
                                for b in &desc[objects::TYPE5_ARGS..objects::TYPE5_ARGS + n] {
                                    use std::fmt::Write as _;
                                    let _ = write!(args_hex, "{b:02x}");
                                }
                                if args_n > n {
                                    args_hex.push('…');
                                }
                            }
                            crate::observe::line(format!(
                                "compute_stage_tex type5 ref={texture_ref} sid={sid} ensure={} owner_task={owner_task} desc_len={} args_n={args_n} args_hex={args_hex}",
                                ok as u8,
                                desc.len(),
                            ));
                        }
                    }
                }
            }
        } else if entry.object_type == objects::OBJECT_TYPE_SURFACE {
            // Direct type-4 surface bind (same id space as present mids).
            from_type4_direct = true;
            let _ = objects::ensure_surface_for_present(state, host, stage_ref);
        }
    }

    // Type-5 / direct type-4: surface id **is** the mapping mid. Never call
    // resolve_type11_ref(task, sid) — task object-list indices collide with texture refs.
    let mapping_id_opt = if from_type5 || from_type4_direct {
        if stage_ref != 0 && state.mappings.contains_key(&stage_ref) {
            Some(stage_ref)
        } else {
            None
        }
    } else if ref_is_linear {
        // Linear texture: never fall back to the mapping registry (id-space
        // collision with type-4 surface mids). Force the type-2/3 path.
        None
    } else {
        objects::resolve_type11_ref(state, host, task_id, stage_ref).or_else(|| {
            if stage_ref != 0 && state.mappings.contains_key(&stage_ref) {
                Some(stage_ref)
            } else {
                None
            }
        })
    };
    if mapping_id_opt.is_none() && from_type5 {
        crate::observe::fail(format!(
            "compute_stage_tex type5_no_map ref={texture_ref} sid={stage_ref}"
        ));
        return Err(ComputeStatus::MissingTexture(
            "compute_stage_tex_type5_no_map",
        ));
    }
    if let Some(mapping_id) = mapping_id_opt {
        let _ = mapper::ensure_resolved_for_scanout(state, host, mapping_id);
        // Geom/format: a type-5 record is the exact Metal texture view over
        // the IOSurface bytes. It is authoritative even for a stageable
        // single-plane mapping: the live BGRA8 desktop target is exposed as a
        // row-byte-equivalent, quarter-width RGBA32Uint view. Type-4 direct
        // refs use base mapping geometry. Type-11 refs may prefer the
        // IOSurface descriptor on this task's object list.
        if view_level != 0 {
            crate::observe::fail(format!(
                "compute_stage_tex view_fail reason=type11_mip ref={texture_ref} base={stage_ref} level={view_level} mapping={mapping_id}"
            ));
            return Err(ComputeStatus::Unsupported("compute_view_type11_mip"));
        }
        let (width, height, format) = if from_type5 || from_type4_direct {
            let m = state
                .mappings
                .get(&mapping_id)
                .ok_or(ComputeStatus::MissingTexture(
                    "compute_stage_tex_mapping_gone",
                ))?;
            let multiplanar = objects::mapping_is_multiplanar(m);
            let mapping_stageable =
                m.has_geom && m.width != 0 && m.height != 0 && m.format != 0 && !multiplanar;
            if let Some(rec) = type5_record {
                // `type11_sample_window` below matches actual plane records by
                // geometry+bpe and otherwise verifies a packed row-compatible
                // view over the same bytes. Per-bind measurement (view vs base
                // geom), not a failure — verbose-gated to keep the always-on sink
                // for genuine failures.
                crate::observe::line(format!(
                    "compute_stage_tex type5_view mapping={mapping_id} view={}x{} fmt={:#x} base={}x{} fmt={:#x} multiplanar={}",
                    rec.width,
                    rec.height,
                    rec.pixel_format,
                    m.width,
                    m.height,
                    m.format,
                    multiplanar as u8
                ));
                (rec.width, rec.height, rec.pixel_format)
            } else if !mapping_stageable {
                if !m.has_geom || m.width == 0 || m.height == 0 {
                    crate::observe::fail(format!(
                        "compute_stage_tex type11_fail reason=no_geom mapping={mapping_id} pages={} has_geom={}",
                        m.page_entries.len(),
                        m.has_geom as u8
                    ));
                    return Err(ComputeStatus::MissingTexture(
                        "compute_stage_tex_type11_no_geom",
                    ));
                } else if multiplanar {
                    // Multi-plane IOSurface without a plane record: fail closed,
                    // do not invent BGRA sample of the whole surface.
                    crate::observe::fail(format!(
                        "compute_stage_tex type11_fail reason=multiplane mapping={mapping_id} {}x{} fmt={:#x} pages={} (no type-5 plane record)",
                        m.width,
                        m.height,
                        m.format,
                        m.page_entries.len()
                    ));
                    return Err(ComputeStatus::Unsupported("stage_tex_multiplane_no_plane"));
                } else {
                    // Single-plane unknown format: fail closed (no BGRA invent).
                    crate::observe::fail(format!(
                        "compute_stage_tex type11_fail reason=fmt_unknown mapping={mapping_id} {}x{} pages={}",
                        m.width,
                        m.height,
                        m.page_entries.len()
                    ));
                    return Err(ComputeStatus::Unsupported("stage_tex_fmt_unknown"));
                }
            } else {
                (m.width, m.height, m.format)
            }
        } else {
            // Three ways the surface's own IOSurface descriptor can fail to
            // answer — no list entry, no descriptor bytes, or bytes that do not
            // decode as an IOSurfaceTexture — and all three fall back to the
            // mapping's latched geometry. Kept sequential rather than chained so
            // the `&mut state` the lookups need does not overlap the `&state` the
            // fallback reads.
            let mut from_descriptor = None;
            if let Some(entry) = objects::lookup_list_entry(state, host, task_id, stage_ref) {
                if let Some(desc_bytes) = objects::read_descriptor(state, host, task_id, &entry) {
                    if let Ok(ResourceDescriptor::IOSurfaceTexture {
                        width,
                        height,
                        pixel_format,
                        ..
                    }) = crate::runtime::decode::resource::decode_iosurface_texture_descriptor(
                        &desc_bytes,
                    ) {
                        from_descriptor = Some((width, height, or_bgra8(pixel_format)));
                    }
                }
            }
            match from_descriptor {
                Some(geom) => geom,
                None => mapping_geom_format(state, mapping_id)?,
            }
        };
        if width == 0 || height == 0 {
            return Err(ComputeStatus::MissingTexture("compute_stage_tex_zero_geom"));
        }
        // sRGB color-renderable surfaces stage as unorm storage (same bpp).
        let Some(view_format) =
            crate::runtime::metal_draw::effective_view_sample_format(format, view_pixel_format)
        else {
            crate::observe::fail(format!(
                "compute_stage_tex view_fail reason=format_incompatible ref={texture_ref} base={stage_ref} base_fmt={format:#x} view_fmt={view_pixel_format:?} mapping={mapping_id}"
            ));
            return Err(ComputeStatus::Unsupported("compute_view_format"));
        };
        let stage_fmt = match view_format {
            pixel_format::MTL_FORMAT_BGRA8_UNORM_SRGB => pixel_format::MTL_FORMAT_BGRA8_UNORM,
            pixel_format::MTL_FORMAT_RGBA8_UNORM_SRGB => pixel_format::MTL_FORMAT_RGBA8_UNORM,
            other => other,
        };
        let bpp = match pixel_format::bytes_per_pixel(stage_fmt) {
            Some(v) => v,
            None => {
                crate::observe::fail(format!(
                    "compute_stage_tex type11_fail reason=fmt_bytes mapping={mapping_id} {width}x{height} fmt={format:#x}"
                ));
                return Err(ComputeStatus::Unsupported("stage_tex_fmt_bytes"));
            }
        };
        let storage_selector = pixel_format::storage_selector(stage_fmt).map(|s| s as u32);
        if is_storage && storage_selector.is_none() {
            crate::observe::fail(format!(
                "compute_stage_tex type11_fail reason=fmt_storage mapping={mapping_id} {width}x{height} fmt={format:#x}"
            ));
            return Err(ComputeStatus::Unsupported("stage_tex_fmt_storage"));
        }
        let m = state
            .mappings
            .get(&mapping_id)
            .ok_or(ComputeStatus::MissingTexture(
                "compute_stage_tex_mapping_gone",
            ))?;
        #[cfg(feature = "backend-vulkan")]
        let map_generation = m.map_generation;
        #[cfg(feature = "backend-vulkan")]
        let mut seed_generation = m.content_generation;
        let pages_n = m.page_entries.len();
        // Wire type-4 `length` (page-aligned getResidentSize), stashed as device_desc.alloc_size.
        // Independent of plane w/h and of MapMemory2 IOAccelMemory length — measure-only.
        let wire_len = crate::contract::iosurface_pages::decode_device_surface(&m.device_desc)
            .map(|s| s.alloc_size as u64)
            .unwrap_or(0);
        // A type-5 record names its IOSurface plane on the wire (record `+0x20`,
        // the `newTextureWithDescriptor:iosurface:plane:` argument), so the
        // plane is decided, not inferred. Type-11 carries no such field and must
        // still match a plane record by geometry — which is ambiguous whenever
        // two planes share dims and bytes-per-element (v0a8 Y and alpha), and
        // resolves to the invented packed window over plane 0 or to nothing at
        // all. The draw path already binds type-5 views by index; this is the
        // same resolution on the staging path.
        let window = match type5_record {
            Some(rec) => {
                mapping_write::type5_sample_window(m, rec.plane_index, width, height, stage_fmt)
                    .map(|(offset, bpr, end, from_device)| {
                        if !from_device {
                            mapping_write::note_type5_plane_invent(
                                mapping_id,
                                rec.plane_index,
                                width,
                                height,
                                stage_fmt,
                                (offset, bpr),
                                "compute_stage_tex",
                            );
                        }
                        (offset, bpr, end)
                    })
            }
            None => mapping_write::type11_sample_window(m, mapping_id, width, height, stage_fmt),
        };
        let (surface_offset, surface_bpr, span_end) = match window {
            Some(w) => w,
            None => {
                // Measure type4_len_vs_plane: which window path rejected (device bpr vs invent span).
                let ds = crate::contract::iosurface_pages::decode_device_surface(&m.device_desc);
                let (dw, dh, dbpr, dalloc) = ds
                    .as_ref()
                    .map(|s| (s.width, s.height, s.bytes_per_row, s.alloc_size))
                    .unwrap_or((0, 0, 0, 0));
                let invent_end =
                    crate::contract::iosurface_pages::sample_window(0, stage_fmt, width, height)
                        .map(|(_, _, e)| e)
                        .unwrap_or(0);
                crate::observe::fail(format!(
                    "compute_stage_tex type11_fail reason=window mapping={mapping_id} {width}x{height} fmt={stage_fmt:#x} pages={pages_n} wire_len={wire_len} desc={dw}x{dh} bpr={dbpr} alloc={dalloc} invent_end={invent_end}"
                ));
                return Err(ComputeStatus::MissingTexture(
                    "compute_stage_tex_type11_window",
                ));
            }
        };
        let tight = (width as u64)
            .checked_mul(bpp as u64)
            .ok_or(ComputeStatus::Unsupported("stage_tex_tight_bpr_overflow"))?
            as u32;
        if from_type5 && type5_record.is_some() {
            // Per-bind type-5 sample-window measurement, not a failure — verbose-gated
            // (was a per-bind always-on line). Genuine window failures above emit
            // `type11_fail reason=window` always-on.
            crate::observe::line(format!(
                "compute_stage_tex type5_view_window mapping={mapping_id} view={width}x{height} fmt={stage_fmt:#x} bpp={bpp} tight={tight} surface_off={surface_offset} surface_bpr={surface_bpr} span_end={span_end}"
            ));
        }
        let need_u64 = (tight as u64)
            .checked_mul(height as u64)
            .ok_or(ComputeStatus::Unsupported("stage_tex_need_overflow"))?;
        let Some(need) = host_alloc_len(need_u64) else {
            crate::observe::fail(format!(
                "compute_stage_tex type11_fail reason=host_len mapping={mapping_id} need={need_u64}"
            ));
            return Err(ComputeStatus::Unsupported("stage_tex_host_len"));
        };
        let page_bytes = (pages_n as u64).saturating_mul(1u64 << state.page_shift);
        if page_bytes < span_end {
            crate::observe::fail(format!(
                "compute_stage_tex type11_fail reason=span mapping={mapping_id} {width}x{height} pages={pages_n} page_bytes={page_bytes} span_end={span_end} bpr={surface_bpr} wire_len={wire_len}"
            ));
            return Err(ComputeStatus::GuestIo("compute_stage_tex_type11_span"));
        }
        #[cfg(feature = "backend-vulkan")]
        let residency_key = crate::model::ComputeStorageResidencyKey {
            mapping_id,
            map_generation,
            surface_offset,
            surface_bpr,
            span_end,
            width,
            height,
            pixel_format: stage_fmt,
            texture_ref: 0,
        };
        // Chained-dispatch restage skip: when guest pages still hold exactly
        // our own last writeback for THIS WINDOW (mirror entry survives only
        // while no intersecting guest write lands — `DeviceState::
        // invalidate_storage_residency_window`, called from mapping_write and
        // mapper, drops every mirror entry whose byte window overlaps the
        // write and keeps the disjoint siblings) AND the engine still holds the
        // resident image at the mirror's generation, reading ~15 MB from guest
        // pages reproduces what the GPU already has. The mapping-level content
        // generation may have advanced via disjoint sibling windows
        // (ping-pong canvases), so the gate pairs mirror↔engine directly.
        // The zero placeholder is never seeded — the engine fails visibly
        // with `vk_compute_exec_resident_seed_generation_lost` if the resident
        // vanishes by acquire time.
        // Copy-on-sample is the same gate: a sampled input of a window whose
        // current content the engine already holds GPU-resident (a prior
        // dispatch's storage output — live class: the dispatch samples the very
        // window it storage-writes) never needs the guest read either.
        #[cfg(feature = "backend-vulkan")]
        let (seed_skipped, sample_resident) = match state
            .compute_storage_residency
            .get(&residency_key)
            .copied()
            .and_then(|mirror_generation| {
                resident_serve(residency_key, mirror_generation, is_storage, stage_fmt)
            }) {
            Some(ResidentServe::Seed(generation)) => {
                seed_generation = generation;
                crate::observe::off(format!(
                    "compute_stage_resident_skip mapping={mapping_id} {width}x{height} fmt={stage_fmt:#x} gen={seed_generation} bytes={need}"
                ));
                (true, None)
            }
            Some(ResidentServe::Sample(key, generation)) => {
                crate::observe::off(format!(
                    "compute_stage_resident_sample mapping={mapping_id} {width}x{height} fmt={stage_fmt:#x} gen={generation} bytes={need}"
                ));
                (false, Some((key, generation)))
            }
            None => (false, None),
        };
        #[cfg(not(feature = "backend-vulkan"))]
        let (seed_skipped, sample_resident): (
            bool,
            Option<(crate::model::ComputeStorageResidencyKey, u32)>,
        ) = (false, None);
        let mut bytes = vec![0u8; need];
        if !seed_skipped
            && sample_resident.is_none()
            && !mapping_write::read_rect_raw_at(
                state,
                host,
                mapping_id,
                surface_offset,
                surface_bpr,
                span_end,
                0,
                0,
                width,
                height,
                bpp,
                &mut bytes,
                tight,
            )
        {
            crate::observe::fail(format!(
                "compute_stage_tex type11_fail reason=read mapping={mapping_id} {width}x{height} off={surface_offset} bpr={surface_bpr} span_end={span_end} pages={pages_n}"
            ));
            return Err(ComputeStatus::GuestIo("compute_stage_tex_type11_read"));
        }
        let writeback = if is_storage {
            TextureWriteback::Type11 {
                mapping_id,
                surface_offset,
                surface_bpr,
                span_end,
                width,
                height,
                bpp,
            }
        } else {
            TextureWriteback::None
        };
        if from_type5 {
            // Per-bind type-5 stage SUCCESS census — not a failure; verbose-gated
            // (was always-on, ~300/boot). Genuine type-5 stage failures above emit
            // `type11_fail reason=<slug>` always-on.
            crate::observe::line(format!(
                "compute_stage_tex type5_ok ref={texture_ref} sid={mapping_id} {width}x{height} fmt={stage_fmt:#x} pages={pages_n}"
            ));
        }
        return Ok(StagedTexture {
            binding,
            pixel_format: stage_fmt,
            storage_selector,
            width,
            height,
            bytes,
            is_storage,
            #[cfg(feature = "backend-vulkan")]
            residency: is_storage.then_some(ComputeStorageResidencyCandidate {
                key: residency_key,
                seed_generation,
            }),
            #[cfg(feature = "backend-vulkan")]
            seed_skipped,
            #[cfg(feature = "backend-vulkan")]
            sample_resident,
            writeback,
        });
    }

    // Type-2/3 linear. Fail-visible: name which gate rejected (live class:
    // silent ot=2 MissingTexture, journal 2026-07-14 compute census).
    // The reason travels *in* the status now, so this line and the caller's
    // both name the registered slug rather than a local shorthand only this
    // closure understood.
    let linear_fail = |st: ComputeStatus, detail: String| {
        crate::observe::fail(format!(
            "compute_stage_tex linear_fail reason={} ref={texture_ref} {detail}",
            st.reason()
        ));
        Err(st)
    };
    let Some(entry) = objects::lookup_list_entry(state, host, task_id, stage_ref) else {
        return linear_fail(
            ComputeStatus::MissingTexture("compute_linear_tex_no_entry"),
            String::new(),
        );
    };
    if entry.object_type != OBJECT_TYPE_TEXTURE && entry.object_type != OBJECT_TYPE_TEXTURE_VARIANT
    {
        return linear_fail(
            ComputeStatus::MissingTexture("compute_linear_tex_not_texture"),
            format!("ot={}", entry.object_type),
        );
    }
    let Some(desc_bytes) = objects::read_descriptor(state, host, task_id, &entry) else {
        return linear_fail(
            ComputeStatus::MissingTexture("compute_linear_tex_no_desc"),
            String::new(),
        );
    };
    let Ok(tex) = decode_texture_descriptor(&desc_bytes) else {
        return linear_fail(
            ComputeStatus::MissingTexture("compute_linear_tex_desc_decode"),
            format!("len={}", desc_bytes.len()),
        );
    };
    if !tex.has_pixel_format {
        return linear_fail(
            ComputeStatus::Unsupported("linear_tex_no_fmt"),
            String::new(),
        );
    }
    let Some(stage_format) = crate::runtime::metal_draw::effective_view_sample_format(
        tex.pixel_format,
        view_pixel_format,
    ) else {
        return linear_fail(
            ComputeStatus::Unsupported("linear_tex_view_format"),
            format!(
                "base={stage_ref} base_fmt={:#x} view_fmt={view_pixel_format:?}",
                tex.pixel_format
            ),
        );
    };
    let Some(bpp) = pixel_format::bytes_per_pixel(stage_format) else {
        return linear_fail(
            ComputeStatus::Unsupported("linear_tex_fmt_bytes"),
            format!("fmt={stage_format:#x}"),
        );
    };
    let storage_selector = pixel_format::storage_selector(stage_format).map(|s| s as u32);
    if is_storage && storage_selector.is_none() {
        return linear_fail(
            ComputeStatus::Unsupported("linear_tex_fmt_storage"),
            format!("fmt={stage_format:#x}"),
        );
    }
    let Some((gva, layout)) = tex.level_gva(view_level, state.page_shift) else {
        return linear_fail(
            ComputeStatus::MissingTexture("compute_linear_tex_no_level"),
            format!(
                "base={stage_ref} level={view_level} handle={:#x} alloc={} levels={} data_off={} page_shift={}",
                tex.handle,
                tex.allocation_size,
                tex.levels.len(),
                tex.data_offset,
                state.page_shift
            ),
        );
    };
    let w = layout.width;
    let h = layout.height;
    if w == 0 || h == 0 || layout.row_stride == 0 {
        return linear_fail(
            ComputeStatus::MissingTexture("compute_linear_tex_zero_geom"),
            format!("{w}x{h} stride={}", layout.row_stride),
        );
    }
    let Some(tight) = (w as u64).checked_mul(bpp as u64).map(|v| v as usize) else {
        return linear_fail(
            ComputeStatus::Unsupported("linear_tex_tight_overflow"),
            format!("{w}x{h} bpp={bpp}"),
        );
    };
    if layout.row_stride < tight as u64 {
        return linear_fail(
            ComputeStatus::MissingTexture("compute_linear_tex_stride_lt_tight"),
            format!("stride={} tight={tight} {w}x{h}", layout.row_stride),
        );
    }
    let Some(need) = tight.checked_mul(h as usize) else {
        return linear_fail(
            ComputeStatus::Unsupported("linear_tex_need_overflow"),
            format!("{w}x{h} bpp={bpp}"),
        );
    };
    // Linear-window residency identity — mirrors the host_linear_textures
    // entry exactly. Absent when the stride overflows the key field (no live
    // class; such a window simply stays on the bytes path).
    #[cfg(feature = "backend-vulkan")]
    let span = layout.row_stride.saturating_mul(h as u64);
    #[cfg(feature = "backend-vulkan")]
    let linear_key = (layout.row_stride <= u32::MAX as u64).then(|| {
        crate::model::ComputeStorageResidencyKey::linear(
            task_id,
            stage_ref,
            gva,
            layout.row_stride as u32,
            span,
            w,
            h,
            stage_format,
        )
    });
    let mut bytes = vec![0u8; need];
    #[cfg_attr(
        not(feature = "backend-vulkan"),
        allow(
            unused_mut,
            reason = "the Vulkan resident-window block below assigns it"
        )
    )]
    let mut have_bytes = false;
    // Resident-authoritative window (deferred linear writeback): consume the
    // engine resident without bytes when possible; otherwise flush it into the
    // entry first — falling through to the raw guest read would silently serve
    // the pre-chain seed pages.
    #[cfg(feature = "backend-vulkan")]
    let resident = match (
        linear_key,
        crate::runtime::surface_cache::linear_texture_resident_gen(
            state,
            task_id,
            stage_ref,
            gva,
            stage_format,
            w,
            h,
            layout.row_stride,
        ),
    ) {
        (Some(key), Some(resident_gen)) => Some((
            key,
            resident_gen,
            resident_serve(key, resident_gen, is_storage, stage_format),
        )),
        _ => None,
    };
    #[cfg(feature = "backend-vulkan")]
    let (seed_skipped, seed_generation, sample_resident) = match resident {
        Some((_, _, Some(ResidentServe::Seed(generation)))) => {
            crate::observe::off(format!(
                "compute_stage_linear_resident_seed task={task_id} ref={texture_ref} gva={gva:#x} fmt={:#x} dims={w}x{h} gen={generation}",
                tex.pixel_format
            ));
            (true, generation, None)
        }
        Some((_, _, Some(ResidentServe::Sample(key, generation)))) => {
            crate::observe::off(format!(
                "compute_stage_linear_resident_sample task={task_id} ref={texture_ref} gva={gva:#x} fmt={:#x} dims={w}x{h} gen={generation}",
                stage_format
            ));
            (false, 0u32, Some((key, generation)))
        }
        _ => (false, 0u32, None),
    };
    #[cfg(not(feature = "backend-vulkan"))]
    let (seed_skipped, sample_resident): (
        bool,
        Option<(crate::model::ComputeStorageResidencyKey, u32)>,
    ) = (false, None);
    #[cfg(feature = "backend-vulkan")]
    if let Some((key, resident_gen, None)) = resident {
        // A bytes consumer (format-mismatched view, non-vulkan reuse):
        // land the resident into the cache entry (and any owed guest
        // write) through the one flush path, then serve the bytes.
        if crate::runtime::storage_flush::flush_linear_one(state, host, &key, resident_gen) {
            if let Some(cached) = crate::runtime::surface_cache::get_linear_texture(
                state,
                task_id,
                stage_ref,
                gva,
                stage_format,
                w,
                h,
                layout.row_stride,
            ) {
                bytes.copy_from_slice(cached);
                have_bytes = true;
                crate::observe::off(format!(
                        "compute_linear_flush task={task_id} ref={texture_ref} gva={gva:#x} fmt={:#x} dims={w}x{h} gen={resident_gen}",
                        stage_format
                    ));
            }
        }
        if !have_bytes {
            // Deferred content is unrecoverable — name the loss, clear
            // the marker, and fall back to the coherent stale seed.
            // (flush_linear_one already fail-logged the engine loss.)
            crate::observe::fail(format!(
                    "compute_stage_tex linear_resident_lost task={task_id} ref={texture_ref} gva={gva:#x} fmt={:#x} dims={w}x{h} gen={resident_gen}",
                    stage_format
                ));
            if let Some(e) = state.host_linear_textures.get_mut(&(task_id, stage_ref)) {
                e.resident_gen = 0;
            }
        }
    }
    if seed_skipped || sample_resident.is_some() || have_bytes {
        // Engine resident serves this window; no cache/guest read.
    } else if let Some(cached) = crate::runtime::surface_cache::get_linear_texture(
        state,
        task_id,
        stage_ref,
        gva,
        stage_format,
        w,
        h,
        layout.row_stride,
    ) {
        bytes.copy_from_slice(cached);
        crate::observe::off(format!(
            "compute_stage_tex linear_cache task={task_id} ref={texture_ref} gva={gva:#x} fmt={:#x} dims={w}x{h} row_stride={}",
            stage_format, layout.row_stride
        ));
    } else {
        // Deferred-writeback flush-on-access: the bulk/row reads below walk
        // raw task GVAs and bypass the mapping-keyed hooks — land any
        // resident-authoritative window aliasing the sampled span first.
        crate::runtime::storage_flush::flush_intersecting_task_gva(
            state,
            host,
            task_id,
            gva,
            layout.row_stride.saturating_mul(h as u64),
        );
        if read_linear_texture_bulk(
            state,
            host,
            task_id,
            gva,
            layout.row_stride,
            tight,
            h,
            &mut bytes,
        ) {
            // One cached-view walk for the whole span (render-path bulk analog).
        } else {
            let mut row = vec![0u8; tight];
            for y in 0..h {
                let row_gva = gva
                    .checked_add((y as u64).checked_mul(layout.row_stride).ok_or(
                        ComputeStatus::GuestIo("compute_stage_tex_linear_row_offset"),
                    )?)
                    .ok_or(ComputeStatus::GuestIo("compute_stage_tex_linear_row_gva"))?;
                if let Err(e) = gva_mem::read_task_gva_by_id(
                    host,
                    &state.tasks,
                    task_id,
                    row_gva,
                    &mut row,
                    state.page_shift,
                ) {
                    // First failing row only — full walk status for one-boot diagnosis.
                    if y == 0 {
                        let walk = gva_mem::diagnose_gva_walk(
                            host,
                            &state.tasks,
                            task_id,
                            row_gva,
                            state.page_shift,
                        );
                        crate::observe::fail(format!(
                            "compute_stage_tex_gva task={task_id} ref={texture_ref} gva={row_gva:#x} y=0 page_shift={} err={e:?} | {walk}",
                            state.page_shift
                        ));
                    }
                    return Err(ComputeStatus::GuestIo("compute_stage_tex_linear_row_read"));
                }
                let off = (y as usize) * tight;
                bytes[off..off + tight].copy_from_slice(&row);
            }
        }
    }
    let writeback = if is_storage {
        TextureWriteback::Linear {
            texture_ref: stage_ref,
            gva,
            pixel_format: stage_format,
            row_stride: layout.row_stride,
            width: w,
            height: h,
            bpp,
            pages: staged_window_pages(state, host, task_id, gva, layout.row_stride, h),
        }
    } else {
        TextureWriteback::None
    };
    // Deferred-writeback candidacy: a linear storage output of a format the
    // BGRA mirror ignores keeps the engine resident authoritative — the
    // readback, cache store, and next chained upload all disappear (the
    // fade-window blur pyramid class). If the GVA is
    // mapped at writeback time (the sync path would have written guest
    // pages), the deferred-writeback arm records a flush obligation with a
    // defer-time page index so aliased raw-GVA readers land it first.
    #[cfg(feature = "backend-vulkan")]
    let mut residency = None;
    #[cfg(feature = "backend-vulkan")]
    if is_storage {
        if let Some(key) = linear_key {
            if !crate::runtime::surface_cache::linear_mirrorable(stage_format) {
                let seed = if seed_skipped {
                    seed_generation
                } else {
                    state
                        .host_linear_textures
                        .get(&(task_id, stage_ref))
                        .map(|e| e.host_gen)
                        .unwrap_or(0)
                };
                residency = Some(ComputeStorageResidencyCandidate {
                    key,
                    seed_generation: seed,
                });
            }
        }
    }
    Ok(StagedTexture {
        binding,
        pixel_format: stage_format,
        storage_selector,
        width: w,
        height: h,
        bytes,
        is_storage,
        #[cfg(feature = "backend-vulkan")]
        residency,
        #[cfg(feature = "backend-vulkan")]
        seed_skipped,
        #[cfg(feature = "backend-vulkan")]
        sample_resident,
        writeback,
    })
}

/// Read a strided linear texture span through one cached GVA view (a single
/// page-table walk for the whole texture), de-striding rows into `bytes`
/// (tight rows). Returns `false` when the span cannot be packed — the caller
/// falls back to the per-row walk. Live transition cost of the per-row walk
/// was ~8–23 ms of `stage_us` per Core Image dispatch.
#[allow(
    clippy::too_many_arguments,
    reason = "the bulk path keeps the decoded texture window and row layout explicit"
)]
fn read_linear_texture_bulk<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    gva: u64,
    row_stride: u64,
    tight: usize,
    height: u32,
    bytes: &mut [u8],
) -> bool {
    if height == 0 || tight == 0 || bytes.len() < (height as usize).saturating_mul(tight) {
        return false;
    }
    if row_stride == tight as u64 {
        return crate::runtime::gva_view::read_span(state, host, task_id, gva, bytes);
    }
    let Some(span_len) = (height as u64 - 1)
        .checked_mul(row_stride)
        .and_then(|v| v.checked_add(tight as u64))
    else {
        return false;
    };
    let Some((ptr, avail)) =
        crate::runtime::gva_view::host_ptr_for_span(state, host, task_id, gva, span_len)
    else {
        return false;
    };
    if (avail as u64) < span_len {
        return false;
    }
    for y in 0..height as usize {
        let src = (y as u64).saturating_mul(row_stride) as usize;
        let dst = y * tight;
        // SAFETY: host_ptr_for_span guarantees `span_len` readable bytes at
        // `ptr`; `src + tight <= span_len` for every row by construction.
        unsafe {
            std::ptr::copy_nonoverlapping(
                ptr.add(src),
                bytes[dst..dst + tight].as_mut_ptr(),
                tight,
            );
        }
    }
    true
}

/// Write tight rows of a linear storage texture through one fresh-walked
/// span mapping. Stride padding bytes are left untouched —
/// consumers address rows by `row_stride`, so padding is dead space and
/// writing it is never observable. Returns `false` when the span cannot be
/// packed or the write is outside the task's recorded map spans — the caller
/// falls back to the per-row walk (which fails visibly per contract).
#[allow(
    clippy::too_many_arguments,
    reason = "the bulk path keeps the decoded texture window and row layout explicit"
)]
fn write_linear_texture_bulk<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    gva: u64,
    row_stride: u64,
    tight: usize,
    height: u32,
    bytes: &[u8],
    allowed: crate::runtime::gva_view::WindowPages<'_>,
) -> bool {
    if height == 0 || tight == 0 || bytes.len() < (height as usize).saturating_mul(tight) {
        return false;
    }
    let Some(span_len) = (height as u64 - 1)
        .checked_mul(row_stride)
        .and_then(|v| v.checked_add(tight as u64))
    else {
        return false;
    };
    // Fresh PT walk at write time — never a cached view (stale-view class) —
    // carrying `allowed` so a deferred window's bytes cannot reach a page
    // outside the set it was armed on, however the guest re-points the range
    // between the flush decision and this walk.
    let Some(span_map) = crate::runtime::gva_view::map_fresh_span_within(
        state, host, task_id, gva, span_len, allowed,
    ) else {
        return false;
    };
    let ptr = span_map.ptr;
    // Sampled payload shape, once for the call. This rail writes rows through a
    // `FreshSpan`, so it reaches neither `mapper::write_mapping_bytes` nor
    // `gva_view::write_gva_bytes` and would otherwise be absent from a census
    // whose only use is answering whether a `0xff`-filled victim could be ours.
    crate::observe::footprint::note_written_payload(bytes);
    for y in 0..height as usize {
        let src = y * tight;
        let dst = (y as u64).saturating_mul(row_stride) as usize;
        // SAFETY: map_fresh_span guarantees `span_len` writable bytes at
        // `ptr`; `dst + tight <= span_len` for every row by construction.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes[src..src + tight].as_ptr(), ptr.add(dst), tight);
        }
    }
    crate::runtime::gva_view::unmap_fresh_span(host, span_map);
    true
}

fn writeback_texture<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    tex: &StagedTexture,
) -> Result<(), ComputeStatus> {
    match &tex.writeback {
        TextureWriteback::None => Ok(()),
        TextureWriteback::Linear {
            texture_ref,
            gva,
            pixel_format,
            row_stride,
            width,
            height,
            bpp,
            pages,
        } => {
            let tight = (*width as usize) * (*bpp as usize);
            let required = tight.saturating_mul(*height as usize);
            if tight > *row_stride as usize || tex.bytes.len() < required {
                crate::observe::fail(format!(
                    "compute_writeback_tex fail reason=linear_layout bind={} gva={gva:#x} dims={}x{} bpp={} row_stride={} tight={} bytes={} required={required}",
                    tex.binding,
                    width,
                    height,
                    bpp,
                    row_stride,
                    tight,
                    tex.bytes.len()
                ));
                return Err(ComputeStatus::GuestIo("compute_wb_tex_linear_layout"));
            }
            if !crate::runtime::surface_cache::store_linear_texture(
                state,
                task_id,
                *texture_ref,
                *gva,
                *pixel_format,
                *width,
                *height,
                *row_stride,
                &tex.bytes,
            ) {
                crate::observe::fail(format!(
                    "compute_writeback_tex fail reason=linear_cache_store task={task_id} ref={texture_ref} bind={} gva={gva:#x} fmt={pixel_format:#x} dims={}x{} bpp={} row_stride={} bytes={}",
                    tex.binding,
                    width,
                    height,
                    bpp,
                    row_stride,
                    tex.bytes.len()
                ));
                return Err(ComputeStatus::GuestIo("compute_wb_tex_linear_cache_store"));
            }
            crate::runtime::surface_cache::mirror_linear_color_cache(
                state,
                host,
                task_id,
                *texture_ref,
                *gva,
                *pixel_format,
                *width,
                *height,
                &tex.bytes,
            );
            // Kept although the span is no longer needed here: the overflow is
            // a real refusal with a name, and `write_linear_guest` would only
            // return a bare `false` for it.
            let Some(_span) = row_stride.checked_mul(*height as u64) else {
                crate::observe::fail(format!(
                    "compute_writeback_tex fail reason=linear_span_overflow task={task_id} ref={texture_ref} bind={} gva={gva:#x} dims={}x{} row_stride={row_stride}",
                    tex.binding, width, height
                ));
                return Err(ComputeStatus::GuestIo(
                    "compute_wb_tex_linear_span_overflow",
                ));
            };
            // This used to return early on `reason=linear_unmapped` whenever the
            // range fell outside the task's notified spans, which on a live boot
            // discarded six glyph-atlas writebacks a boot (79x52, 90x20, 8x8 …)
            // whose pages were mapped the whole time — only the notification had
            // not arrived. The graceful degradation it provided is real and is
            // kept; what changed is that it is now keyed on the condition itself
            // rather than on a proxy that also catches healthy writes.
            match write_linear_guest_within(
                state,
                host,
                task_id,
                *gva,
                *row_stride,
                tight,
                *height,
                &tex.bytes,
                &format!("bind={}", tex.binding),
                (!pages.is_empty()).then_some(pages),
            ) {
                LinearWrite::Written => Ok(()),
                // Nothing resolves under this task, so there is nowhere to put
                // the result. The host cache keeps the authoritative bytes and
                // sampling still serves them, so failing the whole dispatch
                // would cost more than it protects.
                LinearWrite::Unmapped => {
                    crate::observe::fail(format!(
                        "compute_writeback_tex cache_only reason=linear_unmapped task={task_id} ref={texture_ref} bind={} gva={gva:#x} fmt={pixel_format:#x} dims={}x{} bpp={} row_stride={row_stride}",
                        tex.binding, width, height, bpp
                    ));
                    Ok(())
                }
                LinearWrite::Failed => {
                    Err(ComputeStatus::GuestIo("compute_wb_tex_linear_guest_write"))
                }
            }
        }
        TextureWriteback::Type11 {
            mapping_id,
            surface_offset,
            surface_bpr,
            span_end,
            width,
            height,
            bpp,
        } => {
            let tight = width.saturating_mul(*bpp);
            if !mapping_write::write_full_rect_raw_at(
                state,
                host,
                *mapping_id,
                *surface_offset,
                *surface_bpr,
                *span_end,
                *width,
                *height,
                *bpp,
                &tex.bytes,
                tight,
            ) {
                crate::observe::fail(format!(
                    "compute_writeback_tex fail reason=type11_mapping_write task={task_id} bind={} mid={} surface_offset={surface_offset:#x} surface_bpr={} span_end={span_end:#x} dims={}x{} bpp={} bytes={} tight={tight}",
                    tex.binding,
                    mapping_id,
                    surface_bpr,
                    width,
                    height,
                    bpp,
                    tex.bytes.len()
                ));
                return Err(ComputeStatus::GuestIo("compute_wb_tex_type11_write"));
            }
            Ok(())
        }
    }
}

/// What a linear guest writeback did, for callers that must tell "there is
/// nowhere to put this" apart from "putting it there went wrong".
///
/// A bare `bool` collapsed those, and the collapse was load-bearing: the only
/// caller able to degrade gracefully was doing so off a *different* condition
/// (the range being outside the task's notified spans) that also caught healthy
/// writes. `-> bool` crossing a module boundary is exactly where that regrows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LinearWrite {
    /// Every row landed in guest memory.
    Written,
    /// The task's page tables resolve nothing at this GVA, so no write was
    /// possible. Callers keep the host cache and carry on.
    Unmapped,
    /// A write was attempted and did not complete — bad layout, an arithmetic
    /// overflow, or a per-row refusal. Already fail-logged with its own reason.
    Failed,
}

/// Write tight-row `bytes` into a strided linear guest window through fresh
/// task page-table walks (bulk view when packable, per-row fallback), bounded
/// to the guest pages the caller was authorised to write. Fail lines carry
/// `ctx` for the call site.
///
/// There is no unbounded sibling. Both doors onto this rail — the deferred
/// flush and the post-dispatch writeback — hand content produced earlier to a
/// walk taken later, so both need the bound; a wrapper passing `None` would
/// only be a way to reach the rail without one.
///
/// The linear compute rail defers exactly as the GVA render rail does, and it
/// re-walks at flush time for the same reason, so it has the same hazard and
/// takes the same answer: the armed page set travels into the walk that
/// resolves the destination, and both the bulk view and the per-row fallback
/// carry it. Leaving the bound on one of the two would make it depend on how
/// the guest happened to lay the pages out.
#[allow(
    clippy::too_many_arguments,
    reason = "the linear writer mirrors the window's guest geometry"
)]
pub(crate) fn write_linear_guest_within<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    gva: u64,
    row_stride: u64,
    tight: usize,
    height: u32,
    bytes: &[u8],
    ctx: &str,
    allowed: crate::runtime::gva_view::WindowPages<'_>,
) -> LinearWrite {
    if write_linear_texture_bulk(
        state, host, task_id, gva, row_stride, tight, height, bytes, allowed,
    ) {
        return LinearWrite::Written;
    }
    // The bulk path declines for several reasons and the per-row fallback below
    // covers all but one of them. The exception is "nothing is mapped here",
    // which no amount of retrying per row can fix, so it is answered once here
    // rather than discovered `height` times.
    if !crate::runtime::gva_mem::any_task_gva_page_resolves(
        host,
        &state.tasks,
        task_id,
        gva,
        1,
        state.page_shift,
    ) {
        return LinearWrite::Unmapped;
    }
    let mut row = vec![0u8; row_stride as usize];
    for y in 0..height {
        let src_off = (y as usize) * tight;
        row[..tight].copy_from_slice(&bytes[src_off..src_off + tight]);
        // Pad rest of row with zeros already present.
        let Some(row_offset) = (y as u64).checked_mul(row_stride) else {
            crate::observe::fail(format!(
                "compute_writeback_tex fail reason=linear_row_offset_overflow {ctx} gva={gva:#x} y={y} row_stride={row_stride}"
            ));
            return LinearWrite::Failed;
        };
        let Some(row_gva) = gva.checked_add(row_offset) else {
            crate::observe::fail(format!(
                "compute_writeback_tex fail reason=linear_gva_overflow {ctx} gva={gva:#x} y={y} row_offset={row_offset:#x}"
            ));
            return LinearWrite::Failed;
        };
        if let Err(e) = gva_mem::write_task_gva_product_within(
            state,
            host,
            task_id,
            row_gva,
            &row[..row_stride as usize],
            allowed,
        ) {
            crate::observe::fail(format!(
                "compute_writeback_tex fail reason=linear_gva_write task={task_id} {ctx} gva={row_gva:#x} y={y} row_stride={row_stride} height={height} err={e:?}"
            ));
            return LinearWrite::Failed;
        }
    }
    LinearWrite::Written
}

fn writeback_buffer<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    pipe_ref: Option<u32>,
    context: &str,
    staged: &StagedBuffer,
) -> Result<(), ComputeStatus> {
    if let Err(e) = gva_mem::write_task_gva_product_within(
        state,
        host,
        task_id,
        staged.gva,
        &staged.bytes,
        (!staged.pages.is_empty()).then_some(&staged.pages),
    ) {
        crate::observe::fail(format!(
            "compute_writeback_buf fail reason=task_gva_write task={task_id} pipe={} context={context} idx={} ref={} gva={:#x} len={} off={:#x} err={e:?}",
            pipe_ref
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".into()),
            staged.bind.index,
            staged.bind.buffer_ref,
            staged.gva,
            staged.bytes.len(),
            staged.bind.offset
        ));
        return Err(ComputeStatus::GuestIo("compute_wb_buf_task_gva_write"));
    }
    Ok(())
}

/// An absent IOSurface pixel format means BGRA8: a type-11 surface the guest
/// mapped without a format word is scanout-ordered by the display contract, and
/// this is the one place that default is written down.
fn or_bgra8(pixel_format: u16) -> u16 {
    if pixel_format != 0 {
        pixel_format
    } else {
        pixel_format::MTL_FORMAT_BGRA8_UNORM
    }
}

/// Latched geometry and pixel format of a type-11 mapping, for a surface whose
/// own IOSurface descriptor could not be read.
///
/// Three separate descriptor failures share this fallback, and spelling it out at
/// each of them made one block of nineteen lines appear three times in a row.
fn mapping_geom_format(
    state: &DeviceState,
    mapping_id: u32,
) -> Result<(u32, u32, u16), ComputeStatus> {
    let m = state
        .mappings
        .get(&mapping_id)
        .ok_or(ComputeStatus::MissingTexture(
            "compute_stage_tex_mapping_gone",
        ))?;
    if !m.has_geom || m.width == 0 || m.height == 0 {
        return Err(ComputeStatus::MissingTexture(
            "compute_stage_tex_mapping_no_geom",
        ));
    }
    Ok((m.width, m.height, or_bgra8(m.format)))
}

fn u32_dim(v: u64) -> Result<u32, ComputeStatus> {
    if v == 0 || v > u32::MAX as u64 {
        Err(ComputeStatus::BadGrid("compute_grid_dim_range"))
    } else {
        Ok(v as u32)
    }
}

/// Execute a direct or indirect dispatch against the current compute accum state.
pub fn execute_dispatch<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    acc: &ComputeAccum,
    cmd: &ComputeCommand,
) -> ComputeStatus {
    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    {
        execute_dispatch_metal(state, host, task_id, acc, cmd, None)
    }
    #[cfg(feature = "backend-vulkan")]
    {
        execute_dispatch_linux(state, host, task_id, acc, cmd)
    }
}

/// Nested dispatch onto an open multi-record control-flow session encoder.
pub(crate) fn execute_dispatch_nested<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    acc: &ComputeAccum,
    cmd: &ComputeCommand,
    session: &mut crate::runtime::compute_session::ComputeSession,
) -> ComputeStatus {
    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    {
        execute_dispatch_metal(state, host, task_id, acc, cmd, Some(session))
    }
    #[cfg(feature = "backend-vulkan")]
    {
        // Nested/control-flow SPI has no Linux compute path. Fail-visible via
        // the returned status: `exec.rs::note_compute_refusal` names the slug
        // at the rail boundary for every non-`Ok` compute record.
        let _ = (state, host, task_id, acc, cmd, session);
        ComputeStatus::NoMetal("compute_nested_no_vulkan_path")
    }
}

/// One nested dispatch's deferred writeback (GPU → host staging → GVA after session commit).
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
pub(crate) struct NestedDispatchJob {
    staged_bufs: Vec<StagedBuffer>,
    /// Storage textures only (sampled need no writeback).
    storage_tex: Vec<StagedTexture>,
    mtl_buffers: Vec<metal::Buffer>,
    mtl_storage: Vec<metal::Texture>,
}

/// Build a deferred writeback job for ICB-filled kernel buffers (no storage textures).
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
pub(crate) fn nested_job_from_icb_buffers(
    staged_bufs: Vec<StagedBuffer>,
    mtl_buffers: Vec<metal::Buffer>,
) -> NestedDispatchJob {
    nested_job_from_icb_resources(staged_bufs, mtl_buffers, Vec::new(), Vec::new())
}

/// Staged compute buffers as the C ABI records the Metal encoder reads.
///
/// The pointers borrow `staged`, so the returned vector must not outlive it.
/// `backing_*` stay null: a staged buffer owns its bytes, and only the
/// indirect-argument path fills a backing allocation in afterwards.
#[cfg(all(target_os = "macos", feature = "backend-metal"))]
fn abi_buffers(staged: &mut [StagedBuffer]) -> Vec<crate::backend::metal::abi::ReimsVgpuBuffer> {
    use crate::backend::metal::abi::ReimsVgpuBuffer;
    staged
        .iter_mut()
        .map(|s| ReimsVgpuBuffer {
            binding: s.bind.index,
            data: s.bytes.as_mut_ptr(),
            len: s.bytes.len(),
            attribute_stride: s.bind.attribute_stride,
            has_attribute_stride: u32::from(s.bind.has_attribute_stride),
            reserved0: 0,
            backing_data: std::ptr::null_mut(),
            backing_len: 0,
            backing_offset: 0,
        })
        .collect()
}

/// Deferred writeback for parent-encoder ICB inheritance (buffers + storage textures).
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
pub(crate) fn nested_job_from_icb_resources(
    staged_bufs: Vec<StagedBuffer>,
    mtl_buffers: Vec<metal::Buffer>,
    storage_tex: Vec<StagedTexture>,
    mtl_storage: Vec<metal::Texture>,
) -> NestedDispatchJob {
    NestedDispatchJob {
        staged_bufs,
        storage_tex,
        mtl_buffers,
        mtl_storage,
    }
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
pub(crate) fn flush_nested_jobs<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    jobs: &mut [NestedDispatchJob],
) -> ComputeStatus {
    use crate::backend::metal::abi::ReimsVgpuStorageImage;
    use crate::backend::metal::compute::compute_writeback_from_mtl;

    let mut err_buf = [0i8; 256];
    for job in jobs.iter_mut() {
        let mut reims_vgpu_bufs = abi_buffers(&mut job.staged_bufs);
        let mut storage: Vec<ReimsVgpuStorageImage> = job
            .storage_tex
            .iter_mut()
            .map(|t| ReimsVgpuStorageImage {
                binding: t.binding,
                format: t
                    .storage_selector
                    .expect("storage texture staged with a storage selector"),
                width: t.width,
                height: t.height,
                data: t.bytes.as_mut_ptr(),
                len: t.bytes.len(),
            })
            .collect();
        let st = compute_writeback_from_mtl(
            &mut reims_vgpu_bufs,
            &job.mtl_buffers,
            &mut storage,
            &job.mtl_storage,
            (err_buf.as_mut_ptr(), err_buf.len()),
        );
        if !st.is_ok() {
            return ComputeStatus::MetalFailed("compute_nested_writeback_metal");
        }
        for s in &job.staged_bufs {
            if let Err(e) = writeback_buffer(state, host, task_id, None, "nested_flush", s) {
                return e;
            }
        }
        for t in &job.storage_tex {
            if let Err(e) = writeback_texture(state, host, task_id, t) {
                return e;
            }
        }
    }
    ComputeStatus::Ok
}

type DispatchDims = (u32, u32, u32, u32, u32, u32, bool);

/// [`resolve_dispatch_dims`], with the refusal named on the always-on log.
///
/// Both dispatch executors want the same thing on failure: the decline, the
/// command kind, the wire grid and threadgroup, and how many textures were
/// bound. Naming it once is what keeps the Metal and Vulkan arms from
/// drifting into two spellings of the same refusal.
fn resolve_dispatch_dims_reported<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &M,
    task_id: u32,
    cmd: &ComputeCommand,
    acc: &ComputeAccum,
) -> Result<DispatchDims, ComputeStatus> {
    resolve_dispatch_dims(state, host, task_id, cmd).inspect_err(|e| {
        crate::observe::line(format!(
            "compute_resolve_dims fail {e:?} kind={:?} grid=[{},{},{}] tg=[{},{},{}] ntex={}",
            cmd.kind,
            cmd.grid.x,
            cmd.grid.y,
            cmd.grid.z,
            cmd.threads_per_threadgroup.x,
            cmd.threads_per_threadgroup.y,
            cmd.threads_per_threadgroup.z,
            acc.textures.len()
        ));
    })
}

/// Resolve grid/threadgroup dims for direct or indirect dispatches.
fn resolve_dispatch_dims<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &M,
    task_id: u32,
    cmd: &ComputeCommand,
) -> Result<DispatchDims, ComputeStatus> {
    match cmd.kind {
        // Every dimension comes from the wire. `u32_dim` refuses `0` and
        // anything past `u32::MAX` with `BadGrid("compute_grid_dim_range")`, so
        // a malformed grid is a named refusal rather than a substitution.
        Kind::DispatchThreadgroups => Ok((
            u32_dim(cmd.grid.x)?,
            u32_dim(cmd.grid.y)?,
            u32_dim(cmd.grid.z)?,
            u32_dim(cmd.threads_per_threadgroup.x)?,
            u32_dim(cmd.threads_per_threadgroup.y)?,
            u32_dim(cmd.threads_per_threadgroup.z)?,
            false,
        )),
        Kind::DispatchThreads => Ok((
            u32_dim(cmd.grid.x)?,
            u32_dim(cmd.grid.y)?,
            u32_dim(cmd.grid.z)?,
            u32_dim(cmd.threads_per_threadgroup.x)?,
            u32_dim(cmd.threads_per_threadgroup.y)?,
            u32_dim(cmd.threads_per_threadgroup.z)?,
            true,
        )),
        Kind::DispatchThreadgroupsIndirect => {
            let raw = read_buffer_window(
                state,
                host,
                task_id,
                cmd.indirect_buffer_ref,
                cmd.indirect_buffer_offset,
                INDIRECT_THREADGROUPS_ARGS_LEN,
            )?;
            let gx = ld32(&raw[0..]);
            let gy = ld32(&raw[4..]);
            let gz = ld32(&raw[8..]);
            Ok((
                u32_dim(gx as u64)?,
                u32_dim(gy as u64)?,
                u32_dim(gz as u64)?,
                u32_dim(cmd.threads_per_threadgroup.x)?,
                u32_dim(cmd.threads_per_threadgroup.y)?,
                u32_dim(cmd.threads_per_threadgroup.z)?,
                false,
            ))
        }
        Kind::DispatchThreadsIndirect => {
            let raw = read_buffer_window(
                state,
                host,
                task_id,
                cmd.indirect_buffer_ref,
                cmd.indirect_buffer_offset,
                INDIRECT_THREADS_ARGS_LEN,
            )?;
            // MTLDispatchThreadsIndirectArguments: threadsPerGrid[3], threadsPerThreadgroup[3].
            Ok((
                u32_dim(ld32(&raw[0..]) as u64)?,
                u32_dim(ld32(&raw[4..]) as u64)?,
                u32_dim(ld32(&raw[8..]) as u64)?,
                u32_dim(ld32(&raw[12..]) as u64)?,
                u32_dim(ld32(&raw[16..]) as u64)?,
                u32_dim(ld32(&raw[20..]) as u64)?,
                true,
            ))
        }
        _ => Err(ComputeStatus::Unsupported("resolve_dims_unknown_kind")),
    }
}

/// Linux product compute path (doorbell / BQL).
///
/// Stages buffers/textures with device `page_shift`, translates the kernel AIR
/// via [`crate::runtime::m2v_cache::translate_cached_kernel_reflected`], dispatches on the
/// process-global [`crate::backend::vulkan::engine`] (shared GRAPHICS|COMPUTE
/// device), then writebacks GVA / type-11.
///
/// Nested/ICB/stage-in stay Unsupported (engine surface is storage buffers +
/// storage images only).
#[cfg(feature = "backend-vulkan")]
fn execute_dispatch_linux<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    acc: &ComputeAccum,
    cmd: &ComputeCommand,
) -> ComputeStatus {
    use crate::backend::vulkan::engine::{
        self as vk_engine, ComputeBufferResource, ComputeRequest, ComputeSampledImageResource,
        ComputeStorageImageResource, DrawError,
    };

    const TEXTURE_BIND_BASE: u32 = 32;

    if acc.pipeline_ref == 0 {
        return ComputeStatus::MissingPipeline("compute_vk_pipeline_ref_zero");
    }
    let Some(pipeline) = load_compute_pipeline(state, host, task_id, acc.pipeline_ref) else {
        return ComputeStatus::MissingPipeline("compute_vk_pipeline_load");
    };
    if pipeline.stage_input.is_some() || acc.imageblock.is_some() || acc.stage_in_region.is_some() {
        crate::observe::fail(format!(
            "compute_linux unsupported pipe={} stage_in={} imageblock={} (need SPI parity)",
            acc.pipeline_ref,
            pipeline.stage_input.is_some() as u8,
            acc.imageblock.is_some() as u8
        ));
        return ComputeStatus::Unsupported("linux_stage_in_imageblock");
    }
    // Dims first (cheap; proves sentinel recovery without m2v/vk).
    let (grid_x, grid_y, grid_z, tg_x, tg_y, tg_z, dispatch_threads) =
        match resolve_dispatch_dims_reported(state, host, task_id, cmd, acc) {
            Ok(v) => v,
            Err(e) => return e,
        };
    if tg_x == 0 || tg_y == 0 || tg_z == 0 || grid_x == 0 || grid_y == 0 || grid_z == 0 {
        return ComputeStatus::BadGrid("compute_vk_zero_dims");
    }

    // Stage buffers first (page_shift-correct). Texture staging follows kernel
    // translation because sampled-vs-storage access is a SPIR-V interface fact.
    // The translation cache keeps warm dispatches cheap; no Vulkan work occurs
    // until every declared resource has staged successfully.
    let mut staged_bufs: Vec<StagedBuffer> = Vec::new();
    for b in &acc.buffers {
        match stage_buffer(state, host, task_id, b) {
            Ok(s) => staged_bufs.push(s),
            Err(e) => {
                // `st={e:?}` alone was not greppable: the Debug spelling was
                // the only handle on which of stage_buffer's eight checks
                // refused. `reason=` names it.
                crate::observe::fail(format!(
                    "compute_linux stage_buf fail reason={} pipe={} idx={} ref={} off={:#x} class={}",
                    e.reason(),
                    acc.pipeline_ref,
                    b.index,
                    b.buffer_ref,
                    b.offset,
                    e.class()
                ));
                return e;
            }
        }
    }
    // MTLB → AIR → SPIR-V (LocalSize = threadgroup dims).
    let Some(mtlb) = load_mtlb(state, host, task_id, pipeline.kernel_func_ref) else {
        return ComputeStatus::MissingMtlb("compute_vk_mtlb_load");
    };
    // The function blob is an MTLB container; llvm-dis needs the wrapped AIR
    // bitcode member (same extract the render path does — passing the raw
    // container was the live `llvm-dis: file doesn't start with bitcode
    // header` MetalFailed class).
    let air = match crate::runtime::mtlb::extract_air(&mtlb) {
        Ok(a) => a,
        Err(e) => {
            crate::observe::Emit::decline("compute_linux_air_extract", &e)
                .field("pipe", acc.pipeline_ref)
                .fail_once(acc.pipeline_ref as u64);
            return ComputeStatus::MetalFailed("compute_vk_air_extract");
        }
    };
    let kernel_shader = match crate::runtime::m2v_cache::translate_cached_kernel_reflected(
        air,
        [tg_x, tg_y, tg_z],
        acc.pipeline_ref,
    ) {
        Ok(b) => b,
        Err(e) => {
            crate::observe::Emit::decline("compute_linux_m2v", &e)
                .field("pipe", acc.pipeline_ref)
                .fail_once(acc.pipeline_ref as u64);
            return ComputeStatus::MetalFailed("compute_vk_translate");
        }
    };
    // An invalid module must not reach the driver.
    //
    // The translator can emit an `OpCompositeInsert` that puts an image handle
    // into a struct, which the Logical addressing model has no representation
    // for — the type at the indexed path is a pointer, never the handle's own
    // type. `spirv-val` rejects it, and creating a shader module anyway is
    // licence for the driver to do whatever it likes.
    //
    // It took that licence. Three consecutive boots of a macOS desktop stopped
    // being served at the compute pipeline carrying exactly this instruction,
    // and the disassembly of the module dumped from the third names it:
    //
    //     %193 = OpCompositeInsert %_struct_51 %84 %55 0 0
    //
    // Nothing else reported it. No panic, no `VkResult` error, no device loss,
    // no host fault record, and a guest kernel log still healthy after the host
    // process was gone. Declining costs this kernel's dispatches; not declining
    // costs every dispatch after it, because the process is no longer there.
    //
    // The detector agreed with `spirv-val` on all 15 modules captured from a
    // live boot, firing on the one it rejects and on none of the rest.
    if kernel_shader.shape.opaque_in_composite {
        crate::observe::fail(format!(
            "compute_linux m2v_invalid_module reason=opaque_handle_in_composite              pipe={} tg=[{tg_x},{tg_y},{tg_z}]              (OpCompositeInsert/Extract of an image or sampler handle;               spirv-val rejects the module)",
            acc.pipeline_ref
        ));
        return ComputeStatus::MetalFailed("compute_vk_invalid_module");
    }
    let mut spirv = match spirv_words_le(&kernel_shader.spirv) {
        Ok(w) => w,
        Err(e) => {
            crate::observe::Emit::decline("compute_linux_spirv_parse", &e)
                .field("pipe", acc.pipeline_ref)
                .fail_once(acc.pipeline_ref as u64);
            return ComputeStatus::MetalFailed("compute_vk_spirv_parse");
        }
    };

    let mut buffer_accesses = Vec::with_capacity(staged_bufs.len());
    let mut buffer_readonly_count = 0usize;
    let mut buffer_writable_count = 0usize;
    let mut buffer_unused_count = 0usize;
    for s in &staged_bufs {
        use crate::runtime::spirv_bind::BufferAccess;
        match crate::runtime::spirv_bind::buffer_access(&spirv, s.bind.index) {
            Some(BufferAccess::ReadOnly) => {
                buffer_readonly_count += 1;
                buffer_accesses.push((s.bind.index, false));
            }
            Some(BufferAccess::Writable) => {
                buffer_writable_count += 1;
                buffer_accesses.push((s.bind.index, true));
            }
            Some(BufferAccess::PointerEscape) => {
                crate::observe::fail(format!(
                    "compute_linux buffer_access fail reason=spirv_pointer_escape pipe={} idx={} ref={}",
                    acc.pipeline_ref, s.bind.index, s.bind.buffer_ref
                ));
                return ComputeStatus::Unsupported("buffer_spirv_pointer_escape");
            }
            Some(BufferAccess::AmbiguousBinding) => {
                crate::observe::fail(format!(
                    "compute_linux buffer_access fail reason=spirv_ambiguous_binding pipe={} idx={} ref={}",
                    acc.pipeline_ref, s.bind.index, s.bind.buffer_ref
                ));
                return ComputeStatus::Unsupported("buffer_spirv_ambiguous_binding");
            }
            None => {
                buffer_unused_count += 1;
                crate::observe::line(format!(
                    "compute_linux buffer_unused pipe={} idx={} ref={}",
                    acc.pipeline_ref, s.bind.index, s.bind.buffer_ref
                ));
            }
        }
    }

    let mut staged_tex: Vec<StagedTexture> = Vec::new();
    let mut storage_writeonly_count = 0usize;
    for t in &acc.textures {
        use crate::runtime::spirv_bind::{
            ImageAccess, ReflectedComputeTexture, StorageImageAccess,
        };
        let binding = TEXTURE_BIND_BASE + t.index;
        // Both the sampled-vs-storage class and the shape come solely from the
        // translator's reflection — the declared Metal texture type, exact at
        // translate time. The always-on `census_reflection_wellformed` guard
        // proves the reflection is internally consistent per translate.
        let is_storage = match crate::runtime::spirv_bind::reflected_compute_texture(
            &kernel_shader.reflection,
            binding,
        ) {
            ReflectedComputeTexture::Plain2d(ImageAccess::Sampled) => false,
            ReflectedComputeTexture::Plain2d(ImageAccess::Storage) => true,
            ReflectedComputeTexture::Absent => {
                // Metal permits unused bound resources. If reflection lists no
                // texture shape at this binding, the shader does not sample/write
                // it — do not stage or invent access/writeback semantics for it.
                crate::observe::line(format!(
                    "compute_linux texture_unused pipe={} i={} ref={} bind={}",
                    acc.pipeline_ref, t.index, t.texture_ref, binding
                ));
                continue;
            }
            ReflectedComputeTexture::UnstageableShape { axis } => {
                // The rail stages one flat plane window or one linear GVA level
                // per binding, so it can only ever produce a single-layer 2D
                // image. Binding that to a shader image declared with a slice,
                // depth, or sample axis is a descriptor-type mismatch — refuse
                // by name instead of dispatching against the wrong view.
                crate::observe::fail(format!(
                    "compute_linux texture_shape fail reason=unstageable_{axis} pipe={} i={} ref={} bind={binding}",
                    acc.pipeline_ref, t.index, t.texture_ref
                ));
                return ComputeStatus::Unsupported("texture_shape_unstageable");
            }
        };
        let storage_access = if is_storage {
            match crate::runtime::spirv_bind::storage_image_access(&spirv, binding) {
                Some(StorageImageAccess::WriteOnly) => Some("write_only"),
                Some(StorageImageAccess::ReadOnly) => Some("read_only"),
                Some(StorageImageAccess::ReadWrite) => Some("read_write"),
                Some(StorageImageAccess::Unknown) => Some("unknown"),
                Some(StorageImageAccess::AmbiguousBinding) => {
                    crate::observe::fail(format!(
                        "compute_linux texture_access fail reason=spirv_storage_ambiguous_binding pipe={} i={} ref={} bind={binding}",
                        acc.pipeline_ref, t.index, t.texture_ref
                    ));
                    return ComputeStatus::Unsupported("texture_spirv_storage_ambiguous_binding");
                }
                None => {
                    crate::observe::fail(format!(
                        "compute_linux texture_access fail reason=spirv_storage_access_missing pipe={} i={} ref={} bind={binding}",
                        acc.pipeline_ref, t.index, t.texture_ref
                    ));
                    return ComputeStatus::Unsupported("texture_spirv_storage_access_missing");
                }
            }
        } else {
            None
        };
        match stage_texture_raw(state, host, task_id, t.texture_ref, binding, is_storage) {
            Ok(s) => {
                if let Some(storage_access) = storage_access {
                    if storage_access == "write_only" {
                        storage_writeonly_count += 1;
                    }
                    let bytes = (s.width as u64)
                        .saturating_mul(s.height as u64)
                        .saturating_mul(
                            pixel_format::bytes_per_pixel(s.pixel_format).unwrap_or(0) as u64
                        );
                    log_storage_image_access(acc.pipeline_ref, binding, storage_access, bytes);
                }
                staged_tex.push(s);
            }
            Err(e) => {
                let ot = objects::lookup_list_entry(state, host, task_id, t.texture_ref)
                    .map(|en| en.object_type)
                    .unwrap_or(0);
                crate::observe::fail(format!(
                    "compute_linux stage_tex fail reason={} pipe={} i={} ref={} ot={} bind={} access={} class={}",
                    e.reason(),
                    acc.pipeline_ref,
                    t.index,
                    t.texture_ref,
                    ot,
                    binding,
                    if is_storage { "storage" } else { "sampled" },
                    e.class()
                ));
                return e;
            }
        }
    }

    let mut sampled_count = 0usize;
    let mut storage_count = 0usize;
    for t in &staged_tex {
        if t.is_storage {
            storage_count += 1;
        } else {
            sampled_count += 1;
        }
    }
    // A dispatch that staged its resources is expected control flow; the
    // refusals on this path each emit their own typed decline.
    crate::observe::line(format!(
        "compute_linux stage_ok pipe={} nbuf={} bro={} brw={} bunused={} ntex={} sampled={} storage={} swo={} grid=[{grid_x},{grid_y},{grid_z}] tg=[{tg_x},{tg_y},{tg_z}] encode=engine",
        acc.pipeline_ref,
        staged_bufs.len(),
        buffer_readonly_count,
        buffer_writable_count,
        buffer_unused_count,
        staged_tex.len(),
        sampled_count,
        storage_count,
        storage_writeonly_count,
    ));

    // Workgroup counts: DispatchThreadgroups already is groups; DispatchThreads
    // is total threads → ceil-div by LocalSize.
    let (wg_x, wg_y, wg_z) = if dispatch_threads {
        (
            grid_x.div_ceil(tg_x).max(1),
            grid_y.div_ceil(tg_y).max(1),
            grid_z.div_ceil(tg_z).max(1),
        )
    } else {
        (grid_x, grid_y, grid_z)
    };

    let mut storage_buffers = Vec::with_capacity(buffer_accesses.len());
    for s in &mut staged_bufs {
        let Some((_, writable)) = buffer_accesses
            .iter()
            .find(|(binding, _)| *binding == s.bind.index)
        else {
            continue;
        };
        storage_buffers.push(ComputeBufferResource {
            binding: s.bind.index,
            bytes: std::mem::take(&mut s.bytes),
            writable: *writable,
        });
    }
    let mut sampled_images = Vec::with_capacity(sampled_count);
    let mut storage_images = Vec::with_capacity(storage_count);
    let mut storage_formats = Vec::with_capacity(storage_count);
    // Device support for format-less storage writes decides whether a guest
    // BGRA8Unorm storage surface can composite into a B8G8R8A8_UNORM view (no
    // R/B swap) or must degrade to the swapped Rgba8Unorm view.
    let write_without_format = vk_engine::supports_storage_image_write_without_format();
    for t in staged_tex.iter().filter(|texture| texture.is_storage) {
        let Some(selector) = t.storage_selector else {
            crate::observe::fail(format!(
                "compute_linux unsupported storage_format reason=no_storage_selector pipe={} bind={} fmt={:#x}",
                acc.pipeline_ref, t.binding, t.pixel_format
            ));
            return ComputeStatus::Unsupported("storage_no_selector_specialize");
        };
        let Some(guest_fmt) = simg_u32_to_engine_storage(selector) else {
            crate::observe::fail(format!(
                "compute_linux unsupported storage_format reason=selector_unknown pipe={} bind={} simg={selector} fmt={:#x}",
                acc.pipeline_ref, t.binding, t.pixel_format
            ));
            return ComputeStatus::Unsupported("storage_selector_unknown_specialize");
        };
        let Some(shader_decl) = crate::runtime::spirv_bind::image_format(&spirv, t.binding) else {
            crate::observe::fail(format!(
                "compute_linux storage_format fail reason=spirv_format_missing pipe={} bind={} guest={guest_fmt:?} simg={}",
                acc.pipeline_ref, t.binding, selector
            ));
            return ComputeStatus::Unsupported("storage_spirv_format_missing");
        };
        let specialized = match specialized_storage_image_format(
            guest_fmt,
            shader_decl,
            write_without_format,
        ) {
            Ok(format) => format,
            Err(reason) => {
                crate::observe::fail(format!(
                        "compute_linux storage_format fail reason={reason} pipe={} bind={} spirv={shader_decl:?} guest={guest_fmt:?} simg={} guest_bpp={} shader_bpp={}",
                        acc.pipeline_ref,
                        t.binding,
                        selector,
                        guest_fmt.bytes_per_texel(),
                        spirv_image_format_to_engine_storage(shader_decl)
                            .map(|format| format.bytes_per_texel())
                            .unwrap_or(0)
                    ));
                return ComputeStatus::Unsupported("storage_format_specialize_mismatch");
            }
        };
        storage_formats.push((t.binding, guest_fmt, shader_decl, specialized));
    }
    let specialization_requests: Vec<_> = storage_formats
        .iter()
        .map(|(binding, _, _, specialized)| (*binding, *specialized))
        .collect();
    if let Err(error) =
        crate::runtime::spirv_bind::specialize_image_formats(&mut spirv, &specialization_requests)
    {
        let error: crate::runtime::spirv_bind::ImageFormatSpecializeError = error;
        crate::observe::Emit::decline("compute_linux_storage_format", &error)
            .field("pipe", acc.pipeline_ref)
            .fail();
        return ComputeStatus::Unsupported("storage_format_specialize_error");
    }
    // A guest BGRA8Unorm storage surface retargets to an `Unknown`-format
    // storage image (viewed B8G8R8A8_UNORM) so the composite writes land in the
    // guest's channel order — that write is only legal if the module declares
    // `StorageImageWriteWithoutFormat`. Inject it once when any binding took the
    // Unknown path (idempotent; the translator declares only Shader/Float16/…).
    if storage_formats.iter().any(|(_, _, _, specialized)| {
        matches!(
            specialized,
            crate::runtime::spirv_bind::ImageFormat::Unknown
        )
    }) {
        crate::runtime::spirv_bind::ensure_storage_write_without_format_capability(&mut spirv);
    }
    // Compute-side analog of the render resident gates: a deferred storage
    // writeback leaves guest-visible bytes GPU-resident-only until a flush
    // choke point lands them, so it requires the device's
    // `deferred_gpu_only_content` capability (off on portability-subset /
    // MoltenVK, where guest pages stay authoritative and the writeback runs
    // synchronously in this call).
    let deferred_content_allowed = vk_engine::deferred_gpu_only_content_allowed();
    for t in &mut staged_tex {
        if t.is_storage {
            let Some(selector) = t.storage_selector else {
                crate::observe::fail(format!(
                    "compute_linux unsupported storage_format reason=no_storage_selector pipe={} bind={} fmt={:#x}",
                    acc.pipeline_ref, t.binding, t.pixel_format
                ));
                return ComputeStatus::Unsupported("storage_no_selector_writeback");
            };
            let Some(guest_fmt) = simg_u32_to_engine_storage(selector) else {
                crate::observe::fail(format!(
                    "compute_linux unsupported storage_format reason=selector_unknown pipe={} bind={} simg={selector} fmt={:#x}",
                    acc.pipeline_ref, t.binding, t.pixel_format
                ));
                return ComputeStatus::Unsupported("storage_selector_unknown_writeback");
            };
            let Some((_, _, shader_decl, specialized)) = storage_formats
                .iter()
                .find(|(binding, _, _, _)| *binding == t.binding)
            else {
                crate::observe::fail(format!(
                    "compute_linux storage_format fail reason=spirv_format_specialize_internal pipe={} bind={} simg={}",
                    acc.pipeline_ref, t.binding, selector
                ));
                return ComputeStatus::Unsupported("storage_format_specialize_internal");
            };
            // An `Unknown`-format storage image carries no SPIR-V texel format;
            // its engine format (and thus VkImageView) is the guest surface's
            // own format — here BGRA8Unorm → B8G8R8A8_UNORM — so the composite
            // write lands in guest channel order (the R/B-swap fix). Every other
            // format takes its engine format from the specialized SPIR-V format.
            let shader_fmt = if matches!(
                specialized,
                crate::runtime::spirv_bind::ImageFormat::Unknown
            ) {
                // Always-on proxy for the BGRA-storage-composite R/B class: this
                // line fires only on the corrected (without_format) path. Its
                // absence together with a `degraded_rb_swap` line below is the
                // regression signal that a swap is being emitted.
                crate::observe::off(format!(
                    "compute_linux bgra_storage_composite pipe={} bind={} mode=without_format guest={guest_fmt:?} view=B8G8R8A8_UNORM {}x{}",
                    acc.pipeline_ref, t.binding, t.width, t.height
                ));
                guest_fmt
            } else {
                let Some(fmt) = spirv_image_format_to_engine_storage(*specialized) else {
                    crate::observe::fail(format!(
                        "compute_linux storage_format fail reason=spirv_storage_format_unsupported pipe={} bind={} spirv={specialized:?} guest={guest_fmt:?} simg={}",
                        acc.pipeline_ref, t.binding, selector
                    ));
                    return ComputeStatus::Unsupported("storage_spirv_format_unsupported");
                };
                // Degraded path: a BGRA8Unorm guest fell back to a Rgba8Unorm
                // view because `shaderStorageImageWriteWithoutFormat` is absent —
                // the composite output is R/B-swapped. Fail-visible so the class
                // is never silent on an unsupported device.
                if matches!(
                    guest_fmt,
                    crate::backend::vulkan::engine::StorageImageFormat::Bgra8Unorm
                ) && matches!(
                    fmt,
                    crate::backend::vulkan::engine::StorageImageFormat::Rgba8Unorm
                ) {
                    crate::observe::fail(format!(
                        "compute_linux bgra_storage_composite pipe={} bind={} mode=degraded_rb_swap reason=no_storage_image_write_without_format {}x{}",
                        acc.pipeline_ref, t.binding, t.width, t.height
                    ));
                }
                fmt
            };
            if specialized != shader_decl {
                crate::observe::off(format!(
                    "compute_linux storage_format_specialize pipe={} bind={} spirv={shader_decl:?} specialized={specialized:?} engine={shader_fmt:?} guest={guest_fmt:?} simg={} guest_bpp={} shader_bpp={}",
                    acc.pipeline_ref,
                    t.binding,
                    selector,
                    guest_fmt.bytes_per_texel(),
                    spirv_image_format_to_engine_storage(*shader_decl)
                        .map(|format| format.bytes_per_texel())
                        .unwrap_or(0)
                ));
            }
            // Deferred writeback: a resident type-11 output skips the engine
            // readback and the CPU guest writeback entirely — the pinned
            // resident is authoritative and every host access of the window
            // flushes first (storage_flush choke points). Linear windows only
            // carry `residency` when their defer gate passed at stage time
            // (cache-only + non-mirrorable), so residency alone qualifies
            // them. Direct writeback is moot when deferring.
            let defer_readback = compute_defer_readback_allowed(
                deferred_content_allowed,
                t.residency.is_some(),
                matches!(
                    t.writeback,
                    TextureWriteback::Type11 { .. } | TextureWriteback::Linear { .. }
                ),
            );
            storage_images.push(ComputeStorageImageResource {
                binding: t.binding,
                format: shader_fmt,
                width: t.width,
                height: t.height,
                bytes: std::mem::take(&mut t.bytes),
                residency: t.residency.map(|candidate| {
                    crate::backend::vulkan::engine::ComputeStorageResidency {
                        identity: candidate.key,
                        seed_generation: candidate.seed_generation,
                        output_generation: next_mapping_content_generation(
                            candidate.seed_generation,
                        ),
                    }
                }),
                seed_skipped: t.seed_skipped,
                defer_readback,
            });
        } else {
            let Some(sampled_fmt) = mtl_to_engine_sampled(t.pixel_format) else {
                crate::observe::fail(format!(
                    "compute_linux sampled_format fail reason=mtl_format_unsupported pipe={} bind={} fmt={:#x}",
                    acc.pipeline_ref, t.binding, t.pixel_format
                ));
                return ComputeStatus::Unsupported("sampled_format_unsupported");
            };
            sampled_images.push(ComputeSampledImageResource {
                binding: t.binding,
                format: sampled_fmt,
                width: t.width,
                height: t.height,
                bytes: std::mem::take(&mut t.bytes),
                resident_bind: t.sample_resident.map(|(identity, generation)| {
                    crate::backend::vulkan::engine::ComputeResidentSampleBind {
                        identity,
                        generation,
                    }
                }),
            });
        }
    }

    let mut samplers = Vec::new();
    for s in &acc.samplers {
        let binding = crate::runtime::spirv_bind::SAMPLER_BINDING_BASE + s.index;
        if !crate::runtime::spirv_bind::sampler_bindings(&spirv).contains(&binding) {
            continue;
        }
        let mut sampler = match crate::runtime::metal_draw::load_vulkan_sampler(
            state,
            host,
            task_id,
            s.sampler_ref,
            binding,
        ) {
            Ok(v) => v,
            Err(reason) => {
                crate::observe::Emit::decline("compute_linux_sampler", &reason)
                    .field("pipe", acc.pipeline_ref)
                    .fail_once((u64::from(s.sampler_ref) << 32) | u64::from(binding));
                return ComputeStatus::MissingSampler("compute_vk_sampler_load");
            }
        };
        if s.has_lod_clamp {
            sampler.lod_min = s.lod_min_bits;
            sampler.lod_max = s.lod_max_bits;
        }
        samplers.push(sampler);
    }
    for binding in crate::runtime::spirv_bind::sampler_bindings(&spirv) {
        if !samplers.iter().any(|sampler| sampler.binding == binding) {
            samplers
                .push(crate::backend::vulkan::engine::SamplerResource::normalized_default(binding));
        }
    }

    let req = ComputeRequest {
        spirv,
        entry: "main".into(),
        grid: [wg_x, wg_y, wg_z],
        storage_buffers,
        sampled_images,
        samplers,
        storage_images,
    };
    let run_engine = |req: &ComputeRequest| {
        let engine_done = spawn_compute_engine_stall_watchdog(
            acc.pipeline_ref,
            req,
            std::time::Duration::from_millis(COMPUTE_ENGINE_STALL_PROXY_MS),
        );
        let out = vk_engine::execute_compute_request(req);
        engine_done.store(true, std::sync::atomic::Ordering::Release);
        out
    };
    let out_result = run_engine(&req);
    let out = match out_result {
        Ok(o) => o,
        Err(e) => {
            let unsupported = matches!(&e, DrawError::Unsupported(_));
            crate::observe::Emit::decline("compute_linux_engine", &e)
                .field("pipe", acc.pipeline_ref)
                .fail_once(u64::from(acc.pipeline_ref));
            if unsupported {
                return ComputeStatus::Unsupported("engine_run_unsupported");
            }
            return ComputeStatus::MetalFailed("compute_vk_engine_run");
        }
    };
    if out.buffers.len() != buffer_writable_count
        || out.images.len() != storage_count
        || out.images_deferred.len() != storage_count
    {
        crate::observe::fail(format!(
            "compute_linux readback count mismatch pipe={} buf={}/{} img={}/{} deferred={}/{}",
            acc.pipeline_ref,
            out.buffers.len(),
            buffer_writable_count,
            out.images.len(),
            storage_count,
            out.images_deferred.len(),
            storage_count
        ));
        return ComputeStatus::MetalFailed("compute_vk_readback_count");
    }
    let vk_engine::ComputeOutput {
        buffers: output_buffers,
        images: output_images,
        images_deferred: output_images_deferred,
    } = out;
    for buffer in output_buffers {
        let Some(s) = staged_bufs
            .iter_mut()
            .find(|staged| staged.bind.index == buffer.binding)
        else {
            crate::observe::fail(format!(
                "compute_linux readback binding mismatch pipe={} bind={} bytes={}",
                acc.pipeline_ref,
                buffer.binding,
                buffer.bytes.len()
            ));
            return ComputeStatus::MetalFailed("compute_vk_readback_binding");
        };
        s.bytes = buffer.bytes;
        if let Err(e) = writeback_buffer(
            state,
            host,
            task_id,
            Some(acc.pipeline_ref),
            "vulkan_dispatch",
            s,
        ) {
            return e;
        }
    }
    for ((t, bytes), deferred) in staged_tex
        .iter_mut()
        .filter(|texture| texture.is_storage)
        .zip(output_images)
        .zip(output_images_deferred)
    {
        if deferred {
            // Deferred linear window: the pinned resident is the whole story —
            // today's sync path never wrote guest pages either (cache-only),
            // so the only bookkeeping is the cache entry's resident marker.
            if let (
                Some(candidate),
                TextureWriteback::Linear {
                    texture_ref,
                    gva,
                    pixel_format,
                    row_stride,
                    width,
                    height,
                    ..
                },
            ) = (t.residency, &t.writeback)
            {
                let generation = next_mapping_content_generation(candidate.seed_generation);
                if !crate::runtime::surface_cache::note_linear_texture_resident(
                    state,
                    task_id,
                    *texture_ref,
                    *gva,
                    *pixel_format,
                    *width,
                    *height,
                    *row_stride,
                    generation,
                ) {
                    crate::observe::fail(format!(
                        "compute_writeback_deferred fail reason=linear_note task={task_id} ref={texture_ref} gva={gva:#x} fmt={pixel_format:#x} dims={width}x{height} gen={generation}"
                    ));
                    return ComputeStatus::MetalFailed("compute_vk_deferred_linear_note");
                }
                // The sync path writes guest pages when the GVA is mapped —
                // record the flush obligation with a defer-time page index so
                // aliased raw-GVA readers land the content first. Any prior
                // obligation for this identity is superseded content. Pages
                // resolve fully at the defer edge (never at sample time —
                // the boot-19 guard-v1 regression).
                let key = candidate.key;
                state.disarm_linear_deferred_window(&key);
                let span = key.span_end;
                // The window is armed whatever the guest has notified: `pages`
                // is the reading that matters here and it comes from the page
                // tables — an empty index is a window over memory nothing
                // resolves.
                let mut pages = std::collections::HashSet::new();
                pages.extend(crate::runtime::gva_mem::task_gva_page_gpa_set(
                    host,
                    &state.tasks,
                    task_id,
                    *gva,
                    span,
                    state.page_shift,
                ));
                let indexed = pages.len();
                state.arm_linear_deferred_window(key, generation, pages);
                crate::observe::off(format!(
                    "compute_writeback_deferred kind=linear pipe={} bind={} task={task_id} ref={texture_ref} gva={gva:#x} {width}x{height} fmt={pixel_format:#x} gen={generation} pages={indexed}",
                    acc.pipeline_ref,
                    t.binding,
                ));
                continue;
            }
            // The pinned engine resident is authoritative; guest pages are now
            // stale until a flush choke point lands the content
            // (storage_flush::flush_intersecting). Keep the protocol
            // bookkeeping the CPU write would do, then register the window in
            // the deferred-flush map.
            let (Some(candidate), TextureWriteback::Type11 { mapping_id, .. }) =
                (t.residency, &t.writeback)
            else {
                crate::observe::fail(format!(
                    "compute_writeback_deferred fail reason=missing_identity pipe={} bind={}",
                    acc.pipeline_ref, t.binding
                ));
                return ComputeStatus::MetalFailed("compute_vk_deferred_identity");
            };
            let key = candidate.key;
            let generation = next_mapping_content_generation(candidate.seed_generation);
            // Superseded stale windows intersecting this one are dead content:
            // drop them (never flush over the newer output) and release their
            // pins — except our own *storage* identity, which the engine
            // re-pinned.
            //
            // The `victim != key` exemption is about that re-pin, so it applies
            // only to a compute window. A render window can sit at the very same
            // key — same mapping, geometry, format and plane window — while its
            // pixels are in a target resident the engine has not touched, so
            // skipping it there would leak a display-sized pin for the boot.
            // `release_window_pin` picks the registry from the owner.
            for (victim, victim_owner) in
                state.take_deferred_flush_windows(*mapping_id, key.surface_offset, key.span_end)
            {
                let ours = victim == key
                    && matches!(victim_owner, crate::model::DeferredOwner::Storage { .. });
                if !ours {
                    crate::observe::off(format!(
                        "compute_writeback_deferred supersede mapping={mapping_id} victim={}x{} fmt={:#x} owner={}",
                        victim.width,
                        victim.height,
                        victim.pixel_format,
                        crate::runtime::storage_flush::owner_slug(&victim_owner)
                    ));
                    crate::runtime::storage_flush::release_window_pin(&victim, &victim_owner);
                }
            }
            state.compute_deferred_flush.insert(
                key,
                crate::model::DeferredOwner::Storage {
                    generation,
                    armed_stamp_seq: state.completion_stamp_seq,
                },
            );
            state.index_deferred_alias_pages(*mapping_id);
            let _ = state.mark_mapping_written(*mapping_id);
            note_storage_residency_writeback(state, t);
            crate::observe::off(format!(
                "compute_writeback_deferred pipe={} bind={} mapping={mapping_id} {}x{} fmt={:#x} gen={generation}",
                acc.pipeline_ref, t.binding, key.width, key.height, key.pixel_format
            ));
            continue;
        }
        t.bytes = bytes;
        if let Err(e) = writeback_texture(state, host, task_id, t) {
            return e;
        }
        note_storage_residency_writeback(state, t);
    }

    // A dispatch that completed is expected control flow; every refusal on this
    // path emits its own typed decline. The fields are this dispatch's own
    // shape — process-cumulative engine totals belong to the parity tests that
    // take a snapshot around a known workload, not to a per-dispatch line that
    // would pay a global engine lock to print them.
    crate::observe::line(format!(
        "compute_linux ok pipe={} wg=[{wg_x},{wg_y},{wg_z}] nbuf={} bro={} brw={} bunused={} ntex={}",
        acc.pipeline_ref,
        staged_bufs.len(),
        buffer_readonly_count,
        buffer_writable_count,
        buffer_unused_count,
        staged_tex.len(),
    ));
    ComputeStatus::Ok
}

#[cfg(feature = "backend-vulkan")]
const COMPUTE_ENGINE_STALL_PROXY_MS: u64 = 2_000;

/// Measurement-only watchdog for backend calls that cannot be bounded by a
/// Vulkan fence timeout (notably pipeline creation and some driver submits).
/// It never changes execution. A fired proxy preserves the private request
/// inputs under /tmp so the stall can be reproduced without another VM boot.
#[cfg(feature = "backend-vulkan")]
fn spawn_compute_engine_stall_watchdog(
    pipeline_ref: u32,
    req: &crate::backend::vulkan::engine::ComputeRequest,
    threshold: std::time::Duration,
) -> std::sync::Arc<std::sync::atomic::AtomicBool> {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let done = Arc::new(AtomicBool::new(false));
    let thread_done = Arc::clone(&done);
    let spirv = req.spirv.clone();
    let grid = req.grid;
    let buffers = req.storage_buffers.len();
    let images = req.storage_images.len();
    let image_geometry: Vec<_> = req
        .storage_images
        .iter()
        .map(|img| (img.binding, img.width, img.height))
        .collect();
    std::thread::spawn(move || {
        std::thread::sleep(threshold);
        if thread_done.load(Ordering::Acquire) {
            return;
        }
        let elapsed_ms = threshold.as_millis();
        crate::observe::fail(format!(
            "compute_engine_stall reason=backend_call_unreturned pipe={pipeline_ref} elapsed_ms={elapsed_ms} grid={grid:?} nbuf={buffers} nimg={images} image_geom={image_geometry:?}"
        ));
        let base = format!("/tmp/reims-vgpu-compute-stall-pipe-{pipeline_ref}");
        let mut bytes = Vec::with_capacity(spirv.len().saturating_mul(4));
        for word in spirv {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        if let Err(e) = std::fs::write(format!("{base}.spv"), &bytes) {
            crate::observe::fail(format!(
                "compute_engine_stall reason=spv_dump_failed pipe={pipeline_ref} err={e}"
            ));
        }
        let meta = format!(
            "pipe={pipeline_ref}\nelapsed_ms={elapsed_ms}\ngrid={grid:?}\nnbuf={buffers}\nnimg={images}\nimage_geom={image_geometry:?}\n"
        );
        if let Err(e) = std::fs::write(format!("{base}.txt"), meta) {
            crate::observe::fail(format!(
                "compute_engine_stall reason=metadata_dump_failed pipe={pipeline_ref} err={e}"
            ));
        }
    });
    done
}

#[cfg(feature = "backend-vulkan")]
fn spirv_words_le(bytes: &[u8]) -> Result<Vec<u32>, ComputeSpirvDecline> {
    const HEADER_LEN: usize = 20;
    const WORD_ALIGNMENT: usize = 4;
    if bytes.len() < HEADER_LEN {
        return Err(ComputeSpirvDecline::HeaderTooShort {
            len: bytes.len(),
            minimum: HEADER_LEN,
        });
    }
    if !bytes.len().is_multiple_of(WORD_ALIGNMENT) {
        return Err(ComputeSpirvDecline::LengthMisaligned {
            len: bytes.len(),
            alignment: WORD_ALIGNMENT,
        });
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Thin `Option` adapters over the canonical tables in
/// [`crate::backend::vulkan::translate::pixel`].
///
/// These two used to *be* the tables — a second copy of the selector→engine and
/// Metal→engine mappings living in the compute path, where nothing checked them
/// against the pixel table they had to agree with. The call sites below are all
/// `if let Some(..)` / `let Some(..) else`, so the adapters keep that shape; the
/// decision itself now happens in exactly one place.
#[cfg(feature = "backend-vulkan")]
fn simg_u32_to_engine_storage(
    simg: u32,
) -> Option<crate::backend::vulkan::engine::StorageImageFormat> {
    crate::backend::vulkan::translate::pixel::storage_image_from_selector(simg).ok()
}

#[cfg(feature = "backend-vulkan")]
fn mtl_to_engine_sampled(
    format: u16,
) -> Option<crate::backend::vulkan::engine::StorageImageFormat> {
    crate::backend::vulkan::translate::pixel::storage_image(format).ok()
}

#[cfg(feature = "backend-vulkan")]
fn spirv_image_format_to_engine_storage(
    format: crate::runtime::spirv_bind::ImageFormat,
) -> Option<crate::backend::vulkan::engine::StorageImageFormat> {
    use crate::backend::vulkan::engine::StorageImageFormat as V;
    use crate::runtime::spirv_bind::ImageFormat as S;
    Some(match format {
        S::Rgba32Float => V::Rgba32Float,
        S::Rgba16Float => V::Rgba16Float,
        S::R16Float => V::R16Float,
        S::Rgba16Uint => V::Rgba16Uint,
        S::Rgba8Uint => V::Rgba8Uint,
        S::Rgba8Sint => V::Rgba8Sint,
        S::Rgba8Unorm => V::Rgba8Unorm,
        S::Rg16Float => V::Rg16Float,
        S::R8Unorm => V::R8Unorm,
        S::Rg8Unorm => V::Rg8Unorm,
        S::Rgba32Uint => V::Rgba32Uint,
        S::R32Float => V::R32Float,
        S::R32ui => V::R32Uint,
        // Format-less (`Unknown`) storage images carry no engine texel format —
        // their view format comes from the guest surface, resolved by the caller.
        S::Unknown | S::Unsupported(_) => return None,
    })
}

#[cfg(feature = "backend-vulkan")]
/// Numeric class of a guest storage format: 0 normalized/float, 1 unsigned
/// integer, 2 signed integer.
///
/// Kept apart from the specialization table below because that table also
/// refuses formats whose storage path is unproven, and the class of a format is
/// a fact about it that holds whether or not we are willing to target it.
fn guest_numeric_class(guest: crate::backend::vulkan::engine::StorageImageFormat) -> u8 {
    use crate::backend::vulkan::engine::StorageImageFormat as V;
    match guest {
        V::Rgba32Float
        | V::Rgba16Float
        | V::R16Float
        | V::Rgba8Unorm
        | V::Bgra8Unorm
        | V::Rg16Float
        | V::R8Unorm
        | V::Rg8Unorm
        | V::R32Float
        | V::R16Unorm
        | V::Rg16Unorm
        | V::Rgb9e5Ufloat => 0,
        V::Rgba16Uint | V::Rgba8Uint | V::Rgba32Uint | V::R32Uint | V::Rg16Uint => 1,
        V::Rgba8Sint | V::R32Sint => 2,
    }
}

#[cfg(feature = "backend-vulkan")]
fn specialized_storage_image_format(
    guest: crate::backend::vulkan::engine::StorageImageFormat,
    shader: crate::runtime::spirv_bind::ImageFormat,
    write_without_format: bool,
) -> Result<crate::runtime::spirv_bind::ImageFormat, &'static str> {
    use crate::backend::vulkan::engine::StorageImageFormat as V;
    use crate::runtime::spirv_bind::ImageFormat as S;

    let Some(shader_engine) = spirv_image_format_to_engine_storage(shader) else {
        return Err("spirv_storage_format_unsupported");
    };
    // A guest BGRA8Unorm surface written by a normalized (float/unorm-class)
    // shader is a color store. SPIR-V has no `Bgra8` storage format, so a
    // concrete `Rgba8Unorm` view would store the shader's red at the guest's
    // blue byte — the resolution-independent R/B swap. Retarget to a format-less
    // `Unknown` storage image; the engine views it `B8G8R8A8_UNORM` (guest
    // channel order) and the GPU converts the written vec4 to BGRA natively, so
    // every downstream consumer (writeback, resident export, sampling) sees the
    // correct bytes with no per-frame swizzle. Requires
    // `StorageImageWriteWithoutFormat`; when absent we degrade to the swapped
    // `Rgba8Unorm` view and the caller logs the degraded class.
    //
    // A uint/sint shader over BGRA is instead a deliberate raw byte view (byte
    // order preserved, no conversion) and must keep its raw format — it falls
    // through to the raw-view / class-matched logic below, unchanged.
    if matches!(guest, V::Bgra8Unorm) {
        let normalized_color_store = matches!(
            shader,
            S::Rgba8Unorm
                | S::Rgba32Float
                | S::Rgba16Float
                | S::R16Float
                | S::R32Float
                | S::Rg16Float
                | S::R8Unorm
                | S::Rg8Unorm
        );
        if normalized_color_store {
            return Ok(if write_without_format {
                S::Unknown
            } else {
                S::Rgba8Unorm
            });
        }
    }
    // Nothing to specialize when the translator already named the guest's own
    // format. Stated before the class rules below so a guest surface whose
    // storage path is otherwise unproven (`R32Float`, `R32Sint`) is not refused
    // for a shader that declares exactly it.
    if shader_engine == guest {
        return Ok(shader);
    }

    let shader_class = match shader {
        S::Rgba32Float
        | S::Rgba16Float
        | S::R16Float
        | S::R32Float
        | S::Rgba8Unorm
        | S::Rg16Float
        | S::R8Unorm
        | S::Rg8Unorm => 0,
        S::Rgba32Uint | S::Rgba16Uint | S::Rgba8Uint | S::R32ui => 1,
        S::Rgba8Sint => 2,
        // A shader that itself declared `Unknown` (format-less) storage is not a
        // class we specialize by numeric class; the caller only mints `Unknown`
        // deliberately for the BGRA path, which returns above.
        S::Unknown | S::Unsupported(_) => return Err("spirv_storage_format_unsupported"),
    };
    // An integer-class shader over a normalized/float-class guest surface of the
    // same texel width is a deliberate raw byte view — Metal `BGRA8Unorm` bound
    // to a `texture2d<uint, write>` and translated as `Rgba8Uint` writes bytes,
    // not colours, and re-targeting it would convert values that were never meant
    // to be converted. The reverse (a float shader over an integer surface) has
    // never been captured and is refused below rather than guessed at.
    //
    // Within one class equal width means nothing, and the store is a value store:
    // `R32Float` and `Rg16Float` are both four float bytes and mean different
    // things. A `float4` written through the former stores lane `.x` as one f32,
    // which the guest then reads as two halves — so a two-channel write loses its
    // second channel outright and corrupts its first. That is measured, not
    // hypothetical: the guest's decode-time HEIC downsample writes chroma with
    // `OpVectorShuffle … 1 2 1 2` into an `Rg16Float` surface the translator
    // declared `R32f`, and the picture speckles.
    //
    // The `R32Uint` guest case used to be carved out of a bare width test by name
    // for the same reason (`Rgba8Uint` declared over one 32-bit uint channel,
    // storing only the low byte of each lane). It needs no exception now: uint
    // over uint is one class, so it reaches the class-matched table below.
    if guest.bytes_per_texel() == shader_engine.bytes_per_texel()
        && shader_class != 0
        && guest_numeric_class(guest) == 0
    {
        return Ok(shader);
    }

    let (guest_class, specialized) = match guest {
        // R32-single-channel: R32Uint is supported as a storage image by
        // re-targeting the SPIR-V to `R32ui` (its class must still match the
        // shader's numeric class below — a uint-write shader). The remaining
        // R32 sint/float and the packed Rgb9e5 stay sampled-only until a live
        // capture justifies enabling their storage path.
        V::R32Uint => (1, S::R32ui),
        // The 16-bit single- and two-channel normalized/uint formats reach the
        // engine only as sampled textures — they have no storage selector — so
        // a shader declaring one as a storage image is refused here rather than
        // given a view whose storage support was never established.
        V::R32Sint
        | V::R32Float
        | V::Rgb9e5Ufloat
        | V::R16Unorm
        | V::Rg16Unorm
        | V::Rg16Uint => {
            return Err("spirv_sampled_only_format_as_storage");
        }
        V::Rgba32Float => (0, S::Rgba32Float),
        V::Rgba16Float => (0, S::Rgba16Float),
        V::R16Float => (0, S::R16Float),
        // Bgra8Unorm normally returns above (Unknown/B8G8R8A8 view, or the
        // degraded Rgba8Unorm) before reaching here; this arm is only the
        // class/bytes fallthrough for Rgba8Unorm and a defensive default.
        V::Rgba8Unorm | V::Bgra8Unorm => (0, S::Rgba8Unorm),
        V::Rg16Float => (0, S::Rg16Float),
        V::R8Unorm => (0, S::R8Unorm),
        V::Rg8Unorm => (0, S::Rg8Unorm),
        V::Rgba32Uint => (1, S::Rgba32Uint),
        V::Rgba16Uint => (1, S::Rgba16Uint),
        V::Rgba8Uint => (1, S::Rgba8Uint),
        V::Rgba8Sint => (2, S::Rgba8Sint),
    };
    if shader_class != guest_class {
        return Err("spirv_guest_numeric_class_mismatch");
    }
    Ok(specialized)
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn stage_input_to_apv(
    si: &ComputeStageInputDescriptor,
) -> crate::backend::metal::abi::ReimsVgpuComputeStageInputDescriptor {
    use crate::backend::metal::abi::{
        ReimsVgpuComputeStageInputAttribute, ReimsVgpuComputeStageInputDescriptor,
        ReimsVgpuComputeStageInputLayout, REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_ATTRIBUTES,
        REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_LAYOUTS,
    };
    let mut out = ReimsVgpuComputeStageInputDescriptor {
        word0: si.word0,
        header0: si.header0,
        header1: si.header1,
        attribute_count: si.attributes.len() as u32,
        layout_count: si.layouts.len() as u32,
        index_type: si.index_type,
        index_buffer_index: si.index_buffer_index,
        attributes: [ReimsVgpuComputeStageInputAttribute {
            raw_bits: 0,
            location: 0,
            format: 0,
            offset: 0,
            buffer_index: 0,
            reserved0: 0,
        }; REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_ATTRIBUTES],
        layouts: [ReimsVgpuComputeStageInputLayout {
            raw_bits: 0,
            buffer_index: 0,
            step_function: 0,
            step_rate: 0,
            stride: 0,
        }; REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_LAYOUTS],
    };
    for (i, a) in si
        .attributes
        .iter()
        .enumerate()
        .take(REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_ATTRIBUTES)
    {
        out.attributes[i] = ReimsVgpuComputeStageInputAttribute {
            raw_bits: a.raw_bits,
            location: a.location,
            format: a.format,
            offset: a.offset,
            buffer_index: a.buffer_index,
            reserved0: 0,
        };
    }
    for (i, l) in si
        .layouts
        .iter()
        .enumerate()
        .take(REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_LAYOUTS)
    {
        out.layouts[i] = ReimsVgpuComputeStageInputLayout {
            raw_bits: l.raw_bits,
            buffer_index: l.buffer_index,
            step_function: l.step_function,
            step_rate: l.step_rate,
            stride: l.stride,
        };
    }
    out
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
fn execute_dispatch_metal<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    acc: &ComputeAccum,
    cmd: &ComputeCommand,
    session: Option<&mut crate::runtime::compute_session::ComputeSession>,
) -> ComputeStatus {
    use crate::backend::metal::abi::{
        ReimsVgpuComputeImageblockDimensions, ReimsVgpuComputeSampledImage,
        ReimsVgpuComputeStageInRegion, ReimsVgpuComputeStageInRegionIndirectArguments,
        ReimsVgpuComputeTextureUsage, ReimsVgpuSampler, ReimsVgpuStorageImage,
        ReimsVgpuThreadgroupMemory, REIMS_VGPU_BINDING_SAMPLER_BASE,
        REIMS_VGPU_BINDING_TEXTURE_BASE, REIMS_VGPU_COMPUTE_DISPATCH_KIND_THREADGROUPS,
        REIMS_VGPU_COMPUTE_DISPATCH_KIND_THREADS, REIMS_VGPU_COMPUTE_TEXTURE_ACCESS_READ,
        REIMS_VGPU_COMPUTE_TEXTURE_ACCESS_READ_WRITE, REIMS_VGPU_MTL_DISPATCH_TYPE_CONCURRENT,
        REIMS_VGPU_MTL_DISPATCH_TYPE_SERIAL,
    };
    use crate::backend::metal::compute::{
        compute_core, compute_encode_on_encoder, reflect_compute_textures_mtlb,
    };
    if acc.pipeline_ref == 0 {
        return ComputeStatus::MissingPipeline("compute_mtl_pipeline_ref_zero");
    }
    let Some(pipeline) = load_compute_pipeline(state, host, task_id, acc.pipeline_ref) else {
        return ComputeStatus::MissingPipeline("compute_mtl_pipeline_load");
    };
    let Some(mtlb) = load_mtlb(state, host, task_id, pipeline.kernel_func_ref) else {
        return ComputeStatus::MissingMtlb("compute_mtl_mtlb_load");
    };

    let (grid_x, grid_y, grid_z, tg_x, tg_y, tg_z, dispatch_threads) =
        match resolve_dispatch_dims_reported(state, host, task_id, cmd, acc) {
            Ok(v) => v,
            Err(e) => return e,
        };

    let dispatch_kind = if dispatch_threads {
        REIMS_VGPU_COMPUTE_DISPATCH_KIND_THREADS
    } else {
        REIMS_VGPU_COMPUTE_DISPATCH_KIND_THREADGROUPS
    };
    let dispatch_type = if acc.dispatch_type == REIMS_VGPU_MTL_DISPATCH_TYPE_CONCURRENT {
        REIMS_VGPU_MTL_DISPATCH_TYPE_CONCURRENT
    } else {
        REIMS_VGPU_MTL_DISPATCH_TYPE_SERIAL
    };

    // Stage-input descriptor from pipeline (optional).
    let reims_vgpu_stage_input = pipeline.stage_input.as_ref().map(stage_input_to_apv);

    // Direct / indirect stage-in region.
    let direct_region = acc
        .stage_in_region
        .as_ref()
        .map(|r| ReimsVgpuComputeStageInRegion {
            origin_x: r.origin_x,
            origin_y: r.origin_y,
            origin_z: r.origin_z,
            size_x: r.size_x,
            size_y: r.size_y,
            size_z: r.size_z,
        });
    let mut indirect_region_args: Option<ReimsVgpuComputeStageInRegionIndirectArguments> = None;
    if let Some(ind) = &acc.stage_in_region_indirect {
        let raw = match read_buffer_window(
            state,
            host,
            task_id,
            ind.buffer_ref,
            ind.buffer_offset,
            STAGE_IN_INDIRECT_ARGS_LEN,
        ) {
            Ok(b) => b,
            Err(e) => return e,
        };
        indirect_region_args = Some(ReimsVgpuComputeStageInRegionIndirectArguments {
            origin_x: ld32(&raw[0..]),
            origin_y: ld32(&raw[4..]),
            origin_z: ld32(&raw[8..]),
            size_x: ld32(&raw[12..]),
            size_y: ld32(&raw[16..]),
            size_z: ld32(&raw[20..]),
        });
    }
    let imageblock = acc
        .imageblock
        .as_ref()
        .map(|d| ReimsVgpuComputeImageblockDimensions {
            width: d.width,
            height: d.height,
        });
    let tg_mem: Vec<ReimsVgpuThreadgroupMemory> = acc
        .threadgroup_memory
        .iter()
        .map(|t| ReimsVgpuThreadgroupMemory {
            index: t.index,
            length: t.length,
        })
        .collect();

    let mut staged_bufs: Vec<StagedBuffer> = Vec::new();
    for b in &acc.buffers {
        match stage_buffer(state, host, task_id, b) {
            Ok(s) => staged_bufs.push(s),
            Err(e) => return e,
        }
    }

    // Texture reflection: access decides storage vs sampled materialization.
    let mut usages = vec![
        ReimsVgpuComputeTextureUsage {
            binding: 0,
            access: 0
        };
        32
    ];
    let mut usage_count = 0usize;
    let mut err_buf = [0i8; 256];
    if !acc.textures.is_empty() {
        let st = reflect_compute_textures_mtlb(
            &mtlb,
            usages.as_mut_ptr(),
            usages.len(),
            &mut usage_count,
            (err_buf.as_mut_ptr(), err_buf.len()),
        );
        if !st.is_ok() {
            return ComputeStatus::MetalBackend(st);
        }
        usages.truncate(usage_count);
    } else {
        usages.clear();
    }

    let access_for = |binding: u32| -> Option<u32> {
        usages
            .iter()
            .find(|u| u.binding == binding)
            .map(|u| u.access)
    };

    let mut staged_tex: Vec<StagedTexture> = Vec::new();
    for t in &acc.textures {
        let binding = REIMS_VGPU_BINDING_TEXTURE_BASE + t.index;
        let access = access_for(binding).unwrap_or(REIMS_VGPU_COMPUTE_TEXTURE_ACCESS_READ_WRITE);
        let is_storage = access != REIMS_VGPU_COMPUTE_TEXTURE_ACCESS_READ;
        let stage_call_started = std::time::Instant::now();
        match stage_texture_raw(state, host, task_id, t.texture_ref, binding, is_storage) {
            Ok(s) => {
                // Measure-only: localize per-texture stage cost (the
                // transition-window guest stall).
                let us = stage_call_started.elapsed().as_micros() as u64;
                if us > 1500 {
                    crate::observe::off(format!(
                        "compute_stage_slow pipe={} ref={} bind={binding} storage={} {}x{} fmt={:#x} us={us}",
                        acc.pipeline_ref,
                        t.texture_ref,
                        is_storage as u8,
                        s.width,
                        s.height,
                        s.pixel_format
                    ));
                }
                staged_tex.push(s)
            }
            Err(e) => return e,
        }
    }

    // Samplers.
    let mut reims_vgpu_samplers: Vec<ReimsVgpuSampler> = Vec::new();
    for s in &acc.samplers {
        let entry = match objects::lookup_list_entry(state, host, task_id, s.sampler_ref) {
            Some(e) => e,
            None => return ComputeStatus::MissingSampler("compute_mtl_sampler_no_entry"),
        };
        if entry.object_type != OBJECT_TYPE_TYPE7 {
            return ComputeStatus::MissingSampler("compute_mtl_sampler_wrong_type");
        }
        let desc = match objects::read_descriptor(state, host, task_id, &entry) {
            Some(d) => d,
            None => return ComputeStatus::MissingSampler("compute_mtl_sampler_no_desc"),
        };
        if desc.len() < 4 || ld32(&desc) != TYPE7_OBJECT_SAMPLER {
            return ComputeStatus::MissingSampler("compute_mtl_sampler_bad_tag");
        }
        let sd = match decode_sampler_descriptor(&desc) {
            Ok(v) => v,
            Err(_) => return ComputeStatus::MissingSampler("compute_mtl_sampler_decode"),
        };
        reims_vgpu_samplers.push(crate::runtime::metal_draw::sampler_record(
            REIMS_VGPU_BINDING_SAMPLER_BASE + s.index,
            &sd,
            s.has_lod_clamp.then_some((s.lod_min_bits, s.lod_max_bits)),
            false,
        ));
    }

    let mut reims_vgpu_bufs = abi_buffers(&mut staged_bufs);

    let mut storage: Vec<ReimsVgpuStorageImage> = Vec::new();
    let mut sampled: Vec<ReimsVgpuComputeSampledImage> = Vec::new();
    // Keep raw pointers valid: build storage/sampled from staged_tex after mut split.
    for t in &mut staged_tex {
        let Some(selector) = t.storage_selector else {
            crate::observe::fail(format!(
                "compute_metal texture_format fail reason=no_backend_selector pipe={} bind={} fmt={:#x}",
                acc.pipeline_ref, t.binding, t.pixel_format
            ));
            return ComputeStatus::Unsupported("metal_no_backend_selector");
        };
        if t.is_storage {
            storage.push(ReimsVgpuStorageImage {
                binding: t.binding,
                format: selector,
                width: t.width,
                height: t.height,
                data: t.bytes.as_mut_ptr(),
                len: t.bytes.len(),
            });
        } else {
            sampled.push(ReimsVgpuComputeSampledImage {
                binding: t.binding,
                format: selector,
                width: t.width,
                height: t.height,
                data: t.bytes.as_ptr(),
                len: t.bytes.len(),
                has_swizzle: 0,
                swizzle: [2, 3, 4, 5], // identity RGBA selectors
            });
        }
    }

    // Nested: encode onto open session encoder; writeback after segment commit.
    if let Some(sess) = session {
        let retain = match compute_encode_on_encoder(
            &sess.device,
            &sess.encoder,
            &mtlb,
            &mut reims_vgpu_bufs,
            &mut storage,
            &sampled,
            &reims_vgpu_samplers,
            &tg_mem,
            direct_region.as_ref(),
            indirect_region_args.as_ref(),
            imageblock.as_ref(),
            reims_vgpu_stage_input.as_ref(),
            dispatch_kind,
            grid_x,
            grid_y,
            grid_z,
            tg_x,
            tg_y,
            tg_z,
            (err_buf.as_mut_ptr(), err_buf.len()),
        ) {
            Ok(r) => r,
            Err(st) => return ComputeStatus::MetalBackend(st),
        };
        // Split storage textures out of staged_tex for deferred writeback alignment.
        let storage_tex: Vec<StagedTexture> =
            staged_tex.into_iter().filter(|t| t.is_storage).collect();
        if storage_tex.len() != retain.images.len() {
            return ComputeStatus::MetalFailed("compute_mtl_retain_image_count");
        }
        sess.retained.extend(retain.buffers.iter().cloned());
        sess.retained.extend(retain.indirect.iter().cloned());
        sess.nested_jobs.push(NestedDispatchJob {
            staged_bufs,
            storage_tex,
            mtl_buffers: retain.buffers,
            mtl_storage: retain.images,
        });
        return ComputeStatus::Ok;
    }

    let st = compute_core(
        &mtlb,
        &mut reims_vgpu_bufs,
        &mut storage,
        &sampled,
        &reims_vgpu_samplers,
        &tg_mem,
        direct_region.as_ref(),
        indirect_region_args.as_ref(),
        imageblock.as_ref(),
        reims_vgpu_stage_input.as_ref(),
        dispatch_kind,
        dispatch_type,
        grid_x,
        grid_y,
        grid_z,
        tg_x,
        tg_y,
        tg_z,
        (err_buf.as_mut_ptr(), err_buf.len()),
    );
    if !st.is_ok() {
        return ComputeStatus::MetalBackend(st);
    }

    for s in &staged_bufs {
        if let Err(e) = writeback_buffer(
            state,
            host,
            task_id,
            Some(acc.pipeline_ref),
            "metal_dispatch",
            s,
        ) {
            return e;
        }
    }
    for t in &staged_tex {
        if let Err(e) = writeback_texture(state, host, task_id, t) {
            return e;
        }
    }
    ComputeStatus::Ok
}

#[cfg(test)]
mod tests;
