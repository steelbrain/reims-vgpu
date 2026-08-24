//! Product-path compute bind/dispatch for `SEGMENT_TYPE_COMPUTE`.
//!
//! Executable surface:
//! - `0xd0` set compute pipeline (descriptor → kernel function MTLB + optional stage-input)
//! - `0xcb` / `0xd9` set buffers (+ optional attribute stride for dynamic stage-input layouts)
//! - `0xcf` / `0xda` set buffer offset (+ optional attribute stride)
//! - `0xce` set textures (type-2/3 GVA + IOSurface texture; sample vs storage via reflection)
//! - `0xcc` / `0xcd` set samplers (+ optional LOD clamp)
//! - `0xd1` direct stage-in region / `0xd2` indirect stage-in region (guest buffer args)
//! - `0xd3` threadgroup memory length
//! - `0xd8` imageblock dimensions
//! - `0xc8`/`0xca` direct dispatch; `0xc9`/`0xe6` indirect (guest args → direct encode)
//! - `0xdb` dispatch type (serial/concurrent)
//!
//! Fences use the stream walk (`fence_exec`). Control-flow (`0xdc`–`0xe2`) and
//! ICB execution (`0xe4`/`0xe5`) are decoded but return typed unsupported
//! refusals. Memory barriers are resolved into the next dispatch; compressed
//! texture flush remains an ordered no-op.
//!
//! Direct dispatch uses the Vulkan engine. Buffer and storage-image writeback
//! is staged through GVA or IOSurface texture mappings.

use crate::model::LoadedComputePipeline;
use crate::runtime::decode::compute::{
    BufferBinding, Command as ComputeCommand, Kind, RefBinding, SamplerBinding,
};
use crate::runtime::decode::resource::{
    decode_serializer_resource, ComputeStageInputDescriptor, Descriptor as ResourceDescriptor,
    ObjectKind,
};
use crate::runtime::draw::{host_alloc_len, StoreTargetPages};
use crate::runtime::gva_mem;
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::mapper;
use crate::runtime::mapping_write;
use crate::runtime::objects;
use crate::runtime::Device;
use reims_vgpu_core::endian::ld32;
use reims_vgpu_core::pixel_format;

/// Compute buffer slot count from the guest serializer's argument-table limit.
pub const MAX_COMPUTE_BUFFER_SLOTS: u32 = 31;
/// Cap on compute texture stream indices (Metal bind = `TEXTURE_BINDING_BASE +
/// index`). Metal's compute texture argument table, and Apple's serializer's:
/// metal2vulkan's reflected descriptor layout carries the complete range.
pub const MAX_COMPUTE_TEXTURE_SLOTS: u32 = 128;
/// Cap on compute sampler stream indices (Metal bind = `SAMPLER_BINDING_BASE +
/// index`). Metal's sampler argument table, which is genuinely 16.
pub const MAX_COMPUTE_SAMPLER_SLOTS: u32 = 16;

// The three caps above hold the same three measured numbers as
// `reims_vgpu_wire::ops::bind_limit`, and until this gate nothing compared them.
// `bind_limit`'s own module doc says the truncation "is a property of the
// stage's argument table, not of an encoder" and names
// `compute_set_textures_over_bind_limit`, `compute_set_buffers_over_bind_limit`
// and `compute_set_samplers_over_bind_limit` as the captures it was read from —
// so these are compute-rail measurements, not render ones borrowed.
//
// Only one direction is a bug. A cap **below** Apple's table is guest work this
// device refuses: `ComputeBindOverflow` reports it, but a dispatch still runs
// missing that bind, and the render rail already carries the identical gate
// (`exec::apply_binds`' three `const` assertions) for the identical fact. A cap
// **above** it is headroom, which costs nothing and is why this is `<=` rather
// than the render rail's `==` — the other direction, a slot this device accepts
// but cannot name in the descriptor band, is what the two assertions directly
// above already refuse.
//
// A drift here would otherwise surface only as dropped compute binds on a live
// guest, with correct-looking output everywhere the kernel happened not to read
// the missing slot.
const _: () = assert!(reims_vgpu_wire::ops::bind_limit::BUFFER <= MAX_COMPUTE_BUFFER_SLOTS);
const _: () = assert!(reims_vgpu_wire::ops::bind_limit::TEXTURE <= MAX_COMPUTE_TEXTURE_SLOTS);
const _: () = assert!(reims_vgpu_wire::ops::bind_limit::SAMPLER <= MAX_COMPUTE_SAMPLER_SLOTS);
/// `MTLDispatchThreadgroupsIndirectArguments` = three `uint32_t` (12 bytes).
pub const INDIRECT_THREADGROUPS_ARGS_LEN: usize = 12;
/// `MTLDispatchThreadsIndirectArguments` = six `uint32_t` (24 bytes).
pub const INDIRECT_THREADS_ARGS_LEN: usize = 24;
/// `MTLStageInRegionIndirectArguments` = six `uint32_t` (24 bytes).
pub const STAGE_IN_INDIRECT_ARGS_LEN: usize = 24;

/// A compute resource bind dropped because its slot index exceeds the
/// argument-table cap.
///
/// The guest bound a real resource (`ref != 0`, or a non-empty threadgroup
/// allocation) at a slot this device cannot represent, so the dispatch runs
/// *missing that bind* — wrong compute output with no other symptom.
///
/// The cap comparison is exclusive (`index >= MAX_*`) because each `MAX_*` is
/// the serializer's slot count, so slot `MAX` is out of range.
///
/// # It is a `Decline` rather than a `format!`, and that is the point
///
/// This was a hand-rolled line: `observe::fail(format!(…))` behind a private
/// `Mutex<HashSet<(table, index)>>`. Both halves were a second spelling of
/// something the crate already owns — `Emit::fail_once` latches on
/// `(slug, discriminant)` in one process-global set, which is the same dedup
/// with the same shape.
///
/// Keeping a private one had a cost beyond the duplication. The four slugs
/// below lived inside a format string, where nobody looking for this crate's
/// decline vocabulary would find them: a future decline spelling
/// `sampler_index_overflow` would have shared this path's latch and silenced
/// one of the two for the life of the boot, and nothing would have failed. They
/// are `slug()` bodies now for that reason.
///
/// The rendered line is unchanged but for the trailing parenthetical, which the
/// `k=v` shape has no room for and which this doc now carries instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ComputeBindOverflow {
    Buffer { index: u32, arg: u32, cap: u32 },
    Texture { index: u32, arg: u32, cap: u32 },
    Sampler { index: u32, arg: u32, cap: u32 },
}

impl ComputeBindOverflow {
    fn parts(&self) -> (u32, u32, u32) {
        match *self {
            Self::Buffer { index, arg, cap }
            | Self::Texture { index, arg, cap }
            | Self::Sampler { index, arg, cap } => (index, arg, cap),
        }
    }

    /// Emit on the fail channel, once per `(table, slot)` this boot.
    ///
    /// Runs on the drain worker (off the QEMU main core). The latch is what
    /// keeps a repeating dispatch from flooding; a healthy guest — one binding
    /// within the Metal argument-table caps — never reaches here at all.
    fn emit(self) {
        let (index, ..) = self.parts();
        crate::observe::Emit::decline("compute_bind_overflow", &self).fail_once(u64::from(index));
    }
}

impl crate::observe::Decline for ComputeBindOverflow {
    fn slug(&self) -> &'static str {
        match self {
            Self::Buffer { .. } => "buffer_index_overflow",
            Self::Texture { .. } => "texture_index_overflow",
            Self::Sampler { .. } => "sampler_index_overflow",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        let (index, arg, cap) = self.parts();
        vec![
            ("index", index.to_string()),
            ("arg", arg.to_string()),
            ("cap", cap.to_string()),
        ]
    }
}

/// The sampled-image bindings that must remain explicitly null: those the module
/// statically uses and `bound` does not cover.
///
/// Vulkan requires the pipeline layout to contain a descriptor for every
/// resource the module statically uses, and the layout this device builds is
/// assembled from what the guest bound — so a texture the kernel samples and the
/// guest left empty is absent from the layout entirely, not an unwritten slot in
/// it. Besides being undefined by the specification, that hole is fatal on one
/// of the two iGPU vendors this device supports: Mesa's Intel driver scores each
/// used binding as `(use_count << 7) / array_size` over an array it sized to
/// `max_binding + 1` and zero-filled, so it divides by zero and the host process
/// dies of `SIGFPE` inside `vkCreateComputePipelines`.
///
/// Only [`DescriptorUse::Used`] is returned, which is the bar the specification
/// actually sets. A declared-and-never-referenced variable is legal to omit and
/// must stay omitted, or the census that separated those two populations cannot
/// tell them apart any more. `Ambiguous` — two variables on one binding — is its
/// own defect and is not repaired by picking one of them.
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
    /// A bind this accumulator could not hold, and so did not record.
    ///
    /// The three bind walks skip an index past their argument table — there is
    /// no slot to put it in — and that used to be the whole of it: the walk
    /// `continue`d and the dispatch went ahead with the guest's binding simply
    /// absent, which is a wrong result rather than a refused one. Nothing
    /// downstream refuses on a missing binding, because a kernel that does not
    /// sample the slot is indistinguishable from one whose bind landed.
    ///
    /// Recording it here is what lets [`resolve_dispatch_dims_reported`] — the
    /// one gate both dispatch executors pass through — refuse instead. Sticky
    /// for the accumulator's life on purpose: the binding stays unrepresentable
    /// until the guest clears that slot, and every dispatch in between would
    /// run without it.
    pub(crate) refused_bind: Option<ComputeBindOverflow>,
}

impl ComputeAccum {
    pub fn set_pipeline(&mut self, pipeline_ref: u32) {
        if pipeline_ref != 0 {
            self.pipeline_ref = pipeline_ref;
        }
    }

    /// Retire a recorded refusal the guest has just cleared.
    ///
    /// A nil bind at the slot that overflowed says the guest no longer wants
    /// anything there, so what this accumulator holds is once again what the
    /// guest asked for and the dispatch is representable again. Without this
    /// the sticky refusal would outlive the condition that caused it and refuse
    /// every later dispatch in the encoder over a slot nobody is binding — a
    /// remembered refusal gone stale, which is a class this tree already has a
    /// scan for.
    ///
    /// Matched on the index alone. The three tables are disjoint slot spaces so
    /// a clear could in principle name another class's slot, but only one
    /// refusal is ever held and it carries the class it came from, so the pair
    /// cannot be misread.
    fn clear_refusal_at(&mut self, index: u32) {
        if self.refused_bind.is_some_and(|r| r.parts().0 == index) {
            self.refused_bind = None;
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
                self.clear_refusal_at(index);
                crate::runtime::drain::note_store_route("compute_unbind_buffer");
                continue;
            }
            if index >= MAX_COMPUTE_BUFFER_SLOTS {
                let over = ComputeBindOverflow::Buffer {
                    index,
                    arg: e.ref_,
                    cap: MAX_COMPUTE_BUFFER_SLOTS,
                };
                over.emit();
                self.refused_bind.get_or_insert(over);
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
                self.clear_refusal_at(index);
                crate::runtime::drain::note_store_route("compute_unbind_texture");
                continue;
            }
            if index >= MAX_COMPUTE_TEXTURE_SLOTS {
                let over = ComputeBindOverflow::Texture {
                    index,
                    arg: e.ref_,
                    cap: MAX_COMPUTE_TEXTURE_SLOTS,
                };
                over.emit();
                self.refused_bind.get_or_insert(over);
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
                self.clear_refusal_at(index);
                crate::runtime::drain::note_store_route("compute_unbind_sampler");
                continue;
            }
            if index >= MAX_COMPUTE_SAMPLER_SLOTS {
                let over = ComputeBindOverflow::Sampler {
                    index,
                    arg: e.ref_,
                    cap: MAX_COMPUTE_SAMPLER_SLOTS,
                };
                over.emit();
                self.refused_bind.get_or_insert(over);
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

    /// Record a `setThreadgroupMemoryLength:atIndex:` for the next dispatch.
    ///
    /// **No bound here, on purpose.** The three bind setters above each refuse a
    /// slot past a cap because the protocol states one — the guest's serializer
    /// truncates a plural bind at exactly those counts, so a record naming a
    /// higher slot cannot have come from a well-formed guest. This record is
    /// singular, carries a full `u32`, and the guest applies no bound to it, so
    /// there is no protocol cap to compare against.
    ///
    /// The Vulkan rail consumes none of these binds — SPIR-V declares workgroup shared
    /// memory statically — so a cap applied here would have taken slots away
    /// from an arm that has no table to run out of. That is the mistake
    /// [`crate::runtime::draw::MAX_SAMPLER_BIND_SLOTS`]' doc names, and a cap of
    /// 16 sat here making it until the host table size was known.
    pub fn set_threadgroup_memory(&mut self, index: u32, length: u64) {
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
/// `BackendFailed` alone spoke for 38 checks, `MissingTexture` for 25 — so a live
/// `compute_dispatches_fail` counter told you a dispatch died and nothing else.
/// The slug is what makes the class greppable; the class is what decides the
/// caller's recovery.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComputeStatus {
    Ok,

    MissingPipeline(&'static str),
    MissingMtlb(&'static str),
    MissingBuffer(&'static str),
    MissingTexture(&'static str),
    MissingSampler(&'static str),
    BadGrid(&'static str),
    GuestIo(&'static str),
    BackendFailed(&'static str),
    Unsupported(&'static str),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ComputeBarrierRefusal {
    DecodeMalformed {
        opcode: u32,
    },
    ScopeUnsupported {
        raw: u16,
    },
    ResourceUnavailable {
        index: u32,
        object_ref: u32,
    },
    FenceWaitPending {
        fence_ref: u32,
    },
    FenceUnsupported {
        fence_ref: u32,
        reason: &'static str,
    },
}

impl ComputeBarrierRefusal {
    fn latch(self) -> u64 {
        match self {
            Self::DecodeMalformed { opcode } => (2 << 60) | u64::from(opcode),
            Self::ScopeUnsupported { raw } => u64::from(raw),
            Self::ResourceUnavailable { index, object_ref } => {
                (1 << 63) | (u64::from(index) << 32) | u64::from(object_ref)
            }
            Self::FenceWaitPending { fence_ref } => (3 << 60) | u64::from(fence_ref),
            Self::FenceUnsupported { fence_ref, .. } => (4 << 60) | u64::from(fence_ref),
        }
    }
}

impl crate::observe::Decline for ComputeBarrierRefusal {
    fn slug(&self) -> &'static str {
        match self {
            Self::DecodeMalformed { .. } => "compute_barrier_decode_malformed",
            Self::ScopeUnsupported { .. } => "compute_barrier_scope_unsupported",
            Self::ResourceUnavailable { .. } => "compute_barrier_resource_unavailable",
            Self::FenceWaitPending { .. } => "compute_fence_wait_pending",
            Self::FenceUnsupported { reason, .. } => reason,
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::DecodeMalformed { opcode } => vec![("opcode", format!("{opcode:#x}"))],
            Self::ScopeUnsupported { raw } => vec![("raw", raw.to_string())],
            Self::ResourceUnavailable { index, object_ref } => vec![
                ("index", index.to_string()),
                ("object_ref", format!("{object_ref:#x}")),
            ],
            Self::FenceWaitPending { fence_ref } => {
                vec![("fence", format!("{fence_ref:#x}"))]
            }
            Self::FenceUnsupported { fence_ref, .. } => {
                vec![("fence", format!("{fence_ref:#x}"))]
            }
        }
    }
}

pub(crate) fn latch_malformed_compute_barrier(
    opcode: u32,
    seg: &mut crate::runtime::compute_session::ComputeSegment,
) -> bool {
    if !matches!(
        opcode,
        reims_vgpu_wire::ops::compute::OPCODE_MEMORY_BARRIER_RESOURCES
            | reims_vgpu_wire::ops::compute::OPCODE_MEMORY_BARRIER_SCOPE
    ) {
        return false;
    }
    seg.barrier_block
        .get_or_insert(ComputeBarrierRefusal::DecodeMalformed { opcode });
    true
}

impl crate::observe::Refusal for ComputeStatus {
    fn refusal(&self) -> Option<&'static str> {
        match self {
            // The only non-refusal. Keeping it in the same enum is what makes
            // `Emit::refusal` unable to log a success by accident.
            Self::Ok => None,

            Self::MissingPipeline(slug)
            | Self::MissingMtlb(slug)
            | Self::MissingBuffer(slug)
            | Self::MissingTexture(slug)
            | Self::MissingSampler(slug)
            | Self::BadGrid(slug)
            | Self::GuestIo(slug)
            | Self::BackendFailed(slug)
            | Self::Unsupported(slug) => Some(slug),
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        // The class next to the reason: `MissingTexture` vs `BackendFailed` is
        // what the caller acted on, and a reader correlating a log line with a
        // recovery path needs both.

        vec![("class", self.class().to_string())]
    }
}

impl ComputeStatus {
    /// The variant name, for the `class=` field and for call sites that render
    /// their own line.
    pub fn class(&self) -> &'static str {
        match self {
            Self::Ok => "ok",

            Self::MissingPipeline(_) => "missing_pipeline",
            Self::MissingMtlb(_) => "missing_mtlb",
            Self::MissingBuffer(_) => "missing_buffer",
            Self::MissingTexture(_) => "missing_texture",
            Self::MissingSampler(_) => "missing_sampler",
            Self::BadGrid(_) => "bad_grid",
            Self::GuestIo(_) => "guest_io",
            Self::BackendFailed(_) => "backend_failed",
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

/// A reflected kernel resource whose Vulkan ABI this runtime cannot yet
/// populate. Kept separate from malformed SPIR-V: the translation is valid,
/// but executing it without decoding the owner argument buffer would bind the
/// wrong resource.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComputeReflectionDecline {
    ReflectedResourceUnsupported {
        pipeline_ref: u32,
        index: u32,
        binding: Option<u32>,
        kind: &'static str,
    },
    ReflectedInterfaceUnsupported {
        pipeline_ref: u32,
        feature: &'static str,
        count: usize,
    },
}

impl crate::observe::Decline for ComputeReflectionDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::ReflectedResourceUnsupported { .. } => "compute_reflection_resource_unsupported",
            Self::ReflectedInterfaceUnsupported { .. } => {
                "compute_reflection_interface_unsupported"
            }
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::ReflectedResourceUnsupported {
                pipeline_ref,
                index,
                binding,
                kind,
            } => vec![
                ("pipeline_ref", pipeline_ref.to_string()),
                ("index", index.to_string()),
                (
                    "binding",
                    binding.map_or_else(|| "none".to_string(), |value| value.to_string()),
                ),
                ("kind", (*kind).to_string()),
            ],
            Self::ReflectedInterfaceUnsupported {
                pipeline_ref,
                feature,
                count,
            } => vec![
                ("pipeline_ref", pipeline_ref.to_string()),
                ("feature", (*feature).to_string()),
                ("count", count.to_string()),
            ],
        }
    }
}

crate::observe::decline_display!(ComputeReflectionDecline);

impl std::error::Error for ComputeReflectionDecline {}

/// Apply one decoded compute command to accum, or run a dispatch / sequencing op.
///
/// `seg` carries the whole segment's mutable state: the accum this record
/// updates, the multi-record encoder a dispatch encodes onto when one is open,
/// and the latched sequencing failure (ICB / control encode error) that refuses
/// later dispatches.
pub fn apply_record<M: HostMemory + HostOps>(
    state: &mut Device,
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

/// The `MTLDispatchType` the guest declared, or `Serial` with the substitution
/// named in the always-on log.
///
/// `WRITE_DESCRIPTOR` carries this ordinal straight off the wire and nothing
/// bounds it. An earlier consumer silently treated every value other than
/// `Concurrent` as `Serial`; this boundary instead records an unrecognised
/// ordinal before applying that compatibility substitution.
///
/// The substitution is kept rather than turned into a decline, deliberately. The
/// Metal SDK's `MTLDispatchType` has exactly `Serial` and `Concurrent`, so an
/// out-of-range ordinal here is far more likely to be *this device* reading the
/// wrong wire offset than a guest asking for something new. So it is reported
/// and counted first. If
/// `compute_dispatch_type_unknown` is ever seen, the evidence to decide arrives
/// before the behaviour change does.
fn accepted_dispatch_type(task_id: u32, declared: u32) -> u32 {
    use reims_vgpu_protocol::dispatch::{
        is_declared_dispatch_type, MTL_DISPATCH_TYPE_CONCURRENT, MTL_DISPATCH_TYPE_SERIAL,
    };
    if is_declared_dispatch_type(declared) {
        return declared;
    }
    // Counted per occurrence, reported once per value: the magnitude belongs to
    // the counter, and a second line for the same ordinal says nothing the first
    // did not.
    crate::runtime::drain::note_store_route("compute_dispatch_type_unknown");
    if crate::observe::first_sight("compute_dispatch_type_unknown", u64::from(declared)) {
        crate::observe::fail(format!(
            "compute_dispatch_type reason=compute_dispatch_type_unknown task={task_id} \
             declared={declared} (the segment is encoded Serial; MTLDispatchType has only \
             Serial={MTL_DISPATCH_TYPE_SERIAL} and \
             Concurrent={MTL_DISPATCH_TYPE_CONCURRENT})"
        ));
    }
    MTL_DISPATCH_TYPE_SERIAL
}

fn resolved_compute_barrier<M: HostMemory + HostOps>(
    state: &Device,
    host: &M,
    task_id: u32,
    cmd: &ComputeCommand,
) -> Result<Option<reims_vgpu_core::ComputeBarrier>, ComputeBarrierRefusal> {
    match cmd.kind {
        Kind::BarrierResources => {
            let mut resources = Vec::with_capacity(cmd.resources.len());
            for (index, object_ref) in cmd.resources.iter().enumerate() {
                let raw = object_ref.get();
                let resource =
                    objects::resolve_resource(state, host, task_id, raw).map_err(|_| {
                        ComputeBarrierRefusal::ResourceUnavailable {
                            index: index as u32,
                            object_ref: raw,
                        }
                    })?;
                let id =
                    resource
                        .semantic_id()
                        .ok_or(ComputeBarrierRefusal::ResourceUnavailable {
                            index: index as u32,
                            object_ref: raw,
                        })?;
                resources.push(reims_vgpu_core::BarrierResource {
                    id,
                    lifetime: resource.lifetime(),
                });
            }
            if resources.is_empty() {
                Ok(None)
            } else {
                Ok(Some(reims_vgpu_core::ComputeBarrier::Resources(
                    resources.into(),
                )))
            }
        }
        Kind::BarrierScope => {
            let scope = reims_vgpu_core::MemoryBarrierScope::from_bits(cmd.barrier_scope).ok_or(
                ComputeBarrierRefusal::ScopeUnsupported {
                    raw: cmd.barrier_scope,
                },
            )?;
            if scope.is_empty() {
                Ok(None)
            } else {
                Ok(Some(reims_vgpu_core::ComputeBarrier::Scope(scope)))
            }
        }
        _ => unreachable!("compute barrier kind checked by caller"),
    }
}

fn retire_pending_compute_barriers(
    status: ComputeStatus,
    pending: &mut Vec<reims_vgpu_core::ComputeBarrier>,
) -> ComputeStatus {
    if status == ComputeStatus::Ok {
        pending.clear();
    }
    status
}

fn apply_record_inner<M: HostMemory + HostOps>(
    state: &mut Device,
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
            seg.acc.dispatch_type = accepted_dispatch_type(task_id, cmd.dispatch_type);
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
            if let Some(refusal) = seg.barrier_block {
                return Some(ComputeStatus::Unsupported(crate::observe::Decline::slug(
                    &refusal,
                )));
            }
            // Open multi-record session (control-flow SPI): encode on that encoder.
            if let Some(sess) = seg.session.as_mut() {
                return Some(execute_dispatch_nested(
                    state, host, task_id, &seg.acc, cmd, sess,
                ));
            }
            let status =
                execute_dispatch(state, host, task_id, &seg.acc, cmd, &seg.pending_barriers);
            Some(retire_pending_compute_barriers(
                status,
                &mut seg.pending_barriers,
            ))
        }
        Kind::UpdateFence => {
            let status = crate::runtime::fence_exec::execute_fence(
                state,
                task_id,
                reims_vgpu_core::SynchronizationDomain::ComputeFence,
                cmd.fence_ref,
                reims_vgpu_core::FenceAction::Update,
            );
            if let crate::runtime::fence_exec::FenceStatus::Unsupported(reason) = status {
                seg.barrier_block
                    .get_or_insert(ComputeBarrierRefusal::FenceUnsupported {
                        fence_ref: cmd.fence_ref,
                        reason,
                    });
            }
            None
        }
        Kind::WaitFence => {
            use crate::runtime::fence_exec::FenceStatus;
            let status = crate::runtime::fence_exec::execute_fence(
                state,
                task_id,
                reims_vgpu_core::SynchronizationDomain::ComputeFence,
                cmd.fence_ref,
                reims_vgpu_core::FenceAction::Wait,
            );
            match status {
                FenceStatus::Ok => seg
                    .pending_barriers
                    .push(reims_vgpu_core::ComputeBarrier::Fence),
                FenceStatus::Pending => {
                    let refusal = ComputeBarrierRefusal::FenceWaitPending {
                        fence_ref: cmd.fence_ref,
                    };
                    crate::observe::Emit::decline("compute_fence", &refusal)
                        .field("task", task_id)
                        .fail_once(refusal.latch());
                    seg.barrier_block.get_or_insert(refusal);
                }
                FenceStatus::Unsupported(reason) => {
                    seg.barrier_block
                        .get_or_insert(ComputeBarrierRefusal::FenceUnsupported {
                            fence_ref: cmd.fence_ref,
                            reason,
                        });
                }
                FenceStatus::Missing => {}
            }
            None
        }
        Kind::BarrierResources | Kind::BarrierScope => {
            match resolved_compute_barrier(state, host, task_id, cmd) {
                Ok(Some(barrier)) => {
                    seg.pending_barriers.push(barrier);
                    crate::runtime::drain::note_store_route("compute_barrier_pending");
                }
                Ok(None) => crate::runtime::drain::note_store_route("compute_barrier_empty"),
                Err(refusal) => {
                    crate::observe::Emit::decline("compute_barrier", &refusal)
                        .field("task", task_id)
                        .fail_once(refusal.latch());
                    seg.barrier_block.get_or_insert(refusal);
                }
            }
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

/// What a compute pipeline's stage-input block means for the pipeline carrying it.
///
/// Three outcomes, and the whole point of naming them is that two of them are
/// not the same: [`Self::Absent`] is a kernel that declares no per-thread input,
/// and [`Self::OverCap`] is one that declares more than this decoder kept. They
/// used to collapse into one `None`.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum StageInputVerdict {
    /// No block, or a block naming neither an attribute nor a layout.
    Absent,
    /// Carry it to the backend.
    Use,
    /// The decoder dropped entries. Refuse the pipeline.
    OverCap,
}

/// Classify a decoded stage-input block. Free function so the distinction above
/// is testable without a device, a host or a resolvable descriptor.
pub(crate) fn classify_stage_input(si: Option<&ComputeStageInputDescriptor>) -> StageInputVerdict {
    let Some(si) = si else {
        return StageInputVerdict::Absent;
    };
    if si.dropped_attributes != 0 || si.dropped_layouts != 0 {
        return StageInputVerdict::OverCap;
    }
    if si.attributes.is_empty() && si.layouts.is_empty() {
        return StageInputVerdict::Absent;
    }
    StageInputVerdict::Use
}

pub(crate) fn load_compute_pipeline<M: HostMemory + HostOps>(
    state: &Device,
    host: &M,
    task_id: u32,
    pipeline_ref: u32,
) -> Option<std::sync::Arc<LoadedComputePipeline>> {
    // ref==0 is "no pipeline bound" (legitimate) — silent. Other None = a bound
    // pipeline that failed to materialize → caller's coarse MissingPipeline; log
    // the reason (audit).
    if pipeline_ref == 0 {
        return None;
    }
    if let Some(pipeline) = state.task_objects.compute_pipelines.get(
        task_id,
        reims_vgpu_protocol::SerializerRef::new(pipeline_ref),
    ) {
        crate::runtime::drain::note_store_route("compute_pipeline_state_hit");
        return Some(pipeline);
    }
    crate::runtime::drain::note_store_route("compute_pipeline_state_miss");
    let report = crate::observe::RungReport::new("compute_load_pipeline", "pipe_ref");
    let miss = |reason: &str, detail: String| -> Option<std::sync::Arc<LoadedComputePipeline>> {
        report.reason(task_id, pipeline_ref, reason, &detail);
        None
    };
    let (_entry, desc) = match objects::resolve_descriptor(
        state,
        host,
        task_id,
        pipeline_ref,
        &[ObjectKind::SerializerResource],
    ) {
        Ok(found) => found,
        Err(rung) => {
            report.rung(task_id, pipeline_ref, rung);
            return None;
        }
    };
    let Ok(decoded) = decode_serializer_resource(&desc) else {
        return miss(
            crate::observe::ladder_slug!("", desc_decode),
            format!("desc_len={}", desc.len()),
        );
    };
    match decoded {
        ResourceDescriptor::ComputePipeline(cp) if cp.kernel_func_ref != 0 => {
            // A descriptor that named more entries than the decoder kept refuses
            // the whole pipeline. Dropping only the stage-input is not "failing
            // closed": `stage_input: None` is what a kernel declaring no
            // per-thread input looks like, so the two become indistinguishable
            // and the dispatch runs with its stage_in fetch silently absent. On
            // the Vulkan arm it is worse than wrong output — `compute_linux`
            // refuses any pipeline carrying a stage-input, and a dropped one
            // walked straight past that refusal.
            let stage_input = match classify_stage_input(cp.stage_input.as_ref()) {
                StageInputVerdict::Absent => None,
                StageInputVerdict::Use => cp.stage_input,
                StageInputVerdict::OverCap => {
                    let si = cp.stage_input.as_ref().expect("OverCap implies a block");
                    return miss(
                        "stage_input_over_cap",
                        format!(
                            "attrs={} dropped_attrs={} layouts={} dropped_layouts={}",
                            si.attributes.len(),
                            si.dropped_attributes,
                            si.layouts.len(),
                            si.dropped_layouts
                        ),
                    );
                }
            };
            let Some(kernel_mtlb) = crate::runtime::mtlb::load_mtlb(
                state,
                host,
                task_id,
                cp.kernel_func_ref,
                crate::runtime::mtlb::AirLoadRail::Compute,
            ) else {
                return miss("kernel_function_missing", String::new());
            };
            let pipeline = std::sync::Arc::new(LoadedComputePipeline {
                kernel_func_ref: cp.kernel_func_ref,
                kernel_mtlb,
                stage_input,
            });
            Some(state.task_objects.compute_pipelines.register(
                task_id,
                reims_vgpu_protocol::SerializerRef::new(pipeline_ref),
                pipeline,
            ))
        }
        ResourceDescriptor::ComputePipeline(_) => miss("kernel_func_zero", String::new()),
        _ => miss("not_compute_pipeline", String::new()),
    }
}

/// Read `len` bytes from a type-1 buffer at `offset` (product + session helpers).
pub(crate) fn read_buffer_window<M: HostMemory + HostOps>(
    state: &Device,
    host: &M,
    task_id: u32,
    buffer_ref: u32,
    offset: u64,
    len: usize,
) -> Result<Vec<u8>, ComputeStatus> {
    // `ref == 0` is the crate-wide unbound sentinel, not object-list index 0 —
    // every sibling loader guards it and `objects::resolve_descriptor`'s doc says
    // so. Kept as its own refusal rather than folded into the rungs: "the guest
    // bound no buffer" and "the guest named a buffer that is not there" are
    // different statements, and only the second is a resolution failure.
    if buffer_ref == 0 {
        return Err(ComputeStatus::MissingBuffer("compute_buf_win_ref_unbound"));
    }
    // Every other refusal gets its own name too. This used to call a local
    // `Option`-returning helper and label all four `compute_buf_win_no_backing` —
    // the *last* of the four, and so wrong about a ref that names nothing, a ref
    // holding some other object, and a descriptor that would not read or decode.
    let (base, size) =
        objects::resolve_buffer_span(state, host, task_id, buffer_ref).map_err(|refusal| {
            ComputeStatus::MissingBuffer(match refusal {
                objects::BufferSpanRefusal::Rung(rung) => {
                    crate::observe::ladder_slugs!("compute_buf_win")(rung)
                }
                objects::BufferSpanRefusal::Decode => {
                    crate::observe::ladder_slug!("compute_buf_win", desc_decode)
                }
                objects::BufferSpanRefusal::NoBacking => "compute_buf_win_no_backing",
            })
        })?;
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
    /// Metal input and Vulkan host-fallback output. Vulkan guest-backed input
    /// lives in `input` and leaves this empty until a host readback occurs.
    pub bytes: Vec<u8>,
    input: VulkanBufferInput,
    /// Guest pages this buffer resolved to when it was staged — before the
    /// dispatch, and before a nested session accumulated however many more
    /// jobs before flushing. `writeback_buffer` runs at the far end of that
    /// gap, so a walk taken there answers where the address points now rather
    /// than whether it is still this buffer's memory. Empty when the
    /// stage-time walk resolved nothing, which leaves the write unbounded as
    /// it was; the writer's own walk then fails closed on its own terms.
    pub pages: std::collections::HashSet<u64>,
}

enum VulkanBufferInput {
    HostBytes(Vec<u8>),
    GuestPages(reims_vgpu_memory::GuestRunSource),
}

#[derive(Clone, Copy)]
struct BufferStagePlan {
    base_gva: u64,
    size: u64,
    is_private: bool,
    full: u64,
    avail: u64,
    want: usize,
    gva: u64,
}

fn resolve_buffer_stage_plan<M: HostMemory + HostOps>(
    state: &Device,
    host: &M,
    task_id: u32,
    bind: &ComputeBufferBind,
    extent_cap: Option<u64>,
) -> Result<BufferStagePlan, ComputeStatus> {
    // Eight distinct checks answer with `MissingBuffer`; the status carries
    // which one, so the caller's line and this one name the same slug.
    let miss = |st: ComputeStatus, detail: String| -> Result<BufferStagePlan, ComputeStatus> {
        crate::observe::fail(format!(
            "compute_stage_buf fail reason={} ref={} off={:#x} {detail}",
            st.reason(),
            bind.buffer_ref,
            bind.offset
        ));
        Err(st)
    };
    let resource = match objects::resolve_resource(state, host, task_id, bind.buffer_ref) {
        Ok(resource) if resource.entry().kind == ObjectKind::Buffer => resource,
        Ok(resource) => {
            return miss(
                ComputeStatus::MissingBuffer(crate::observe::ladder_slug!(
                    "compute_stage_buf",
                    wrong_type
                )),
                format!("ot={}", resource.entry().kind),
            );
        }
        Err(rung) => {
            return miss(
                ComputeStatus::MissingBuffer(crate::observe::ladder_slugs!("compute_stage_buf")(
                    rung,
                )),
                match rung {
                    objects::LadderRung::WrongType { got } => format!("ot={got}"),
                    objects::LadderRung::NoListEntry | objects::LadderRung::DescRead { .. } => {
                        String::new()
                    }
                },
            )
        }
    };
    let Ok(ResourceDescriptor::Buffer(desc)) = objects::decoded_resource(&resource) else {
        return miss(
            ComputeStatus::MissingBuffer(crate::observe::ladder_slug!(
                "compute_stage_buf",
                desc_decode
            )),
            format!("desc_len={}", resource.descriptor().len()),
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
    let full = size - bind.offset;
    let avail = extent_cap.map_or(full, |cap| full.min(cap));
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
    Ok(BufferStagePlan {
        base_gva,
        size,
        is_private: desc.is_private,
        full,
        avail,
        want,
        gva,
    })
}

pub(crate) fn stage_buffer_with_extent<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    bind: &ComputeBufferBind,
    extent_cap: Option<u64>,
) -> Result<StagedBuffer, ComputeStatus> {
    let plan = resolve_buffer_stage_plan(state, host, task_id, bind, extent_cap)?;
    {
        let BufferStagePlan {
            base_gva,
            size,
            is_private,
            full,
            avail,
            want,
            gva,
        } = plan;
        if !is_private
            && crate::runtime::bound_buffers::ensure_packed_resource(
                state,
                host,
                task_id,
                bind.buffer_ref,
                base_gva,
                size,
                crate::runtime::bound_buffers::PackedResourceUse::Buffer,
            )
        {
            let guest = state
                .bound_buffers
                .packed_available(task_id, bind.buffer_ref, base_gva, size)
                .and_then(|packed| {
                    Some((
                        packed.buffer_source(bind.offset, want as u64)?,
                        packed.window_pages(bind.offset, want as u64)?,
                    ))
                });
            if let Some((source, pages)) = guest {
                if avail < full {
                    crate::runtime::drain::note_store_route("compute_buffer_extent_narrowed");
                    crate::runtime::drain::note_store_route_n(
                        "compute_buffer_extent_saved_bytes",
                        full - avail,
                    );
                }
                crate::runtime::drain::note_store_route("compute_buffer_guest_pages");
                return Ok(StagedBuffer {
                    bind: bind.clone(),
                    gva,
                    bytes: Vec::new(),
                    input: VulkanBufferInput::GuestPages(source),
                    pages,
                });
            }
        }
    }

    materialize_buffer_host(state, host, task_id, bind, plan)
}

fn materialize_buffer_host<M: HostMemory + HostOps>(
    state: &Device,
    host: &M,
    task_id: u32,
    bind: &ComputeBufferBind,
    plan: BufferStagePlan,
) -> Result<StagedBuffer, ComputeStatus> {
    let BufferStagePlan {
        base_gva,
        size,
        is_private: _,
        full,
        avail,
        want,
        gva,
    } = plan;
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
    // Count only a cap that actually staged. A failed walk saved no traffic and
    // must not make the rail look effective merely because reflection answered.
    if avail < full {
        crate::runtime::drain::note_store_route("compute_buffer_extent_narrowed");
        crate::runtime::drain::note_store_route_n(
            "compute_buffer_extent_saved_bytes",
            full - avail,
        );
    }
    let pages = staged_span_pages(state, host, task_id, gva, bytes.len() as u64);
    let input = VulkanBufferInput::HostBytes(std::mem::take(&mut bytes));
    Ok(StagedBuffer {
        bind: bind.clone(),
        gva,
        bytes,
        input,
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
        ///
        /// Ordered as well as a membership set, because the GPU-direct arm reads
        /// index `i` as page `i` of the window. See [`staged_window_pages`].
        pages: StoreTargetPages,
    },
    IOSurface {
        mapping_id: u32,
        /// The window this bind was staged against — a byte offset into the
        /// mapping, the surface's row pitch, and one past the last byte the
        /// window may touch.
        ///
        /// Resolved once, at stage time, through the plane the bind actually
        /// names: `iosurface_plane_view_sample_window` when the wire carried a IOSurface plane view view's
        /// plane index, `iosurface_texture_sample_window` otherwise. Both the read that
        /// seeds the image and the write that lands it use exactly these three
        /// numbers, so the two cannot name different bytes of one surface.
        surface_offset: u64,
        surface_bpr: u32,
        span_end: u64,
        width: u32,
        height: u32,
        /// The guest pixel format the window above was resolved against, and
        /// the only texel measurement this record carries.
        ///
        /// It is the bind's own staged format rather than the mapping's current
        /// declaration, because every byte offset above is arithmetic over it:
        /// judging the same window under a declaration that has since changed
        /// would be judging a different window from the one staged. Bytes per
        /// texel is derived from it at each consumer rather than carried
        /// alongside, so the two cannot disagree.
        format: u16,
    },
}

/// Guest pages a linear storage window resolves to at stage time.
///
/// Taken before the dispatch so the record names the memory the *command* was
/// issued against, not whatever the address points at once the GPU is done. An
/// empty record means the walk resolved nothing and the writeback stays
/// unbounded, which is what it was before this existed.
///
/// # Why this keeps the walk's order and not just its membership
///
/// The walk visits every page of the span in guest-virtual order and reports an
/// unresolved one as `None` rather than skipping it. Collecting straight into a
/// `HashSet` — which this did — throws both of those away, and neither can be
/// recovered afterwards: sorting the set yields ascending *physical* order,
/// which is not the window's order once the guest's mapping is scattered, and
/// nothing in a set says whether a page went missing.
///
/// The row-by-row host writer only ever asked "is this page one of mine?", so
/// the loss did not show. A GPU-direct copy asks the other question — it reads
/// index `i` as page `i` of the window — and a short or reordered vector lands
/// the frame at the wrong guest addresses with nothing noticing, because the
/// copy converts nothing and checks nothing. [`StoreTargetPages`] is the render
/// rail's existing answer to exactly this and carries both forms from the one
/// walk, so this rail takes that type rather than growing a third spelling.
fn staged_window_pages<M: HostMemory>(
    state: &Device,
    host: &M,
    task_id: u32,
    gva: u64,
    row_stride: u64,
    height: u32,
) -> StoreTargetPages {
    let Some(span) = row_stride.checked_mul(height as u64) else {
        return StoreTargetPages::empty();
    };
    if gva == 0 || span == 0 {
        return StoreTargetPages::empty();
    }
    let ordered = crate::runtime::gva_mem::task_gva_page_gpas(
        host,
        &state.tasks,
        task_id,
        gva,
        span,
        state.page_shift,
    );
    StoreTargetPages::from_ordered(&ordered, span)
}

/// Where one compute storage image's output should land.
///
/// `GuestPages` when this device can put the dispatch's own copy straight into
/// the guest's RAM, `Host` when it has to read the pixels back and write them
/// itself. **Every decline here costs a device→host crossing and no guest
/// work**: `Host` is the general path, is what a host without the guest-RAM
/// import runs for everything, and lands identical bytes. So these are routed
/// on the `OFF` channel as a census rather than refused on the fail channel —
/// nothing is lost, and the counters are what say how much of a boot's compute
/// readback this arm can actually reach.
///
/// Two conditions, and each names a contract term rather than an observation:
///
/// - the writeback must be a **guest-linear plane**. An IOSurface texture destination is a
///   tiled surface mapping, which [`crate::runtime::render_writeback::GvaPlaneDestination`]
///   cannot describe and the licence therefore cannot walk. It is the largest
///   class this arm does not reach, so [`note_iosurface_texture_shape`] bands how much of
///   it a raw copy could ever serve — see that function for why the route
///   counter alone does not say.
/// - the licence must be granted. That is where the format, the complete page
///   walk, the texel alignment and the guest-RAM references are all checked, in
///   the one place both GPU-side writers of a guest plane meet them.
///
/// # Residency is not a third condition
///
/// It was, for one boot, and that restriction reached 81 of the 89 linear
/// windows a driven macos-13 boot produces — so a rule written to be safe was
/// most of the traffic this arm exists to remove. What it was protecting against
/// is real but is not the reclaim: both reclaim paths already skip a resident
/// whose `gpu_only_content` holds, and every executed dispatch sets that flag.
/// The actual window is a **re-key**, which destroys the held image when the same
/// identity arrives at a new shape, and the pin is what refuses it. The engine
/// now takes that pin itself when it arms the write debt — see
/// `GuestWriteSource::ResidentStorage` — and releases it from the ring slot's
/// cleanup, after the fence. So a resident is held for exactly the window a
/// submitted-not-waited copy needs, and the destination no longer has to care.
fn direct_destination<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    tex: &StagedTexture,
    held: reims_vgpu_protocol::StorageImageFormat,
) -> reims_vgpu_core::ComputeImageDestination {
    use reims_vgpu_core::ComputeImageDestination;
    let TextureWriteback::Linear {
        texture_ref,
        gva,
        pixel_format,
        row_stride,
        width,
        height,
        pages,
        ..
    } = &tex.writeback
    else {
        return iosurface_texture_destination(state, host, tex, held);
    };
    let Ok(row_stride) = u32::try_from(*row_stride) else {
        crate::runtime::drain::note_store_route("compute_dst_host_stride_width");
        return ComputeImageDestination::Host;
    };
    let plane = crate::runtime::render_writeback::GvaPlaneDestination {
        target_gva: *gva,
        width: *width,
        height: *height,
        row_stride,
        format: *pixel_format,
        texture_ref: *texture_ref,
    };
    match crate::runtime::render_writeback::licence_gva_plane(
        state,
        host,
        held,
        &plane,
        Some(pages),
    ) {
        Ok(licence) => {
            crate::runtime::drain::note_store_route("compute_dst_guest_pages");
            // A split of the line above, so the two add up to it. Worth counting
            // separately because the resident half is the half that needs the
            // engine's pin, and it is the half that used to read back: a boot
            // where it stays at zero is a boot where the pin never had to work.
            crate::runtime::drain::note_store_route(if tex.residency.is_some() {
                "compute_dst_guest_pages_resident"
            } else {
                "compute_dst_guest_pages_transient"
            });
            ComputeImageDestination::GuestPages {
                target: Box::new(licence.target),
                pages: licence.gpas,
            }
        }
        Err(decline) => {
            // Named, because the reasons are not interchangeable and the census
            // above cannot tell them apart: a format the copy cannot land raw is
            // a different thing to learn about this rail than a page walk that
            // came up short.
            crate::observe::off(format!(
                "compute_dst host bind={} gva={gva:#x} dims={width}x{height} fmt={pixel_format:#x} reason={decline:?}",
                tex.binding
            ));
            crate::runtime::drain::note_store_route("compute_dst_host_unlicensed");
            ComputeImageDestination::Host
        }
    }
}

/// [`direct_destination`] for an IOSurface texture surface mapping.
///
/// A tiled surface mapping is not a guest-linear plane and the GVA licence
/// cannot describe one — but it is not therefore unreachable, and treating it as
/// such is what this arm used to do. It answered `Host` before looking at
/// anything, and on a driven macos-13 boot that was 35 of the 51 storage
/// destinations, every one of them a device→host crossing.
///
/// The destination that *can* describe it already existed on the render rail,
/// resolving the sample window, walking the mapping's page entries and building
/// the same [`reims_vgpu_memory::GuestPageTarget`] this rail wants.
/// It is now [`crate::runtime::mapping_write::licence_iosurface_texture_surface`] and both
/// rails ask it, so the surface geometry, the format rule, the page walk and the
/// guest-RAM references have one spelling rather than two.
///
/// Every decline is a routing answer on the `OFF` channel, not a loss: readback
/// lands identical bytes, and on a host without the guest-RAM import it is the
/// only rail there is.
fn iosurface_texture_destination<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    tex: &StagedTexture,
    held: reims_vgpu_protocol::StorageImageFormat,
) -> reims_vgpu_core::ComputeImageDestination {
    use reims_vgpu_core::ComputeImageDestination;
    let TextureWriteback::IOSurface {
        mapping_id,
        surface_offset,
        surface_bpr,
        span_end,
        width,
        height,
        format,
        ..
    } = &tex.writeback
    else {
        // A storage image the guest gave nowhere to land. Not a destination this
        // arm declined — there is no destination.
        crate::runtime::drain::note_store_route("compute_dst_no_writeback");
        return ComputeImageDestination::Host;
    };
    // The window this bind staged against, not one resolved here. It is already
    // plane-correct for a IOSurface plane view view and already a sub-rectangle where the
    // dispatch writes one, and it is the same window the readback rail lands
    // through — so the two rails cannot name different bytes of one surface.
    match crate::runtime::mapping_write::licence_iosurface_texture_surface(
        state,
        host,
        held,
        &crate::runtime::mapping_write::IOSurfaceDestination {
            mapping_id: *mapping_id,
            base_off: *surface_offset,
            bpr: *surface_bpr,
            span_end: *span_end,
            width: *width,
            height: *height,
            format: *format,
        },
    ) {
        Ok(licence) => {
            crate::runtime::drain::note_store_route("compute_dst_guest_pages");
            crate::runtime::drain::note_store_route(if tex.residency.is_some() {
                "compute_dst_guest_pages_iosurface_texture_resident"
            } else {
                "compute_dst_guest_pages_iosurface_texture_transient"
            });
            ComputeImageDestination::GuestPages {
                target: Box::new(licence.target),
                pages: licence.gpas,
            }
        }
        Err(decline) => {
            // Named, because the reasons are not interchangeable and the route
            // counter cannot tell them apart. The one that dominates is
            // `ResidentFormatMismatch` — a storage image's format comes from the
            // specialized SPIR-V texel format and owes the mapping's declaration
            // nothing — and no copy can serve those, so a boot where it is most
            // of this counter is this arm working rather than failing.
            crate::observe::off(format!(
                "compute_dst_iosurface_texture bind={} mid={mapping_id} dims={width}x{height} held={held:?} reason={decline:?}",
                tex.binding
            ));
            crate::runtime::drain::note_store_route(
                "compute_dst_host_iosurface_texture_unlicensed",
            );
            ComputeImageDestination::Host
        }
    }
}

/// [`staged_window_pages`] for a flat span — the buffer rail's shape.
fn staged_span_pages<M: HostMemory>(
    state: &Device,
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
    /// Object-list reference resolved at staging time. Content effects use
    /// this identity after the executor receipt arrives; the destination
    /// shape below is only a placement and must not stand in for identity.
    pub resource_ref: u32,
    pub binding: u32,
    pub array_element: u32,
    pub descriptor_count: u32,

    /// Raw Metal pixel format from the exact texture/view descriptor.
    pub pixel_format: u16,
    /// Semantic storage-image format for this texture, or `None` when the Metal
    /// format is sampled-only. This exact value is supplied to metal2vulkan's
    /// runtime specialization API.
    pub storage_format: Option<reims_vgpu_protocol::StorageImageFormat>,
    /// Component mapping declared by a semantic texture view. The base Metal
    /// format's own mapping is composed with this when the sampled request is
    /// built; storage binds require identity and refuse earlier.
    pub view_swizzle: reims_vgpu_protocol::SwizzlePlan,
    pub width: u32,
    pub height: u32,
    /// Shader-required sample axis. Multisampled binds are admitted only from
    /// the exact render-target resident; flat guest bytes cannot represent it.
    pub multisampled: bool,
    /// post-dispatch host result only; the pre-dispatch source is the typed
    /// `input` below and never consults this field.
    pub bytes: Vec<u8>,
    pub is_storage: bool,
    residency: Option<ComputeStorageResidencyCandidate>,
    input: VulkanTextureInput,
    writeback: TextureWriteback,
}

enum VulkanTextureInput {
    HostBytes(Vec<u8>),
    GuestPages(reims_vgpu_memory::GuestRunSource),
    GuestImage(reims_vgpu_memory::GuestImageSource),
    TargetResident(crate::model::TargetIdentity),
    Resident(ResidentServe),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ComputeTextureStage {
    Sampled2d,
    Sampled2dMultisample,
    Storage2d,
}

impl ComputeTextureStage {
    const fn is_storage(self) -> bool {
        matches!(self, Self::Storage2d)
    }

    const fn is_multisampled(self) -> bool {
        matches!(self, Self::Sampled2dMultisample)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ComputeStorageResidencyCandidate {
    key: crate::model::ComputeStorageResidencyKey,
    seed_generation: u32,
}

fn note_storage_residency_writeback(state: &mut Device, texture: &StagedTexture) {
    let Some(candidate) = texture.residency else {
        return;
    };
    // Linear windows keep their authority in the host_linear_textures entry
    // (resident_gen), never in the mapping-keyed mirror.
    if candidate.key.is_linear() {
        return;
    }
    if candidate.key.is_heap() {
        state.content.compute_residency.publish(
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
    // invalidation — kept here as defense in depth). Disjoint siblings are
    // independent live content and remain until their mapping/resource
    // lifetime or a write to their own window retires them.
    let Some((mapping_id, surface_offset, span_end)) = candidate.key.surface_window() else {
        return;
    };
    state.invalidate_storage_residency_window(mapping_id, surface_offset, span_end);
    state
        .content
        .compute_residency
        .publish(candidate.key, generation);
}

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
fn log_storage_image_access(pipe: u32, binding: u32, access: &str, bytes: u64) {
    crate::observe::off(format!(
        "compute_linux storage_access pipe={pipe} bind={binding} access={access} seed=1 bytes={bytes}"
    ));
}

/// What the engine-resident image of a window can serve one staged binding.
///
/// `Seed` means a storage binding's output is already GPU-resident at this
/// generation, so the guest read that would seed it is unnecessary. `Sample`
/// names the resident key a sampled binding reads directly instead.
///
/// Which variant a binding can receive is fixed by `is_storage`, not chosen:
/// [`resident_serve`]'s two arms are the two variants. That is why the
/// consumers split the same way — the storage rail reads only the seed and the
/// sampled rail only the source.
///
/// Declared unconditionally although only the Vulkan backend can produce one,
/// name its type. Each used to substitute its own loose tuple of the same
/// fields in a second backend arm, spelled out once per rail.
/// [`resident_serve`] is the only producer and it is gated on the Vulkan
/// nowhere. The rails still read the type — `serve` is `None` there and their
/// accessor calls compile unchanged — which is the whole point of declaring it
/// unconditionally.
#[derive(Clone, Copy)]
pub(crate) enum ResidentServe {
    Seed(u32),
    Sample(crate::model::ComputeStorageResidencyKey, u32),
}

impl ResidentServe {
    /// The generation a seeded resident is held at, or `None` for a sampled
    /// one — whose generation belongs to its key rather than to the guest read
    /// this binding skipped.
    pub(crate) fn seed_generation(self) -> Option<u32> {
        match self {
            Self::Seed(generation) => Some(generation),
            Self::Sample(..) => None,
        }
    }

    /// The resident a sampled binding reads directly, or `None` for a seeded
    /// one.
    pub(crate) fn sample_source(self) -> Option<(crate::model::ComputeStorageResidencyKey, u32)> {
        match self {
            Self::Sample(key, generation) => Some((key, generation)),
            Self::Seed(_) => None,
        }
    }
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
pub(crate) fn resident_serve(
    executor: &dyn crate::runtime::executor::Executor,
    key: crate::model::ComputeStorageResidencyKey,
    mirror_generation: u32,
    is_storage: bool,
    pixel_format: u16,
) -> Option<ResidentServe> {
    if is_storage {
        return (executor.compute_resident_storage_generation(&key) == Some(mirror_generation))
            .then_some(ResidentServe::Seed(mirror_generation));
    }
    let (engine_generation, engine_format) = executor.compute_resident_sample_source(&key)?;
    (engine_generation == mirror_generation
        && mtl_to_engine_sampled(pixel_format).is_some_and(|f| f.storage() == engine_format))
    .then_some(ResidentServe::Sample(key, mirror_generation))
}

/// Load tight raw texels for a compute texture binding (type-2/3, IOSurface plane view→surface, or IOSurface texture).
///
/// IOSurface plane view (`RefTextureHandle`) is the live CI wallpaper path (`compute_stage_tex … ot=5`).
/// RE (IOSurface plane view wire + `runtime::draw` sample path): surfaceID@0 is a surface backing object id (= mapping
/// mid). Product draw samples call [`objects::ensure_surface_for_present`] on that id and
/// stage from the **mapping registry**, never re-resolving the surface id through the
/// compute task's object list (that list uses a separate texture-ref namespace — live
/// ensure=1 then MissingTexture/GuestIo class when `resolve_iosurface_texture_ref(task, sid)` hit a
/// different IOSurface texture slot).
pub(crate) fn stage_texture_raw<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    binding: u32,
    stage: ComputeTextureStage,
) -> Result<StagedTexture, ComputeStatus> {
    let is_storage = stage.is_storage();
    // IOSurface plane view RefTextureHandle → surface_id (live CI binds ot5).
    let mut stage_ref = texture_ref;
    let mut from_iosurface_plane_view = false;
    let mut from_surface_backing_direct = false;
    let mut iosurface_plane_view_record: Option<objects::IOSurfacePlaneViewDescriptor> = None;
    let mut view_level = 0;
    let mut view_pixel_format = None;
    let mut view_swizzle = reims_vgpu_protocol::SwizzlePlan::default();
    let mut heap_texture = None;
    let mut buffer_texture = None;
    // A linear texture object (type-2/3) must resolve through its own
    // descriptor, never through the mapping registry: its numeric ref shares
    // the id space with surface backing surface mids, so the `mappings.contains(ref)`
    // fallback below would wrongly grab a same-numbered surface (live class:
    // `ref=N ot=2` dragged into the IOSurface texture path and failing silently against
    // the biplanar wallpaper mid). Same collision the IOSurface plane view path documents.
    // Resolve the object-list entry once: `ref_is_linear` and the iosurface_plane_view/surface_backing
    // classification below both read it for the same ref, and the guest object
    // list is immutable for the life of the dispatch (the device never writes
    // those pages). `ListObjectEntry` is `Copy`, so one guest-DMA read+decode
    // serves both instead of two.
    let ref_entry = objects::lookup_list_entry(state, host, task_id, texture_ref);
    if let Some(entry) = ref_entry {
        if entry.kind == ObjectKind::TextureView {
            let resource =
                objects::resolve_resource(state, host, task_id, texture_ref).map_err(|rung| {
                    crate::observe::fail(format!(
                        "compute_stage_tex view_fail reason={} ref={texture_ref} desc_len={}",
                        crate::observe::ladder_slugs!("compute_stage_tex_view")(rung),
                        entry.descriptor_length
                    ));
                    ComputeStatus::MissingTexture(crate::observe::ladder_slugs!(
                        "compute_stage_tex_view"
                    )(rung))
                })?;
            match objects::decoded_resource(&resource) {
                Ok(ResourceDescriptor::HeapTexture(record)) => {
                    let (heap_ref, use_offset, offset) =
                        (record.heap.get(), record.use_offset, record.offset);
                    if heap_ref == 0 {
                        crate::observe::fail(format!(
                            "compute_stage_tex heap_fail reason=zero_heap ref={texture_ref}"
                        ));
                        return Err(ComputeStatus::MissingTexture(
                            "compute_stage_tex_heap_zero_ref",
                        ));
                    }
                    heap_texture = Some((heap_ref, use_offset, offset, record.declaration));
                }
                Ok(ResourceDescriptor::BufferTexture(record)) => {
                    // A texture over an MTLBuffer's own storage. It is not a
                    // view over another texture and has no surface behind it,
                    // so it skips every path below and goes straight to the
                    // linear rail, which is what it already is.
                    buffer_texture = Some(*record);
                }
                Ok(ResourceDescriptor::TextureView(_)) => {
                    let view = match crate::runtime::draw::resolve_texture_view_reasoned(
                        state,
                        host,
                        task_id,
                        texture_ref,
                    ) {
                        Ok(view) => view,
                        Err(reason) => {
                            crate::observe::Emit::decline(
                                "compute_stage_tex_view_resolve",
                                &reason,
                            )
                            .field("ref", texture_ref)
                            .fail_once(texture_ref as u64);
                            return Err(ComputeStatus::MissingTexture(
                                "compute_stage_tex_view_resolve",
                            ));
                        }
                    };
                    if is_storage
                        && view
                            .swizzle
                            .as_ref()
                            .is_some_and(|plan| !pixel_format::swizzle_is_identity(plan))
                    {
                        crate::observe::fail(format!(
                        "compute_stage_tex view_fail reason=swizzle_unsupported ref={texture_ref} base={} storage={}",
                        view.base_texture_ref, is_storage as u8
                    ));
                        return Err(ComputeStatus::Unsupported(
                            "compute_view_swizzle_unsupported",
                        ));
                    }
                    view_swizzle = view.swizzle.unwrap_or_default();
                    stage_ref = view.base_texture_ref;
                    let Some(level) = view.single_non_array_level() else {
                        crate::observe::fail(format!(
                            "compute_stage_tex view_fail reason=subresource_range_unsupported ref={texture_ref} base={} range={:?} storage={}",
                            view.base_texture_ref, view.range, is_storage as u8
                        ));
                        return Err(ComputeStatus::Unsupported(
                            "compute_view_subresource_range_unsupported",
                        ));
                    };
                    view_level = level;
                    view_pixel_format = view.pixel_format;
                }
                Err(error) => {
                    crate::observe::Emit::decline(
                        "compute_stage_tex_view",
                        &crate::runtime::decode::resource::DecodeDecline(*error),
                    )
                    .field("ref", texture_ref)
                    .field("len", resource.descriptor().len())
                    .fail();
                    return Err(ComputeStatus::MissingTexture(
                        "compute_stage_tex_view_bad_record",
                    ));
                }
                Ok(_) => {
                    return Err(ComputeStatus::MissingTexture(
                        "compute_stage_tex_view_kind_mismatch",
                    ));
                }
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
        let storage_format = pixel_format::storage_image_format(format);
        if is_storage && storage_format.is_none() {
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
        let Some(origin) = state
            .task_objects
            .resources
            .heap_storage_origin(task_id, texture_ref)
        else {
            return Err(ComputeStatus::MissingTexture(
                "compute_heap_storage_identity",
            ));
        };
        let key = crate::model::ComputeStorageResidencyKey {
            origin,
            width,
            height,
            pixel_format: format,
        };
        let serve = match state.content.compute_residency.generation(&key) {
            None => None,
            Some(generation) => {
                match resident_serve(state.executor.as_ref(), key, generation, is_storage, format) {
                    // A heap texture has no guest window to re-read: once the mirror
                    // claims a resident, the engine's copy is the only content, so a
                    // resident the engine can no longer serve is a loss, not a
                    // fallback. The window-backed rails below fall through to the
                    // guest read here instead; this is the arm that must not.
                    None => {
                        crate::observe::fail(format!(
                            "compute_stage_tex heap_fail reason=resident_lost ref={texture_ref} heap={heap_ref} fmt={format:#x} {width}x{height} gen={generation} use_offset={} offset={offset:#x}",
                            use_offset as u8
                        ));
                        return Err(ComputeStatus::MissingTexture(
                            "compute_stage_tex_heap_resident_lost",
                        ));
                    }
                    serve => serve,
                }
            }
        };
        let seed_generation = serve.and_then(ResidentServe::seed_generation).unwrap_or(0);
        crate::observe::off(format!(
            "compute_stage_tex heap_ok ref={texture_ref} heap={heap_ref} fmt={format:#x} {width}x{height} storage={} seed_gen={seed_generation} resident_sample={} use_offset={} offset={offset:#x}",
            is_storage as u8,
            serve.and_then(ResidentServe::sample_source).is_some() as u8,
            use_offset as u8
        ));
        return Ok(StagedTexture {
            resource_ref: stage_ref,
            binding,
            array_element: 0,
            descriptor_count: 1,

            pixel_format: format,
            storage_format,
            view_swizzle,
            width,
            height,
            multisampled: false,
            bytes: Vec::new(),
            is_storage,
            residency: is_storage.then_some(ComputeStorageResidencyCandidate {
                key,
                seed_generation,
            }),
            input: serve
                .map(VulkanTextureInput::Resident)
                .unwrap_or_else(|| VulkanTextureInput::HostBytes(vec![0; need])),
            writeback: TextureWriteback::None,
        });
    }
    // Before the surface classification, not after it: a buffer-backed
    // texture's ref shares its id space with surface backing mids, so the
    // `mappings.contains(ref)` fallback below would hand it a same-numbered
    // surface. That is the collision the type-2/3 arm already forces past with
    // `ref_is_linear`, and this form has no reason to enter the question at
    // all — it names its storage outright.
    if let Some(record) = buffer_texture {
        let placement = buffer_texture_placement(state, host, task_id, texture_ref, &record)?;
        return stage_linear_placement(
            state,
            host,
            task_id,
            texture_ref,
            binding,
            stage,
            view_pixel_format,
            view_swizzle,
            placement,
        );
    }
    let stage_entry = objects::lookup_list_entry(state, host, task_id, stage_ref);
    let ref_is_linear = stage_entry
        .map(|e| e.kind == ObjectKind::Texture)
        .unwrap_or(false);
    if let Some(entry) = stage_entry {
        if entry.kind == ObjectKind::IOSurfacePlaneView {
            if let Ok(resource) = objects::resolve_resource(state, host, task_id, stage_ref) {
                if let Ok(crate::runtime::decode::resource::Descriptor::IOSurfacePlaneView(t5)) =
                    objects::decoded_resource(&resource)
                {
                    let sid = t5.surface.get();
                    if sid != 0 {
                        stage_ref = sid;
                        from_iosurface_plane_view = true;
                        iosurface_plane_view_record = t5.view;
                        let ok = objects::ensure_surface_for_present(state, host, sid);
                        // Optional verbose observation uses only the decoded contract. Raw
                        // descriptor bytes stop at the protocol boundary.
                        if crate::observe::draw_log_enabled() {
                            crate::observe::line(format!(
                                "compute_stage_tex iosurface_plane_view ref={texture_ref} sid={sid} ensure={} owner_task={} operation_kind={:?} operation_len={:?} decode_state={:?} view={:?}",
                                ok as u8,
                                t5.owner_task.get(),
                                t5.operation_kind,
                                t5.operation_length,
                                t5.decode_state,
                                t5.view,
                            ));
                        }
                    }
                }
            }
        } else if entry.kind == ObjectKind::SurfaceBacking {
            // Direct surface backing surface bind (same id space as present mids).
            from_surface_backing_direct = true;
            let _ = objects::ensure_surface_for_present(state, host, stage_ref);
        }
    }

    // IOSurface plane view / direct surface backing: surface id **is** the mapping mid. Never call
    // resolve_iosurface_texture_ref(task, sid) — task object-list indices collide with texture refs.
    let mapping_id_opt = if from_iosurface_plane_view || from_surface_backing_direct {
        if stage_ref != 0 && state.surfaces.mappings.contains_key(&stage_ref) {
            Some(stage_ref)
        } else {
            None
        }
    } else if ref_is_linear {
        // Linear texture: never fall back to the mapping registry (id-space
        // collision with surface backing surface mids). Force the type-2/3 path.
        None
    } else {
        objects::resolve_iosurface_texture_ref(state, host, task_id, stage_ref).or_else(|| {
            if stage_ref != 0 && state.surfaces.mappings.contains_key(&stage_ref) {
                Some(stage_ref)
            } else {
                None
            }
        })
    };
    if mapping_id_opt.is_none() && from_iosurface_plane_view {
        crate::observe::fail(format!(
            "compute_stage_tex iosurface_plane_view_no_map ref={texture_ref} sid={stage_ref}"
        ));
        return Err(ComputeStatus::MissingTexture(
            "compute_stage_tex_iosurface_plane_view_no_map",
        ));
    }
    if let Some(mapping_id) = mapping_id_opt {
        let _ = mapper::ensure_resolved_for_scanout(state, host, mapping_id);
        // Geom/format: a IOSurface plane view record is the exact Metal texture view over
        // the IOSurface bytes. It is authoritative even for a stageable
        // single-plane mapping: the live BGRA8 desktop target is exposed as a
        // row-byte-equivalent, quarter-width RGBA32Uint view. Surface backing direct
        // refs use base mapping geometry. IOSurface texture refs may prefer the
        // IOSurface descriptor on this task's object list.
        if view_level != 0 {
            crate::observe::fail(format!(
                "compute_stage_tex view_fail reason=iosurface_texture_mip ref={texture_ref} base={stage_ref} level={view_level} mapping={mapping_id}"
            ));
            return Err(ComputeStatus::Unsupported(
                "compute_view_iosurface_texture_mip",
            ));
        }
        let (width, height, format) = if from_iosurface_plane_view || from_surface_backing_direct {
            let m =
                state
                    .surfaces
                    .mappings
                    .get(&mapping_id)
                    .ok_or(ComputeStatus::MissingTexture(
                        "compute_stage_tex_mapping_gone",
                    ))?;
            let multiplanar = objects::mapping_is_multiplanar(m);
            let mapping_stageable = m.has_geometry()
                && m.width_or_zero() != 0
                && m.height_or_zero() != 0
                && m.format_or_zero() != 0
                && !multiplanar;
            if let Some(rec) = iosurface_plane_view_record {
                // `iosurface_texture_sample_window` below matches actual plane records by
                // geometry+bpe and otherwise verifies a packed row-compatible
                // view over the same bytes. Per-bind measurement (view vs base
                // geom), not a failure — verbose-gated to keep the always-on sink
                // for genuine failures.
                crate::observe::line(format!(
                    "compute_stage_tex iosurface_plane_view mapping={mapping_id} view={}x{} fmt={:#x} base={}x{} fmt={:#x} multiplanar={}",
                    rec.width,
                    rec.height,
                    rec.pixel_format,
                    m.width_or_zero(),
                    m.height_or_zero(),
                    m.format_or_zero(),
                    multiplanar as u8
                ));
                (rec.width, rec.height, rec.pixel_format)
            } else if !mapping_stageable {
                if !m.has_geometry() || m.width_or_zero() == 0 || m.height_or_zero() == 0 {
                    crate::observe::fail(format!(
                        "compute_stage_tex iosurface_texture_fail reason=no_geom mapping={mapping_id} pages={} has_geom={}",
                        m.pages.entries.len(),
                        m.has_geometry() as u8
                    ));
                    return Err(ComputeStatus::MissingTexture(
                        "compute_stage_tex_iosurface_texture_no_geom",
                    ));
                } else if multiplanar {
                    // Multi-plane IOSurface without a plane record: fail closed,
                    // do not invent BGRA sample of the whole surface.
                    crate::observe::fail(format!(
                        "compute_stage_tex iosurface_texture_fail reason=multiplane mapping={mapping_id} {}x{} fmt={:#x} pages={} (no IOSurface plane view plane record)",
                        m.width_or_zero(),
                        m.height_or_zero(),
                        m.format_or_zero(),
                        m.pages.entries.len()
                    ));
                    return Err(ComputeStatus::Unsupported("stage_tex_multiplane_no_plane"));
                } else {
                    // Single-plane unknown format: fail closed (no BGRA invent).
                    crate::observe::fail(format!(
                        "compute_stage_tex iosurface_texture_fail reason=fmt_unknown mapping={mapping_id} {}x{} pages={}",
                        m.width_or_zero(),
                        m.height_or_zero(),
                        m.pages.entries.len()
                    ));
                    return Err(ComputeStatus::Unsupported("stage_tex_fmt_unknown"));
                }
            } else {
                (m.width_or_zero(), m.height_or_zero(), m.format_or_zero())
            }
        } else {
            // Three ways the surface's own IOSurface descriptor can fail to
            // answer — no list entry, no descriptor bytes, or bytes that do not
            // decode as an IOSurfaceTexture — and all three fall back to the
            // mapping's latched geometry. Kept sequential rather than chained so
            // the `&mut state` the lookups need does not overlap the `&state` the
            // fallback reads.
            let from_descriptor = objects::resolve_resource(state, host, task_id, stage_ref)
                .ok()
                .and_then(|resource| match objects::decoded_resource(&resource) {
                    Ok(ResourceDescriptor::MapperIOSurfaceTextureView(view)) => Some((
                        view.declaration.width,
                        view.declaration.height,
                        or_bgra8(view.declaration.pixel_format),
                    )),
                    _ => None,
                });
            match from_descriptor {
                Some(geom) => geom,
                None => mapping_geom_format(state, mapping_id)?,
            }
        };
        if width == 0 || height == 0 {
            return Err(ComputeStatus::MissingTexture("compute_stage_tex_zero_geom"));
        }
        // sRGB color-renderable surfaces stage as unorm storage (same bpp).
        let view_format = match crate::runtime::draw::effective_view_sample_format_reasoned(
            format,
            view_pixel_format,
        ) {
            Ok(view_format) => view_format,
            Err(refusal) => {
                // `term=` is what says whether this is a gap in this crate's
                // format table or the guest asking for something Metal forbids,
                // and `role=` says which rail would have had to take it — the
                // two questions the next reader has, and the two the old
                // `format_incompatible` could not answer. The bind dies here,
                // before the storage check, so without `role=` the log cannot
                // say whether the missing rail is a sampled layout or a storage
                // selector.
                crate::observe::fail(format!(
                    "compute_stage_tex view_fail reason=format_incompatible term={refusal} \
                     role={} ref={texture_ref} base={stage_ref} base_fmt={format:#x} \
                     view_fmt={view_pixel_format:?} {width}x{height} mapping={mapping_id}",
                    if is_storage { "storage" } else { "sampled" }
                ));
                return Err(ComputeStatus::Unsupported("compute_view_format"));
            }
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
                    "compute_stage_tex iosurface_texture_fail reason=fmt_bytes mapping={mapping_id} {width}x{height} fmt={format:#x}"
                ));
                return Err(ComputeStatus::Unsupported("stage_tex_fmt_bytes"));
            }
        };
        let storage_format = pixel_format::storage_image_format(stage_fmt);
        if is_storage && storage_format.is_none() {
            crate::observe::fail(format!(
                "compute_stage_tex iosurface_texture_fail reason=fmt_storage mapping={mapping_id} {width}x{height} fmt={format:#x}"
            ));
            return Err(ComputeStatus::Unsupported("stage_tex_fmt_storage"));
        }
        let was_render_target = state
            .task_objects
            .resources
            .get(task_id, texture_ref)
            .is_some_and(|resource| resource.was_render_target());
        let target_resident = if stage.is_multisampled() && was_render_target {
            Some(crate::runtime::present_identity::surface_identity(
                state, mapping_id, width, height,
            ))
        } else {
            (!is_storage)
                .then(|| {
                    crate::runtime::draw::compute_iosurface_resident_sample(
                        state,
                        host,
                        task_id,
                        texture_ref,
                        mapping_id,
                        width,
                        height,
                    )
                })
                .flatten()
        };
        let m = state
            .surfaces
            .mappings
            .get(&mapping_id)
            .ok_or(ComputeStatus::MissingTexture(
                "compute_stage_tex_mapping_gone",
            ))?;
        let map_generation = m.lifecycle.generation;
        let mut seed_generation = m.content.guest_page_generation;
        let pages_n = m.pages.entries.len();
        // Wire surface backing `length` (page-aligned getResidentSize), stashed as device_desc.alloc_size.
        // Independent of plane w/h and of MapMemory2 IOAccelMemory length — measure-only.
        let wire_len = reims_vgpu_protocol::decode_device_surface(m.device_desc_bytes())
            .map(|s| s.alloc_size as u64)
            .unwrap_or(0);
        // A IOSurface plane view record names its IOSurface plane on the wire (record `+0x20`,
        // the `newTextureWithDescriptor:iosurface:plane:` argument), so the
        // plane is decided, not inferred. IOSurface texture carries no such field and must
        // still match a plane record by geometry — which is ambiguous whenever
        // two planes share dims and bytes-per-element (v0a8 Y and alpha), and
        // declines rather than picking one. The draw path already binds IOSurface plane view
        // views by index; this is the same resolution on the staging path.
        let window = match iosurface_plane_view_record {
            Some(rec) => mapping_write::iosurface_plane_view_sample_window(
                m,
                rec.plane_index,
                width,
                height,
                stage_fmt,
            ),
            None => mapping_write::iosurface_texture_sample_window(m, width, height, stage_fmt),
        };
        let (surface_offset, surface_bpr, span_end) = match window {
            Some(w) => w,
            None => {
                // What the descriptor said, so a refusal names which of its
                // fields the texture could not be placed against. `reach` is the
                // byte count this geometry needs; a descriptor whose alloc is
                // smaller is a different failure from one whose plane records
                // matched nothing.
                let ds = reims_vgpu_protocol::decode_device_surface(m.device_desc_bytes());
                let (dw, dh, dbpr, dalloc) = ds
                    .as_ref()
                    .map(|s| (s.width, s.height, s.bytes_per_row, s.alloc_size))
                    .unwrap_or((0, 0, 0, 0));
                let reach = reims_vgpu_protocol::packed_span_estimate(stage_fmt, width, height)
                    .unwrap_or(0);
                crate::observe::fail(format!(
                    "compute_stage_tex iosurface_texture_fail reason=window mapping={mapping_id} {width}x{height} fmt={stage_fmt:#x} pages={pages_n} wire_len={wire_len} desc={dw}x{dh} bpr={dbpr} alloc={dalloc} reach={reach}"
                ));
                return Err(ComputeStatus::MissingTexture(
                    "compute_stage_tex_iosurface_texture_window",
                ));
            }
        };
        let tight = (width as u64)
            .checked_mul(bpp as u64)
            .ok_or(ComputeStatus::Unsupported("stage_tex_tight_bpr_overflow"))?
            as u32;
        if from_iosurface_plane_view && iosurface_plane_view_record.is_some() {
            // Per-bind IOSurface plane view sample-window measurement, not a failure — verbose-gated
            // (was a per-bind always-on line). Genuine window failures above emit
            // `iosurface_texture_fail reason=window` always-on.
            crate::observe::line(format!(
                "compute_stage_tex iosurface_plane_view_window mapping={mapping_id} view={width}x{height} fmt={stage_fmt:#x} bpp={bpp} tight={tight} surface_off={surface_offset} surface_bpr={surface_bpr} span_end={span_end}"
            ));
        }
        let need_u64 = (tight as u64)
            .checked_mul(height as u64)
            .ok_or(ComputeStatus::Unsupported("stage_tex_need_overflow"))?;
        let Some(need) = host_alloc_len(need_u64) else {
            crate::observe::fail(format!(
                "compute_stage_tex iosurface_texture_fail reason=host_len mapping={mapping_id} need={need_u64}"
            ));
            return Err(ComputeStatus::Unsupported("stage_tex_host_len"));
        };
        let page_bytes = (pages_n as u64).saturating_mul(1u64 << state.page_shift);
        if page_bytes < span_end {
            crate::observe::fail(format!(
                "compute_stage_tex iosurface_texture_fail reason=span mapping={mapping_id} {width}x{height} pages={pages_n} page_bytes={page_bytes} span_end={span_end} bpr={surface_bpr} wire_len={wire_len}"
            ));
            return Err(ComputeStatus::GuestIo(
                "compute_stage_tex_iosurface_texture_span",
            ));
        }
        let residency_key = crate::model::ComputeStorageResidencyKey::surface(
            mapping_id,
            map_generation,
            surface_offset,
            surface_bpr,
            span_end,
            width,
            height,
            stage_fmt,
        );
        // Chained-dispatch restage skip: when guest pages still hold exactly
        // our own last writeback for THIS WINDOW (mirror entry survives only
        // while no intersecting guest write lands — `Device::
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
        let serve = state
            .content
            .compute_residency
            .generation(&residency_key)
            .and_then(|mirror_generation| {
                resident_serve(
                    state.executor.as_ref(),
                    residency_key,
                    mirror_generation,
                    is_storage,
                    stage_fmt,
                )
            });
        // Unlike the heap and linear rails, this one's fallback generation is
        // the mapping's own content generation rather than zero, so a seed
        // overwrites it and anything else leaves it alone. Gated with the
        // generation it writes.
        if let Some(generation) = serve.and_then(ResidentServe::seed_generation) {
            seed_generation = generation;
            crate::observe::off(format!(
                "compute_stage_resident_skip mapping={mapping_id} {width}x{height} fmt={stage_fmt:#x} gen={seed_generation} bytes={need}"
            ));
        } else if let Some((_, generation)) = serve.and_then(ResidentServe::sample_source) {
            crate::observe::off(format!(
                "compute_stage_resident_sample mapping={mapping_id} {width}x{height} fmt={stage_fmt:#x} gen={generation} bytes={need}"
            ));
        }
        let mut bytes = vec![0u8; need];
        let input = if let Some(identity) = target_resident {
            VulkanTextureInput::TargetResident(identity)
        } else if let Some(resident) = serve {
            VulkanTextureInput::Resident(resident)
        } else if let Some(source) = span_end.checked_sub(surface_offset).and_then(|span| {
            let row_length_texels = if surface_bpr == tight {
                0
            } else {
                surface_bpr.checked_div(bpp)?
            };
            crate::runtime::mapper::guest_texel_source(
                state,
                host,
                mapping_id,
                surface_offset,
                span,
                row_length_texels,
            )
        }) {
            VulkanTextureInput::GuestPages(source)
        } else {
            if !mapping_write::read_rect_raw_at(
                state,
                host,
                mapping_id,
                mapping_write::SurfaceWindow {
                    base_off: surface_offset,
                    bpr: surface_bpr,
                    span_end,
                    bpp,
                },
                mapping_write::Rect {
                    origin_x: 0,
                    origin_y: 0,
                    width,
                    height,
                },
                &mut bytes,
                tight,
            ) {
                crate::observe::fail(format!(
                    "compute_stage_tex iosurface_texture_fail reason=read mapping={mapping_id} {width}x{height} off={surface_offset} bpr={surface_bpr} span_end={span_end} pages={pages_n}"
                ));
                return Err(ComputeStatus::GuestIo(
                    "compute_stage_tex_iosurface_texture_read",
                ));
            }
            VulkanTextureInput::HostBytes(std::mem::take(&mut bytes))
        };
        let writeback = if is_storage {
            TextureWriteback::IOSurface {
                mapping_id,
                surface_offset,
                surface_bpr,
                span_end,
                width,
                height,
                format: stage_fmt,
            }
        } else {
            TextureWriteback::None
        };
        if from_iosurface_plane_view {
            // Per-bind IOSurface plane view stage SUCCESS census — not a failure; verbose-gated
            // (was always-on, ~300/boot). Genuine IOSurface plane view stage failures above emit
            // `iosurface_texture_fail reason=<slug>` always-on.
            crate::observe::line(format!(
                "compute_stage_tex iosurface_plane_view_ok ref={texture_ref} sid={mapping_id} {width}x{height} fmt={stage_fmt:#x} pages={pages_n}"
            ));
        }
        return Ok(StagedTexture {
            resource_ref: stage_ref,
            binding,
            array_element: 0,
            descriptor_count: 1,

            pixel_format: stage_fmt,
            storage_format,
            view_swizzle,
            width,
            height,
            multisampled: false,
            bytes,
            is_storage,
            residency: is_storage.then_some(ComputeStorageResidencyCandidate {
                key: residency_key,
                seed_generation,
            }),
            input,
            writeback,
        });
    }

    // Type-2/3 linear. The buffer-backed form took the same rail above, from
    // its own placement — see [`LinearPlacement`].
    let placement =
        linear_texture_placement(state, host, task_id, texture_ref, stage_ref, view_level)?;
    stage_linear_placement(
        state,
        host,
        task_id,
        texture_ref,
        binding,
        stage,
        view_pixel_format,
        view_swizzle,
        placement,
    )
}

/// A texture whose texels are a strided window over one guest allocation.
///
/// Two guest constructions produce one of these, and the only difference
/// between them is which descriptor was read. A type-2/3 linear texture object
/// names its own allocation and a mip level within it. An opcode-9
/// buffer-backed texture — `newTextureWithDescriptor:offset:bytesPerRow:` —
/// names an `MTLBuffer`, a byte offset into it, and a row pitch. Past this
/// point they are the same object: a first row at a GVA, a stride, an extent,
/// and a format. Everything the rail does after that — the window cache, the
/// residency key, the packed guest alias, the storage writeback — is arithmetic
/// over exactly these fields, which is why it takes this rather than either
/// descriptor.
///
/// The two refs are separate because for a buffer-backed texture they are two
/// different guest objects, and each is the right key for a different question.
struct LinearPlacement {
    /// The texture object the shader bound. Identity for the host-bytes window
    /// cache, the residency key, and the storage writeback — all questions
    /// about *this texture's* content.
    texture_ref: u32,
    /// The object that owns the guest allocation holding the texels: the
    /// texture itself for a linear texture object, the **buffer** for a
    /// buffer-backed one.
    ///
    /// This keys the packed alias, so a buffer the guest binds both as a buffer
    /// and as a texture over the same bytes resolves to one alias rather than
    /// two aliases of one allocation — which is the whole point of the
    /// construction, and the case a guest uses it for.
    storage_ref: u32,
    /// First byte of the allocation named by `storage_ref`, and its length.
    allocation_gva: u64,
    allocation_size: u64,
    /// First byte of this texture's first row.
    gva: u64,
    /// The pixel format the descriptor declared, before any view override.
    declared_format: u16,
    width: u32,
    height: u32,
    row_stride: u64,
    /// Complete base-texture allocation when the sampled bind names a mip
    /// chain. A single-level view deliberately leaves this absent.
    sampled_allocation: Option<(
        reims_vgpu_memory::GuestImageAllocationLayout,
        reims_vgpu_memory::GuestImageViewRange,
    )>,
}

/// Complete sampled-mip transfer independent of host-pointer import.
///
/// A retained packed resource is the direct answer when available. The copying
/// rail still owns the same allocation contract when it is not: walking the
/// task's current page plan produces a full-allocation run source rather than
/// narrowing the bind to level zero.
fn complete_mip_transfer_source<M: HostMemory + HostOps>(
    state: &Device,
    host: &mut M,
    task_id: u32,
    allocation_gva: u64,
    allocation_size: u64,
    packed: Option<reims_vgpu_memory::GuestRunSource>,
) -> Result<reims_vgpu_memory::GuestRunSource, crate::runtime::draw::WindowRefusal> {
    packed.map_or_else(
        || {
            crate::runtime::draw::task_gva_guest_run_source(
                state,
                host,
                task_id,
                allocation_gva,
                allocation_size,
            )
            .map(|(_, source)| source)
        },
        Ok,
    )
}

/// A render-target resident currently represents one linear level. It can
/// replace a sampled guest allocation only when that bind names no complete
/// mip chain; otherwise explicit LODs above zero would silently lose the
/// allocation levels that exist only in guest storage.
fn can_bind_linear_target_resident(
    is_storage: bool,
    has_complete_mip_allocation: bool,
    was_render_target: bool,
) -> bool {
    !is_storage && !has_complete_mip_allocation && was_render_target
}

/// Name which gate a linear placement or stage failed at.
///
/// Shared by both constructors and the rail so a refusal reads the same
/// whichever descriptor produced the placement — the reason travels in the
/// status, so this line and the caller's name one registered slug.
fn linear_fail<T>(bound_ref: u32, status: ComputeStatus, detail: &str) -> Result<T, ComputeStatus> {
    crate::observe::fail(format!(
        "compute_stage_tex linear_fail reason={} ref={bound_ref} {detail}",
        status.reason()
    ));
    Err(status)
}

/// Place a type-2/3 linear texture object's `view_level`.
///
/// Fail-visible: name which gate rejected (live class: silent ot=2
/// MissingTexture, journal 2026-07-14 compute census).
fn linear_texture_placement<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    bound_ref: u32,
    stage_ref: u32,
    view_level: u32,
) -> Result<LinearPlacement, ComputeStatus> {
    let resource = match objects::resolve_resource(state, host, task_id, stage_ref) {
        Ok(resource) if resource.entry().kind == ObjectKind::Texture => resource,
        Ok(resource) => {
            return linear_fail(
                bound_ref,
                ComputeStatus::MissingTexture(crate::observe::ladder_slug!(
                    "compute_linear_tex",
                    wrong_type
                )),
                &format!("ot={}", resource.entry().kind),
            );
        }
        Err(rung) => {
            return linear_fail(
                bound_ref,
                ComputeStatus::MissingTexture(crate::observe::ladder_slugs!("compute_linear_tex")(
                    rung,
                )),
                &match rung {
                    objects::LadderRung::WrongType { got } => format!("ot={got}"),
                    objects::LadderRung::NoListEntry | objects::LadderRung::DescRead { .. } => {
                        String::new()
                    }
                },
            );
        }
    };
    let Ok(ResourceDescriptor::Texture(tex)) = objects::decoded_resource(&resource) else {
        return linear_fail(
            bound_ref,
            ComputeStatus::MissingTexture(crate::observe::ladder_slug!(
                "compute_linear_tex",
                desc_decode
            )),
            &format!("len={}", resource.descriptor().len()),
        );
    };
    let Some(declared_format) = tex.declared_pixel_format() else {
        return linear_fail(
            bound_ref,
            ComputeStatus::Unsupported("linear_tex_no_fmt"),
            "",
        );
    };
    // `level_gva` derives the level's first row from this same base, so a
    // placement can only fail here if it would have failed there — naming it
    // separately says *which* of the two the descriptor lacked.
    let Some(allocation_gva) = tex.allocation_base_gva(state.page_shift) else {
        return linear_fail(
            bound_ref,
            ComputeStatus::MissingTexture("compute_linear_tex_no_allocation"),
            &format!(
                "base={stage_ref} handle={:#x} page_shift={}",
                tex.handle, state.page_shift
            ),
        );
    };
    let Some((gva, layout)) = tex.level_gva(view_level, state.page_shift) else {
        return linear_fail(
            bound_ref,
            ComputeStatus::MissingTexture("compute_linear_tex_no_level"),
            &format!(
                "base={stage_ref} level={view_level} handle={:#x} alloc={} levels={} base_off={} page_shift={}",
                tex.handle,
                tex.allocation_size,
                tex.levels.len(),
                tex.base_offset,
                state.page_shift
            ),
        );
    };
    let sampled_allocation = if view_level == 0 && tex.mipmap_level_count > 1 {
        let Some(bytes_per_texel) = pixel_format::bytes_per_pixel(declared_format) else {
            return linear_fail(
                bound_ref,
                ComputeStatus::Unsupported("compute_linear_mip_format"),
                &format!("fmt={declared_format:#x}"),
            );
        };
        let shape = reims_vgpu_core::sampled_image_shape(reims_vgpu_core::SampledImageKind::D2)
            .expect("2-D sampled images are representable");
        Some(
            crate::runtime::draw::declared_guest_image_allocation(
                shape,
                tex,
                None,
                None,
                u64::from(bytes_per_texel),
            )
            .ok_or(ComputeStatus::Unsupported(
                "compute_linear_mip_allocation_invalid",
            ))?,
        )
    } else {
        None
    };
    Ok(LinearPlacement {
        texture_ref: stage_ref,
        storage_ref: stage_ref,
        allocation_gva,
        allocation_size: tex.allocation_size,
        gva,
        declared_format,
        width: layout.width,
        height: layout.height,
        row_stride: layout.row_stride,
        sampled_allocation,
    })
}

/// Place an opcode-9 buffer-backed texture over its `MTLBuffer`.
///
/// `newTextureWithDescriptor:offset:bytesPerRow:` admits exactly one shape —
/// one mip level, one slice, one sample, one depth plane — so anything else
/// here is a descriptor this device decoded wrongly rather than a texture it
/// could place, and it refuses by name instead of placing a window whose
/// arithmetic would be a guess.
fn buffer_texture_placement<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    bound_ref: u32,
    record: &reims_vgpu_protocol::BufferTextureDescriptor,
) -> Result<LinearPlacement, ComputeStatus> {
    let resource = objects::resolve_resource(state, host, task_id, bound_ref).map_err(|rung| {
        ComputeStatus::MissingTexture(crate::observe::ladder_slugs!("compute_buffer_tex")(rung))
    })?;
    let level = objects::resolve_buffer_texture_placement_from_resource(state, &resource)
        .map_err(|reason| match reason {
            objects::BufferTexturePlacementRefusal::Decode => ComputeStatus::MissingTexture(
                crate::observe::ladder_slug!("compute_buffer_tex", desc_decode),
            ),
            objects::BufferTexturePlacementRefusal::SemanticKind => {
                ComputeStatus::MissingTexture("compute_buffer_tex_semantic_kind")
            }
            objects::BufferTexturePlacementRefusal::Buffer(objects::BufferSpanRefusal::Rung(
                rung,
            )) => ComputeStatus::MissingTexture(crate::observe::ladder_slugs!(
                "compute_buffer_tex"
            )(rung)),
            objects::BufferTexturePlacementRefusal::Buffer(objects::BufferSpanRefusal::Decode) => {
                ComputeStatus::MissingTexture(crate::observe::ladder_slug!(
                    "compute_buffer_tex",
                    desc_decode
                ))
            }
            objects::BufferTexturePlacementRefusal::Buffer(
                objects::BufferSpanRefusal::NoBacking,
            ) => ComputeStatus::MissingTexture("compute_buffer_tex_no_backing"),
            objects::BufferTexturePlacementRefusal::PastAllocation => {
                ComputeStatus::MissingTexture("compute_buffer_tex_span_oob")
            }
            objects::BufferTexturePlacementRefusal::AddressOverflow => {
                ComputeStatus::MissingTexture("compute_buffer_tex_offset_overflow")
            }
            objects::BufferTexturePlacementRefusal::InvalidShape => {
                ComputeStatus::Unsupported("compute_buffer_tex_shape")
            }
            objects::BufferTexturePlacementRefusal::MissingFormat => {
                ComputeStatus::Unsupported("compute_buffer_tex_no_fmt")
            }
            objects::BufferTexturePlacementRefusal::UnsupportedFormat => {
                ComputeStatus::Unsupported("compute_buffer_tex_fmt_bytes")
            }
            objects::BufferTexturePlacementRefusal::RowStrideTooSmall => {
                ComputeStatus::Unsupported("compute_buffer_tex_bpr_short")
            }
            objects::BufferTexturePlacementRefusal::ReachOverflow => {
                ComputeStatus::Unsupported("compute_buffer_tex_reach_overflow")
            }
        })?
        .ok_or(ComputeStatus::MissingTexture(
            "compute_buffer_tex_semantic_kind",
        ))?;
    Ok(LinearPlacement {
        // The bound texture object is this content's identity even though the
        // bytes belong to the buffer: two textures over one buffer at different
        // offsets are two textures, and the window cache and residency mirror
        // must not confuse them.
        texture_ref: bound_ref,
        storage_ref: record.buffer_ref,
        allocation_gva: level.base_gva,
        allocation_size: level.alloc_size,
        gva: level.base_gva + level.level_offset,
        declared_format: level.pixel_format,
        width: level.width,
        height: level.height,
        row_stride: level.row_stride,
        sampled_allocation: None,
    })
}

/// Stage a linear placement: resident if the engine already holds this window,
/// a zero-copy guest alias if the pages can be packed, host bytes otherwise.
#[allow(
    clippy::too_many_arguments,
    reason = "the bind's access mode and view format decide the staged format, and neither belongs in the placement"
)]
fn stage_linear_placement<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    bound_ref: u32,
    binding: u32,
    stage: ComputeTextureStage,
    view_pixel_format: Option<u16>,
    view_swizzle: reims_vgpu_protocol::SwizzlePlan,
    placement: LinearPlacement,
) -> Result<StagedTexture, ComputeStatus> {
    let is_storage = stage.is_storage();
    let LinearPlacement {
        texture_ref,
        storage_ref,
        allocation_gva,
        allocation_size,
        gva,
        declared_format,
        width: w,
        height: h,
        row_stride,
        sampled_allocation,
    } = placement;
    let Some(stage_format) =
        crate::runtime::draw::effective_view_sample_format(declared_format, view_pixel_format)
    else {
        return linear_fail(
            bound_ref,
            ComputeStatus::Unsupported("linear_tex_view_format"),
            &format!(
                "base={texture_ref} base_fmt={declared_format:#x} view_fmt={view_pixel_format:?}"
            ),
        );
    };
    let Some(bpp) = pixel_format::bytes_per_pixel(stage_format) else {
        return linear_fail(
            bound_ref,
            ComputeStatus::Unsupported("linear_tex_fmt_bytes"),
            &format!("fmt={stage_format:#x}"),
        );
    };
    let storage_format = pixel_format::storage_image_format(stage_format);
    if is_storage && storage_format.is_none() {
        return linear_fail(
            bound_ref,
            ComputeStatus::Unsupported("linear_tex_fmt_storage"),
            &format!("fmt={stage_format:#x}"),
        );
    }
    if w == 0 || h == 0 || row_stride == 0 {
        return linear_fail(
            bound_ref,
            ComputeStatus::MissingTexture("compute_linear_tex_zero_geom"),
            &format!("{w}x{h} stride={row_stride}"),
        );
    }
    let Some(tight) = (w as u64).checked_mul(bpp as u64).map(|v| v as usize) else {
        return linear_fail(
            bound_ref,
            ComputeStatus::Unsupported("linear_tex_tight_overflow"),
            &format!("{w}x{h} bpp={bpp}"),
        );
    };
    if row_stride < tight as u64 {
        return linear_fail(
            bound_ref,
            ComputeStatus::MissingTexture("compute_linear_tex_stride_lt_tight"),
            &format!("stride={row_stride} tight={tight} {w}x{h}"),
        );
    }
    let Some(need) = tight.checked_mul(h as usize) else {
        return linear_fail(
            bound_ref,
            ComputeStatus::Unsupported("linear_tex_need_overflow"),
            &format!("{w}x{h} bpp={bpp}"),
        );
    };
    // The identity every cache question below asks about. Built once so the
    // resident probe, the flush-and-serve, and the plain serve cannot drift
    // from each other.
    let window = crate::runtime::surface_cache::LinearWindow {
        task_id,
        texture_ref,
        gva,
        pixel_format: stage_format,
        width: w,
        height: h,
        row_stride,
    };
    // Linear-window residency identity — mirrors the host_linear_textures
    // entry exactly. Absent when the stride overflows the key field (no live
    // class; such a window simply stays on the bytes path).
    let span = row_stride.saturating_mul(h as u64);
    let linear_key = (row_stride <= u32::MAX as u64)
        .then(|| state.task_objects.resources.identity(task_id, texture_ref))
        .flatten()
        .map(|resource| {
            crate::model::ComputeStorageResidencyKey::linear(
                resource,
                gva,
                row_stride as u32,
                span,
                w,
                h,
                stage_format,
            )
        });
    let serve = match (
        linear_key,
        crate::runtime::surface_cache::linear_texture_resident_gen(state, &window),
    ) {
        (Some(key), Some(generation)) => resident_serve(
            state.executor.as_ref(),
            key,
            generation,
            is_storage,
            stage_format,
        ),
        _ => None,
    };
    let may_bind_target = can_bind_linear_target_resident(
        is_storage,
        sampled_allocation.is_some(),
        state
            .task_objects
            .resources
            .get(task_id, texture_ref)
            .is_some_and(|resource| resource.was_render_target()),
    );
    let target_resident = if stage.is_multisampled() && may_bind_target {
        crate::runtime::writeback_debt::resource_key(state, task_id, texture_ref).and_then(|key| {
            let generation = crate::runtime::writeback_debt::gva_resource_generation(
                state,
                host,
                key,
                gva,
                row_stride.saturating_mul(u64::from(h)),
            );
            (generation != 0).then(|| crate::model::TargetIdentity::Gva {
                gva,
                width: w,
                height: h,
                generation,
                format: crate::runtime::draw::gva_resident_format(
                    state.executor.as_ref(),
                    stage_format,
                ),
            })
        })
    } else if may_bind_target {
        u32::try_from(row_stride).ok().and_then(|row_stride| {
            crate::runtime::draw::compute_gva_resident_sample(
                state,
                host,
                task_id,
                texture_ref,
                gva,
                row_stride,
                w,
                h,
                declared_format,
            )
        })
    } else {
        None
    };
    let mut bytes = vec![0u8; need];
    let input = if let Some(identity) = target_resident {
        VulkanTextureInput::TargetResident(identity)
    } else if let Some(resident) = serve {
        VulkanTextureInput::Resident(resident)
    } else {
        // A pending Store is queue work, not host bytes. Submit it before this
        // source read so both operations are ordered on the engine queue.
        crate::runtime::writeback_debt::pay_for_texture(state, host, task_id, bound_ref);
        // A buffer-backed texture is two contract references over one
        // allocation — the texture the shader binds and the buffer that owns
        // the storage — and a synchronize may name either, so a debt may be
        // armed under either. Both are paid. No-op when they are the same
        // reference, which is every other placement.
        if storage_ref != bound_ref {
            crate::runtime::writeback_debt::pay_for_texture(state, host, task_id, storage_ref);
        }
        let exact_span = (h.saturating_sub(1) as u64)
            .checked_mul(row_stride)
            .and_then(|prefix| prefix.checked_add(tight as u64));
        let guest_image = if is_storage {
            None
        } else if let Some((allocation, view)) = sampled_allocation.as_ref() {
            let packed = crate::runtime::bound_buffers::ensure_packed_resource(
                state,
                host,
                task_id,
                storage_ref,
                allocation_gva,
                allocation_size,
                crate::runtime::bound_buffers::PackedResourceUse::ComputeTexture,
            )
            .then(|| state.bound_buffers.packed(task_id, storage_ref))
            .flatten()
            .and_then(|packed| match packed {
                crate::runtime::bound_buffers::PackedBufferResolution::Available(packed) => {
                    packed.texel_source(0, allocation_size, 0)
                }
                crate::runtime::bound_buffers::PackedBufferResolution::Unavailable { .. } => None,
            });
            let transfer = match complete_mip_transfer_source(
                state,
                host,
                task_id,
                allocation_gva,
                allocation_size,
                packed,
            ) {
                Ok(source) => source,
                Err(refusal) => {
                    let reason = match refusal {
                        crate::runtime::draw::WindowRefusal::NoAlias => {
                            "compute_linear_mip_no_alias"
                        }
                        crate::runtime::draw::WindowRefusal::SpanUnmapped => {
                            "compute_linear_mip_span_unmapped"
                        }
                        crate::runtime::draw::WindowRefusal::Untileable => {
                            "compute_linear_mip_untileable"
                        }
                    };
                    return linear_fail(
                        bound_ref,
                        ComputeStatus::GuestIo(reason),
                        &format!("base={allocation_gva:#x} alloc={allocation_size}"),
                    );
                }
            };
            Some(reims_vgpu_memory::GuestImageSource {
                direct: None,
                allocation: allocation.clone(),
                view: *view,
                transfer,
            })
        } else {
            None
        };
        let guest = exact_span.and_then(|span| {
            let level_offset = gva.checked_sub(allocation_gva)?;
            let row_length_texels = if row_stride == tight as u64 {
                0
            } else {
                u32::try_from(row_stride.checked_div(u64::from(bpp))?).ok()?
            };
            if !crate::runtime::bound_buffers::ensure_packed_resource(
                state,
                host,
                task_id,
                storage_ref,
                allocation_gva,
                allocation_size,
                crate::runtime::bound_buffers::PackedResourceUse::ComputeTexture,
            ) {
                return None;
            }
            let crate::runtime::bound_buffers::PackedBufferResolution::Available(packed) =
                state.bound_buffers.packed(task_id, storage_ref)?
            else {
                return None;
            };
            packed.texel_source(level_offset, span, row_length_texels)
        });
        if let Some(source) = guest_image {
            VulkanTextureInput::GuestImage(source)
        } else if let Some(source) = guest {
            VulkanTextureInput::GuestPages(source)
        } else if let Some(cached) =
            crate::runtime::surface_cache::get_linear_texture(state, &window)
        {
            VulkanTextureInput::HostBytes(cached.to_vec())
        } else {
            crate::runtime::render_writeback::settle_guest_writes(
                state.executor.as_ref(),
                crate::runtime::render_writeback::SettleSite::ComputeStageTexture,
            );
            if read_linear_texture_bulk(state, host, task_id, gva, row_stride, tight, h, &mut bytes)
            {
                // One cached-view walk for the whole span.
            } else {
                let mut row = vec![0u8; tight];
                for y in 0..h {
                    let row_gva = gva
                        .checked_add((y as u64).checked_mul(row_stride).ok_or(
                            ComputeStatus::GuestIo("compute_stage_tex_linear_row_offset"),
                        )?)
                        .ok_or(ComputeStatus::GuestIo("compute_stage_tex_linear_row_gva"))?;
                    gva_mem::read_task_gva_by_id(
                        host,
                        &state.tasks,
                        task_id,
                        row_gva,
                        &mut row,
                        state.page_shift,
                    )
                    .map_err(|_| ComputeStatus::GuestIo("compute_stage_tex_linear_row_read"))?;
                    let off = (y as usize) * tight;
                    bytes[off..off + tight].copy_from_slice(&row);
                }
            }
            VulkanTextureInput::HostBytes(std::mem::take(&mut bytes))
        }
    };
    let writeback = if is_storage {
        TextureWriteback::Linear {
            texture_ref,
            gva,
            pixel_format: stage_format,
            row_stride,
            width: w,
            height: h,
            bpp,
            pages: staged_window_pages(state, host, task_id, gva, row_stride, h),
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
    let mut residency = None;
    if is_storage {
        if let Some(key) = linear_key {
            if !crate::runtime::surface_cache::linear_mirrorable(stage_format) {
                let seed = serve
                    .and_then(ResidentServe::seed_generation)
                    .unwrap_or_else(|| {
                        state
                            .host_replicas
                            .linear_host_generation(task_id, texture_ref)
                            .unwrap_or(0)
                    });
                residency = Some(ComputeStorageResidencyCandidate {
                    key,
                    seed_generation: seed,
                });
            }
        }
    }
    Ok(StagedTexture {
        resource_ref: texture_ref,
        binding,
        array_element: 0,
        descriptor_count: 1,

        pixel_format: stage_format,
        storage_format,
        view_swizzle,
        width: w,
        height: h,
        multisampled: false,
        bytes,
        is_storage,
        residency,
        input,
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
    state: &mut Device,
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
    state: &mut Device,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GuestMaterialization {
    Materialized,
    HostOnly,
}

fn writeback_texture<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    tex: &StagedTexture,
) -> Result<GuestMaterialization, ComputeStatus> {
    // Which destination namespace a compute storage output lands in, and — on
    // the linear arm — whether its guest rows are dense. Both are properties of
    // the guest's own window rather than of this device, and neither is
    // otherwise reported: `rectwr_*` and `linear*` are shared with several
    // other rails, so they cannot be read as this rail's split. Observed only;
    // nothing branches on these.
    match &tex.writeback {
        TextureWriteback::None => crate::runtime::drain::note_store_route("compute_wb_none"),
        TextureWriteback::Linear {
            width,
            bpp,
            row_stride,
            ..
        } => {
            crate::runtime::drain::note_store_route("compute_wb_linear");
            // A dense window is one `VkBufferCopy` run per guest run; a padded
            // one needs a rectangle copy per run per row fragment, which is the
            // difference between a handful of regions and a few hundred.
            crate::runtime::drain::note_store_route(
                if u64::from(*width) * u64::from(*bpp) == *row_stride {
                    "compute_wb_linear_dense"
                } else {
                    "compute_wb_linear_padded"
                },
            );
        }
        TextureWriteback::IOSurface {
            width,
            format,
            surface_bpr,
            ..
        } => {
            crate::runtime::drain::note_store_route("compute_wb_iosurface_texture");
            let tight = pixel_format::bytes_per_pixel(*format).map(|bpp| width.saturating_mul(bpp));
            crate::runtime::drain::note_store_route(if tight == Some(*surface_bpr) {
                "compute_wb_iosurface_texture_dense"
            } else {
                "compute_wb_iosurface_texture_padded"
            });
        }
    }

    match &tex.writeback {
        TextureWriteback::None => Ok(GuestMaterialization::HostOnly),
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
            let window = crate::runtime::surface_cache::LinearWindow {
                task_id,
                texture_ref: *texture_ref,
                gva: *gva,
                pixel_format: *pixel_format,
                width: *width,
                height: *height,
                row_stride: *row_stride,
            };
            if !crate::runtime::surface_cache::store_linear_texture(state, &window, &tex.bytes) {
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
                state, host, &window, &tex.bytes,
            );
            // Kept although the span is no longer needed here: the overflow is
            // a real refusal with a name, and `write_linear_guest_within` would only
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
                (!pages.membership().is_empty()).then_some(pages.membership()),
            ) {
                LinearWrite::Written => {
                    // The mirror above cached these bytes as unevictable
                    // because the write had not happened yet. It has, so the
                    // guest can re-derive them and the byte cap may reclaim the
                    // entry. The `Unmapped` arm below deliberately does not:
                    // its own comment is that the host cache keeps the
                    // authoritative bytes.
                    crate::runtime::surface_cache::note_gva_landed(state, *gva);
                    Ok(GuestMaterialization::Materialized)
                }
                // Nothing resolves under this task, so there is nowhere to put
                // the result. The host cache keeps the authoritative bytes and
                // sampling still serves them, so failing the whole dispatch
                // would cost more than it protects.
                LinearWrite::Unmapped => {
                    crate::observe::fail(format!(
                        "compute_writeback_tex cache_only reason=linear_unmapped task={task_id} ref={texture_ref} bind={} gva={gva:#x} fmt={pixel_format:#x} dims={}x{} bpp={} row_stride={row_stride}",
                        tex.binding, width, height, bpp
                    ));
                    Ok(GuestMaterialization::HostOnly)
                }
                LinearWrite::Failed => {
                    Err(ComputeStatus::GuestIo("compute_wb_tex_linear_guest_write"))
                }
            }
        }
        TextureWriteback::IOSurface {
            mapping_id,
            surface_offset,
            surface_bpr,
            span_end,
            width,
            height,
            format,
        } => {
            // Derived here rather than carried beside the format, so the record
            // holds one answer about its texel size. Staging refused this bind
            // outright if the format had no byte width, so a `None` here is a
            // format that changed identity between stage and landing rather
            // than an unsupported one — refuse it by name instead of writing
            // rows at a width nothing declared.
            let Some(bpp) = pixel_format::bytes_per_pixel(*format) else {
                crate::observe::fail(format!(
                    "compute_writeback_tex fail reason=iosurface_texture_format_unsized task={task_id} bind={} mid={mapping_id} fmt={format:#x}",
                    tex.binding
                ));
                return Err(ComputeStatus::GuestIo(
                    "compute_wb_tex_iosurface_texture_format",
                ));
            };
            let tight = width.saturating_mul(bpp);
            if !mapping_write::write_full_rect_raw_at(
                state,
                host,
                *mapping_id,
                *surface_offset,
                *surface_bpr,
                *span_end,
                *width,
                *height,
                bpp,
                &tex.bytes,
                tight,
            ) {
                crate::observe::fail(format!(
                    "compute_writeback_tex fail reason=iosurface_mapping_write task={task_id} bind={} mid={} surface_offset={surface_offset:#x} surface_bpr={} span_end={span_end:#x} dims={}x{} bpp={} bytes={} tight={tight}",
                    tex.binding,
                    mapping_id,
                    surface_bpr,
                    width,
                    height,
                    bpp,
                    tex.bytes.len()
                ));
                return Err(ComputeStatus::GuestIo(
                    "compute_wb_tex_iosurface_texture_write",
                ));
            }
            Ok(GuestMaterialization::Materialized)
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
    state: &mut Device,
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
    state: &mut Device,
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

/// An absent IOSurface pixel format means BGRA8: an IOSurface texture surface the guest
/// mapped without a format word is scanout-ordered by the display contract, and
/// this is the one place that default is written down.
fn or_bgra8(pixel_format: u16) -> u16 {
    if pixel_format != 0 {
        pixel_format
    } else {
        pixel_format::MTL_FORMAT_BGRA8_UNORM
    }
}

/// Latched geometry and pixel format of an IOSurface texture mapping, for a surface whose
/// own IOSurface descriptor could not be read.
///
/// Three separate descriptor failures share this fallback, and spelling it out at
/// each of them made one block of nineteen lines appear three times in a row.
fn mapping_geom_format(state: &Device, mapping_id: u32) -> Result<(u32, u32, u16), ComputeStatus> {
    let m = state
        .surfaces
        .mappings
        .get(&mapping_id)
        .ok_or(ComputeStatus::MissingTexture(
            "compute_stage_tex_mapping_gone",
        ))?;
    if !m.has_geometry() || m.width_or_zero() == 0 || m.height_or_zero() == 0 {
        return Err(ComputeStatus::MissingTexture(
            "compute_stage_tex_mapping_no_geom",
        ));
    }
    Ok((
        m.width_or_zero(),
        m.height_or_zero(),
        or_bgra8(m.format_or_zero()),
    ))
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
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    acc: &ComputeAccum,
    cmd: &ComputeCommand,
    barriers: &[reims_vgpu_core::ComputeBarrier],
) -> ComputeStatus {
    {
        execute_dispatch_linux(state, host, task_id, acc, cmd, barriers)
    }
}

/// Nested dispatch onto an open multi-record control-flow session encoder.
pub(crate) fn execute_dispatch_nested<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    acc: &ComputeAccum,
    cmd: &ComputeCommand,
    session: &mut crate::runtime::compute_session::ComputeSession,
) -> ComputeStatus {
    {
        // Nested/control-flow SPI has no Linux compute path. Fail-visible via
        // the returned status: `exec.rs::note_compute_refusal` names the slug
        // at the rail boundary for every non-`Ok` compute record.
        let _ = (state, host, task_id, acc, cmd, session);
        ComputeStatus::Unsupported("compute_nested_session_unimplemented")
    }
}

/// The dispatch extents, narrowed from the wire's `u64` by [`u32_dim`].
///
/// The protocol-owned type crosses both this decoder and the backend call;
/// keeping it only around construction would protect the half of the journey
/// where the fields are not yet interchangeable.
use reims_vgpu_protocol::Extent3;

/// Runtime sources from which this composition layer can construct a semantic
/// protocol extent.
trait ResolveExtent: Sized {
    fn from_wire(s: crate::runtime::decode::compute::Size3) -> Result<Self, ComputeStatus>;
    fn from_indirect(raw: &[u8], at: usize) -> Result<Self, ComputeStatus>;
}

impl ResolveExtent for Extent3 {
    /// From a decoded wire `Size3`, refusing each component out of range.
    fn from_wire(s: crate::runtime::decode::compute::Size3) -> Result<Self, ComputeStatus> {
        Ok(Self {
            x: u32_dim(s.x)?,
            y: u32_dim(s.y)?,
            z: u32_dim(s.z)?,
        })
    }

    /// From three consecutive LE `u32`s of an indirect-arguments buffer at
    /// `at`. One stride expression rather than six offset literals: the
    /// literals were `0, 4, 8` and `12, 16, 20` written out, where a
    /// transposition is invisible.
    fn from_indirect(raw: &[u8], at: usize) -> Result<Self, ComputeStatus> {
        Ok(Self {
            x: u32_dim(u64::from(ld32(&raw[at..])))?,
            y: u32_dim(u64::from(ld32(&raw[at + 4..])))?,
            z: u32_dim(u64::from(ld32(&raw[at + 8..])))?,
        })
    }
}

/// Grid and threadgroup extents for one dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DispatchDims {
    grid: Extent3,
    threadgroup: Extent3,
    /// The guest asked for `dispatchThreads` — an exact thread count — rather
    /// than whole threadgroups.
    dispatch_threads: bool,
}

/// [`resolve_dispatch_dims`], with the refusal named on the always-on log.
///
/// Both dispatch executors want the same thing on failure: the decline, the
/// command kind, the wire grid and threadgroup, and how many textures were
/// bound. Naming it once is what keeps the Metal and Vulkan arms from
/// drifting into two spellings of the same refusal.
fn resolve_dispatch_dims_reported<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &M,
    task_id: u32,
    cmd: &ComputeCommand,
    acc: &ComputeAccum,
) -> Result<DispatchDims, ComputeStatus> {
    // A bind the accumulator could not hold refuses the dispatch here, before
    // either executor reads the state. It is checked at this gate rather than
    // at the bind because the bind walk has no dispatch to refuse — and a
    // dispatch that runs with the guest's binding simply absent is a wrong
    // result the guest is never told about, which is the one thing this device
    // is not allowed to do. The slot is past Metal's own argument table, so a
    // firing is a record Apple's serializer cannot emit; refusing costs a
    // healthy zero and buys the guarantee.
    if let Some(over) = acc.refused_bind {
        let (index, arg, cap) = over.parts();
        crate::observe::Emit::decline("compute_dispatch", &over)
            .field("kind", format!("{:?}", cmd.kind))
            .field("refused_index", index)
            .field("refused_arg", arg)
            .field("table", cap)
            .fail_once(u64::from(index));
        return Err(ComputeStatus::Unsupported(
            "compute_dispatch_bind_past_table",
        ));
    }
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
    state: &mut Device,
    host: &M,
    task_id: u32,
    cmd: &ComputeCommand,
) -> Result<DispatchDims, ComputeStatus> {
    match cmd.kind {
        // Every dimension comes from the wire. `u32_dim` refuses `0` and
        // anything past `u32::MAX` with `BadGrid("compute_grid_dim_range")`, so
        // a malformed grid is a named refusal rather than a substitution.
        Kind::DispatchThreadgroups => Ok(DispatchDims {
            grid: Extent3::from_wire(cmd.grid)?,
            threadgroup: Extent3::from_wire(cmd.threads_per_threadgroup)?,
            dispatch_threads: false,
        }),
        Kind::DispatchThreads => Ok(DispatchDims {
            grid: Extent3::from_wire(cmd.grid)?,
            threadgroup: Extent3::from_wire(cmd.threads_per_threadgroup)?,
            dispatch_threads: true,
        }),
        Kind::DispatchThreadgroupsIndirect => {
            let raw = read_buffer_window(
                state,
                host,
                task_id,
                cmd.indirect_buffer_ref,
                cmd.indirect_buffer_offset,
                INDIRECT_THREADGROUPS_ARGS_LEN,
            )?;
            Ok(DispatchDims {
                grid: Extent3::from_indirect(&raw, 0)?,
                threadgroup: Extent3::from_wire(cmd.threads_per_threadgroup)?,
                dispatch_threads: false,
            })
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
            Ok(DispatchDims {
                grid: Extent3::from_indirect(&raw, 0)?,
                threadgroup: Extent3::from_indirect(&raw, 12)?,
                dispatch_threads: true,
            })
        }
        _ => Err(ComputeStatus::Unsupported("resolve_dims_unknown_kind")),
    }
}

/// Linux product compute path (doorbell / BQL).
///
/// Stages buffers/textures with device `page_shift`, translates the kernel AIR
/// via [`crate::runtime::executor::ShaderTranslationService::translate_compute`], dispatches on the
/// the device-owned executor's shared GRAPHICS|COMPUTE queue, then writes back
/// GVA / IOSurface texture results.
///
/// Nested/ICB/stage-in stay Unsupported (engine surface is storage buffers +
/// storage images only).
fn execute_dispatch_linux<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    acc: &ComputeAccum,
    cmd: &ComputeCommand,
    barriers: &[reims_vgpu_core::ComputeBarrier],
) -> ComputeStatus {
    use crate::runtime::executor::DrawError;
    use reims_vgpu_core::{
        ComputeBufferResource, ComputeImageResult, ComputeOutput, ComputeRequest,
        ComputeSampledImageResource, ComputeStorageImageResource,
    };

    if acc.pipeline_ref == 0 {
        return ComputeStatus::MissingPipeline("compute_vk_pipeline_ref_zero");
    }
    let Some(pipeline) = load_compute_pipeline(state, host, task_id, acc.pipeline_ref) else {
        return ComputeStatus::MissingPipeline("compute_vk_pipeline_load");
    };
    if let Some(stage_input) = pipeline.stage_input.as_ref() {
        if crate::observe::first_sight("compute_stage_input_contract", u64::from(acc.pipeline_ref))
        {
            crate::observe::off(format!(
                "compute_stage_input_contract pipe={} function={} attrs={:?} layouts={:?} index_type={} \
                 index_buffer={}",
                acc.pipeline_ref,
                pipeline.kernel_func_ref,
                stage_input.attributes,
                stage_input.layouts,
                stage_input.index_type,
                stage_input.index_buffer_index,
            ));
        }
    }
    // A stage-in region this rail proceeds past — see
    // `linux_stage_input_or_imageblock_unsupported`, which explains why that is
    // lossless on a pipeline with no stage input. Counted rather than assumed.
    if acc.stage_in_region.is_some() || acc.stage_in_region_indirect.is_some() {
        crate::runtime::drain::note_store_route("compute_stage_in_region_unused");
    }
    if linux_stage_input_or_imageblock_unsupported(pipeline.stage_input.is_some(), acc) {
        crate::observe::fail(format!(
            "compute_linux unsupported pipe={} stage_in_desc={} stage_in_direct={} \
             stage_in_indirect={} imageblock={} (need SPI parity)",
            acc.pipeline_ref,
            pipeline.stage_input.is_some() as u8,
            acc.stage_in_region.is_some() as u8,
            acc.stage_in_region_indirect.is_some() as u8,
            acc.imageblock.is_some() as u8
        ));
        return ComputeStatus::Unsupported("linux_stage_in_imageblock");
    }
    // Dims first (cheap; proves sentinel recovery without m2v/vk).
    let DispatchDims {
        grid,
        threadgroup: tg,
        dispatch_threads,
    } = match resolve_dispatch_dims_reported(state, host, task_id, cmd, acc) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let (grid_x, grid_y, grid_z) = (grid.x, grid.y, grid.z);
    let (tg_x, tg_y, tg_z) = (tg.x, tg.y, tg.z);
    // Resolved here, before any staging, so a record with no work costs nothing
    // — but computed by the same function that refuses the zero, because the two
    // are one rule. See [`reims_vgpu_protocol::dispatch::workgroup_counts`] for why
    // splitting them put an unreachable `.max(1)` on the quotients.
    let Some(plan) = reims_vgpu_protocol::dispatch::workgroup_counts(
        [grid_x, grid_y, grid_z],
        [tg_x, tg_y, tg_z],
        dispatch_threads,
    ) else {
        return ComputeStatus::BadGrid("compute_vk_zero_dims");
    };
    let [wg_x, wg_y, wg_z] = plan.counts;

    // Translate before staging buffers. The final adopted SPIR-V carries the
    // conservative byte footprint that decides how much of each allocation the
    // dispatch can touch; staging first discarded that answer and copied every
    // bind through the end of its allocation.
    //
    // MTLB → AIR → SPIR-V (LocalSize = threadgroup dims).
    let mtlb = std::sync::Arc::clone(&pipeline.kernel_mtlb);
    // The function blob is an MTLB container; llvm-dis needs the wrapped AIR
    // bitcode member (same extract the render path does — passing the raw
    // container was the live `llvm-dis: file doesn't start with bitcode
    // header` BackendFailed class).
    let air = match crate::runtime::mtlb::extract_air(&mtlb) {
        Ok(a) => a,
        Err(e) => {
            crate::observe::Emit::decline("compute_linux_air_extract", &e)
                .field("pipe", acc.pipeline_ref)
                .fail_once(acc.pipeline_ref as u64);
            return ComputeStatus::BackendFailed("compute_vk_air_extract");
        }
    };
    let kernel_shader =
        match state
            .executor
            .translate_compute(air, [tg_x, tg_y, tg_z], acc.pipeline_ref)
        {
            Ok(b) => b,
            Err(e) => {
                crate::observe::Emit::decline("compute_linux_m2v", &e)
                    .field("pipe", acc.pipeline_ref)
                    .fail_once(acc.pipeline_ref as u64);
                return ComputeStatus::BackendFailed("compute_vk_translate");
            }
        };
    if let Some(unsupported) = kernel_shader
        .interface()
        .first_unsupported_interface(reims_vgpu_core::ReflectedShaderStage::Kernel)
    {
        let reason = ComputeReflectionDecline::ReflectedInterfaceUnsupported {
            pipeline_ref: acc.pipeline_ref,
            feature: unsupported.feature,
            count: unsupported.count,
        };
        crate::observe::Emit::decline("compute_linux_reflection", &reason)
            .fail_once(u64::from(acc.pipeline_ref));
        return ComputeStatus::Unsupported(crate::observe::Decline::slug(&reason));
    }
    if let Some(resource) = kernel_shader.interface().first_unsupported_resource() {
        let kind = resource
            .kind
            .unsupported_vulkan_name()
            .expect("helper returned an unsupported Vulkan resource");
        let reason = ComputeReflectionDecline::ReflectedResourceUnsupported {
            pipeline_ref: acc.pipeline_ref,
            index: resource.metal_index,
            binding: resource.descriptor.map(|descriptor| descriptor.binding),
            kind,
        };
        crate::observe::Emit::decline("compute_linux_reflection", &reason)
            .fail_once((u64::from(acc.pipeline_ref) << 32) | u64::from(resource.metal_index));
        return ComputeStatus::Unsupported(crate::observe::Decline::slug(&reason));
    }
    let reflected_local_size = kernel_shader
        .interface()
        .local_size
        .expect("kernel cache admits only the requested reflected local size");
    // Stage buffers only after translation has published the final-module
    // footprint and access. No Vulkan work occurs until every declared resource
    // has staged successfully. A bind reflection calls `Unused` or does not
    // declare is skipped before resolving its descriptor, walking its pages, or
    // allocating its staging Vec.
    let mut staged_bufs: Vec<StagedBuffer> = Vec::new();
    let mut buffer_accesses = Vec::with_capacity(acc.buffers.len());
    let mut buffer_readonly_count = 0usize;
    let mut buffer_writable_count = 0usize;
    let mut buffer_unused_count = 0usize;
    let mut buffer_absent_count = 0usize;
    let mut buffer_unknown_count = 0usize;
    for b in &acc.buffers {
        use reims_vgpu_core::ReflectedBufferAccess;
        let access = kernel_shader.interface().buffer_access(b.index);
        let writable = match access {
            ReflectedBufferAccess::Unused => {
                buffer_unused_count += 1;
                crate::runtime::drain::note_store_route("compute_buffer_unused_reflected");
                continue;
            }
            ReflectedBufferAccess::Absent => {
                buffer_unused_count += 1;
                buffer_absent_count += 1;
                crate::runtime::drain::note_store_route("compute_buffer_absent_reflected");
                continue;
            }
            ReflectedBufferAccess::ReadOnly => {
                buffer_readonly_count += 1;
                false
            }
            ReflectedBufferAccess::Writable => {
                buffer_writable_count += 1;
                true
            }
            ReflectedBufferAccess::Unknown => {
                // A declared descriptor with no access answer stays on the
                // conservative read/write arm. The per-translate reflection
                // guard names the malformed fact; this count shows how often a
                // dispatch had to pay for it.
                buffer_writable_count += 1;
                buffer_unknown_count += 1;
                true
            }
        };
        let extent = kernel_shader.buffer_extent(b.index, [wg_x, wg_y, wg_z], reflected_local_size);
        match stage_buffer_with_extent(state, host, task_id, b, extent) {
            Ok(s) => {
                buffer_accesses.push((b.index, writable));
                staged_bufs.push(s);
            }
            Err(e) => {
                // `st={e:?}` alone was not greppable: the Debug spelling was
                // the only handle on which of stage_buffer's checks refused.
                // `reason=` names it.
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

    let mut staged_tex: Vec<StagedTexture> = Vec::new();
    let mut storage_writeonly_count = 0usize;
    for t in &acc.textures {
        use reims_vgpu_core::{ImageAccess, ReflectedComputeTexture, StorageImageAccess};
        let Some(descriptor) = kernel_shader.interface().texture_descriptor(t.index) else {
            crate::observe::line(format!(
                "compute_linux texture_unused pipe={} i={} ref={}",
                acc.pipeline_ref, t.index, t.texture_ref
            ));
            continue;
        };
        let binding = descriptor.binding;
        // Both the sampled-vs-storage class and the shape come solely from the
        // translator's reflection — the declared Metal texture type, exact at
        // translate time. The always-on `census_reflection_wellformed` guard
        // proves the reflection is internally consistent per translate.
        let stage = match kernel_shader.interface().compute_texture(binding) {
            ReflectedComputeTexture::Plain2d(ImageAccess::Sampled) => {
                ComputeTextureStage::Sampled2d
            }
            ReflectedComputeTexture::Plain2d(ImageAccess::Storage) => {
                ComputeTextureStage::Storage2d
            }
            ReflectedComputeTexture::Multisampled2d => ComputeTextureStage::Sampled2dMultisample,
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
        let is_storage = stage.is_storage();
        let multisampled = stage.is_multisampled();
        let storage_access = if is_storage {
            match kernel_shader.storage_image_access(binding) {
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
        match stage_texture_raw(state, host, task_id, t.texture_ref, binding, stage) {
            Ok(mut s) => {
                if multisampled && !matches!(s.input, VulkanTextureInput::TargetResident(_)) {
                    crate::observe::fail(format!(
                        "compute_linux texture_shape fail reason=multisample_resident_missing pipe={} i={} ref={} bind={binding}",
                        acc.pipeline_ref, t.index, t.texture_ref
                    ));
                    return ComputeStatus::Unsupported("texture_multisample_resident_missing");
                }
                s.array_element = descriptor.array_element;
                s.descriptor_count = descriptor.descriptor_count;
                s.multisampled = multisampled;
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
                    .map(|en| en.kind.to_string())
                    .unwrap_or_else(|| "absent".to_string());
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
        "compute_linux stage_ok pipe={} nbuf={} bro={} brw={} bunused={} babsent={} bunknown={} ntex={} sampled={} storage={} swo={} grid=[{grid_x},{grid_y},{grid_z}] tg=[{tg_x},{tg_y},{tg_z}] encode=engine",
        acc.pipeline_ref,
        staged_bufs.len(),
        buffer_readonly_count,
        buffer_writable_count,
        buffer_unused_count,
        buffer_absent_count,
        buffer_unknown_count,
        staged_tex.len(),
        sampled_count,
        storage_count,
        storage_writeonly_count,
    ));

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
            backing: match std::mem::replace(&mut s.input, VulkanBufferInput::HostBytes(Vec::new()))
            {
                VulkanBufferInput::HostBytes(bytes) => {
                    reims_vgpu_core::ComputeBufferBacking::Bytes(bytes)
                }
                VulkanBufferInput::GuestPages(source) => {
                    reims_vgpu_core::ComputeBufferBacking::GuestPages {
                        source,
                        write_pages: s.pages.iter().copied().collect(),
                    }
                }
            },
            writable: *writable,
        });
    }
    let mut sampled_images = Vec::with_capacity(sampled_count);
    let mut storage_images = Vec::with_capacity(storage_count);
    let mut storage_formats = Vec::with_capacity(storage_count);
    for t in staged_tex.iter().filter(|texture| texture.is_storage) {
        let Some(guest_fmt) = t.storage_format else {
            crate::observe::fail(format!(
                "compute_linux unsupported storage_format reason=no_storage_format pipe={} bind={} fmt={:#x}",
                acc.pipeline_ref, t.binding, t.pixel_format
            ));
            return ComputeStatus::Unsupported("storage_no_format_specialize");
        };
        storage_formats.push((t.binding, guest_fmt));
    }
    let prepared_program = match kernel_shader.prepare_program(&storage_formats) {
        Ok(program) => program,
        Err(error) => {
            crate::observe::Emit::decline("compute_linux_storage_format", &error)
                .field("pipe", acc.pipeline_ref)
                .fail();
            return ComputeStatus::Unsupported("storage_format_specialize_error");
        }
    };
    // Compute-side analog of the render resident gates: a deferred storage
    // writeback leaves guest-visible bytes GPU-resident-only until a flush
    // choke point lands them, so it requires the device's
    // `deferred_gpu_only_content` capability (off on portability-subset /
    // MoltenVK, where guest pages stay authoritative and the writeback runs
    // synchronously in this call).
    for t in &mut staged_tex {
        if t.is_storage {
            let Some(guest_fmt) = t.storage_format else {
                crate::observe::fail(format!(
                "compute_linux unsupported storage_format reason=no_storage_format pipe={} bind={} fmt={:#x}",
                    acc.pipeline_ref, t.binding, t.pixel_format
                ));
                return ComputeStatus::Unsupported("storage_no_format_writeback");
            };
            let Some((_, specialized)) = prepared_program
                .storage_image_formats
                .iter()
                .find(|(binding, _)| *binding == t.binding)
            else {
                crate::observe::fail(format!(
                    "compute_linux storage_format fail reason=spirv_format_specialize_internal pipe={} bind={} guest={guest_fmt:?}",
                    acc.pipeline_ref, t.binding
                ));
                return ComputeStatus::Unsupported("storage_format_specialize_internal");
            };
            // `None` is metal2vulkan's reflected `ImageFormat::Unknown`; the
            // view then uses the guest format whose exact read/write capability
            // facts were supplied to translation.
            let shader_fmt = if specialized.is_none() {
                crate::observe::off(format!(
                    "compute_linux bgra_storage_composite pipe={} bind={} mode=without_format guest={guest_fmt:?} view=B8G8R8A8_UNORM {}x{}",
                    acc.pipeline_ref, t.binding, t.width, t.height
                ));
                guest_fmt
            } else {
                specialized.expect("concrete specialization checked above")
            };
            let shader_decl = kernel_shader.interface().storage_image_format(t.binding);
            if *specialized != shader_decl {
                crate::observe::off(format!(
                    "compute_linux storage_format_specialize pipe={} bind={} spirv={shader_decl:?} specialized={specialized:?} engine={shader_fmt:?} guest={guest_fmt:?} guest_bpp={}",
                    acc.pipeline_ref,
                    t.binding,
                    guest_fmt.bytes_per_texel()
                ));
            }
            let seed =
                match std::mem::replace(&mut t.input, VulkanTextureInput::HostBytes(Vec::new())) {
                    VulkanTextureInput::HostBytes(bytes) => {
                        reims_vgpu_core::ComputeStorageImageSeed::Bytes(bytes)
                    }
                    VulkanTextureInput::GuestPages(source) => {
                        reims_vgpu_core::ComputeStorageImageSeed::GuestPages(source)
                    }
                    VulkanTextureInput::GuestImage(_) => {
                        crate::observe::fail(format!(
                        "compute_linux internal reason=sampled_image_on_storage pipe={} bind={}",
                        acc.pipeline_ref, t.binding
                    ));
                        return ComputeStatus::Unsupported("compute_storage_source_role");
                    }
                    VulkanTextureInput::TargetResident(_) => {
                        crate::observe::fail(format!(
                        "compute_linux internal reason=target_resident_on_storage pipe={} bind={}",
                        acc.pipeline_ref, t.binding
                    ));
                        return ComputeStatus::Unsupported("compute_storage_source_role");
                    }
                    VulkanTextureInput::Resident(ResidentServe::Seed(_)) => {
                        reims_vgpu_core::ComputeStorageImageSeed::Resident
                    }
                    VulkanTextureInput::Resident(ResidentServe::Sample(..)) => {
                        crate::observe::fail(format!(
                        "compute_linux internal reason=sample_resident_on_storage pipe={} bind={}",
                        acc.pipeline_ref, t.binding
                    ));
                        return ComputeStatus::Unsupported("compute_storage_source_role");
                    }
                };
            storage_images.push(ComputeStorageImageResource {
                binding: t.binding,
                array_element: t.array_element,
                descriptor_count: t.descriptor_count,
                format: shader_fmt,
                width: t.width,
                height: t.height,
                seed,
                // The guest window this output belongs to is on `t.writeback`,
                // so the destination is decided from the window rather than
                // from anything about this dispatch. `Host` needs no host
                // capability; the direct arm needs the guest-RAM import, and
                // where that is absent the licence declines by name and this
                // reads back exactly as it always did.
                destination: direct_destination(state, host, t, shader_fmt),
                residency: t
                    .residency
                    .map(|candidate| reims_vgpu_core::ComputeStorageResidency {
                        identity: candidate.key,
                        seed_generation: candidate.seed_generation,
                        output_generation: next_mapping_content_generation(
                            candidate.seed_generation,
                        ),
                    }),
            });
        } else {
            let Some(sampled_fmt) = mtl_to_engine_sampled(t.pixel_format) else {
                crate::observe::fail(format!(
                    "compute_linux sampled_format fail reason=mtl_format_unsupported pipe={} bind={} fmt={:#x}",
                    acc.pipeline_ref, t.binding, t.pixel_format
                ));
                return ComputeStatus::Unsupported("sampled_format_unsupported");
            };
            let sampled_fmt =
                sampled_fmt.with_swizzle(t.view_swizzle.after(&sampled_fmt.swizzle()));
            let source =
                match std::mem::replace(&mut t.input, VulkanTextureInput::HostBytes(Vec::new())) {
                    VulkanTextureInput::HostBytes(bytes) => {
                        reims_vgpu_core::ComputeSampledImageSource::Bytes(bytes)
                    }
                    VulkanTextureInput::GuestPages(source) => {
                        reims_vgpu_core::ComputeSampledImageSource::GuestPages(source)
                    }
                    VulkanTextureInput::GuestImage(source) => {
                        reims_vgpu_core::ComputeSampledImageSource::GuestImage(source)
                    }
                    VulkanTextureInput::TargetResident(identity) => {
                        reims_vgpu_core::ComputeSampledImageSource::TargetResident(identity)
                    }
                    VulkanTextureInput::Resident(ResidentServe::Sample(identity, generation)) => {
                        reims_vgpu_core::ComputeSampledImageSource::Resident(
                            reims_vgpu_core::ComputeResidentSampleBind {
                                identity,
                                generation,
                            },
                        )
                    }
                    VulkanTextureInput::Resident(ResidentServe::Seed(_)) => {
                        crate::observe::fail(format!(
                            "compute_linux internal reason=seed_resident_on_sample pipe={} bind={}",
                            acc.pipeline_ref, t.binding
                        ));
                        return ComputeStatus::Unsupported("compute_sampled_source_role");
                    }
                };
            sampled_images.push(ComputeSampledImageResource {
                binding: t.binding,
                array_element: t.array_element,
                descriptor_count: t.descriptor_count,
                format: sampled_fmt,
                width: t.width,
                height: t.height,
                multisampled: t.multisampled,
                source,
                content: state
                    .task_objects
                    .resources
                    .content_stamp(task_id, t.resource_ref)
                    .map(|(resource, version)| reims_vgpu_core::ContentStamp { resource, version }),
            });
        }
    }

    // Vulkan requires the pipeline layout to contain a descriptor for every
    // resource the module *statically uses*. The layout this device builds is
    // assembled from what the guest bound, so a texture the kernel samples and
    // the guest left empty is absent from the layout entirely — not an unwritten
    // slot in it. That is undefined behaviour by the specification and it is
    // worse than that in practice: Mesa's Intel driver sizes its binding array
    // to `max_binding + 1`, zero-fills every number nothing declared, and scores
    // each used binding as `(use_count << 7) / array_size` when it picks
    // binding-table slots. A hole under a used binding divides by zero, so the
    // whole process dies of `SIGFPE` inside `vkCreateComputePipelines` with no
    // error for this device to decline on. Fill it the way the sampler class
    // below already fills its own.
    //
    // Only `Used` is filled. A declared-and-unused variable is legal to omit and
    // must stay omitted, or the census that separated those two populations
    // cannot tell them apart any more; `Ambiguous` is two variables on one
    // binding, which is its own defect and is not repaired by picking one.
    let bound: Vec<u32> = sampled_images.iter().map(|img| img.binding).collect();
    for binding in kernel_shader.null_sampled_image_bindings(&bound) {
        crate::runtime::drain::note_store_route("compute_null_texture");
        sampled_images.push(ComputeSampledImageResource {
            binding,
            array_element: 0,
            descriptor_count: 1,
            format: reims_vgpu_protocol::SampledImageFormat::linear(
                reims_vgpu_protocol::StorageImageFormat::Rgba8Unorm,
                reims_vgpu_protocol::SwizzlePlan::default(),
            ),
            width: 0,
            height: 0,
            multisampled: false,
            source: reims_vgpu_core::ComputeSampledImageSource::Null,
            content: None,
        });
    }

    // Reflection is the sampler interface emitted alongside this exact module.
    // Derive it once per dispatch instead of walking every SPIR-V instruction
    // once to filter guest samplers and again to provision defaults.
    let reflected_samplers = kernel_shader.samplers();
    let mut samplers = Vec::new();
    for s in &acc.samplers {
        let Some(binding) = reflected_samplers
            .iter()
            .find(|sampler| sampler.metal_index == s.index)
            .map(|sampler| sampler.binding)
        else {
            continue;
        };
        let mut sampler = match crate::runtime::draw::load_vulkan_sampler(
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
    for reflected in reflected_samplers.iter() {
        if !samplers
            .iter()
            .any(|sampler| sampler.binding == reflected.binding)
        {
            if let Some(state) = reflected.static_state {
                let sampler = match crate::runtime::draw::reflected_static_sampler_resource(
                    "kernel",
                    reflected.binding,
                    state,
                ) {
                    Ok(sampler) => sampler,
                    Err(reason) => {
                        crate::observe::Emit::decline("compute_linux_static_sampler", &reason)
                            .field("pipe", acc.pipeline_ref)
                            .fail_once(
                                (u64::from(acc.pipeline_ref) << 32) | u64::from(reflected.binding),
                            );
                        return ComputeStatus::Unsupported("compute_static_sampler_unsupported");
                    }
                };
                samplers.push(sampler);
            } else {
                samplers.push(reims_vgpu_core::SamplerResource::null(reflected.binding));
            }
        }
    }

    let req = ComputeRequest {
        program: prepared_program.stage.clone(),
        entry: "main".into(),
        dispatch: plan,
        barriers: barriers.to_vec(),
        storage_buffers,
        sampled_images,
        samplers,
        storage_images,
    };
    let run_engine = |req: ComputeRequest| {
        let engine_done = spawn_compute_engine_stall_watchdog(
            acc.pipeline_ref,
            &req,
            std::time::Duration::from_millis(COMPUTE_ENGINE_STALL_PROXY_MS),
        );
        let executor = std::sync::Arc::clone(&state.executor);
        let submission = crate::runtime::executor::context_for(state, task_id);
        let out = crate::runtime::executor::execute_compute(executor.as_ref(), submission, req);
        engine_done.store(true, std::sync::atomic::Ordering::Release);
        out
    };
    let out_result = run_engine(req);
    let (completed_submission, out) = match out_result {
        Ok(receipt) => {
            state
                .task_objects
                .resources
                .record_gpu_materializations(receipt.gpu_materialized.iter().copied());
            (receipt.submission.id, receipt.output)
        }
        Err(e) => {
            let unsupported = matches!(&e, DrawError::Unsupported(_));
            crate::observe::Emit::decline("compute_linux_engine", &e)
                .field("pipe", acc.pipeline_ref)
                .fail_once(u64::from(acc.pipeline_ref));
            if unsupported {
                return ComputeStatus::Unsupported("engine_run_unsupported");
            }
            return ComputeStatus::BackendFailed("compute_vk_engine_run");
        }
    };
    if out.buffers.len() != buffer_writable_count || out.images.len() != storage_count {
        crate::observe::fail(format!(
            "compute_linux readback count mismatch pipe={} buf={}/{} img={}/{}",
            acc.pipeline_ref,
            out.buffers.len(),
            buffer_writable_count,
            out.images.len(),
            storage_count
        ));
        return ComputeStatus::BackendFailed("compute_vk_readback_count");
    }
    let ComputeOutput {
        buffers: output_buffers,
        images: output_images,
    } = out;
    // A resource can occupy more than one writable binding. Complete its GPU
    // version once, immediately after the first successful output, and let a
    // later alias strengthen that same version to guest-materialized. Publishing
    // here (rather than after the whole loop) also preserves output A when
    // writeback B fails.
    let mut content_effects = std::collections::BTreeMap::<
        u32,
        (
            reims_vgpu_protocol::ResourceId<reims_vgpu_protocol::ResourceObject>,
            reims_vgpu_protocol::ContentVersion,
            bool,
        ),
    >::new();
    let mut record_content_effect = |state: &Device,
                                     resource_ref: u32,
                                     guest_materialized: bool| {
        if let Some((id, version, already_materialized)) = content_effects.get_mut(&resource_ref) {
            if guest_materialized && !*already_materialized {
                *already_materialized = state
                    .task_objects
                    .resources
                    .record_gpu_to_guest_copy(*id, *version);
            }
            return;
        }
        let Some((id, version)) = state.task_objects.resources.record_completed_gpu_store(
            task_id,
            resource_ref,
            completed_submission,
        ) else {
            return;
        };
        let materialized = guest_materialized
            && state
                .task_objects
                .resources
                .record_gpu_to_guest_copy(id, version);
        content_effects.insert(resource_ref, (id, version, materialized));
    };
    for buffer in output_buffers {
        let Some(s) = staged_bufs
            .iter_mut()
            .find(|staged| staged.bind.index == buffer.binding)
        else {
            crate::observe::fail(format!(
                "compute_linux readback binding mismatch pipe={} bind={}",
                acc.pipeline_ref, buffer.binding
            ));
            return ComputeStatus::BackendFailed("compute_vk_readback_binding");
        };
        match buffer.result {
            reims_vgpu_core::ComputeBufferResult::Bytes(bytes) => {
                s.bytes = bytes;
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
            reims_vgpu_core::ComputeBufferResult::Landed { bytes } => {
                crate::runtime::drain::note_store_route("compute_buffer_wb_landed");
                crate::runtime::drain::note_store_route_n("compute_buffer_wb_landed_bytes", bytes);
            }
        }
        record_content_effect(state, s.bind.buffer_ref, true);
    }
    for (t, result) in staged_tex
        .iter_mut()
        .filter(|texture| texture.is_storage)
        .zip(output_images)
    {
        let guest_materialized = match result {
            ComputeImageResult::Bytes(bytes) => {
                t.bytes = bytes;
                match writeback_texture(state, host, task_id, t) {
                    Ok(materialization) => materialization == GuestMaterialization::Materialized,
                    Err(e) => return e,
                }
            }
            // The engine copied straight into the guest's pages, so there is no
            // writeback to do and no bytes to do it from.
            ComputeImageResult::Landed { bytes } => {
                if matches!(t.writeback, TextureWriteback::None) {
                    crate::observe::fail(format!(
                        "compute_linux readback destination mismatch pipe={} bind={} result=landed destination=none",
                        acc.pipeline_ref, t.binding
                    ));
                    return ComputeStatus::BackendFailed("compute_vk_landed_without_destination");
                }
                crate::runtime::drain::note_store_route("compute_wb_landed");
                let _ = bytes;
                // The guest's pages are the only place this frame exists now,
                // so no host cache may go on naming one. This arm writes
                // neither cache — both are on the readback path — but a
                // previous dispatch's readback may have left an entry, and it
                // is stale by exactly one frame. Same call, same reason, as
                // both arms of the render rail's GVA Store.
                match &t.writeback {
                    TextureWriteback::Linear {
                        gva, texture_ref, ..
                    } => crate::runtime::render_writeback::forget_gva_host_copies(
                        state,
                        task_id,
                        *gva,
                        *texture_ref,
                    ),
                    // The surface-keyed rail owes more than a cache forget — a
                    // resident storage window over these bytes and the mapping's
                    // own written mark — and the render Store that shares this
                    // destination owes exactly the same set. Both call it.
                    //
                    // The offsets are the staged ones rather than the licence's,
                    // and they are the same offsets: `licence_iosurface_texture_surface`
                    // resolves the window through `iosurface_texture_sample_window`, which
                    // is where these came from when the texture was staged.
                    TextureWriteback::IOSurface {
                        mapping_id,
                        surface_offset,
                        span_end,
                        ..
                    } => {
                        let _ = crate::runtime::mapping_write::note_iosurface_texture_landed(
                            state,
                            *mapping_id,
                            *surface_offset,
                            *span_end,
                        );
                    }
                    TextureWriteback::None => {}
                }
                true
            }
        };
        // Only an actual guest write makes the resident reproducible. A heap
        // texture has no guest destination: its host readback is transient,
        // and declaring that copy sufficient would let reclaim discard the
        // resource's only durable content.
        if guest_materialized {
            if let Some(candidate) = t.residency {
                state
                    .executor
                    .note_resident_storage_copied_out(&candidate.key);
            }
        }
        note_storage_residency_writeback(state, t);
        record_content_effect(state, t.resource_ref, guest_materialized);
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

/// Whether this dispatch asks for something the Vulkan compute rail cannot do.
///
/// Only two things refuse: a pipeline carrying a stage-input descriptor, and an
/// imageblock. Both name storage this rail has no representation for, so the
/// dispatch would compute against memory that does not exist.
///
/// **`stage_in_region` and `stage_in_region_indirect` deliberately do not
/// refuse.** They bound the stage-in grid a stage-input pipeline walks, so on a
/// pipeline that declares no stage input there is nothing for them to bound and
/// executing the dispatch loses no guest work. That is a claim about the
/// contract rather than a measurement, which is why the caller counts the case
/// (`compute_stage_in_region_unused`) instead of staying silent about it: if a
/// guest ever pairs a region with a stage-input-free pipeline *and* depends on
/// it, the counter is what says so.
pub(crate) fn linux_stage_input_or_imageblock_unsupported(
    pipeline_stage_input: bool,
    acc: &ComputeAccum,
) -> bool {
    pipeline_stage_input || acc.imageblock.is_some()
}

const COMPUTE_ENGINE_STALL_PROXY_MS: u64 = 2_000;

/// Measurement-only watchdog for backend calls that cannot be bounded by a
/// Vulkan fence timeout (notably pipeline creation and some driver submits).
/// It never changes execution. A fired proxy preserves the private request
/// inputs under /tmp so the stall can be reproduced without another VM boot.
fn spawn_compute_engine_stall_watchdog(
    pipeline_ref: u32,
    req: &reims_vgpu_core::ComputeRequest,
    threshold: std::time::Duration,
) -> std::sync::Arc<std::sync::atomic::AtomicBool> {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let done = Arc::new(AtomicBool::new(false));
    let thread_done = Arc::clone(&done);
    let grid = req.dispatch.counts;
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

/// Thin adapters over the canonical tables in
/// [`reims_vgpu_core::pixel_format`].
///
/// These two used to *be* the tables — a second copy of the selector→engine and
/// Metal→semantic mappings living in the compute path, where nothing checked them
/// against the pixel table they had to agree with. The call sites below are all
/// `if let Some(..)` / `let Some(..) else`, so the adapters keep that shape; the
/// decision itself now happens in exactly one place.
fn mtl_to_engine_sampled(format: u16) -> Option<reims_vgpu_protocol::SampledImageFormat> {
    // The *sampled* admission, not the storage one. Asking `storage_image` here
    // cost macOS 14 and macOS 15 a whole `DispatchThreadgroups` a boot on
    // `MTLPixelFormatR16Unorm`, which is sampleable everywhere and is not a
    // storage format — see `translate::pixel::sampled_image`.
    pixel_format::compute_sampled_image_format(format)
}

#[cfg(test)]
mod tests;
