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
    decode_heap_texture, decode_texture_descriptor, decode_type7_descriptor, texture_type8_opcode,
    ComputeStageInputDescriptor, Descriptor as ResourceDescriptor, HEAP_TEXTURE_OPCODE,
    HEAP_TEXTURE_WIDE_OPCODE, OBJECT_TYPE_BUFFER, OBJECT_TYPE_TEXTURE, OBJECT_TYPE_TEXTURE_VARIANT,
    OBJECT_TYPE_TEXTURE_VIEW, OBJECT_TYPE_TYPE7, TEXTURE_VIEW_OPCODE_BUFFER_TEXTURE,
    TEXTURE_VIEW_OPCODE_BUFFER_TEXTURE_WIDE,
};
use crate::runtime::draw::{host_alloc_len, StoreTargetPages};
use crate::runtime::gva_mem;
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::mapper;
use crate::runtime::mapping_write;
use crate::runtime::mtlb::{load_mtlb, AirLoadRail};
use crate::runtime::objects;

/// Cap on Metal compute buffer slots (matches backend `REIMS_VGPU_METAL_MAX_BUFFERS`).
pub const MAX_COMPUTE_BUFFER_SLOTS: u32 = 31;
/// Cap on compute texture stream indices (Metal bind = `TEXTURE_BINDING_BASE +
/// index`). Metal's compute texture argument table, and Apple's serializer's:
/// this rail refused indices 31..127 only because the descriptor binding band
/// was that narrow, which `spirv_bind::widen_sampled_bands` fixed.
pub const MAX_COMPUTE_TEXTURE_SLOTS: u32 = 128;
/// Cap on compute sampler stream indices (Metal bind = `SAMPLER_BINDING_BASE +
/// index`). Metal's sampler argument table, which is genuinely 16.
pub const MAX_COMPUTE_SAMPLER_SLOTS: u32 = 16;

// The two caps above are what keeps a stream index inside its own descriptor
// band: this rail binds a texture at `TEXTURE_BINDING_BASE + index` and a
// sampler at `SAMPLER_BINDING_BASE + index`, so a cap that let an index reach
// the next base would make a texture resolve against a sampler's reflection
// entry — and `reflected_compute_texture` would answer `Absent` for it, which
// this rail treats as "the shader does not use this binding" and skips. A
// silent drop, from two constants that never name each other.
//
// `backend::metal::constants` states the same relation for the Metal argument
// tables, in the same form and for the same reason; this side had the caps and
// the bands in two modules with nothing between them.
const _: () = assert!(
    crate::runtime::spirv_bind::TEXTURE_BINDING_BASE + MAX_COMPUTE_TEXTURE_SLOTS
        <= crate::runtime::spirv_bind::SAMPLER_BINDING_BASE
);
const _: () = assert!(
    crate::runtime::spirv_bind::SAMPLER_BINDING_BASE + MAX_COMPUTE_SAMPLER_SLOTS
        <= crate::runtime::spirv_bind::COLOR_INPUT_BINDING_BASE
);

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
/// The cap comparison is exclusive (`index >= MAX_*`) to match the backend,
/// which sizes its argument-table arrays to exactly these counts
/// (`[false; REIMS_VGPU_METAL_MAX_BUFFERS]`) and guards
/// `idx >= REIMS_VGPU_METAL_MAX_*` before indexing — so slot `MAX` is out of
/// range and a bind there is a genuine drop, not a boundary the accum should
/// have accepted.
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

/// The sampled-image bindings that need a neutral texture: those the module
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
#[cfg(feature = "backend-vulkan")]
fn neutral_sampled_image_bindings(spirv: &[u32], bound: &[u32]) -> Vec<u32> {
    crate::runtime::spirv_bind::sampled_image_bindings(spirv)
        .into_iter()
        .filter(|binding| {
            !bound.contains(binding)
                && crate::runtime::spirv_bind::descriptor_static_use(spirv, *binding).is_violation()
        })
        .collect()
}

/// Side length of the texture substituted for a sampled image the kernel
/// samples and the guest never bound.
///
/// One texel, because there is nothing to derive a size from: the guest supplied
/// no texture, and any larger extent would be a number chosen to look plausible.
/// A kernel that asks this image its size gets 1×1 and that is reported, rather
/// than a guess that reads as data.
#[cfg(feature = "backend-vulkan")]
const NEUTRAL_SAMPLED_IMAGE_EXTENT: u32 = 1;

/// A sampled image the kernel statically uses and the guest never bound, given a
/// neutral transparent texture so the pipeline layout can describe it.
///
/// **A repair that succeeded, not a success**, which is why it goes to the fail
/// channel: the kernel samples a texture whose contents this device invented, and
/// the reliance has to stay measurable so a later session can find out whether
/// the guest ever depended on what was in it. Nothing here claims the read did
/// not matter.
///
/// Omitting the binding instead is not the cheaper option. It is a specification
/// violation, and on Mesa's Intel driver it is a `SIGFPE` that kills the host
/// process — see the walk in [`crate::runtime::spirv_bind::sampled_image_bindings`].
#[cfg(feature = "backend-vulkan")]
struct NeutralSampledImage {
    binding: u32,
    width: u32,
    height: u32,
}

#[cfg(feature = "backend-vulkan")]
impl crate::observe::Decline for NeutralSampledImage {
    fn slug(&self) -> &'static str {
        "compute_neutral_sampled_image_unbound"
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![
            ("binding", self.binding.to_string()),
            ("width", self.width.to_string()),
            ("height", self.height.to_string()),
        ]
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
    /// What does bound it is the *host's* argument table, and only one backend
    /// has one: `backend::metal::compute::bind_threadgroup_memory` refuses at
    /// `REIMS_VGPU_METAL_MAX_THREADGROUP_MEMORY` and names the check. The Vulkan
    /// rail consumes none of these binds — SPIR-V declares workgroup shared
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
    /// The reflected exact-thread dispatch names a push-constant offset whose
    /// payload would not fit the range the translator publishes. Refused rather
    /// than clamped: a truncated range is a shader reading bytes no one wrote.
    DispatchPushRangeUnavailable { pipeline_ref: u32 },
    /// The translator refused to plan this launch's regions, so the dispatch
    /// does not reach the device. Its own text rides the emitter's `detail`
    /// field rather than the reason, which stays a stable slug.
    DispatchPlanRefused { pipeline_ref: u32 },
}

impl crate::observe::Decline for ComputeReflectionDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::ReflectedResourceUnsupported { .. } => "compute_reflection_resource_unsupported",
            Self::ReflectedInterfaceUnsupported { .. } => {
                "compute_reflection_interface_unsupported"
            }
            Self::DispatchPushRangeUnavailable { .. } => {
                "compute_reflection_dispatch_push_range_unavailable"
            }
            Self::DispatchPlanRefused { .. } => "compute_reflection_dispatch_plan_refused",
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
            Self::DispatchPushRangeUnavailable { pipeline_ref } => {
                vec![("pipeline_ref", pipeline_ref.to_string())]
            }
            Self::DispatchPlanRefused { pipeline_ref } => {
                vec![("pipeline_ref", pipeline_ref.to_string())]
            }
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

/// The `MTLDispatchType` the guest declared, or `Serial` with the substitution
/// named in the always-on log.
///
/// `WRITE_DESCRIPTOR` carries this ordinal straight off the wire and nothing
/// bounds it: the decoder stores `d.dispatch_type.get()` unexamined, and the
/// accumulator used to store that. The narrowing lived at the far end of the
/// rail instead — inside `execute_dispatch_metal`, as
/// `if acc.dispatch_type == CONCURRENT { CONCURRENT } else { SERIAL }` — which
/// is `Serial` for every value the device does not recognise, chosen silently.
///
/// Three things were wrong with it being there, and all three are why the rule
/// now lives here, beside the field it constrains:
///
/// - **It was invisible.** A guest asking for a dispatch type this device has no
///   contract for got a *serial* encoder and no line anywhere. Serial and
///   concurrent differ in whether Metal may overlap the dispatches in a segment,
///   so the substitution is a real change to what the guest asked for.
/// - **It made a written refusal unreachable.** `backend::metal::compute`'s
///   `mtl_dispatch_type` returns `None` for an unrecognised ordinal and its
///   caller declines with `metal_compute_dispatch_type_invalid` — a typed
///   refusal that could never fire, because the only producer feeding it had
///   already replaced every unrecognised value with `Serial`.
/// - **It only ran on one arm.** `execute_dispatch_metal` is
///   `backend-metal`-gated, so on a Vulkan host the field was accepted, stored
///   and then read by nobody. The value is a *guest contract* fact, not a
///   backend one, so both arms now score it the same way and the check runs on
///   the pathway this repository can boot.
///
/// The substitution is kept rather than turned into a decline, deliberately. The
/// Metal SDK's `MTLDispatchType` has exactly `Serial` and `Concurrent`, so an
/// out-of-range ordinal here is far more likely to be *this device* reading the
/// wrong wire offset than a guest asking for something new — and declining the
/// dispatch would turn a decode bug into lost guest work on a pathway no boot
/// available here can exercise. So it is reported and counted first. If
/// `compute_dispatch_type_unknown` is ever seen, the evidence to decide arrives
/// before the behaviour change does.
fn accepted_dispatch_type(task_id: u32, declared: u32) -> u32 {
    use crate::contract::dispatch::{
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

pub(crate) struct LoadedComputePipeline {
    pub kernel_func_ref: u32,
    /// Product-ready stage-input. `None` means the descriptor declared none —
    /// and only that. A descriptor whose entries exceeded the decoder's caps
    /// refuses the pipeline (`stage_input_over_cap`) rather than landing here as
    /// `None`, because the two are different guest programs.
    pub stage_input: Option<ComputeStageInputDescriptor>,
}

/// What a type-7's stage-input block means for the pipeline carrying it.
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
    let report = crate::observe::RungReport::new("compute_load_pipeline", "pipe_ref");
    let miss = |reason: &str, detail: String| -> Option<LoadedComputePipeline> {
        report.reason(task_id, pipeline_ref, reason, &detail);
        None
    };
    let (_entry, desc) =
        match objects::resolve_descriptor(state, host, task_id, pipeline_ref, &[OBJECT_TYPE_TYPE7])
        {
            Ok(found) => found,
            Err(rung) => {
                report.rung(task_id, pipeline_ref, rung);
                return None;
            }
        };
    let Ok(decoded) = decode_type7_descriptor(&desc) else {
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
            Some(LoadedComputePipeline {
                kernel_func_ref: cp.kernel_func_ref,
                stage_input,
            })
        }
        ResourceDescriptor::ComputePipeline(_) => miss("kernel_func_zero", String::new()),
        _ => miss("not_compute_pipeline", String::new()),
    }
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

/// Conservative whole-allocation staging used by the Metal-direct callers,
/// which do not translate the shader through the reflection-producing path.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
pub(crate) fn stage_buffer<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    bind: &ComputeBufferBind,
) -> Result<StagedBuffer, ComputeStatus> {
    stage_buffer_with_extent(state, host, task_id, bind, None)
}

pub(crate) fn stage_buffer_with_extent<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    bind: &ComputeBufferBind,
    extent_cap: Option<u64>,
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
    let (_entry, desc_bytes) = match objects::resolve_descriptor(
        state,
        host,
        task_id,
        bind.buffer_ref,
        &[OBJECT_TYPE_BUFFER],
    ) {
        Ok(found) => found,
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
    let Ok(desc) = crate::runtime::decode::resource::decode_buffer_descriptor(&desc_bytes) else {
        return miss(
            ComputeStatus::MissingBuffer(crate::observe::ladder_slug!(
                "compute_stage_buf",
                desc_decode
            )),
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
        ///
        /// Ordered as well as a membership set, because the GPU-direct arm reads
        /// index `i` as page `i` of the window. See [`staged_window_pages`].
        pages: StoreTargetPages,
    },
    Type11 {
        mapping_id: u32,
        /// The window this bind was staged against — a byte offset into the
        /// mapping, the surface's row pitch, and one past the last byte the
        /// window may touch.
        ///
        /// Resolved once, at stage time, through the plane the bind actually
        /// names: `type5_sample_window` when the wire carried a type-5 view's
        /// plane index, `type11_sample_window` otherwise. Both the read that
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
    state: &DeviceState,
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
/// - the writeback must be a **guest-linear plane**. A type-11 destination is a
///   tiled surface mapping, which [`crate::runtime::render_writeback::GvaPlaneDestination`]
///   cannot describe and the licence therefore cannot walk. It is the largest
///   class this arm does not reach, so [`note_type11_shape`] bands how much of
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
#[cfg(feature = "backend-vulkan")]
fn direct_destination<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    tex: &StagedTexture,
    held: ash::vk::Format,
) -> crate::backend::vulkan::engine::ComputeImageDestination {
    use crate::backend::vulkan::engine::ComputeImageDestination;
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
        return type11_destination(state, host, tex, held);
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

/// [`direct_destination`] for a type-11 surface mapping.
///
/// A tiled surface mapping is not a guest-linear plane and the GVA licence
/// cannot describe one — but it is not therefore unreachable, and treating it as
/// such is what this arm used to do. It answered `Host` before looking at
/// anything, and on a driven macos-13 boot that was 35 of the 51 storage
/// destinations, every one of them a device→host crossing.
///
/// The destination that *can* describe it already existed on the render rail,
/// resolving the sample window, walking the mapping's page entries and building
/// the same [`crate::backend::vulkan::engine::GuestPageTarget`] this rail wants.
/// It is now [`crate::runtime::mapping_write::licence_type11_surface`] and both
/// rails ask it, so the surface geometry, the format rule, the page walk and the
/// guest-RAM references have one spelling rather than two.
///
/// Every decline is a routing answer on the `OFF` channel, not a loss: readback
/// lands identical bytes, and on a host without the guest-RAM import it is the
/// only rail there is.
#[cfg(feature = "backend-vulkan")]
fn type11_destination<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    tex: &StagedTexture,
    held: ash::vk::Format,
) -> crate::backend::vulkan::engine::ComputeImageDestination {
    use crate::backend::vulkan::engine::ComputeImageDestination;
    let TextureWriteback::Type11 {
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
    // plane-correct for a type-5 view and already a sub-rectangle where the
    // dispatch writes one, and it is the same window the readback rail lands
    // through — so the two rails cannot name different bytes of one surface.
    match crate::runtime::mapping_write::licence_type11_surface(
        state,
        host,
        held,
        &crate::runtime::mapping_write::Type11SurfaceDestination {
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
                "compute_dst_guest_pages_type11_resident"
            } else {
                "compute_dst_guest_pages_type11_transient"
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
                "compute_dst_type11 bind={} mid={mapping_id} dims={width}x{height} held={held:?} reason={decline:?}",
                tex.binding
            ));
            crate::runtime::drain::note_store_route("compute_dst_host_type11_unlicensed");
            ComputeImageDestination::Host
        }
    }
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
    #[cfg(feature = "backend-vulkan")]
    pub array_element: u32,
    #[cfg(feature = "backend-vulkan")]
    pub descriptor_count: u32,
    /// The guest ref this was staged from. Carried so a refusal downstream can
    /// name the object the guest bound and not only the slot it bound it to.
    /// Read by the direct-Metal rail's format refusal; the Vulkan arm reaches
    /// its images by another route and never asks.
    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    pub texture_ref: u32,
    /// Raw Metal pixel format from the exact texture/view descriptor.
    pub pixel_format: u16,
    /// Product storage-selector ABI when this Metal format is storage-capable.
    /// Sample-only formats such as RGB9E5Float intentionally have no selector.
    /// The contract's storage-image selector for this texture's format, or
    /// `None` for a format that is not a storage image.
    ///
    /// Carried as the enum rather than as its `u32` ordinal. It used to be
    /// narrowed to `u32` the moment `pixel_format::storage_selector` produced
    /// it, at three staging sites, which pushed the coverage question past every
    /// compiler that could have answered it: both backends then matched raw
    /// integers, and the Metal one had silently been missing a member.
    pub storage_selector: Option<pixel_format::StorageImageSelector>,
    pub width: u32,
    pub height: u32,
    /// How many mip levels `bytes` carries, base first, packed tightly by
    /// [`crate::contract::extent::tight_pyramid_spans`].
    ///
    /// `1` on every rail but the type-2/3 linear one, and `1` there too for a
    /// storage binding or a view that already names a level: a compute write
    /// names one level and a levelled view exposes one. Where it is greater,
    /// `width`/`height` remain level 0's extent and every other level's is
    /// `mip_extent(width, n)` — the pyramid is a derivation of this geometry
    /// and not a second one, so no level's extent is stored twice.
    pub mip_levels: u32,
    pub bytes: Vec<u8>,
    pub is_storage: bool,
    #[cfg(feature = "backend-vulkan")]
    residency: Option<ComputeStorageResidencyCandidate>,
    /// What the engine could already serve for this binding, so the stage-time
    /// guest read was skipped and `bytes` is a zero placeholder.
    ///
    /// [`ResidentServe::Seed`] — a storage binding whose resident the engine
    /// holds at a verified generation; it must never be seeded from the
    /// placeholder. [`ResidentServe::Sample`] — a sampled input whose window is
    /// a prior dispatch's storage output; the engine seeds the sampled image by
    /// copy-on-sample from that resident, again never from the bytes.
    ///
    /// One field rather than the `bool` and `Option` pair it replaces: those
    /// were the variant tag and the payload of this enum stored apart, so every
    /// producer had to rebuild both halves and nothing made a producer that set
    /// one without the other fail to compile.
    #[cfg(feature = "backend-vulkan")]
    serve: Option<ResidentServe>,
    /// The retained multisample render target this binding is served from.
    ///
    /// Set only for a kernel-declared `texture2d_ms<T, access::read>`, and it is
    /// exclusive with everything above it by construction: `bytes` is empty,
    /// `serve` is `None`, and nothing is staged, because
    /// `engine::types::SampledResource::multisampled` says linear bytes cannot
    /// be uploaded into a multisample image at all. The engine binds this
    /// target's own view.
    #[cfg(feature = "backend-vulkan")]
    multisample_target: Option<crate::backend::vulkan::engine::TargetIdentity>,
    writeback: TextureWriteback,
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
use crate::backend::metal::abi::{ReimsVgpuComputeSampledImage, ReimsVgpuStorageImage};

impl StagedTexture {
    /// The Metal storage-image selector for this texture's guest pixel format,
    /// or a named refusal.
    ///
    /// Sample-only formats such as `RGB9E5Float` have no selector by design, so
    /// this is a real class rather than an internal error — a guest binding one
    /// into a compute slot loses that bind, and the line has to say which
    /// object at which slot in which format.
    ///
    /// Three sites asked this one question and each carried its own answer:
    /// `reason=metal_selector_missing` twice and `reason=no_backend_selector`
    /// once, under two event names, returning three different refusal slugs,
    /// with one line carrying `ref`, another `storage` and the third neither.
    /// A grep for any of the three names found a third of the occurrences.
    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    pub(crate) fn storage_selector_or_refuse(
        &self,
        task_id: u32,
        pipeline_ref: u32,
    ) -> Result<pixel_format::StorageImageSelector, ComputeStatus> {
        self.storage_selector.ok_or_else(|| {
            crate::observe::fail(format!(
                "compute_texture_format fail reason=no_backend_selector task={task_id} \
                 pipe={pipeline_ref} bind={} ref={} fmt={:#x} storage={}",
                self.binding, self.texture_ref, self.pixel_format, self.is_storage as u8
            ));
            ComputeStatus::Unsupported("compute_no_backend_selector")
        })
    }
}

/// Split staged compute textures into the two ABI image lists Metal binds.
///
/// A storage-capable bind becomes a `ReimsVgpuStorageImage` the kernel writes
/// through; everything else becomes a sampled image. Both rails that reach the
/// direct-Metal encoder — the ICB session's inherited binds and the standalone
/// dispatch — carried a copy of this, byte-identical apart from the refusal
/// above.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
#[allow(clippy::type_complexity)]
pub(crate) fn split_staged_textures(
    staged: &mut [StagedTexture],
    task_id: u32,
    pipeline_ref: u32,
) -> Result<
    (
        Vec<ReimsVgpuStorageImage>,
        Vec<ReimsVgpuComputeSampledImage>,
    ),
    ComputeStatus,
> {
    let mut storage: Vec<ReimsVgpuStorageImage> = Vec::new();
    let mut sampled: Vec<ReimsVgpuComputeSampledImage> = Vec::new();
    for t in staged {
        let selector = t.storage_selector_or_refuse(task_id, pipeline_ref)?;
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
            if t.mip_levels > 1 {
                // `ReimsVgpuComputeSampledImage` carries one level's texels, so
                // this rail would bind the base and answer every
                // `read(coord, lod)` above it with nothing. Refuse by name
                // rather than serve a pyramid flattened to its base.
                crate::observe::fail(format!(
                    "compute_stage_tex metal_fail reason=sampled_mip_levels task={task_id}                      pipe={pipeline_ref} bind={} levels={} {}x{}",
                    t.binding, t.mip_levels, t.width, t.height
                ));
                return Err(ComputeStatus::Unsupported("metal_sampled_mip_levels"));
            }
            sampled.push(ReimsVgpuComputeSampledImage::unswizzled(
                t.binding,
                selector,
                t.width,
                t.height,
                t.bytes.as_ptr(),
                t.bytes.len(),
            ));
        }
    }
    Ok((storage, sampled))
}

#[cfg(feature = "backend-vulkan")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ComputeStorageResidencyCandidate {
    key: crate::model::ComputeStorageResidencyKey,
    seed_generation: u32,
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
///
/// Which variant a binding can receive is fixed by `is_storage`, not chosen:
/// [`resident_serve`]'s two arms are the two variants. That is why the
/// consumers split the same way — the storage rail reads only the seed and the
/// sampled rail only the source.
///
/// Declared unconditionally although only the Vulkan backend can produce one,
/// so the rails that carry the answer through a `backend-metal` build can still
/// name its type. Each used to substitute its own loose tuple of the same
/// fields under `cfg(not(backend-vulkan))`, spelled out once per rail.
/// [`resident_serve`] is the only producer and it is gated on the Vulkan
/// backend, so on a `backend-metal` build both variants are constructed
/// nowhere. The rails still read the type — `serve` is `None` there and their
/// accessor calls compile unchanged — which is the whole point of declaring it
/// unconditionally.
#[cfg_attr(not(feature = "backend-vulkan"), allow(dead_code))]
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

/// Stage an opcode-9 buffer-backed texture: tight raw texels read out of the
/// type-1 buffer the guest named, at its declared offset and row pitch.
///
/// The contract is `newTextureWithBuffer:descriptor:offset:bytesPerRow:` — the
/// texels *are* the buffer's bytes, reinterpreted through the embedded texture
/// descriptor. [`crate::runtime::draw`] executes the same record for the draw
/// rail; this is its compute twin and reads the same fields the same way,
/// because one wire form with two disagreeing readers is the defect shape this
/// repository keeps finding. This arm used to refuse the form outright.
///
/// Unlike the draw twin this does **not** convert to RGBA8. [`StagedTexture`]
/// carries `pixel_format` beside `bytes`, so the native texels survive; the
/// draw arm narrows because its consumer takes RGBA8, and reports the loss as
/// `buftex_narrowed`. Here there is no loss to report.
///
/// De-pitching is the whole of the work: the guest's rows are `bytes_per_row`
/// apart and only the leading tight row is texels. The rest is padding the
/// guest may have written anything into, and folding it into the image is
/// exactly the failure the conformance battery fills padding with a distinct
/// pattern to catch.
fn stage_buffer_texture<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    binding: u32,
    is_storage: bool,
    bt: &crate::runtime::decode::resource::BufferTextureDescriptor,
) -> Result<StagedTexture, ComputeStatus> {
    let (width, height) = (bt.desc.width, bt.desc.height);
    if width == 0 || height == 0 {
        crate::observe::fail(format!(
            "compute_stage_tex buftex_fail reason=zero_geom ref={texture_ref} buf={} {width}x{height}",
            bt.buffer_ref
        ));
        return Err(ComputeStatus::Unsupported("compute_buftex_zero_geom"));
    }
    // A storage binding would have to write *back* through the buffer, which is
    // a destination contract this arm has no evidence for: no case in the
    // battery binds a buffer-backed texture writable, and inventing a writeback
    // here would widen the repair past what it can show. Refused under its own
    // name so the two questions stay separable in the log.
    if is_storage {
        crate::observe::fail(format!(
            "compute_stage_tex buftex_fail reason=storage_destination ref={texture_ref} buf={} {width}x{height}",
            bt.buffer_ref
        ));
        return Err(ComputeStatus::Unsupported(
            "compute_buffer_texture_storage_unsupported",
        ));
    }
    let format = if bt.desc.pixel_format != 0 {
        bt.desc.pixel_format
    } else {
        crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM
    };
    let Some(tight) = pixel_format::tight_row_bytes(width, format) else {
        crate::observe::fail(format!(
            "compute_stage_tex buftex_fail reason=unknown_fmt ref={texture_ref} buf={} fmt={format:#x} {width}x{height}",
            bt.buffer_ref
        ));
        return Err(ComputeStatus::Unsupported("compute_buftex_fmt"));
    };
    // A declared `bytesPerRow` of 0 means tight rows — the API default a
    // single-row or unpadded texture serializes as. Same reading as the draw
    // twin; the two arms must not differ on it.
    let bpr = if bt.bytes_per_row == 0 {
        u64::from(tight)
    } else {
        bt.bytes_per_row
    };
    if bpr < u64::from(tight) {
        crate::observe::fail(format!(
            "compute_stage_tex buftex_fail reason=bpr_short ref={texture_ref} buf={} bpr={bpr} tight={tight} {width}x{height} fmt={format:#x}",
            bt.buffer_ref
        ));
        return Err(ComputeStatus::Unsupported("compute_buftex_bpr_short"));
    }
    // The span the guest's rows actually occupy. Every row is `bpr` apart, but
    // the last one only needs its texels: demanding a full trailing pitch would
    // refuse a texture whose final row sits at the very end of the allocation.
    let Some(span) = bpr
        .checked_mul(u64::from(height) - 1)
        .and_then(|s| s.checked_add(u64::from(tight)))
        .and_then(|s| usize::try_from(s).ok())
    else {
        crate::observe::fail(format!(
            "compute_stage_tex buftex_fail reason=span_overflow ref={texture_ref} buf={} bpr={bpr} {width}x{height}",
            bt.buffer_ref
        ));
        return Err(ComputeStatus::Unsupported("compute_buftex_span"));
    };
    // A buffer-backed texture is two contract references over one allocation —
    // the type-8 texture the guest binds and the type-1 buffer that owns the
    // storage — and a debt may be armed under either. The draw twin pays for
    // both; so does this.
    crate::runtime::writeback_debt::pay_for_texture(state, host, task_id, texture_ref);
    let raw = read_buffer_window(state, host, task_id, bt.buffer_ref, bt.offset, span)?;

    let tight = tight as usize;
    let bpr = bpr as usize;
    let mut bytes = vec![0u8; tight * height as usize];
    for y in 0..height as usize {
        let src = y * bpr;
        bytes[y * tight..(y + 1) * tight].copy_from_slice(&raw[src..src + tight]);
    }
    crate::observe::off(format!(
        "compute_stage_tex buftex_ok ref={texture_ref} buf={} fmt={format:#x} {width}x{height} off={} bpr={bpr} tight={tight}",
        bt.buffer_ref, bt.offset
    ));
    Ok(StagedTexture {
        // Every staging rail produces bytes; the multisample source is
        // not staged and is set at the classification site instead.
        #[cfg(feature = "backend-vulkan")]
        multisample_target: None,
        binding,
        #[cfg(feature = "backend-vulkan")]
        array_element: 0,
        #[cfg(feature = "backend-vulkan")]
        descriptor_count: 1,
        #[cfg(all(feature = "backend-metal", target_os = "macos"))]
        texture_ref,
        pixel_format: format,
        storage_selector: pixel_format::storage_selector(format),
        // A buffer-backed texture view is one level of one buffer.
        mip_levels: 1,
        width,
        height,
        bytes,
        is_storage,
        #[cfg(feature = "backend-vulkan")]
        residency: None,
        #[cfg(feature = "backend-vulkan")]
        serve: None,
        writeback: TextureWriteback::None,
    })
}

/// Resolve a kernel-declared `texture2d_ms<T, access::read>` binding to the
/// retained multisample target that holds its samples.
///
/// # Why this rail exists beside `stage_texture_raw` rather than inside it
///
/// Every other compute texture binding is *staged*: guest texels are read into
/// a host buffer and uploaded into a pooled transient. A multisample image
/// cannot be filled that way at all —
/// `engine::types::SampledResource::multisampled` states the rule, "such an
/// image can only come from a retained multisample target; linear bytes cannot
/// be uploaded into one with a buffer-to-image copy" — so a staging function
/// asked for one has nothing to do and no way to say so except by refusing.
///
/// That is exactly what the compute rail did: `reflected_compute_texture`
/// classified the shape as `UnstageableShape { axis: "multisampled" }`, beside
/// the 1D, 3D, cube, buffer and arrayed axes, and the dispatch was refused
/// whole. For those five the premise holds — the rail produces a single-layer
/// 2D rectangle and binding it to another declared shape is a descriptor-type
/// mismatch. For this one the premise is about bytes that were never wanted.
///
/// # What it resolves, and why through the same span the render rail uses
///
/// The samples live in the engine resident the render pass wrote, keyed by that
/// target's `TargetIdentity`. It is named through `draw::gva_span_identity`,
/// which is the identity half of the currency test itself, and not rebuilt
/// here: a second derivation of the same registry key is how two rails come to
/// name different residents for one texture.
///
/// What this rail does *not* take is the other half —
/// `draw::gva_resident_if_current`, the currency test the single-sample
/// resident rails share. That test asks whether
/// anything has written the target's guest pages since the Store, making the
/// resident stale against them. A multisample target has no such second copy:
/// no rail of this device writes a multisample target's guest pages, and this
/// device is the only reader of them. With nothing to compare, the witness
/// cannot answer — it reports no observed write rather than a quiet span — and
/// a refusal for want of an answer would cost the guest its dispatch while
/// protecting nothing. The hazard it stands in for on other rails, an absent or
/// unready resident, is carried here by the engine's own `MultisampleSample*`
/// declines at bind time.
///
/// The target's own geometry comes from `draw::color_target_request`, which is
/// the same resolver the render pass resolved its attachment through, so the
/// bind and the render cannot disagree about extent, format, or sample count.
///
/// Every refusal is fail-visible and named for the rung that refused. A guest
/// kernel that reaches here and gets nothing has lost work.
#[cfg(feature = "backend-vulkan")]
#[allow(
    clippy::too_many_arguments,
    reason = "the binding's descriptor identity plus the guest object it names"
)]
fn multisample_sampled_texture<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    pipeline_ref: u32,
    texture_ref: u32,
    binding: u32,
    descriptor: crate::runtime::spirv_bind::ReflectedTextureDescriptor,
) -> Result<StagedTexture, ComputeStatus> {
    let refuse = |reason: &'static str, detail: String, status: ComputeStatus| {
        crate::observe::fail(format!(
            "compute_linux texture_multisample fail reason={reason} pipe={pipeline_ref} \
             ref={texture_ref} bind={binding} {detail}"
        ));
        status
    };
    // The attachment resolver, not a second reading of the descriptor: the
    // render pass that produced these samples resolved its target through this
    // same function, so its geometry, format and sample count are the ones the
    // resident was created from.
    let Some(req) = crate::runtime::draw::color_target_request(
        state,
        host,
        task_id,
        crate::runtime::decode::render::ColorAttachment {
            texture_ref,
            ..Default::default()
        },
        0,
        0,
        1,
        0,
        0,
        0,
    ) else {
        return Err(refuse(
            "target_unresolved",
            String::new(),
            ComputeStatus::MissingTexture("compute_multisample_target_unresolved"),
        ));
    };
    let c0 = req
        .colors
        .first()
        .expect("color_target_request builds exactly one colour");
    // The texture's own declaration, decoded from its descriptor's trailer. A
    // kernel declaring `texture2d_ms` against a texture that declares one
    // sample is a disagreement between two guest statements, and this device
    // must not pick a side by binding either shape.
    if c0.sample_count <= 1 {
        return Err(refuse(
            "texture_is_single_sample",
            format!(
                "samples={} {}x{} gva={:#x}",
                c0.sample_count, c0.width, c0.height, c0.target_gva
            ),
            ComputeStatus::Unsupported("compute_multisample_texture_is_single_sample"),
        ));
    }
    // Asked here rather than left to the request builder below, which would
    // refuse the whole dispatch under a name that says nothing about which
    // texture carried the format.
    if mtl_to_engine_sampled(c0.format).is_none() {
        return Err(refuse(
            "mtl_format_unsupported",
            format!("fmt={:#x}", c0.format),
            ComputeStatus::Unsupported("compute_multisample_format_unsupported"),
        ));
    }
    // The geometry the resident was created from, read once off the resolved
    // attachment and used by the refusals, the identity and the bind alike.
    let (sample_count, width, height, format, target_gva, row_stride) = (
        c0.sample_count,
        c0.width,
        c0.height,
        c0.format,
        c0.target_gva,
        c0.row_stride,
    );
    let span = crate::runtime::draw::GvaSpan {
        texture_ref,
        gva: target_gva,
        row_stride,
        width,
        height,
        format,
    };
    let Some(identity) = crate::runtime::draw::gva_span_identity(state, host, task_id, span) else {
        return Err(refuse(
            "resident_unnamed",
            format!("samples={sample_count} {width}x{height} gva={target_gva:#x}"),
            ComputeStatus::MissingTexture("compute_multisample_resident_unnamed"),
        ));
    };
    crate::runtime::drain::note_store_route("compute_multisample_resident_bind");
    Ok(StagedTexture {
        binding,
        array_element: descriptor.array_element,
        descriptor_count: descriptor.descriptor_count,
        #[cfg(all(feature = "backend-metal", target_os = "macos"))]
        texture_ref,
        pixel_format: format,
        // A multisample image is never a storage image on this rail, so it has
        // no storage selector to carry.
        storage_selector: None,
        width,
        height,
        // A multisample texture has one level by construction.
        mip_levels: 1,
        bytes: Vec::new(),
        is_storage: false,
        residency: None,
        serve: None,
        multisample_target: Some(identity),
        // Read-only: the kernel declares `access::read` or this shape would
        // have been refused as `multisampled_storage` before reaching here.
        writeback: TextureWriteback::None,
    })
}

/// Load tight raw texels for a compute texture binding (type-2/3, type-5→surface, or type-11).
///
/// Type-5 (`RefTextureHandle`) is the live CI wallpaper path (`compute_stage_tex … ot=5`).
/// RE (type-5 wire + `runtime::draw` sample path): surfaceID@0 is a type-4 object id (= mapping
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
    let mut buffer_texture: Option<crate::runtime::decode::resource::BufferTextureDescriptor> =
        None;
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
                return Err(ComputeStatus::MissingTexture(crate::observe::ladder_slug!(
                    "compute_stage_tex_view",
                    desc_read
                )));
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
                        return Err(ComputeStatus::MissingTexture(crate::observe::ladder_slug!(
                            "compute_stage_tex_heap",
                            desc_decode
                        )));
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
                // The same wire form `runtime::draw` decodes and executes, read
                // through the same decoder. This arm used to refuse it.
                let bt =
                    match crate::runtime::decode::resource::decode_buffer_texture_descriptor(&desc)
                    {
                        Ok(bt) => bt,
                        Err(error) => {
                            crate::observe::Emit::decline("compute_stage_tex_buftex", &error)
                                .field("ref", texture_ref)
                                .field("opcode", format!("{opcode:#x}"))
                                .field("len", desc.len())
                                .fail();
                            return Err(ComputeStatus::MissingTexture(
                                "compute_stage_tex_buftex_desc",
                            ));
                        }
                    };
                buffer_texture = Some(bt);
            } else {
                let view = match crate::runtime::draw::resolve_texture_view_reasoned(
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
    if let Some(bt) = buffer_texture {
        return stage_buffer_texture(state, host, task_id, texture_ref, binding, is_storage, &bt);
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
        let storage_selector = pixel_format::storage_selector(format);
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
        #[cfg(feature = "backend-vulkan")]
        let serve = match state.compute_storage_residency.get(&key).copied() {
            None => None,
            Some(generation) => match resident_serve(key, generation, is_storage, format) {
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
            },
        };
        #[cfg(not(feature = "backend-vulkan"))]
        let serve: Option<ResidentServe> = None;
        let seed_generation = serve.and_then(ResidentServe::seed_generation).unwrap_or(0);
        crate::observe::off(format!(
            "compute_stage_tex heap_ok ref={texture_ref} heap={heap_ref} fmt={format:#x} {width}x{height} storage={} seed_gen={seed_generation} resident_sample={} use_offset={} offset={offset:#x}",
            is_storage as u8,
            serve.and_then(ResidentServe::sample_source).is_some() as u8,
            use_offset as u8
        ));
        return Ok(StagedTexture {
            // Every staging rail produces bytes; the multisample source is
            // not staged and is set at the classification site instead.
            #[cfg(feature = "backend-vulkan")]
            multisample_target: None,
            binding,
            #[cfg(feature = "backend-vulkan")]
            array_element: 0,
            #[cfg(feature = "backend-vulkan")]
            descriptor_count: 1,
            #[cfg(all(feature = "backend-metal", target_os = "macos"))]
            texture_ref,
            pixel_format: format,
            storage_selector,
            // The heap arm refuses a descriptor declaring more than one level
            // above, so a heap texture reaching here is single-level.
            mip_levels: 1,
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
            serve,
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
                if let Ok(t5) = reims_vgpu_wire::device_desc::type5_header(&desc) {
                    let sid = t5.surface_id.get();
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
                        crate::observe::when_verbose(|| {
                            // The owner task the view names. `note_type5_owner_task`
                            // is the always-on check on its value; this echo carries
                            // it beside the descriptor it came out of.
                            let owner_task = t5.owner_task.get();
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
                        });
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
                    "compute_stage_tex type11_fail reason=fmt_bytes mapping={mapping_id} {width}x{height} fmt={format:#x}"
                ));
                return Err(ComputeStatus::Unsupported("stage_tex_fmt_bytes"));
            }
        };
        let storage_selector = pixel_format::storage_selector(stage_fmt);
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
        // declines rather than picking one. The draw path already binds type-5
        // views by index; this is the same resolution on the staging path.
        let window = match type5_record {
            Some(rec) => {
                mapping_write::type5_sample_window(m, rec.plane_index, width, height, stage_fmt)
            }
            None => mapping_write::type11_sample_window(m, width, height, stage_fmt),
        };
        let (surface_offset, surface_bpr, span_end) = match window {
            Some(w) => w,
            None => {
                // What the descriptor said, so a refusal names which of its
                // fields the texture could not be placed against. `reach` is the
                // byte count this geometry needs; a descriptor whose alloc is
                // smaller is a different failure from one whose plane records
                // matched nothing.
                let ds = crate::contract::iosurface_pages::decode_device_surface(&m.device_desc);
                let (dw, dh, dbpr, dalloc) = ds
                    .as_ref()
                    .map(|s| (s.width, s.height, s.bytes_per_row, s.alloc_size))
                    .unwrap_or((0, 0, 0, 0));
                let reach = crate::contract::iosurface_pages::packed_span_estimate(
                    stage_fmt, width, height,
                )
                .unwrap_or(0);
                crate::observe::fail(format!(
                    "compute_stage_tex type11_fail reason=window mapping={mapping_id} {width}x{height} fmt={stage_fmt:#x} pages={pages_n} wire_len={wire_len} desc={dw}x{dh} bpr={dbpr} alloc={dalloc} reach={reach}"
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
        let serve = state
            .compute_storage_residency
            .get(&residency_key)
            .copied()
            .and_then(|mirror_generation| {
                resident_serve(residency_key, mirror_generation, is_storage, stage_fmt)
            });
        #[cfg(not(feature = "backend-vulkan"))]
        let serve: Option<ResidentServe> = None;
        // Unlike the heap and linear rails, this one's fallback generation is
        // the mapping's own content generation rather than zero, so a seed
        // overwrites it and anything else leaves it alone. Gated with the
        // generation it writes: `serve` is unconditionally `None` without the
        // Vulkan backend, so this is a no-op there rather than a second policy.
        #[cfg(feature = "backend-vulkan")]
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
        if serve.is_none()
            && !mapping_write::read_rect_raw_at(
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
                format: stage_fmt,
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
            // Every staging rail produces bytes; the multisample source is
            // not staged and is set at the classification site instead.
            #[cfg(feature = "backend-vulkan")]
            multisample_target: None,
            binding,
            #[cfg(feature = "backend-vulkan")]
            array_element: 0,
            #[cfg(feature = "backend-vulkan")]
            descriptor_count: 1,
            #[cfg(all(feature = "backend-metal", target_os = "macos"))]
            texture_ref,
            pixel_format: stage_fmt,
            storage_selector,
            // Metal forbids a mipmapped IOSurface texture.
            mip_levels: 1,
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
            serve,
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
    let (_entry, desc_bytes) = match objects::resolve_descriptor(
        state,
        host,
        task_id,
        stage_ref,
        &[OBJECT_TYPE_TEXTURE, OBJECT_TYPE_TEXTURE_VARIANT],
    ) {
        Ok(found) => found,
        Err(rung) => {
            return linear_fail(
                ComputeStatus::MissingTexture(crate::observe::ladder_slugs!("compute_linear_tex")(
                    rung,
                )),
                match rung {
                    objects::LadderRung::WrongType { got } => format!("ot={got}"),
                    objects::LadderRung::NoListEntry | objects::LadderRung::DescRead { .. } => {
                        String::new()
                    }
                },
            );
        }
    };
    let Ok(tex) = decode_texture_descriptor(&desc_bytes) else {
        return linear_fail(
            ComputeStatus::MissingTexture(crate::observe::ladder_slug!(
                "compute_linear_tex",
                desc_decode
            )),
            format!("len={}", desc_bytes.len()),
        );
    };
    if tex.declared_pixel_format().is_none() {
        return linear_fail(
            ComputeStatus::Unsupported("linear_tex_no_fmt"),
            String::new(),
        );
    }
    let Some(stage_format) =
        crate::runtime::draw::effective_view_sample_format(tex.pixel_format, view_pixel_format)
    else {
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
    let storage_selector = pixel_format::storage_selector(stage_format);
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
    // Which levels of the guest's declared chain this binding serves.
    //
    // A storage write names one level and a levelled view already exposes one,
    // so both stay at the base. Everything else stages the declared pyramid,
    // because `read(coord, lod)` and `sample(_, _, level(lod))` name a level of
    // it and an image built with only the base answers the first with nothing
    // and the second with level 0.
    let mut level_sources = vec![LinearLevelSource {
        gva,
        row_stride: layout.row_stride,
    }];
    if !is_storage && view_level == 0 {
        level_sources.extend(linear_extra_levels(
            &tex,
            state.page_shift,
            w,
            h,
            bpp,
            texture_ref,
        ));
    }
    let Some(pyramid) = crate::contract::extent::tight_pyramid_spans(
        w,
        h,
        level_sources.len() as u32,
        bpp as usize,
    ) else {
        return linear_fail(
            ComputeStatus::Unsupported("linear_tex_pyramid_layout"),
            format!("{w}x{h} bpp={bpp} levels={}", level_sources.len()),
        );
    };
    let Some(pyramid_need) = pyramid
        .last()
        .and_then(|last| last.offset.checked_add(last.len))
    else {
        return linear_fail(
            ComputeStatus::Unsupported("linear_tex_pyramid_span"),
            format!("{w}x{h} bpp={bpp} levels={}", level_sources.len()),
        );
    };
    // Level 0 of the packed pyramid and the single image this window already
    // sized are two independent derivations of one length — `tight * h` here,
    // `mip_extent(w, 0) * mip_extent(h, 0) * bpp` there. If they ever
    // disagreed, the upload would be apportioned to levels by a layout the
    // reader below does not share, and level 1 would hold level 0's tail.
    if pyramid.first().map(|base| base.len) != Some(need) {
        return linear_fail(
            ComputeStatus::Unsupported("linear_tex_pyramid_base"),
            format!(
                "base={:?} need={need} {w}x{h} bpp={bpp}",
                pyramid.first().map(|base| base.len)
            ),
        );
    }
    // The identity every cache question below asks about. One derivation so
    // the resident probe, the flush-and-serve, the plain serve and the
    // per-level read cannot drift from each other — and so a level's cache key
    // is that level's own rows and extent rather than the base's.
    let level_window = |source: &LinearLevelSource,
                        span: &crate::contract::extent::MipLevelSpan| {
        crate::runtime::surface_cache::LinearWindow {
            task_id,
            texture_ref: stage_ref,
            gva: source.gva,
            pixel_format: stage_format,
            width: span.width,
            height: span.height,
            row_stride: source.row_stride,
        }
    };
    // Only the base can be resident: a resident is one window at one level.
    #[cfg(feature = "backend-vulkan")]
    let window = level_window(&level_sources[0], &pyramid[0]);
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
    let mut bytes = vec![0u8; pyramid_need];
    #[cfg_attr(
        not(feature = "backend-vulkan"),
        allow(
            unused_mut,
            reason = "the Vulkan resident-window block below assigns it"
        )
    )]
    let have_bytes = false;
    // Resident-authoritative window (deferred linear writeback): consume the
    // engine resident without bytes when possible; otherwise flush it into the
    // entry first — falling through to the raw guest read would silently serve
    // the pre-chain seed pages.
    #[cfg(feature = "backend-vulkan")]
    let resident = match (
        linear_key,
        crate::runtime::surface_cache::linear_texture_resident_gen(state, &window),
    ) {
        (Some(key), Some(resident_gen)) => Some((
            key,
            resident_gen,
            resident_serve(key, resident_gen, is_storage, stage_format),
        )),
        _ => None,
    };
    #[cfg(feature = "backend-vulkan")]
    // A resident is one window at one level, so it can only answer for the
    // base. Serving a pyramid from it would leave every level above the base
    // unwritten — which is exactly the defect the pyramid repairs — so a
    // multi-level binding reads its own bytes and the engine refuses the pair
    // outright as `vk_compute_exec_resident_sample_is_not_a_pyramid`.
    let serve = if level_sources.len() > 1 {
        None
    } else {
        resident.and_then(|(_, _, serve)| serve)
    };
    #[cfg(not(feature = "backend-vulkan"))]
    let serve: Option<ResidentServe> = None;
    if let Some(generation) = serve.and_then(ResidentServe::seed_generation) {
        crate::observe::off(format!(
            "compute_stage_linear_resident_seed task={task_id} ref={texture_ref} gva={gva:#x} fmt={:#x} dims={w}x{h} gen={generation}",
            tex.pixel_format
        ));
    } else if let Some((_, generation)) = serve.and_then(ResidentServe::sample_source) {
        crate::observe::off(format!(
            "compute_stage_linear_resident_sample task={task_id} ref={texture_ref} gva={gva:#x} fmt={:#x} dims={w}x{h} gen={generation}",
            stage_format
        ));
    }
    if serve.is_some() || have_bytes {
        // Engine resident serves this window; no cache/guest read.
    } else {
        // Level 0 is the window built above; every level after it is the same
        // read against that level's own rows, so the cache is consulted per
        // level and one level's bytes can never answer for another's.
        for (span, source) in pyramid.iter().zip(level_sources.iter()) {
            let level_tight = span.len / span.height.max(1) as usize;
            read_linear_level(
                state,
                host,
                task_id,
                texture_ref,
                &level_window(source, span),
                source.gva,
                source.row_stride,
                level_tight,
                span.height,
                &mut bytes[span.offset..span.offset + span.len],
            )?;
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
                let seed = serve
                    .and_then(ResidentServe::seed_generation)
                    .unwrap_or_else(|| {
                        state
                            .host_linear_textures
                            .get(&(task_id, stage_ref))
                            .map(|e| e.host_gen)
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
        // Every staging rail produces bytes; the multisample source is
        // not staged and is set at the classification site instead.
        #[cfg(feature = "backend-vulkan")]
        multisample_target: None,
        binding,
        #[cfg(feature = "backend-vulkan")]
        array_element: 0,
        #[cfg(feature = "backend-vulkan")]
        descriptor_count: 1,
        #[cfg(all(feature = "backend-metal", target_os = "macos"))]
        texture_ref,
        pixel_format: stage_format,
        storage_selector,
        // The levels actually placed, which is the declared count when the
        // descriptor places all of them and a reported-short prefix when it
        // does not.
        mip_levels: level_sources.len() as u32,
        width: w,
        height: h,
        bytes,
        is_storage,
        #[cfg(feature = "backend-vulkan")]
        residency,
        #[cfg(feature = "backend-vulkan")]
        serve,
        writeback,
    })
}

/// Fill `dst` with one linear texture level's tight rows, from the surface
/// cache when it still holds this exact window and from guest pages otherwise.
///
/// Its own function because a mip chain reads the same thing once per level
/// against a different `gva`, `row_stride` and extent, and a loop that inlined
/// this would have been the place where level `n` was read with level 0's
/// stride.
#[allow(
    clippy::too_many_arguments,
    reason = "a level is its own window, gva, stride, extent and destination, and \
              collapsing them into a struct here would hide which of them a caller varies"
)]
fn read_linear_level<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    window: &crate::runtime::surface_cache::LinearWindow,
    gva: u64,
    row_stride: u64,
    tight: usize,
    height: u32,
    dst: &mut [u8],
) -> Result<(), ComputeStatus> {
    if let Some(cached) = crate::runtime::surface_cache::get_linear_texture(state, window) {
        if cached.len() == dst.len() {
            dst.copy_from_slice(cached);
            crate::observe::off(format!(
                "compute_stage_tex linear_cache task={task_id} ref={texture_ref} gva={gva:#x} fmt={:#x} dims={}x{height} row_stride={row_stride}",
                window.pixel_format, window.width
            ));
            return Ok(());
        }
        // A cache entry keyed to this window whose length is not this window's
        // is a key that stopped identifying its contents. Read the guest pages
        // rather than serve it, and say so — silently trusting it is how one
        // level's texels would reach another's.
        crate::observe::fail(format!(
            "compute_stage_tex linear_cache_len task={task_id} ref={texture_ref} gva={gva:#x} cached={} want={}",
            cached.len(),
            dst.len()
        ));
    }
    // The bulk/row reads below walk raw task GVAs; a Store's
    // guest-page write is submitted and not waited on.
    crate::runtime::writeback_debt::pay_for_texture(state, host, task_id, texture_ref);
    crate::runtime::render_writeback::settle_guest_writes(
        crate::runtime::render_writeback::SettleSite::ComputeStageTexture,
    );
    if read_linear_texture_bulk(state, host, task_id, gva, row_stride, tight, height, dst) {
        // One cached-view walk for the whole span (render-path bulk analog).
        return Ok(());
    }
    let mut row = vec![0u8; tight];
    for y in 0..height {
        let row_gva = gva
            .checked_add(
                (y as u64)
                    .checked_mul(row_stride)
                    .ok_or(ComputeStatus::GuestIo(
                        "compute_stage_tex_linear_row_offset",
                    ))?,
            )
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
        dst[off..off + tight].copy_from_slice(&row);
    }
    Ok(())
}

/// One level of a linear texture as this rail stages it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LinearLevelSource {
    gva: u64,
    row_stride: u64,
}

/// Levels 1.. of a type-2/3 texture's declared mip chain, as far as the
/// descriptor actually places them.
///
/// The prefix, not the set: a level that will not resolve makes every level
/// above it unreachable too, because the packed pyramid the host image is built
/// from has no way to express a hole. Truncation is dropped guest work, so it is
/// reported by name rather than left to read as a texture that simply has fewer
/// levels.
///
/// Extents are checked against [`crate::contract::extent::mip_extent`] because
/// the packed layout is derived from the base geometry alone; a level whose
/// declared extent disagrees would be read at one size and copied at another.
fn linear_extra_levels(
    tex: &crate::runtime::decode::resource::TextureDescriptor,
    page_shift: u32,
    base_width: u32,
    base_height: u32,
    bpp: u32,
    texture_ref: u32,
) -> Vec<LinearLevelSource> {
    let declared = tex.mipmap_level_count.max(1);
    let mut out = Vec::new();
    for level in 1..declared {
        let want_w = crate::contract::extent::mip_extent(base_width, level);
        let want_h = crate::contract::extent::mip_extent(base_height, level);
        let refuse = |reason: &str, detail: String| {
            crate::observe::fail(format!(
                "compute_stage_tex mip_truncated reason={reason} ref={texture_ref} level={level}                  staged={} declared={declared} want={want_w}x{want_h} {detail}",
                level
            ));
        };
        let Some((level_gva, layout)) = tex.level_gva(level, page_shift) else {
            refuse("no_level", String::new());
            break;
        };
        if layout.width != want_w || layout.height != want_h {
            refuse("extent", format!("got={}x{}", layout.width, layout.height));
            break;
        }
        if layout.row_stride < u64::from(want_w).saturating_mul(u64::from(bpp)) {
            refuse("stride_lt_tight", format!("stride={}", layout.row_stride));
            break;
        }
        out.push(LinearLevelSource {
            gva: level_gva,
            row_stride: layout.row_stride,
        });
    }
    out
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
        TextureWriteback::Type11 {
            width,
            format,
            surface_bpr,
            ..
        } => {
            crate::runtime::drain::note_store_route("compute_wb_type11");
            let tight = pixel_format::bytes_per_pixel(*format).map(|bpp| width.saturating_mul(bpp));
            crate::runtime::drain::note_store_route(if tight == Some(*surface_bpr) {
                "compute_wb_type11_dense"
            } else {
                "compute_wb_type11_padded"
            });
        }
    }

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
                    Ok(())
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
                    "compute_writeback_tex fail reason=type11_format_unsized task={task_id} bind={} mid={mapping_id} fmt={format:#x}",
                    tex.binding
                ));
                return Err(ComputeStatus::GuestIo("compute_wb_tex_type11_format"));
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

/// The dispatch extents, narrowed from the wire's `u64` by [`u32_dim`].
///
/// The type is [`crate::contract::extent::Extent3`], which both this decoder
/// and the Metal backend it dispatches through now name. It used to be private
/// here, which protected construction and stopped at the backend call — see its
/// doc for why that was the wrong half of the journey to protect.
use crate::contract::extent::Extent3;

// The two constructors are free functions here rather than an inherent `impl`
// on `Extent3`, because both refuse with `ComputeStatus` and one reads a decoded
// `Size3` — device vocabulary the contract crate cannot name, and Rust's orphan
// rule says so. The extent type stays shared; only the narrowing that produces
// it from *this* device's wire belongs to this decoder.

/// An [`Extent3`] from a decoded wire `Size3`, refusing each component out of
/// range.
fn extent_from_wire(s: crate::runtime::decode::compute::Size3) -> Result<Extent3, ComputeStatus> {
    Ok(Extent3 {
        x: u32_dim(s.x)?,
        y: u32_dim(s.y)?,
        z: u32_dim(s.z)?,
    })
}

/// An [`Extent3`] from three consecutive LE `u32`s of an indirect-arguments
/// buffer at `at`. One stride expression rather than six offset literals: the
/// literals were `0, 4, 8` and `12, 16, 20` written out, where a transposition
/// is invisible.
fn extent_from_indirect(raw: &[u8], at: usize) -> Result<Extent3, ComputeStatus> {
    Ok(Extent3 {
        x: u32_dim(u64::from(ld32(&raw[at..])))?,
        y: u32_dim(u64::from(ld32(&raw[at + 4..])))?,
        z: u32_dim(u64::from(ld32(&raw[at + 8..])))?,
    })
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
    state: &mut DeviceState,
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
    state: &mut DeviceState,
    host: &M,
    task_id: u32,
    cmd: &ComputeCommand,
) -> Result<DispatchDims, ComputeStatus> {
    match cmd.kind {
        // Every dimension comes from the wire. `u32_dim` refuses `0` and
        // anything past `u32::MAX` with `BadGrid("compute_grid_dim_range")`, so
        // a malformed grid is a named refusal rather than a substitution.
        Kind::DispatchThreadgroups => Ok(DispatchDims {
            grid: extent_from_wire(cmd.grid)?,
            threadgroup: extent_from_wire(cmd.threads_per_threadgroup)?,
            dispatch_threads: false,
        }),
        Kind::DispatchThreads => Ok(DispatchDims {
            grid: extent_from_wire(cmd.grid)?,
            threadgroup: extent_from_wire(cmd.threads_per_threadgroup)?,
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
                grid: extent_from_indirect(&raw, 0)?,
                threadgroup: extent_from_wire(cmd.threads_per_threadgroup)?,
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
                grid: extent_from_indirect(&raw, 0)?,
                threadgroup: extent_from_indirect(&raw, 12)?,
                dispatch_threads: true,
            })
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
        self as vk_engine, ComputeBufferResource, ComputeImageResult, ComputeRequest,
        ComputeSampledImageResource, ComputeSampledSource, ComputeStorageImageResource, DrawError,
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
                "compute_stage_input_contract pipe={} attrs={:?} layouts={:?} index_type={} \
                 index_buffer={}",
                acc.pipeline_ref,
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
    // are one rule. See [`crate::contract::dispatch::workgroup_counts`] for why
    // splitting them put an unreachable `.max(1)` on the quotients.
    let Some([wg_x, wg_y, wg_z]) = crate::contract::dispatch::workgroup_counts(
        [grid_x, grid_y, grid_z],
        [tg_x, tg_y, tg_z],
        dispatch_threads,
    ) else {
        return ComputeStatus::BadGrid("compute_vk_zero_dims");
    };

    // Translate before staging buffers. The final adopted SPIR-V carries the
    // conservative byte footprint that decides how much of each allocation the
    // dispatch can touch; staging first discarded that answer and copied every
    // bind through the end of its allocation.
    //
    // MTLB → AIR → SPIR-V (LocalSize = threadgroup dims).
    let Some(mtlb) = load_mtlb(
        state,
        host,
        task_id,
        pipeline.kernel_func_ref,
        AirLoadRail::Compute,
    ) else {
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
    if let Some(unsupported) = crate::runtime::spirv_bind::first_unsupported_vulkan_interface(
        &kernel_shader.reflection,
        metal2vulkan::reflect::ShaderStage::Kernel,
    ) {
        let reason = ComputeReflectionDecline::ReflectedInterfaceUnsupported {
            pipeline_ref: acc.pipeline_ref,
            feature: unsupported.feature,
            count: unsupported.count,
        };
        crate::observe::Emit::decline("compute_linux_reflection", &reason)
            .fail_once(u64::from(acc.pipeline_ref));
        return ComputeStatus::Unsupported(crate::observe::Decline::slug(&reason));
    }
    if let Some(resource) =
        crate::runtime::spirv_bind::first_unsupported_vulkan_resource(&kernel_shader.reflection)
    {
        let kind = crate::runtime::spirv_bind::unsupported_vulkan_resource_kind_name(resource.kind)
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
        .reflection
        .local_size
        .expect("kernel cache admits only the requested reflected local size");
    let Some(kernel_dispatch) = kernel_shader.reflection.kernel_dispatch else {
        return ComputeStatus::Unsupported("compute_kernel_dispatch_missing");
    };
    let dispatch = match kernel_dispatch_launch(
        kernel_dispatch,
        reflected_local_size,
        [wg_x, wg_y, wg_z],
        [tg_x, tg_y, tg_z],
        dispatch_threads.then_some([grid_x, grid_y, grid_z]),
    ) {
        Ok(dispatch) => dispatch,
        Err(decline) => {
            let (status, detail) = match &decline {
                KernelDispatchDecline::GridOverflow => {
                    (ComputeStatus::BadGrid("compute_vk_grid_overflow"), None)
                }
                KernelDispatchDecline::PushRangeUnavailable => (
                    ComputeStatus::Unsupported("compute_kernel_dispatch_push_range"),
                    None,
                ),
                KernelDispatchDecline::PlanRefused(detail) => (
                    ComputeStatus::BadGrid("compute_vk_dispatch_plan"),
                    Some(detail.replace(char::is_whitespace, "_")),
                ),
            };
            let reason = decline.reason(acc.pipeline_ref);
            let mut emit = crate::observe::Emit::decline("compute_linux_kernel_dispatch", &reason);
            if let Some(detail) = detail {
                emit = emit.field("detail", detail);
            }
            emit.fail_once(u64::from(acc.pipeline_ref));
            return status;
        }
    };
    let mut spirv = match spirv_words_le(&kernel_shader.spirv) {
        Ok(w) => w,
        Err(e) => {
            crate::observe::Emit::decline("compute_linux_spirv_parse", &e)
                .field("pipe", acc.pipeline_ref)
                .fail_once(acc.pipeline_ref as u64);
            return ComputeStatus::MetalFailed("compute_vk_spirv_parse");
        }
    };

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
        use crate::runtime::spirv_bind::ReflectedBufferAccess;
        let access =
            crate::runtime::spirv_bind::reflected_buffer_access(&kernel_shader.reflection, b.index);
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
        let extent = crate::runtime::spirv_bind::reflected_compute_buffer_extent(
            &kernel_shader.reflection,
            b.index,
            [wg_x, wg_y, wg_z],
            reflected_local_size,
        );
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
        use crate::runtime::spirv_bind::{
            ImageAccess, ReflectedComputeTexture, StorageImageAccess,
        };
        let Some(descriptor) = crate::runtime::spirv_bind::reflected_texture_descriptor(
            &kernel_shader.reflection,
            t.index,
        ) else {
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
        let is_storage = match crate::runtime::spirv_bind::reflected_compute_texture(
            &kernel_shader.reflection,
            binding,
        ) {
            ReflectedComputeTexture::Plain2d(ImageAccess::Sampled) => false,
            ReflectedComputeTexture::Plain2d(ImageAccess::Storage) => true,
            ReflectedComputeTexture::Multisample2d => {
                // Not staged, because there is nothing to stage from and
                // nothing to stage into: a multisample image is filled by
                // rendering and by nothing else. The retained target that
                // rendered those samples is the whole source, so this arm
                // resolves it and skips `stage_texture_raw` entirely.
                match multisample_sampled_texture(
                    state,
                    host,
                    task_id,
                    acc.pipeline_ref,
                    t.texture_ref,
                    binding,
                    descriptor,
                ) {
                    Ok(staged) => {
                        staged_tex.push(staged);
                        continue;
                    }
                    // Already fail-visible, by the name of the rung that
                    // refused; the caller must not print a second line for one
                    // refusal.
                    Err(status) => return status,
                }
            }
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
            Ok(mut s) => {
                s.array_element = descriptor.array_element;
                s.descriptor_count = descriptor.descriptor_count;
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
        let guest_fmt = selector_to_engine_storage(selector);
        let Some(shader_decl) = crate::runtime::spirv_bind::reflected_storage_image_format(
            &kernel_shader.reflection,
            t.binding,
        ) else {
            crate::observe::fail(format!(
                "compute_linux storage_format fail reason=reflection_format_missing pipe={} bind={} guest={guest_fmt:?} simg={}",
                acc.pipeline_ref, t.binding, selector as u32
            ));
            return ComputeStatus::Unsupported("storage_reflection_format_missing");
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
                        selector as u32,
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
    for t in &mut staged_tex {
        if t.is_storage {
            let Some(selector) = t.storage_selector else {
                crate::observe::fail(format!(
                    "compute_linux unsupported storage_format reason=no_storage_selector pipe={} bind={} fmt={:#x}",
                    acc.pipeline_ref, t.binding, t.pixel_format
                ));
                return ComputeStatus::Unsupported("storage_no_selector_writeback");
            };
            let guest_fmt = selector_to_engine_storage(selector);
            let Some((_, _, shader_decl, specialized)) = storage_formats
                .iter()
                .find(|(binding, _, _, _)| *binding == t.binding)
            else {
                crate::observe::fail(format!(
                    "compute_linux storage_format fail reason=spirv_format_specialize_internal pipe={} bind={} simg={}",
                    acc.pipeline_ref, t.binding, selector as u32
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
                        acc.pipeline_ref, t.binding, selector as u32
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
                    selector as u32,
                    guest_fmt.bytes_per_texel(),
                    spirv_image_format_to_engine_storage(*shader_decl)
                        .map(|format| format.bytes_per_texel())
                        .unwrap_or(0)
                ));
            }
            storage_images.push(ComputeStorageImageResource {
                binding: t.binding,
                array_element: t.array_element,
                descriptor_count: t.descriptor_count,
                format: shader_fmt,
                width: t.width,
                height: t.height,
                bytes: std::mem::take(&mut t.bytes),
                // The guest window this output belongs to is on `t.writeback`,
                // so the destination is decided from the window rather than
                // from anything about this dispatch. `Host` needs no host
                // capability; the direct arm needs the guest-RAM import, and
                // where that is absent the licence declines by name and this
                // reads back exactly as it always did.
                destination: direct_destination(state, host, t, shader_fmt.vk_format()),
                residency: t.residency.map(|candidate| {
                    crate::backend::vulkan::engine::ComputeStorageResidency {
                        identity: candidate.key,
                        seed_generation: candidate.seed_generation,
                        output_generation: next_mapping_content_generation(
                            candidate.seed_generation,
                        ),
                    }
                }),
                seed_skipped: t.serve.and_then(ResidentServe::seed_generation).is_some(),
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
                array_element: t.array_element,
                descriptor_count: t.descriptor_count,
                format: sampled_fmt,
                width: t.width,
                height: t.height,
                mip_levels: t.mip_levels,
                // Asked in the order the sources exclude each other, not as a
                // pair: the producer that sets `multisample_target` is the one
                // rail that stages nothing, and it leaves `serve` and `bytes`
                // empty because there is nothing for either to hold.
                source: match t.multisample_target.take() {
                    Some(identity) => ComputeSampledSource::MultisampleTarget(identity),
                    None => match t.serve.and_then(ResidentServe::sample_source) {
                        Some((identity, generation)) => ComputeSampledSource::ResidentCopy(
                            crate::backend::vulkan::engine::ComputeResidentSampleBind {
                                identity,
                                generation,
                            },
                        ),
                        None => ComputeSampledSource::Bytes(std::mem::take(&mut t.bytes)),
                    },
                },
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
    for binding in neutral_sampled_image_bindings(&spirv, &bound) {
        crate::observe::Emit::decline(
            "compute_linux_sampled",
            &NeutralSampledImage {
                binding,
                width: NEUTRAL_SAMPLED_IMAGE_EXTENT,
                height: NEUTRAL_SAMPLED_IMAGE_EXTENT,
            },
        )
        .field("pipe", acc.pipeline_ref)
        .fail_once((u64::from(acc.pipeline_ref) << 32) | u64::from(binding));
        sampled_images.push(ComputeSampledImageResource {
            binding,
            array_element: 0,
            descriptor_count: 1,
            format: crate::backend::vulkan::engine::StorageImageFormat::Rgba8Unorm,
            width: NEUTRAL_SAMPLED_IMAGE_EXTENT,
            height: NEUTRAL_SAMPLED_IMAGE_EXTENT,
            // A stand-in for a binding the guest left empty is one level.
            mip_levels: 1,
            source: ComputeSampledSource::Bytes(pixel_format::solid_rgba8(
                NEUTRAL_SAMPLED_IMAGE_EXTENT,
                NEUTRAL_SAMPLED_IMAGE_EXTENT,
                &[0.0; 4],
            )),
        });
    }

    // Reflection is the sampler interface emitted alongside this exact module.
    // Derive it once per dispatch instead of walking every SPIR-V instruction
    // once to filter guest samplers and again to provision defaults.
    let reflected_samplers = kernel_shader.variant(false, false).samplers.clone();
    let mut samplers = Vec::new();
    for s in &acc.samplers {
        let binding = crate::runtime::spirv_bind::SAMPLER_BINDING_BASE + s.index;
        if reflected_samplers
            .binary_search_by_key(&binding, |sampler| sampler.binding)
            .is_err()
        {
            continue;
        }
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
                samplers.push(
                    crate::backend::vulkan::engine::SamplerResource::normalized_default(
                        reflected.binding,
                    ),
                );
            }
        }
    }

    let req = ComputeRequest {
        spirv,
        entry: "main".into(),
        dispatch,
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
    if out.buffers.len() != buffer_writable_count || out.images.len() != storage_count {
        crate::observe::fail(format!(
            "compute_linux readback count mismatch pipe={} buf={}/{} img={}/{}",
            acc.pipeline_ref,
            out.buffers.len(),
            buffer_writable_count,
            out.images.len(),
            storage_count
        ));
        return ComputeStatus::MetalFailed("compute_vk_readback_count");
    }
    let vk_engine::ComputeOutput {
        buffers: output_buffers,
        images: output_images,
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
    for (t, result) in staged_tex
        .iter_mut()
        .filter(|texture| texture.is_storage)
        .zip(output_images)
    {
        match result {
            ComputeImageResult::Bytes(bytes) => {
                t.bytes = bytes;
                if let Err(e) = writeback_texture(state, host, task_id, t) {
                    return e;
                }
            }
            // The engine copied straight into the guest's pages, so there is no
            // writeback to do and no bytes to do it from.
            ComputeImageResult::Landed { bytes } => {
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
                    } => crate::runtime::surface_cache::forget_gva_copies(
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
                    // and they are the same offsets: `licence_type11_surface`
                    // resolves the window through `type11_sample_window`, which
                    // is where these came from when the texture was staged.
                    TextureWriteback::Type11 {
                        mapping_id,
                        surface_offset,
                        span_end,
                        ..
                    } => crate::runtime::mapping_write::note_type11_landed(
                        state,
                        *mapping_id,
                        *surface_offset,
                        *span_end,
                    ),
                    TextureWriteback::None => {}
                }
            }
        }
        // The output is in the guest's pages now, so the engine's image has
        // stopped being the only copy and the reclaim paths may take it. The
        // deferred branch above reaches the same edge through its own flush;
        // without this one a synchronously-written resident stayed flagged
        // unreproducible forever and no reclaim could ever touch it.
        if let Some(candidate) = t.residency {
            crate::backend::vulkan::engine::note_resident_storage_copied_out(&candidate.key);
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
#[cfg(feature = "backend-vulkan")]
pub(crate) fn linux_stage_input_or_imageblock_unsupported(
    pipeline_stage_input: bool,
    acc: &ComputeAccum,
) -> bool {
    pipeline_stage_input || acc.imageblock.is_some()
}

/// Why a reflected kernel dispatch contract could not become device work.
#[cfg(feature = "backend-vulkan")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum KernelDispatchDecline {
    /// A `dispatchThreadgroups` record whose workgroup count times its
    /// threadgroup size does not fit the logical thread grid's `u32`.
    GridOverflow,
    /// The reflected exact-thread payload has no representable byte range.
    PushRangeUnavailable,
    /// The translator refused to plan this launch's regions.
    PlanRefused(String),
}

#[cfg(feature = "backend-vulkan")]
impl KernelDispatchDecline {
    fn reason(&self, pipeline_ref: u32) -> ComputeReflectionDecline {
        match self {
            Self::GridOverflow | Self::PlanRefused(_) => {
                ComputeReflectionDecline::DispatchPlanRefused { pipeline_ref }
            }
            Self::PushRangeUnavailable => {
                ComputeReflectionDecline::DispatchPushRangeUnavailable { pipeline_ref }
            }
        }
    }
}

/// Turn one translated kernel's reflected dispatch contract into the device
/// work this launch performs.
///
/// The contract, not the record, decides the shape. A module translated for
/// whole workgroups baked its local size and can only be dispatched as one
/// rounded grid. A module translated for exact threads left its local size
/// specializable, and the translator decomposes the logical thread grid into
/// the interior plus each axis's boundary slab — at most eight regions, each
/// its own dispatch at its own workgroup size. Issuing such a module as a
/// single rounded dispatch would run invocations past the guest's grid; issuing
/// only some of its regions would drop guest threads. Both are why this returns
/// the whole plan rather than a grid and a correction.
///
/// `dispatch_threads_grid` is the exact thread count of a Metal
/// `dispatchThreads` record, and `None` is a `dispatchThreadgroups` record —
/// whose `workgroups * threadgroup` threads decompose to exactly one region at
/// the nominal local size, so one cached translation serves both Metal forms.
#[cfg(feature = "backend-vulkan")]
pub(crate) fn kernel_dispatch_launch(
    kernel_dispatch: metal2vulkan::reflect::KernelDispatch,
    nominal_local_size: [u32; 3],
    workgroups: [u32; 3],
    threadgroup: [u32; 3],
    dispatch_threads_grid: Option<[u32; 3]>,
) -> Result<crate::backend::vulkan::engine::ComputeDispatch, KernelDispatchDecline> {
    use crate::backend::vulkan::engine as vk_engine;
    use metal2vulkan::reflect::KernelDispatch;

    if matches!(kernel_dispatch, KernelDispatch::Workgroups) {
        return Ok(vk_engine::ComputeDispatch::Workgroups(workgroups));
    }
    let threads_per_grid = match dispatch_threads_grid {
        Some(grid) => grid,
        None => {
            let mut threads = [0u32; 3];
            for dimension in 0..3 {
                threads[dimension] = workgroups[dimension]
                    .checked_mul(threadgroup[dimension])
                    .ok_or(KernelDispatchDecline::GridOverflow)?;
            }
            threads
        }
    };
    // The reflected range, not a constructed one: `ThreadsFixed` puts its
    // payload at the translator's default offset while `ThreadsDynamic` names
    // its own, and an offset whose range would not fit is refused rather than
    // truncated — a short range is a shader reading bytes no one wrote.
    let range = kernel_dispatch
        .push_constant_range()
        .ok_or(KernelDispatchDecline::PushRangeUnavailable)?;
    let plan = kernel_dispatch
        .plan(nominal_local_size, Some(threads_per_grid))
        .map_err(KernelDispatchDecline::PlanRefused)?;
    Ok(vk_engine::ComputeDispatch::Regions {
        push_offset: range.offset,
        threadgroups_per_grid: plan.threadgroups_per_grid,
        regions: plan
            .regions
            .iter()
            .map(|region| vk_engine::ComputeDispatchRegion {
                local_size: region.local_size,
                group_count: region.group_count,
                push_constants: plan.push_constants(*region),
            })
            .collect(),
    })
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
    let grid = req.dispatch.threadgroups_per_grid();
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
/// The engine's storage format for a contract selector.
///
/// Total, because the translate layer's map is. It used to take the selector's
/// `u32` ordinal and hand back an `Option`, and both of its call sites carried a
/// `reason=selector_unknown` refusal for the `None` — a decline that could only
/// have fired if two enums in this crate had drifted, which is not a thing the
/// guest can cause and not a thing a run-time check should be watching for.
/// Those two refusals are gone with the `Option`.
fn selector_to_engine_storage(
    selector: pixel_format::StorageImageSelector,
) -> crate::backend::vulkan::engine::StorageImageFormat {
    crate::backend::vulkan::translate::pixel::storage_image_from_selector(selector)
}

#[cfg(feature = "backend-vulkan")]
fn mtl_to_engine_sampled(
    format: u16,
) -> Option<crate::backend::vulkan::engine::StorageImageFormat> {
    // The *sampled* admission, not the storage one. Asking `storage_image` here
    // cost macOS 14 and macOS 15 a whole `DispatchThreadgroups` a boot on
    // `MTLPixelFormatR16Unorm`, which is sampleable everywhere and is not a
    // storage format — see `translate::pixel::sampled_image`.
    crate::backend::vulkan::translate::pixel::sampled_image(format).ok()
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
        | V::Rgb9e5Ufloat
        | V::R16Unorm
        | V::Rg16Unorm
        | V::Rgba16Unorm
        | V::Rgb10a2Unorm
        | V::Bgr10a2Unorm
        | V::A8Unorm
        | V::Rg11b10Float => 0,
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
        V::R32Sint
        | V::R32Float
        | V::Rgb9e5Ufloat
        | V::R16Unorm
        | V::Rg16Unorm
        // The integer member of that family, sampled-only for the same reason
        // and not for its class: `STORAGE_IMAGE` is no more mandatory for
        // `R16G16_UINT` than for `R16G16_UNORM`.
        | V::Rg16Uint
        | V::Rgba16Unorm
        // The packed 32-bit colour formats join them: Vulkan mandates no
        // `STORAGE_IMAGE` support for any of the three, and one of them is not
        // in the mandatory table at all.
        | V::Rgb10a2Unorm
        | V::Bgr10a2Unorm
        // `A8Unorm` joins them by contract rather than by capability:
        // `storage_selector` has no entry for it, so no guest storage binding
        // can name it and its view mapping would be illegal on one.
        | V::A8Unorm
        | V::Rg11b10Float => {
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
    use crate::backend::metal::abi::texture_binds_as_storage;
    use crate::backend::metal::abi::{
        ReimsVgpuComputeImageblockDimensions, ReimsVgpuComputeStageInRegion,
        ReimsVgpuComputeStageInRegionIndirectArguments, ReimsVgpuComputeTextureUsage,
        ReimsVgpuSampler, ReimsVgpuThreadgroupMemory, REIMS_VGPU_BINDING_SAMPLER_BASE,
        REIMS_VGPU_BINDING_TEXTURE_BASE,
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
    let Some(mtlb) = load_mtlb(
        state,
        host,
        task_id,
        pipeline.kernel_func_ref,
        AirLoadRail::Compute,
    ) else {
        return ComputeStatus::MissingMtlb("compute_mtl_mtlb_load");
    };

    let DispatchDims {
        grid,
        threadgroup: tg,
        dispatch_threads,
    } = match resolve_dispatch_dims_reported(state, host, task_id, cmd, acc) {
        Ok(v) => v,
        Err(e) => return e,
    };

    // No narrowing here: `accepted_dispatch_type` scored this ordinal when the
    // record was applied, on both arms, and named the substitution if it made
    // one. Re-deciding it at the encode would be the same rule in a second
    // place, and the second place is the one that could not report.
    let dispatch_type = acc.dispatch_type;

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
        match stage_buffer_with_extent(state, host, task_id, b, None) {
            Ok(s) => staged_bufs.push(s),
            Err(e) => return e,
        }
    }

    // Texture reflection: access decides storage vs sampled materialization.
    // The reflection owns its own list — no caller-side capacity, so a kernel
    // declaring more bindings than some local buffer happened to hold is not a
    // refused dispatch.
    let mut err_buf = [0i8; 256];
    let usages: Vec<ReimsVgpuComputeTextureUsage> = if acc.textures.is_empty() {
        Vec::new()
    } else {
        match reflect_compute_textures_mtlb(&mtlb, (err_buf.as_mut_ptr(), err_buf.len())) {
            Ok(u) => u,
            Err(st) => return ComputeStatus::MetalBackend(st),
        }
    };

    let mut staged_tex: Vec<StagedTexture> = Vec::new();
    for t in &acc.textures {
        let binding = REIMS_VGPU_BINDING_TEXTURE_BASE + t.index;
        let is_storage = texture_binds_as_storage(&usages, binding);
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
        let sampler = match objects::resolve_sampler_state(state, host, task_id, s.sampler_ref) {
            Ok(sampler) => sampler,
            Err(objects::SamplerResolveError::Rung(rung)) => {
                return ComputeStatus::MissingSampler(crate::observe::ladder_slugs!(
                    "compute_mtl_sampler"
                )(rung))
            }
            Err(objects::SamplerResolveError::Decode { .. }) => {
                return ComputeStatus::MissingSampler(crate::observe::ladder_slug!(
                    "compute_mtl_sampler",
                    desc_decode
                ))
            }
        };
        reims_vgpu_samplers.push(crate::runtime::draw::sampler_record(
            REIMS_VGPU_BINDING_SAMPLER_BASE + s.index,
            &sampler.descriptor,
            s.has_lod_clamp.then_some((s.lod_min_bits, s.lod_max_bits)),
            false,
        ));
    }

    let mut reims_vgpu_bufs = abi_buffers(&mut staged_bufs);

    // Keep raw pointers valid: build storage/sampled from staged_tex after mut split.
    let (mut storage, sampled) =
        match split_staged_textures(&mut staged_tex, task_id, acc.pipeline_ref) {
            Ok(split) => split,
            Err(e) => return e,
        };

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
            dispatch_threads,
            grid,
            tg,
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
        dispatch_threads,
        dispatch_type,
        grid,
        tg,
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
