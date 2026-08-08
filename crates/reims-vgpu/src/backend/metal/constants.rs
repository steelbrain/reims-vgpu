//! Metal argument-table caps.
//!
//! These are the sizes this backend's encoders accept, and each is checked
//! before the `setBuffer:`/`setTexture:`/`setSamplerState:` call it guards —
//! Metal answers an out-of-range index with an exception that aborts the
//! process rather than a status this device can decline.

use crate::backend::metal::abi::{
    REIMS_VGPU_BINDING_SAMPLER_BASE, REIMS_VGPU_BINDING_TEXTURE_BASE,
};

pub const REIMS_VGPU_METAL_MAX_ATTRS: usize = 31;
pub const REIMS_VGPU_METAL_MAX_BUFFERS: usize = 31;
/// The texture argument table: Metal's own, and Apple's serializer's.
///
/// It was 32 — the width of the descriptor binding band, not a Metal fact —
/// because a texture at index 32 would have carried
/// [`REIMS_VGPU_BINDING_SAMPLER_BASE`], sampler 0's number, and
/// [`texture_index`](crate::backend::metal::util::texture_index) could not have
/// told the two apart. The sampler band moved up
/// (`spirv_bind::widen_sampled_bands`) so the texture band is 128 wide, and the
/// band assertion below is what holds the two in step.
pub const REIMS_VGPU_METAL_MAX_TEXTURES: usize = 128;
/// The sampler argument table.
///
/// The only one of the three tables whose number stood alone. `MAX_ATTRS` is
/// held equal to `decode::resource::MAX_VERTEX_ATTRS` and `MAX_BUFFERS` to
/// `draw::MAX_BUFFER_BIND_SLOTS`, and both of those carry the derivation at
/// their own declaration — so a reader of either arrives at a basis in one hop.
/// This one was a bare `16` with a mask-width assertion under it, which bounds
/// the number from above and says nothing about where it came from.
///
/// It is Apple's, and it is measured rather than asserted: their serializer
/// truncates a plural sampler bind at 16 per stage, which
/// [`bind_limit`](reims_vgpu_wire::ops::bind_limit::SAMPLER) captured by asking
/// for 200 and reading back 16. The pin below is what makes that basis
/// mechanical instead of a sentence — the capture and this table are one fact,
/// and the failure if they part is a sampler this device refuses to bind and
/// Apple's own driver would have.
pub const REIMS_VGPU_METAL_MAX_SAMPLERS: usize = 16;
const _: () =
    assert!(REIMS_VGPU_METAL_MAX_SAMPLERS == reims_vgpu_wire::ops::bind_limit::SAMPLER as usize);
/// The threadgroup-memory argument table.
///
/// The one table here that is **not** also a serializer truncation limit.
/// `reims_vgpu_wire::ops::bind_limit` captured the guest clamping a plural bind
/// at 31 buffers / 128 textures / 16 samplers, and there is no fourth capture to
/// read: `setThreadgroupMemoryLength:atIndex:` is a singular record carrying a
/// full `u32`, and the guest applies no bound to it on the way out. The
/// negotiated device info describes threadgroup memory in *bytes* and never in
/// slots, so the protocol does not state this bound either.
///
/// It is Metal's, and Metal states it as `maxComputeLocalMemorySizes` — the
/// entry the framework's own limits table fills from the same value it gives
/// `maxComputeBuffers`, on every GPU family, which is why this equals
/// [`REIMS_VGPU_METAL_MAX_BUFFERS`] rather than being a second reading of it.
/// The rule the framework enforces is `index < maxComputeLocalMemorySizes`, and
/// it enforces it by *throwing*, so an over-range index is a process abort
/// rather than a status this device can decline.
///
/// That is what this constant stands in front of, and it is why the accumulator
/// no longer carries a cap of its own. A `MAX_THREADGROUP_MEMORY_SLOTS` of 16
/// used to refuse the bind during stream accumulation: safe against the abort,
/// unjustified as a number, and — being applied before the backend split — it
/// took slots 16..=30 away from the Vulkan arm as well, which does not consume
/// threadgroup-memory binds at all.
///
/// Equal to [`REIMS_VGPU_METAL_MAX_BUFFERS`] and deliberately not written as it:
/// they are two Metal limits that hold the same number, not one limit spelled
/// twice, and nothing says a future family moves them together.
pub const REIMS_VGPU_METAL_MAX_THREADGROUP_MEMORY: usize = 31;
// The sampler table is also carried as a bitmask: `render_reflection_sampler_mask`
// packs one bit per slot into a `u32` and `bind_samplers` reads it back to supply
// a default sampler for every slot the pipeline reflection says is used. A table
// wider than the mask would shift past the end — undefined behaviour rather than
// a slot quietly missing its default — so the mask's width is a bound on this
// constant, stated here rather than at the shift.
const _: () = assert!(REIMS_VGPU_METAL_MAX_SAMPLERS <= u32::BITS as usize);
/// Metal max color attachments per render pass / PSO.
pub const REIMS_VGPU_METAL_MAX_COLOR_RTS: usize = 8;
// Two independent bases for one number, which is why both names stay: this is
// Metal's attachment array, and `PASS_MAX_COLOR_ATTACHMENTS` is the width of the
// colour-slot array in Apple's serialized render-pass record
// (`wire::RENDER_PASS_COLOR_ATTACHMENTS`). They have to agree, and the failure if
// they stop is quiet rather than loud: `fill_render_pso_key` clamps `color_count`
// with `.min(REIMS_VGPU_METAL_MAX_COLOR_RTS)` because it indexes arrays of that
// width, so a wire record carrying more slots than this table would lose the
// extra attachments to that clamp with nothing said — a multi-target draw missing
// its last render target. Pinned here rather than left to the clamp to notice.
const _: () = assert!(
    REIMS_VGPU_METAL_MAX_COLOR_RTS == crate::runtime::decode::render::PASS_MAX_COLOR_ATTACHMENTS
);

/// Metal `MTLBufferLayoutStrideDynamic` == `NSUIntegerMax`.
pub const MTL_BUFFER_LAYOUT_STRIDE_DYNAMIC: u64 = u64::MAX;

// The three binding bands do not overlap. A `const` assertion rather than a
// `#[test]`, for the reason the buffer-bind-limit pin below spells out.
const _: () = assert!(REIMS_VGPU_METAL_MAX_BUFFERS as u32 <= REIMS_VGPU_BINDING_TEXTURE_BASE);
const _: () = assert!(
    REIMS_VGPU_BINDING_TEXTURE_BASE + REIMS_VGPU_METAL_MAX_TEXTURES as u32
        <= REIMS_VGPU_BINDING_SAMPLER_BASE
);

/// The buffer argument table is one Metal limit, so the two spellings of it
/// must stay equal.
///
/// Four bind paths gate on it and they must all refuse the same index: direct
/// compute (`backend::metal::compute` via
/// [`crate::backend::metal::util::valid_buffer_binding`], which reads
/// `REIMS_VGPU_METAL_MAX_BUFFERS`), direct render and render ICB inheritance
/// (both `draw::MAX_BUFFER_BIND_SLOTS`), and compute ICB inheritance
/// (`valid_buffer_binding`). Letting the two constants drift would leave one
/// pair of paths passing an index to `setBuffer:offset:atIndex:` that the other
/// pair rejects, and Metal answers an out-of-range index with an exception that
/// aborts the process rather than a status this device can decline.
///
/// # Why these are `const` assertions and not tests
///
/// They were tests, and on the host this project is developed from they never
/// ran even once. This module is `backend-metal`-gated, so its `#[cfg(test)]`
/// block is compiled out of the Vulkan arm entirely, and `AGENTS.md` runs the
/// `backend-metal` `--lib` suite on Apple hosts only. The check standing between
/// four bind paths and a process-aborting Metal exception was therefore live on
/// no machine that anybody edits this code on.
///
/// A `const` assertion is evaluated by `rustc` whenever this file is compiled,
/// which includes the cross-compiled `--target aarch64-apple-darwin` clippy arm
/// that `AGENTS.md` requires from Linux. Same guarantee, checked everywhere,
/// and it fails the build rather than a suite nobody on this pathway runs.
const _: () =
    assert!(REIMS_VGPU_METAL_MAX_BUFFERS as u32 == crate::runtime::draw::MAX_BUFFER_BIND_SLOTS);

// The texture table and the accumulator's texture bound are one number — the
// band width — reached from two directions. `apply_binds` keeps a slot this
// table cannot hold, or this table holds one no bind record can reach, if they
// part.
const _: () =
    assert!(REIMS_VGPU_METAL_MAX_TEXTURES as u32 == crate::runtime::draw::MAX_TEXTURE_BIND_SLOTS);

// The vertex-attribute table and the decoder's bound on an `MTLVertexDescriptor`
// are one number, and here the cost of them parting is not a dropped attribute
// but a panic.
//
// `fill_render_pso_key` stores the **untruncated** `attrs.len()` as `attr_count`,
// which is right — clamping it would let two descriptors differing only past the
// table collide on one cache key, and the wrong pipeline is worse than a refused
// one. But the key's hash and `RenderPsoKey`'s equality then both walk
// `0..attr_count` over arrays that are `REIMS_VGPU_METAL_MAX_ATTRS` long, so an
// `attrs` longer than this constant indexes off the end of five of them.
//
// The colour-attachment sibling does not need this: `color_count` is `.min`ed at
// the same site precisely because it indexes the same way. Attributes are the one
// class whose safety rests entirely on the decoder having refused first, and it
// does — `parse_vertex_block` answers `res_vertex_attr_count_over` above
// `MAX_VERTEX_ATTRS` rather than truncating. This pins the number it refuses at
// to the width of the arrays that refusal is protecting, so the two cannot drift
// into a process abort. Same reasoning, and same `const`-assertion form, as the
// buffer and texture pins above.
const _: () =
    assert!(REIMS_VGPU_METAL_MAX_ATTRS == crate::runtime::decode::resource::MAX_VERTEX_ATTRS);

// This backend's two band bases are mirrors of `runtime::spirv_bind`'s, which is
// where the widening that set them is written. A mirror that drifts would have
// the two arms encode the same guest bind as two different descriptor bindings,
// and nothing else in the toolchain compares them.
const _: () =
    assert!(REIMS_VGPU_BINDING_TEXTURE_BASE == crate::runtime::spirv_bind::TEXTURE_BINDING_BASE);
const _: () =
    assert!(REIMS_VGPU_BINDING_SAMPLER_BASE == crate::runtime::spirv_bind::SAMPLER_BINDING_BASE);
