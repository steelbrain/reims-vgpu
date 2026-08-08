//! L2–L7 immutable object caches (content/descriptor keyed, negative + hit/miss).

#![allow(unsafe_op_in_unsafe_fn)]

use ash::vk;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::Ordering;

// ash Handle trait not required here.

use super::context::DeviceContext;
use super::counters::EngineCounters;
use super::digest::Digest128;
use super::pools::{DeferredHandle, ResourcePools};
use super::types::{
    BlendKey, ColorWriteMask, CullMode, DepthClipMode, DrawError, FillMode, PrimitiveTopology,
    SamplerStateKey, VertexAttributeFormat, VertexStepFunction,
};
use super::vk_call::{VkCall, VkOp};

/// A device-specific widening of an optional three-component vertex format.
///
/// The draw remains executable, but the pipeline is not byte-for-byte what the
/// guest requested; keep the substitution and the affected attribute visible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct VertexFormatWidenDecline {
    from: vk::Format,
    to: vk::Format,
    location: u32,
    offset: u32,
    stride: u32,
}

impl crate::observe::Decline for VertexFormatWidenDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self { .. } => "vk_vertex_format_widened",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![
            ("from", format!("{:?}", self.from)),
            ("to", format!("{:?}", self.to)),
            ("location", self.location.to_string()),
            ("offset", self.offset.to_string()),
            ("stride", self.stride.to_string()),
        ]
    }
}
use crate::backend::vulkan::translate;
use crate::runtime::spirv_vertex_input::VertexInputWidths;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct AttrKey {
    pub location: u32,
    pub binding: u32,
    pub format: VertexAttributeFormat,
    pub offset: u32,
    pub stride: u32,
    pub step_function: VertexStepFunction,
    pub step_rate: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) struct BindingSig {
    pub binding: u32,
    pub ty: u32, // vk::DescriptorType as u32
    pub stages: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) struct LayoutKey {
    pub bindings: Vec<BindingSig>,
}

/// Max secondary color attachments (MRT slot 1..): every colour slot Apple's
/// serialized render pass can carry, less the primary at slot 0.
///
/// The fourth spelling of one number, and the last one to be pinned. The wire
/// record's colour-slot array is the truth,
/// [`crate::runtime::decode::render::PASS_MAX_COLOR_ATTACHMENTS`] derives from
/// it, and `backend::metal::REIMS_VGPU_METAL_MAX_COLOR_RTS` is held equal to it
/// by an assertion beside itself. This one is that bound minus one, on the arm
/// the other assertion cannot reach — `REIMS_VGPU_METAL_MAX_COLOR_RTS` is behind
/// `feature = "backend-metal"`, so nothing in a Vulkan build compared the two.
///
/// A drift here is refused rather than lost: `execute_draw_inner` returns
/// [`super::reason::DrawReason::SecondaryAttachmentCap`] for a request past this
/// count, so a shortfall costs the whole draw and says so. That makes the
/// failure loud and still wrong — a guest sending the eighth colour slot the
/// wire format allows would have every MRT draw refused — which is what this
/// assertion is for.
pub(crate) const MAX_SECONDARY_ATTACH: usize = 7;
const _: () =
    assert!(1 + MAX_SECONDARY_ATTACH == crate::runtime::decode::render::PASS_MAX_COLOR_ATTACHMENTS);

/// A secondary MRT attachment's contribution to the render-pass / pipeline key.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
pub(crate) struct SecondaryAttachKey {
    pub format: ash::vk::Format,
    /// true = LOAD existing content, false = CLEAR.
    pub load: bool,
}

/// A depth attachment's contribution to the render-pass key. `None` on `PassKey`
/// ⇒ no depth attachment (the 2D UI path). Depth-only uses D32_SFLOAT; when
/// `stencil` is set the attachment is the device-queried combined
/// depth-stencil format (`DeviceContext::depth_stencil_format`) with a live
/// STENCIL aspect (load/store), so it must partition the pass cache.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct DepthAttachKey {
    /// true = LOAD existing depth, false = CLEAR at pass start.
    pub load: bool,
    /// true = combined depth-stencil attachment (stencil test active).
    pub stencil: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct PassKey {
    pub load_seed: bool, // LOAD vs CLEAR (slot 0)
    /// Slot-0 attachment format: true = B8G8R8A8_UNORM (guest scanout order for
    /// zero-copy import-present), false = R8G8B8A8_UNORM.
    pub bgra: bool,
    /// Secondary color attachments (slot 1..). `secondary_count == 0` ⇒ the
    /// classic single-attachment pass, byte-identical to the pre-MRT engine.
    pub secondary: [SecondaryAttachKey; MAX_SECONDARY_ATTACH],
    pub secondary_count: u8,
    /// Depth attachment. `None` ⇒ no depth (byte-identical to the pre-depth
    /// pass); the depth attachment is always appended AFTER color + secondaries
    /// so slot 0 stays the primary color (the zero-copy readback assumes this).
    pub depth: Option<DepthAttachKey>,
    /// Attachment 0 is ALSO referenced as a subpass input (framebuffer fetch).
    /// Both references use GENERAL layout and the subpass carries a BY_REGION
    /// self-dependency — the Vulkan feedback-loop form MoltenVK lowers to Metal
    /// programmable blending. `false` keeps the pass byte-identical.
    pub color_input: bool,
}

impl PassKey {
    /// Single-color-attachment pass (the pre-MRT constructor).
    pub(crate) fn single(load_seed: bool, bgra: bool) -> Self {
        Self {
            load_seed,
            bgra,
            secondary: [SecondaryAttachKey::default(); MAX_SECONDARY_ATTACH],
            secondary_count: 0,
            depth: None,
            color_input: false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) struct PipelineKey {
    pub vert: Digest128,
    pub frag: Digest128,
    pub attrs: Vec<AttrKey>,
    pub topology: PrimitiveTopology,
    pub blend: Option<BlendKey>,
    /// Per-slot blend for secondary colour attachments, parallel to
    /// `pass.secondary[..pass.secondary_count]`. Entries past the count are
    /// `None` and inert.
    ///
    /// This is part of the key, not just the builder input: two draws sharing
    /// shaders and pass shape but blending different secondary slots need
    /// different pipelines, and before this they would have aliased onto
    /// whichever was created first.
    pub secondary_blend: [Option<BlendKey>; MAX_SECONDARY_ATTACH],
    /// Per-slot `MTLColorWriteMask`, index 0 the primary attachment and index
    /// `n` the secondary parallel to `pass.secondary[n - 1]`.
    ///
    /// In the key, not just the builder input: two draws sharing shaders, pass
    /// shape and blend but masking different channels need different
    /// pipelines. Vulkan's write mask is pipeline state with no dynamic
    /// spelling below `VK_EXT_extended_dynamic_state3`.
    pub color_write_mask: [ColorWriteMask; 1 + MAX_SECONDARY_ATTACH],
    pub pass: PassKey,
    /// Face culling. `None` (the 2D UI default) keeps the raster state at
    /// `CULL_NONE`, byte-identical to the pre-cull engine; the key still
    /// participates in hashing so a later culled draw with the same shaders gets
    /// its own pipeline rather than aliasing the no-cull one.
    pub cull_mode: CullMode,
    /// Metal front-facing winding (`true` = counter-clockwise), mapped to a
    /// Vulkan `FrontFace` by [`crate::backend::vulkan::translate::raster::vk_front_face`].
    pub front_face_ccw: bool,
    /// Metal `MTLTriangleFillMode`, mapped to a `VkPolygonMode`. In the key
    /// because Vulkan has no dynamic polygon mode below
    /// `VK_EXT_extended_dynamic_state3`: a wireframe draw and a filled draw
    /// sharing shaders need different pipelines.
    pub fill_mode: FillMode,
    /// Metal `MTLDepthClipMode`, mapped to `depthClampEnable`. In the key for
    /// the same reason as [`Self::fill_mode`].
    pub depth_clip: DepthClipMode,
    /// Depth-test pipeline state. Meaningful only when `pass.depth.is_some()`;
    /// otherwise all-default (test/write off) and no depth-stencil state is
    /// attached, so the color-only pipeline is byte-identical to the pre-depth
    /// engine. Metal `MTLCompareFunction` shares `SamplerCompareFunction`.
    pub depth_test: bool,
    pub depth_write: bool,
    pub depth_compare: super::types::SamplerCompareFunction,
    /// Front/back stencil op state. `Some` only when `pass.depth` carries a
    /// stencil aspect; the reference value is *excluded* (dynamic state) so
    /// distinct references reuse one pipeline. `None` keeps the depth-only /
    /// no-depth pipelines byte-identical to the pre-stencil engine.
    pub stencil: Option<StencilKey>,
    /// How many viewport/scissor slots the pipeline declares.
    ///
    /// In the key because `VkPipelineViewportStateCreateInfo::viewportCount` is
    /// **not** dynamic below `VK_EXT_extended_dynamic_state`
    /// (`vkCmdSetViewportWithCount`), which is core in 1.3 and this device's
    /// floor is 1.2. So the count is baked, `vkCmdSetViewport` must bind exactly
    /// that many, and two draws sharing shaders and pass shape but rasterizing
    /// into different numbers of viewports need different pipelines. It is one
    /// number for both counts because Vulkan requires `scissorCount` to equal
    /// `viewportCount`; [`super::viewport_slot_count`] is the only place that
    /// decides it.
    pub viewport_slots: u32,
    pub layout: LayoutKey,
}

/// Front/back stencil op state baked into the pipeline (everything but the
/// dynamic reference value). See [`super::types::StencilFaceOps`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct StencilKey {
    pub front: super::types::StencilFaceOps,
    pub back: super::types::StencilFaceOps,
}

/// Lc: compute pipeline cache key — SPIR-V content digest + entry name + layout.
/// Never funcId / pipeline ref.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) struct ComputePipelineKey {
    pub spirv: Digest128,
    pub entry: String,
    pub layout: LayoutKey,
}

/// How many distinct never-creatable keys a cache remembers the refusal for.
///
/// This bounds the **negative** map only, and it is the one bound in this file
/// that is not a fidelity question: an evicted negative entry costs a re-attempt
/// of a create that has already been measured to fail, never a dropped guest
/// object. The positive maps are deliberately unbounded — see [`ObjectCache`].
const NEGATIVE_CAP: usize = 1024;

/// A content-keyed cache of immutable Vulkan objects, plus the typed refusal for
/// keys whose create failed.
///
/// **The positive map is unbounded, and that is the contract.** Every key here
/// is a content digest or a full descriptor of guest-decoded state — a shader's
/// SPIR-V digest, a pipeline's complete key, a sampler's state. So the live
/// entry count is the number of *distinct* objects the guest has asked for,
/// which is a property of its own program and state set rather than of how long
/// the device has run.
///
/// It used to hold 1024 (64 for render passes) and evict in **insertion** order.
/// Insertion order is the worst possible choice here for the same reason it was
/// in `runtime::m2v_cache`: the first pipeline a boot creates is the
/// compositor's, and it is bound on every frame until the guest shuts down, so
/// the first thing a cap crossing discards is the entry that is still hot. The
/// re-create is `vkCreateGraphicsPipelines` — a driver-side shader compile, not
/// a lookup — so a thrashing cache pays one compile per frame per evicted
/// pipeline, forever.
///
/// The bound also never engaged on this arm. A driven x86 boot, window-drag
/// probe against Safari, settles at `pipelines=92 shaders=75 layouts=33
/// passes=4 samplers=14 compute_pipelines=16` — read directly off
/// [`ObjectCaches::levels`], which is what the `object_cache_levels` census
/// publishes. Every level is flat from roughly 38 s in through the end of the
/// run, including across the drag probe's compositing, so the caps only stood
/// ready to evict the hot set on a heavier guest.
///
/// Two of those numbers matter beyond this arm. `passes=4` against the 64 this
/// cache carried is the widest margin here; and `pipelines=92` is *above* the
/// 64-slot render-pipeline table the Metal arm carried, which is how that arm's
/// cap was shown to be binding — see [`crate::model::content_cache`].
///
/// Unbounded is also the faithful failure mode. When a guest really does ask for
/// more distinct pipelines than the host can hold, the create itself returns
/// `VK_ERROR_OUT_OF_DEVICE_MEMORY` and that is reported as a typed [`DrawError`].
/// That is a GPU refusing because its memory is full — the behavior we are
/// emulating — rather than a device that silently forgets an object the guest
/// still has bound. It is deliberately *not* remembered; see
/// [`ObjectCache::insert_negative`] for why a refusal about this instant must not
/// outlive the instant.
struct ObjectCache<K, V> {
    map: HashMap<K, V>,
    negative: HashMap<K, DrawError>,
    /// FIFO order for `negative`, bounded by [`NEGATIVE_CAP`]. Negative entries
    /// are only added on create failures that a second identical attempt would
    /// meet again — a Vulkan create call refusing for a reason inherent to the
    /// request (a typed [`VkCall`]) or a device-capability refusal
    /// (`DrawError::Unsupported`, e.g. an unsupported vertex divisor) — empty on
    /// a healthy boot, but a guest that keeps submitting distinct
    /// never-creatable objects would grow `negative` without limit if it were
    /// unbounded. The value is the exact typed [`DrawError`] the create refused
    /// with, so the cheap re-attempt replays that reason — slug and all — rather
    /// than a re-formatted `Vulkan(String)` that dropped it.
    negative_order: VecDeque<K>,
    negative_cap: usize,
}

impl<K: Clone + Eq + std::hash::Hash, V> ObjectCache<K, V> {
    fn new() -> Self {
        Self::with_negative_cap(NEGATIVE_CAP)
    }

    fn with_negative_cap(negative_cap: usize) -> Self {
        Self {
            map: HashMap::new(),
            negative: HashMap::new(),
            negative_order: VecDeque::new(),
            negative_cap,
        }
    }

    fn get(&self, k: &K) -> Option<&V> {
        self.map.get(k)
    }

    fn get_negative(&self, k: &K) -> Option<DrawError> {
        self.negative.get(k).cloned()
    }

    /// Insert. Returns the value a *replace* displaced, so the caller can
    /// destroy the Vulkan object it owned; a fresh key returns `None`. Nothing
    /// is ever displaced for capacity.
    fn insert(&mut self, k: K, v: V) -> Option<V> {
        self.negative.remove(&k);
        self.map.insert(k, v)
    }

    /// Remember a create failure so the next identical ask replays it without
    /// paying the driver call again.
    ///
    /// **A refusal about this instant is not remembered at all.** Out of memory
    /// describes how much the device is holding right now, not anything about
    /// the request — the guest can free a texture atlas and ask for the very
    /// same pipeline a frame later, and by then the create succeeds. Memoizing
    /// one turns a GPU that refuses while full into a GPU that refuses forever,
    /// which is the failure a real one does not have: nothing here can clear a
    /// negative entry short of device teardown, because the lookup consults
    /// `negative` before the create and so the create that would displace it
    /// never runs.
    ///
    /// The predicate is [`DrawError::out_of_memory`], the crate's single
    /// statement of which refusals a second attempt could answer differently;
    /// the resident image and command-buffer allocators already reclaim and
    /// retry on it. Deciding it here rather than at the call sites is
    /// deliberate — thirteen of them insert negatives, and a rule spread over
    /// thirteen sites is a rule that will be half-applied.
    ///
    /// Declining to memoize costs a repeated failing create while the device
    /// stays full. That is the same bargain the resident allocators take, and
    /// it is bounded by the guest's own retry rate rather than by anything
    /// here.
    fn insert_negative(&mut self, k: K, err: DrawError) {
        if err.out_of_memory() {
            return;
        }
        if self.negative.insert(k.clone(), err).is_some() {
            // Already tracked (error refreshed); order stays as-is.
            return;
        }
        self.negative_order.push_back(k);
        // Bound the negative map, oldest-first. Pops skip stale order entries
        // (keys since promoted into the positive map by `insert`).
        while self.negative.len() > self.negative_cap {
            match self.negative_order.pop_front() {
                Some(old) => {
                    self.negative.remove(&old);
                }
                None => break,
            }
        }
        // Compact the order deque if promotions left many stale entries, so it
        // can never itself grow unbounded (rare; error path only).
        if self.negative_order.len() > self.negative_cap.saturating_mul(2) {
            self.negative_order
                .retain(|key| self.negative.contains_key(key));
        }
    }

    fn len(&self) -> usize {
        self.map.len()
    }

    fn clear(&mut self) {
        self.map.clear();
        self.negative.clear();
        self.negative_order.clear();
    }

    fn take_all(&mut self) -> Vec<V> {
        self.negative.clear();
        self.negative_order.clear();
        self.map.drain().map(|(_, v)| v).collect()
    }
}

pub(crate) struct ObjectCaches {
    shaders: ObjectCache<Digest128, vk::ShaderModule>,
    layouts: ObjectCache<LayoutKey, (vk::DescriptorSetLayout, vk::PipelineLayout)>,
    passes: ObjectCache<PassKey, vk::RenderPass>,
    pipelines: ObjectCache<PipelineKey, vk::Pipeline>,
    samplers: ObjectCache<SamplerStateKey, vk::Sampler>,
    /// Lc: compute pipelines (content digest + entry + layout).
    compute_pipelines: ObjectCache<ComputePipelineKey, vk::Pipeline>,
}

impl ObjectCaches {
    pub(crate) fn new() -> Self {
        Self {
            shaders: ObjectCache::new(),
            layouts: ObjectCache::new(),
            passes: ObjectCache::new(),
            pipelines: ObjectCache::new(),
            samplers: ObjectCache::new(),
            compute_pipelines: ObjectCache::new(),
        }
    }

    pub(crate) unsafe fn destroy_all(&mut self, device: &ash::Device) {
        for p in self.pipelines.take_all() {
            device.destroy_pipeline(p, None);
        }
        for p in self.compute_pipelines.take_all() {
            device.destroy_pipeline(p, None);
        }
        for (dsl, pl) in self.layouts.take_all() {
            device.destroy_pipeline_layout(pl, None);
            if dsl != vk::DescriptorSetLayout::null() {
                device.destroy_descriptor_set_layout(dsl, None);
            }
        }
        for rp in self.passes.take_all() {
            device.destroy_render_pass(rp, None);
        }
        for s in self.shaders.take_all() {
            device.destroy_shader_module(s, None);
        }
        for s in self.samplers.take_all() {
            device.destroy_sampler(s, None);
        }
    }

    /// Live entries in each cache, in the order
    /// `(shaders, layouts, passes, pipelines, samplers, compute_pipelines)`.
    ///
    /// Published because [`ObjectCache`] is unbounded on the argument that its
    /// entry count is the guest's distinct object set and therefore plateaus.
    /// That is a claim about a running guest, and this is the reading that can
    /// falsify it: a level that climbs for the life of a boot instead of
    /// settling means some key is carrying per-frame state and the argument is
    /// wrong for that cache. Levels, not deltas — the census line says so.
    pub(crate) fn levels(&self) -> [usize; 6] {
        [
            self.shaders.len(),
            self.layouts.len(),
            self.passes.len(),
            self.pipelines.len(),
            self.samplers.len(),
            self.compute_pipelines.len(),
        ]
    }

    pub(crate) fn clear_logical(&mut self) {
        self.shaders.clear();
        self.layouts.clear();
        self.passes.clear();
        self.pipelines.clear();
        self.samplers.clear();
        self.compute_pipelines.clear();
    }

    pub(crate) unsafe fn get_or_create_shader(
        &mut self,
        ctx: &DeviceContext,
        words: &[u32],
        counters: &EngineCounters,
        pools: &mut ResourcePools,
    ) -> Result<(Digest128, vk::ShaderModule), DrawError> {
        let key = Digest128::of_u32_words(words);
        if let Some(err) = self.shaders.get_negative(&key) {
            counters.shader_misses.fetch_add(1, Ordering::Relaxed);
            return Err(err);
        }
        if let Some(&m) = self.shaders.get(&key) {
            counters.shader_hits.fetch_add(1, Ordering::Relaxed);
            return Ok((key, m));
        }
        counters.shader_misses.fetch_add(1, Ordering::Relaxed);
        let created = ctx
            .device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(words), None);
        let module = created.map_err(|e| {
            let err = DrawError::VkCall(VkCall::new(VkOp::CachesCreateShaderModule, e));
            self.shaders.insert_negative(key, err.clone());
            err
        })?;
        counters.note_create();
        if let Some(old) = self.shaders.insert(key, module) {
            pools.dispose(&ctx.device, DeferredHandle::ShaderModule(old));
        }
        Ok((key, module))
    }

    pub(crate) unsafe fn get_or_create_layout(
        &mut self,
        ctx: &DeviceContext,
        key: &LayoutKey,
        counters: &EngineCounters,
        pools: &mut ResourcePools,
    ) -> Result<(vk::DescriptorSetLayout, vk::PipelineLayout), DrawError> {
        if let Some(err) = self.layouts.get_negative(key) {
            counters.layout_misses.fetch_add(1, Ordering::Relaxed);
            return Err(err);
        }
        if let Some(&(dsl, pl)) = self.layouts.get(key) {
            counters.layout_hits.fetch_add(1, Ordering::Relaxed);
            return Ok((dsl, pl));
        }
        counters.layout_misses.fetch_add(1, Ordering::Relaxed);
        let bindings: Vec<vk::DescriptorSetLayoutBinding<'_>> = key
            .bindings
            .iter()
            .map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b.binding)
                    .descriptor_type(vk::DescriptorType::from_raw(b.ty as i32))
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::from_raw(b.stages))
            })
            .collect();
        let dsl = if bindings.is_empty() {
            vk::DescriptorSetLayout::null()
        } else {
            let d = ctx
                .device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .map_err(|e| {
                    let err =
                        DrawError::VkCall(VkCall::new(VkOp::CachesCreateDescriptorSetLayout, e));
                    self.layouts.insert_negative(key.clone(), err.clone());
                    err
                })?;
            counters.note_create();
            d
        };
        let layouts: Vec<vk::DescriptorSetLayout> = if dsl == vk::DescriptorSetLayout::null() {
            Vec::new()
        } else {
            vec![dsl]
        };
        let pl = ctx
            .device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts),
                None,
            )
            .map_err(|e| {
                if dsl != vk::DescriptorSetLayout::null() {
                    ctx.device.destroy_descriptor_set_layout(dsl, None);
                }
                let err = DrawError::VkCall(VkCall::new(VkOp::CachesCreatePipelineLayout, e));
                self.layouts.insert_negative(key.clone(), err.clone());
                err
            })?;
        counters.note_create();
        if let Some((old_dsl, old_pl)) = self.layouts.insert(key.clone(), (dsl, pl)) {
            pools.dispose(&ctx.device, DeferredHandle::PipelineLayout(old_pl));
            if old_dsl != vk::DescriptorSetLayout::null() {
                pools.dispose(&ctx.device, DeferredHandle::DescriptorSetLayout(old_dsl));
            }
        }
        Ok((dsl, pl))
    }

    pub(crate) unsafe fn get_or_create_pass(
        &mut self,
        ctx: &DeviceContext,
        key: PassKey,
        counters: &EngineCounters,
        pools: &mut ResourcePools,
    ) -> Result<vk::RenderPass, DrawError> {
        if let Some(err) = self.passes.get_negative(&key) {
            counters.pass_misses.fetch_add(1, Ordering::Relaxed);
            return Err(err);
        }
        if let Some(&rp) = self.passes.get(&key) {
            counters.pass_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(rp);
        }
        counters.pass_misses.fetch_add(1, Ordering::Relaxed);
        let target_format = translate::pixel::resident_color(key.bgra);
        let (load_op, initial) = if key.load_seed {
            (
                vk::AttachmentLoadOp::LOAD,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            )
        } else {
            (vk::AttachmentLoadOp::CLEAR, vk::ImageLayout::UNDEFINED)
        };
        // Slot 0 (primary): final layout TRANSFER_SRC_OPTIMAL for readback /
        // present. Secondary attachments (slot 1..) are consumed by a *later*
        // draw as sampled images, but they resolve to COLOR_ATTACHMENT_OPTIMAL —
        // NOT SHADER_READ_ONLY. The consumer's resident-sample barrier is
        // skipped when the tracked layout is already SHADER_READ_ONLY, which
        // would drop the producer(color-write)→consumer(shader-read) dependency;
        // leaving the mask at COLOR_ATTACHMENT_OPTIMAL forces that barrier to
        // fire with a color-write source scope. The registry tracks this layout.
        let mut attachments = vec![vk::AttachmentDescription::default()
            .format(target_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(load_op)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(initial)
            .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)];
        // Framebuffer fetch: when attachment 0 is also a subpass input, BOTH
        // references must use GENERAL (same-attachment color+input requires it);
        // the pass still transitions initial→GENERAL→final automatically.
        let color0_layout = if key.color_input {
            vk::ImageLayout::GENERAL
        } else {
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        };
        let mut color_ref = vec![vk::AttachmentReference::default()
            .attachment(0)
            .layout(color0_layout)];
        for (i, sec) in key.secondary[..key.secondary_count as usize]
            .iter()
            .enumerate()
        {
            let (sload, sinitial) = if sec.load {
                (
                    vk::AttachmentLoadOp::LOAD,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                )
            } else {
                (vk::AttachmentLoadOp::CLEAR, vk::ImageLayout::UNDEFINED)
            };
            attachments.push(
                vk::AttachmentDescription::default()
                    .format(sec.format)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(sload)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(sinitial)
                    .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
            );
            color_ref.push(
                vk::AttachmentReference::default()
                    .attachment(1 + i as u32)
                    .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
            );
        }
        // Depth attachment is appended LAST (after color + secondaries), so its
        // index is the current attachment count and color slot 0 is untouched.
        let depth_ref = key.depth.map(|d| {
            let (dload, dinitial) = if d.load {
                (
                    vk::AttachmentLoadOp::LOAD,
                    vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                )
            } else {
                (vk::AttachmentLoadOp::CLEAR, vk::ImageLayout::UNDEFINED)
            };
            // Stencil test active ⇒ combined format with a live STENCIL aspect
            // (CLEAR/STORE mirroring depth). Depth-only stays D32_SFLOAT with
            // DONT_CARE stencil, byte-identical to the pre-stencil pass.
            let (dformat, sload, sstore) = if d.stencil {
                (
                    ctx.depth_stencil_format,
                    dload,
                    vk::AttachmentStoreOp::STORE,
                )
            } else {
                (
                    translate::pixel::TRANSIENT_DEPTH_FORMAT,
                    vk::AttachmentLoadOp::DONT_CARE,
                    vk::AttachmentStoreOp::DONT_CARE,
                )
            };
            let index = attachments.len() as u32;
            attachments.push(
                vk::AttachmentDescription::default()
                    .format(dformat)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(dload)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(sload)
                    .stencil_store_op(sstore)
                    .initial_layout(dinitial)
                    .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
            );
            vk::AttachmentReference::default()
                .attachment(index)
                .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        });
        let input_ref = [vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::GENERAL)];
        let mut subpass_desc = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_ref);
        if key.color_input {
            subpass_desc = subpass_desc.input_attachments(&input_ref);
        }
        if let Some(depth_ref) = &depth_ref {
            subpass_desc = subpass_desc.depth_stencil_attachment(depth_ref);
        }
        let subpass = [subpass_desc];
        // Explicit subpass dependencies only for the depth pass — the color-only
        // pass keeps relying on the implicit dependencies (byte-identical). The
        // implicit external dependency synchronizes only COLOR_ATTACHMENT_OUTPUT,
        // NOT the EARLY/LATE_FRAGMENT_TESTS stages a depth attachment's
        // load-clear + test + store use, so the depth layout transition and the
        // clear can race without these.
        let depth_deps = [
            vk::SubpassDependency::default()
                .src_subpass(vk::SUBPASS_EXTERNAL)
                .dst_subpass(0)
                .src_stage_mask(
                    vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                        | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                )
                .dst_stage_mask(
                    vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                        | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                )
                .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                ),
            vk::SubpassDependency::default()
                .src_subpass(0)
                .dst_subpass(vk::SUBPASS_EXTERNAL)
                .src_stage_mask(vk::PipelineStageFlags::LATE_FRAGMENT_TESTS)
                .dst_stage_mask(vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
                .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                ),
        ];
        // Framebuffer-fetch feedback loop: the same-pixel color-write →
        // input-read ordering within the one subpass. BY_REGION keeps it
        // framebuffer-local (the form MoltenVK lowers to tile-memory fetch).
        let fetch_dep = vk::SubpassDependency::default()
            .src_subpass(0)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::INPUT_ATTACHMENT_READ)
            .dependency_flags(vk::DependencyFlags::BY_REGION);
        let mut deps: Vec<vk::SubpassDependency> = Vec::new();
        if key.depth.is_some() {
            deps.extend_from_slice(&depth_deps);
        }
        if key.color_input {
            deps.push(fetch_dep);
        }
        let mut rp_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpass);
        if !deps.is_empty() {
            rp_info = rp_info.dependencies(&deps);
        }
        let rp = ctx.device.create_render_pass(&rp_info, None).map_err(|e| {
            let err = DrawError::VkCall(VkCall::new(VkOp::CachesCreateRenderPass, e));
            self.passes.insert_negative(key, err.clone());
            err
        })?;
        counters.note_create();
        if let Some(old) = self.passes.insert(key, rp) {
            pools.dispose(&ctx.device, DeferredHandle::RenderPass(old));
        }
        Ok(rp)
    }

    pub(crate) unsafe fn get_or_create_sampler(
        &mut self,
        ctx: &DeviceContext,
        key: &SamplerStateKey,
        counters: &EngineCounters,
        pools: &mut ResourcePools,
    ) -> Result<vk::Sampler, DrawError> {
        if let Some(err) = self.samplers.get_negative(key) {
            counters.sampler_misses.fetch_add(1, Ordering::Relaxed);
            return Err(err);
        }
        if let Some(&s) = self.samplers.get(key) {
            counters.sampler_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(s);
        }
        counters.sampler_misses.fetch_add(1, Ordering::Relaxed);
        let not_mipmapped = key.mip_filter == super::types::SamplerMipFilter::NotMipmapped;
        let (min_lod, max_lod) = if key.unnormalized_coordinates || not_mipmapped {
            (0.0, 0.0)
        } else {
            (f32::from_bits(key.lod_min), f32::from_bits(key.lod_max))
        };
        let address_uses_zero = [key.address_mode_u, key.address_mode_v, key.address_mode_w]
            .contains(&super::types::SamplerAddressMode::ClampToZero);
        if key.max_anisotropy > 1 && !ctx.sampler_anisotropy {
            let reason = super::reason::DrawReason::SamplerAnisotropyUnsupported;
            // Fail-visible here, at the check, and exactly once per sampler key:
            // the negative cache means a replay returns without reaching this
            // line, and the returned `DrawError` reaches the log only if some
            // caller happens to render it. A capability the host GPU lacks is
            // precisely the class says must
            // never surface as a silently different sampler.
            crate::observe::Emit::decline("vk_engine_sampler", &reason)
                .field("max_anisotropy", key.max_anisotropy)
                .fail();
            // The negative cache stores the typed `DrawError`, so a replay
            // returns this exact decline — slug and all — not a re-rendered
            // `Vulkan(String)` that would drop the reason to `vk_engine_vk_untyped`.
            let err = DrawError::Unsupported(reason);
            self.samplers.insert_negative(*key, err.clone());
            return Err(err);
        }
        // `MTLSamplerAddressModeMirrorClampToEdge` translates to
        // `MIRROR_CLAMP_TO_EDGE`, which needs either the Vulkan 1.2 feature or
        // `VK_KHR_sampler_mirror_clamp_to_edge`. Until this check existed the
        // mode was bound with neither requested — the sampler was created with
        // something the device had not been asked for. Same shape as the
        // anisotropy check above: capability question, typed decline, cached.
        let uses_mirror_clamp = [key.address_mode_u, key.address_mode_v, key.address_mode_w]
            .contains(&super::types::SamplerAddressMode::MirrorClampToEdge);
        if uses_mirror_clamp && !ctx.features.mirror_clamp_to_edge.is_available() {
            let reason = super::reason::DrawReason::SamplerMirrorClampToEdgeUnsupported;
            crate::observe::Emit::decline("vk_engine_sampler", &reason)
                .field("address_u", format!("{:?}", key.address_mode_u))
                .field("address_v", format!("{:?}", key.address_mode_v))
                .field("address_w", format!("{:?}", key.address_mode_w))
                .fail();
            let err = DrawError::Unsupported(reason);
            self.samplers.insert_negative(*key, err.clone());
            return Err(err);
        }
        // Not floored here: every producer of this key either writes a literal
        // 1 (the reflected static sampler) or carries a decoded
        // `SamplerDescriptor`, which `decode_sampler_descriptor` already floors.
        let max_anisotropy = (key.max_anisotropy as f32).min(ctx.max_sampler_anisotropy);
        let sampler = ctx
            .device
            .create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(key.mag_filter.vk())
                    .min_filter(key.min_filter.vk())
                    .mipmap_mode(key.mip_filter.vk())
                    .address_mode_u(key.address_mode_u.vk())
                    .address_mode_v(key.address_mode_v.vk())
                    .address_mode_w(key.address_mode_w.vk())
                    .mip_lod_bias(0.0)
                    .anisotropy_enable(key.max_anisotropy > 1)
                    .max_anisotropy(max_anisotropy)
                    .compare_enable(
                        key.compare_function != super::types::SamplerCompareFunction::Never,
                    )
                    .compare_op(key.compare_function.vk())
                    .min_lod(min_lod)
                    .max_lod(max_lod)
                    .border_color(translate::sampler::vk_border_color_with_clamp_to_zero(
                        key.border_color,
                        address_uses_zero,
                    ))
                    .unnormalized_coordinates(key.unnormalized_coordinates),
                None,
            )
            .map_err(|e| {
                let err = DrawError::VkCall(VkCall::new(VkOp::CachesCreateSampler, e));
                self.samplers.insert_negative(*key, err.clone());
                err
            })?;
        counters.note_create();
        if let Some(old) = self.samplers.insert(*key, sampler) {
            pools.dispose(&ctx.device, DeferredHandle::Sampler(old));
        }
        Ok(sampler)
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "pipeline creation mirrors the Vulkan shader, layout, pass, and cache handles"
    )]
    pub(crate) unsafe fn get_or_create_pipeline(
        &mut self,
        ctx: &DeviceContext,
        key: &PipelineKey,
        vert_module: vk::ShaderModule,
        // The post-relocation words `vert_module` was built from. Read only to
        // answer how wide this shader's stage-in reads are, and only on a host
        // that substitutes a vertex format; see the resolution loop below.
        vert_spirv: &[u32],
        frag_module: vk::ShaderModule,
        pipeline_layout: vk::PipelineLayout,
        render_pass: vk::RenderPass,
        counters: &EngineCounters,
        pools: &mut ResourcePools,
    ) -> Result<vk::Pipeline, DrawError> {
        if let Some(err) = self.pipelines.get_negative(key) {
            counters.pipeline_misses.fetch_add(1, Ordering::Relaxed);
            return Err(err);
        }
        if let Some(&p) = self.pipelines.get(key) {
            counters.pipeline_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(p);
        }
        counters.pipeline_misses.fetch_add(1, Ordering::Relaxed);

        // `MTLBlendFactor` 15-18 are the dual-source factors; Vulkan spells them
        // `SRC1_*` and gates them behind `VkPhysicalDeviceFeatures::dualSrcBlend`.
        // Same shape as the sampler's mirror-clamp check: capability question,
        // typed decline, cached negatively so a replay returns this exact reason.
        //
        // Every attachment is checked, not just slot 0 — the secondaries carry
        // their own decoded blend, and a pipeline is invalid if *any* attachment
        // names a `SRC1_*` factor without the feature.
        if !ctx.features.dual_src_blend {
            let uses_dual_source = std::iter::once(&key.blend)
                .chain(key.secondary_blend.iter())
                .flatten()
                .any(|b| {
                    [b.src_color, b.dst_color, b.src_alpha, b.dst_alpha]
                        .iter()
                        .any(|f| f.is_dual_source())
                });
            if uses_dual_source {
                let reason = super::reason::DrawReason::DualSourceBlendUnsupported;
                crate::observe::Emit::decline("vk_engine_pipeline", &reason).fail();
                let err = DrawError::Unsupported(reason);
                self.pipelines.insert_negative(key.clone(), err.clone());
                return Err(err);
            }
        }

        // `MTLTriangleFillModeLines` and `MTLDepthClipModeClamp` are the two
        // rasterization states whose non-default arm Vulkan makes optional:
        // `VK_POLYGON_MODE_LINE` needs `fillModeNonSolid` and
        // `depthClampEnable` needs `depthClamp`, and naming either without its
        // feature makes the pipeline invalid. Same shape as the two checks
        // above — capability question, typed decline, cached negatively.
        //
        // Refused rather than rasterized the other way, because the other way
        // is a whole pass rendered wrong with nothing to say so: a wireframe
        // filled in, or the geometry a clamped pass wanted kept discarded at
        // the near plane.
        if key.fill_mode != FillMode::default() && !ctx.features.fill_mode_non_solid {
            let reason = super::reason::DrawReason::FillModeNonSolidUnsupported;
            crate::observe::Emit::decline("vk_engine_pipeline", &reason).fail();
            let err = DrawError::Unsupported(reason);
            self.pipelines.insert_negative(key.clone(), err.clone());
            return Err(err);
        }
        if key.depth_clip != DepthClipMode::default() && !ctx.features.depth_clamp {
            let reason = super::reason::DrawReason::DepthClampUnsupported;
            crate::observe::Emit::decline("vk_engine_pipeline", &reason).fail();
            let err = DrawError::Unsupported(reason);
            self.pipelines.insert_negative(key.clone(), err.clone());
            return Err(err);
        }

        // Resolve every attribute against what this device accepts as a vertex
        // buffer format. Vulkan makes the three-component 8/16-bit formats
        // optional, so the format the guest decoded is not automatically
        // bindable; `translate::support` either confirms it, substitutes the
        // mandatory wider sibling, or declines by name.
        //
        // A substitution is only invisible to a shader that does not read the
        // component it oversupplies, so `resolve` asks what this shader
        // declares at the attribute's location. Walked at most once per
        // pipeline miss and only when some attribute really needs substituting:
        // on a host that accepts every format — every host this project has run
        // on — `vert_spirv` is never read at all.
        let mut shader_inputs: Option<VertexInputWidths> = None;
        let mut attribute_formats = Vec::with_capacity(key.attrs.len());
        for attr in &key.attrs {
            let binding =
                match ctx
                    .vertex_formats
                    .resolve(attr.format, attr.offset, attr.stride, || {
                        shader_inputs
                            .get_or_insert_with(|| VertexInputWidths::from_spirv(vert_spirv))
                            .at(attr.location)
                    }) {
                    Ok(binding) => binding,
                    Err(translate_reason) => {
                        let err = DrawError::Unsupported(super::reason::DrawReason::VertexFormat(
                            translate_reason,
                        ));
                        crate::observe::Emit::decline("vk_engine_vertex_format", &translate_reason)
                            .fail_once(
                                (u64::from(attr.location) << 32)
                                    | u64::from(translate_reason.value()),
                            );
                        self.pipelines.insert_negative(key.clone(), err.clone());
                        return Err(err);
                    }
                };
            if let Some(narrow) = binding.widened_from {
                // Fail-visible because a widened attribute is a device-specific
                // difference from what the guest asked for, even though
                // `resolve` has just established that no shader input can
                // observe it: without this line a substitution is invisible in
                // a bug report from a host nobody here owns.
                let decline = VertexFormatWidenDecline {
                    from: narrow,
                    to: binding.format,
                    location: attr.location,
                    offset: attr.offset,
                    stride: attr.stride,
                };
                crate::observe::Emit::decline("vk_engine_vertex_format", &decline).fail_once(
                    (u64::from(attr.location) << 32) | u64::from(narrow.as_raw() as u32),
                );
            }
            attribute_formats.push(binding.format);
            let divisor = match attr.step_function {
                VertexStepFunction::Constant => Some(0),
                VertexStepFunction::PerVertex => None,
                VertexStepFunction::PerInstance if attr.step_rate == 1 => None,
                VertexStepFunction::PerInstance => Some(attr.step_rate),
            };
            if divisor == Some(0) && !ctx.vertex_divisor.zero_divisor {
                let err =
                    DrawError::Unsupported(super::reason::DrawReason::ConstantVertexAttribute);
                self.pipelines.insert_negative(key.clone(), err.clone());
                return Err(err);
            }
            if divisor.is_some_and(|v| v > 1) {
                if !ctx.vertex_divisor.instance_rate_divisor {
                    let err = DrawError::Unsupported(
                        super::reason::DrawReason::InstanceRateDivisorUnsupported {
                            step_rate: attr.step_rate,
                        },
                    );
                    self.pipelines.insert_negative(key.clone(), err.clone());
                    return Err(err);
                }
                if attr.step_rate > ctx.vertex_divisor.max_divisor {
                    let err = DrawError::Unsupported(
                        super::reason::DrawReason::InstanceRateDivisorOverLimit {
                            step_rate: attr.step_rate,
                            limit: ctx.vertex_divisor.max_divisor,
                        },
                    );
                    self.pipelines.insert_negative(key.clone(), err.clone());
                    return Err(err);
                }
            }
        }

        let main_c = super::context::main_entry();
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(&main_c),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module)
                .name(&main_c),
        ];
        let vertex_binding_descs: Vec<_> = key
            .attrs
            .iter()
            .map(|attribute| {
                vk::VertexInputBindingDescription::default()
                    .binding(attribute.binding)
                    .stride(attribute.stride)
                    .input_rate(translate::vertex::vk_input_rate(attribute.step_function))
            })
            .collect();
        let vertex_binding_divisors: Vec<_> = key
            .attrs
            .iter()
            .filter_map(|attribute| {
                let divisor = match attribute.step_function {
                    VertexStepFunction::Constant => 0,
                    VertexStepFunction::PerVertex => return None,
                    VertexStepFunction::PerInstance if attribute.step_rate == 1 => return None,
                    VertexStepFunction::PerInstance => attribute.step_rate,
                };
                Some(
                    vk::VertexInputBindingDivisorDescriptionKHR::default()
                        .binding(attribute.binding)
                        .divisor(divisor),
                )
            })
            .collect();
        let vertex_attribute_descs: Vec<_> = key
            .attrs
            .iter()
            .zip(&attribute_formats)
            .map(|(attribute, format)| {
                vk::VertexInputAttributeDescription::default()
                    .location(attribute.location)
                    .binding(attribute.binding)
                    // The device-resolved format, which equals the attribute's
                    // own on every host seen so far and its mandatory wider
                    // sibling where the device declined the optional one.
                    .format(*format)
                    .offset(attribute.offset)
            })
            .collect();
        let mut vertex_divisor_state = vk::PipelineVertexInputDivisorStateCreateInfoKHR::default()
            .vertex_binding_divisors(&vertex_binding_divisors);
        let mut vtx_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&vertex_binding_descs)
            .vertex_attribute_descriptions(&vertex_attribute_descs);
        if !vertex_binding_divisors.is_empty() {
            vtx_input = vtx_input.push_next(&mut vertex_divisor_state);
        }
        let input_asm =
            vk::PipelineInputAssemblyStateCreateInfo::default().topology(key.topology.vk());
        // Dynamic viewport/scissor so L5 key need not include extent (flip flag is static).
        // Stencil reference is dynamic (Metal's `SetStencilReferenceValue` is a
        // command distinct from the state object) so distinct references reuse
        // one pipeline; only listed for stencil pipelines.
        let mut dynamic_states = vec![vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        if key.stencil.is_some() {
            dynamic_states.push(vk::DynamicState::STENCIL_REFERENCE);
        }
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        // Both counts are the key's one number: the viewports and scissors
        // themselves are dynamic, but how many of them there are is not, and
        // Vulkan requires the two counts to be equal.
        let vp_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(key.viewport_slots)
            .scissor_count(key.viewport_slots);
        // Cull mode, winding, fill mode and depth clip mode all come from the
        // guest; the last two were refused above where the host cannot spell
        // them, so reaching here means both are bindable.
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(translate::raster::vk_polygon_mode(key.fill_mode))
            .depth_clamp_enable(translate::raster::vk_depth_clamp_enable(key.depth_clip))
            .cull_mode(translate::raster::vk_cull_mode(key.cull_mode))
            .front_face(translate::raster::vk_front_face(key.front_face_ccw))
            .line_width(1.0);
        // Pinned rather than unknown, and every render target this backend
        // allocates is single-sampled, so honouring a count would need the
        // attachment path to carry one too.
        //
        // `rasterSampleCount` is a property of `MTLRenderPipelineDescriptor`,
        // so it reaches this device inside the type-7 pipeline's own
        // compact-TLV block, which is the *only* route to it: the render-pass
        // attachment record on the wire carries a resolve ref and no count, and
        // the texture objects are met through the kernel's object list, whose
        // descriptor has no such field either. That tag is now read —
        // `PIPELINE_TAG_RASTER_SAMPLE_COUNT` — and a count this line cannot
        // meet is named as `pipeline_raster_sample_count_degraded` rather than
        // defaulted in silence. So the demand for multisampled attachments is
        // now measurable, which is what widening this would need first.
        //
        // A pass that states a sample count *without* an attachment carrying
        // one — `defaultRasterSampleCount` — is refused rather than rasterized
        // here, as `StreamDrawDrop::PassRasterSampleCountUnsupported`.
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        // One blend attachment state per color attachment; Vulkan requires the
        // count to match the render pass. Every slot uses its own decoded
        // blend, slot 0 from `key.blend` and slot n from
        // `key.secondary_blend[n-1]`.
        //
        // The secondaries used to be forced unblended here, justified by a
        // comment saying the decode side did not carry per-attachment blend
        // state. It did, and had all along — the Metal arm reads exactly these
        // fields per slot. Only this key collapsed them, so a guest MRT
        // pipeline that asked to blend slot 1 silently got a raw store.
        //
        // The colour write mask comes from the guest too, and it is applied on
        // both arms because `MTLColorWriteMask` is independent of
        // `blendingEnabled` — an unblended masked attachment still leaves its
        // unwritten channels alone. Metal's bits are alpha-first and Vulkan's
        // are red-first, so the exchange goes through `vk_color_write_mask`
        // rather than a cast.
        let attachment_blend = |blend: Option<BlendKey>, mask: ColorWriteMask| {
            let write = translate::blend::vk_color_write_mask(mask);
            match blend {
                Some(b) => vk::PipelineColorBlendAttachmentState::default()
                    .color_write_mask(write)
                    .blend_enable(true)
                    .src_color_blend_factor(b.src_color.vk())
                    .dst_color_blend_factor(b.dst_color.vk())
                    .color_blend_op(b.color_op.vk())
                    .src_alpha_blend_factor(b.src_alpha.vk())
                    .dst_alpha_blend_factor(b.dst_alpha.vk())
                    .alpha_blend_op(b.alpha_op.vk()),
                None => vk::PipelineColorBlendAttachmentState::default()
                    .color_write_mask(write)
                    .blend_enable(false),
            }
        };
        let mut blend_att = vec![attachment_blend(key.blend, key.color_write_mask[0])];
        for slot in 0..key.pass.secondary_count as usize {
            blend_att.push(attachment_blend(
                key.secondary_blend[slot],
                key.color_write_mask[slot + 1],
            ));
        }
        let blend_constants = key
            .blend
            .map(|b| b.constants.map(f32::from_bits))
            .unwrap_or([0.0; 4]);
        let blend = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(&blend_att)
            .blend_constants(blend_constants);
        // Depth-stencil state: attached ONLY when the pass carries a depth
        // attachment (Vulkan requires the pipeline's depth-stencil state to be
        // consistent with the subpass). Without it the color-only pipeline is
        // byte-identical to the pre-depth engine. Stencil is enabled only when
        // the bound state requested it (`key.stencil`); the reference field is
        // left 0 here and supplied dynamically per draw.
        let stencil_face = |ops: super::types::StencilFaceOps| {
            vk::StencilOpState::default()
                .fail_op(ops.fail_op.vk())
                .pass_op(ops.pass_op.vk())
                .depth_fail_op(ops.depth_fail_op.vk())
                .compare_op(ops.compare.vk())
                .compare_mask(ops.read_mask)
                .write_mask(ops.write_mask)
                .reference(0)
        };
        let mut depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(key.depth_test)
            .depth_write_enable(key.depth_write)
            .depth_compare_op(key.depth_compare.vk())
            .depth_bounds_test_enable(false)
            .stencil_test_enable(key.stencil.is_some());
        if let Some(s) = key.stencil {
            depth_stencil = depth_stencil
                .front(stencil_face(s.front))
                .back(stencil_face(s.back));
        }
        let mut gpci = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vtx_input)
            .input_assembly_state(&input_asm)
            .viewport_state(&vp_state)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .color_blend_state(&blend)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);
        if key.pass.depth.is_some() {
            gpci = gpci.depth_stencil_state(&depth_stencil);
        }
        let created = ctx
            .device
            .create_graphics_pipelines(ctx.pipeline_cache, &[gpci], None);
        let pipe = created.map_err(|(_, e)| {
            let err = DrawError::VkCall(VkCall::new(VkOp::CachesCreateGraphicsPipelines, e));
            self.pipelines.insert_negative(key.clone(), err.clone());
            err
        })?[0];
        counters.note_create();
        // A fresh pipeline compile grew the VkPipelineCache — persist it so
        // the next boot warm-starts (file write is off-thread, debounced).
        ctx.persist_pipeline_cache();
        if let Some(old) = self.pipelines.insert(key.clone(), pipe) {
            pools.dispose(&ctx.device, DeferredHandle::Pipeline(old));
        }
        Ok(pipe)
    }

    pub(crate) unsafe fn get_or_create_compute_pipeline(
        &mut self,
        ctx: &DeviceContext,
        key: &ComputePipelineKey,
        module: vk::ShaderModule,
        pipeline_layout: vk::PipelineLayout,
        counters: &EngineCounters,
        pools: &mut ResourcePools,
    ) -> Result<vk::Pipeline, DrawError> {
        if let Some(err) = self.compute_pipelines.get_negative(key) {
            counters
                .compute_pipeline_misses
                .fetch_add(1, Ordering::Relaxed);
            return Err(err);
        }
        if let Some(&p) = self.compute_pipelines.get(key) {
            counters
                .compute_pipeline_hits
                .fetch_add(1, Ordering::Relaxed);
            return Ok(p);
        }
        counters
            .compute_pipeline_misses
            .fetch_add(1, Ordering::Relaxed);
        let entry_c = std::ffi::CString::new(key.entry.as_str()).map_err(|_| {
            DrawError::ComputeValidation(
                super::compute_validation::ComputeValidationDecline::EntryInteriorNul,
            )
        })?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(&entry_c);
        let cpci = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout);
        let pipe = ctx
            .device
            .create_compute_pipelines(ctx.pipeline_cache, &[cpci], None)
            .map_err(|(_, e)| {
                let err = DrawError::VkCall(VkCall::new(VkOp::CachesCreateComputePipelines, e));
                self.compute_pipelines
                    .insert_negative(key.clone(), err.clone());
                err
            })?[0];
        counters.note_create();
        // Same warm-start persistence as the graphics path.
        ctx.persist_pipeline_cache();
        if let Some(old) = self.compute_pipelines.insert(key.clone(), pipe) {
            pools.dispose(&ctx.device, DeferredHandle::Pipeline(old));
        }
        Ok(pipe)
    }
}

#[cfg(test)]
mod object_cache_tests {
    use super::*;

    #[test]
    fn vertex_format_widening_names_both_formats_and_attribute() {
        use crate::observe::Decline as _;
        let narrow = translate::vertex::vertex_layout(VertexAttributeFormat::UChar3Normalized).vk;
        let binding = translate::VertexFormatSupport::with_unsupported(&[narrow])
            .resolve(VertexAttributeFormat::UChar3Normalized, 12, 32, || {
                crate::runtime::spirv_vertex_input::InputWidth::Components(3)
            })
            .unwrap();
        let decline = VertexFormatWidenDecline {
            from: binding.widened_from.unwrap(),
            to: binding.format,
            location: 3,
            offset: 12,
            stride: 32,
        };
        assert_eq!(decline.slug(), "vk_vertex_format_widened");
        assert_eq!(
            crate::observe::Emit::decline("vk_engine_vertex_format", &decline).render(),
            "vk_engine_vertex_format reason=vk_vertex_format_widened \
             from=R8G8B8_UNORM to=R8G8B8A8_UNORM location=3 offset=12 stride=32"
        );
    }

    #[test]
    fn negative_map_is_bounded_by_cap() {
        // Negative entries (create failures) must not grow without bound: a
        // guest submitting endless distinct never-creatable objects would
        // otherwise leak one entry per distinct key forever.
        let mut c: ObjectCache<u32, u32> = ObjectCache::with_negative_cap(4);
        for k in 0..100u32 {
            c.insert_negative(
                k,
                DrawError::VkCall(VkCall::new(
                    VkOp::CachesCreateShaderModule,
                    vk::Result::ERROR_UNKNOWN,
                )),
            );
        }
        assert_eq!(c.negative.len(), 4, "negative map bounded by cap");
        assert!(
            c.negative_order.len() <= 8,
            "order deque bounded (<= 2*cap): {}",
            c.negative_order.len()
        );
        // The newest 4 keys survive (oldest-first eviction).
        for k in 96..100u32 {
            assert!(c.get_negative(&k).is_some(), "recent negative {k} retained");
        }
        assert!(c.get_negative(&0).is_none(), "oldest negative evicted");
    }

    /// The first pipeline a boot creates is the compositor's, and it stays bound
    /// for the life of the guest. Under the retired insertion-order cap it was
    /// also the first thing a cap crossing threw away. Drive far past every cap
    /// this file used to carry (1024, and 64 for render passes) and assert the
    /// first key is still served and nothing was displaced for capacity.
    #[test]
    fn the_first_key_survives_far_past_every_retired_capacity() {
        let mut c: ObjectCache<u32, u32> = ObjectCache::new();
        c.insert(0, 0xC0FFEE);
        for k in 1..4096u32 {
            assert!(
                c.insert(k, k).is_none(),
                "a fresh key displaces nothing: {k}"
            );
        }
        assert_eq!(
            c.get(&0),
            Some(&0xC0FFEE),
            "the hot first entry is still served after 4095 later ones"
        );
        assert_eq!(c.map.len(), 4096, "every distinct key retained");
    }

    /// A replace hands the displaced handle back so the caller can destroy it.
    /// The retired implementation overwrote in place and returned `None`, which
    /// leaked the Vulkan object it had just dropped the last reference to.
    #[test]
    fn replacing_a_key_returns_the_displaced_value_to_destroy() {
        let mut c: ObjectCache<u32, u32> = ObjectCache::new();
        assert_eq!(c.insert(1, 10), None);
        assert_eq!(
            c.insert(1, 20),
            Some(10),
            "the displaced handle comes back for disposal"
        );
        assert_eq!(c.get(&1), Some(&20));
    }

    #[test]
    fn positive_insert_clears_negative_for_the_key() {
        // A key that failed then later succeeds must not keep serving the stale
        // negative error.
        let mut c: ObjectCache<u32, u32> = ObjectCache::with_negative_cap(4);
        c.insert_negative(
            7,
            DrawError::VkCall(VkCall::new(
                VkOp::CachesCreateShaderModule,
                vk::Result::ERROR_UNKNOWN,
            )),
        );
        assert!(c.get_negative(&7).is_some());
        c.insert(7, 42);
        assert!(c.get_negative(&7).is_none(), "promotion clears negative");
        assert_eq!(c.get(&7), Some(&42));
    }

    #[test]
    fn reinserting_same_negative_does_not_duplicate_order() {
        // Both results here are inherent to the request, so both are remembered.
        // They used to be the two out-of-memory results, which no longer reach
        // the map at all — see `an_out_of_memory_refusal_is_never_remembered`.
        let mut c: ObjectCache<u32, u32> = ObjectCache::with_negative_cap(4);
        let a = DrawError::VkCall(VkCall::new(
            VkOp::CachesCreateShaderModule,
            vk::Result::ERROR_UNKNOWN,
        ));
        let b = DrawError::VkCall(VkCall::new(
            VkOp::CachesCreateShaderModule,
            vk::Result::ERROR_INITIALIZATION_FAILED,
        ));
        c.insert_negative(1, a);
        c.insert_negative(1, b.clone());
        assert_eq!(c.negative_order.len(), 1, "same key tracked once");
        assert_eq!(c.get_negative(&1), Some(b), "error refreshed");
    }

    /// Out of memory says what the device holds *now*. The lookup consults
    /// `negative` before the create, so a remembered one is never displaced by a
    /// later success — the create that would displace it never runs. Remembering
    /// it turns "refused while full" into "refused forever", which is the failure
    /// mode a real GPU does not have.
    #[test]
    fn an_out_of_memory_refusal_is_never_remembered() {
        for result in [
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
            vk::Result::ERROR_OUT_OF_HOST_MEMORY,
        ] {
            let mut c: ObjectCache<u32, u32> = ObjectCache::with_negative_cap(4);
            let err = DrawError::VkCall(VkCall::new(VkOp::CachesCreateGraphicsPipelines, result));
            assert!(err.out_of_memory(), "{result:?} is the retryable class");
            c.insert_negative(1, err);
            assert_eq!(
                c.get_negative(&1),
                None,
                "{result:?} must not short-circuit the next create"
            );
            assert!(c.negative.is_empty(), "{result:?} left no entry");
            assert!(c.negative_order.is_empty(), "{result:?} left no order slot");
        }
    }

    /// The converse, so the test above cannot pass by disabling the map. A
    /// refusal inherent to the request — malformed SPIR-V, or a capability this
    /// host does not have — is worth remembering, because a second identical
    /// attempt meets it again.
    #[test]
    fn a_refusal_inherent_to_the_request_is_still_remembered() {
        let mut c: ObjectCache<u32, u32> = ObjectCache::with_negative_cap(4);

        let bad_shader = DrawError::VkCall(VkCall::new(
            VkOp::CachesCreateShaderModule,
            vk::Result::ERROR_INVALID_SHADER_NV,
        ));
        assert!(!bad_shader.out_of_memory());
        c.insert_negative(1, bad_shader.clone());
        assert_eq!(c.get_negative(&1), Some(bad_shader));

        let unsupported = DrawError::Unsupported(
            super::super::reason::DrawReason::InstanceRateDivisorUnsupported { step_rate: 3 },
        );
        assert!(!unsupported.out_of_memory());
        c.insert_negative(2, unsupported.clone());
        assert_eq!(c.get_negative(&2), Some(unsupported));
    }

    /// A pipeline the guest still wants is asked for again after the memory it
    /// needed came back. This is the whole point of the rule, written as the
    /// sequence a guest actually produces: create fails while an atlas is
    /// resident, the guest frees the atlas, the guest re-binds the same
    /// pipeline. Before the rule, step three replayed a stale error and the
    /// driver was never asked.
    #[test]
    fn a_key_that_ran_out_of_memory_is_created_on_the_next_ask() {
        let mut c: ObjectCache<u32, u32> = ObjectCache::with_negative_cap(4);
        let key = 0xB0BAu32;

        // Frame N: the create refuses because the device is full.
        c.insert_negative(
            key,
            DrawError::VkCall(VkCall::new(
                VkOp::CachesCreateGraphicsPipelines,
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
            )),
        );

        // Frame N+1: the guest asks again. Nothing short-circuits it, so the
        // caller reaches its create.
        assert_eq!(
            c.get_negative(&key),
            None,
            "the second ask must reach the driver"
        );
        assert_eq!(c.get(&key), None, "and it is still a miss, not a stale hit");

        // The memory came back, so this time the create succeeds.
        c.insert(key, 0x5EED);
        assert_eq!(c.get(&key), Some(&0x5EED));
    }
}
