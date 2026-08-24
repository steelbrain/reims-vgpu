//! L2–L7 immutable object caches (content/descriptor keyed, negative + hit/miss).

#![allow(unsafe_op_in_unsafe_fn)]

use ash::vk;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::Ordering;

// ash Handle trait not required here.

use super::context::DeviceContext;
use super::counters::{CreateSite, EngineCounters};
use super::digest::Digest128;
use super::pools::{DeferredHandle, RecordingPools};
use super::types::{
    BlendKey, ColorWriteMask, CullMode, DepthClipMode, DrawError, FillMode, PrimitiveTopology,
    SamplerStateKey, VertexAttributeFormat, VertexStepFunction,
};
use super::vk_call::{VkCall, VkOp};

pub(crate) fn vk_sample_count(count: u32) -> vk::SampleCountFlags {
    match count {
        2 => vk::SampleCountFlags::TYPE_2,
        4 => vk::SampleCountFlags::TYPE_4,
        8 => vk::SampleCountFlags::TYPE_8,
        16 => vk::SampleCountFlags::TYPE_16,
        32 => vk::SampleCountFlags::TYPE_32,
        64 => vk::SampleCountFlags::TYPE_64,
        _ => vk::SampleCountFlags::TYPE_1,
    }
}

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

impl reims_vgpu_observe::Decline for VertexFormatWidenDecline {
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
use crate::spirv_vertex_input::VertexInputWidths;
use crate::translate;

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
    pub count: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) struct LayoutKey {
    pub bindings: Vec<BindingSig>,
    /// The kernel-grid push-constant range this layout must expose, when the
    /// stage using it is a compute kernel that culls its surplus invocations.
    ///
    /// Part of the key and not a side input: two stages with identical bindings
    /// and different push-constant ranges are different pipeline layouts, and
    /// a graphics stage shares this cache. Leaving it out would let a render
    /// layout satisfy a kernel's lookup and hand back a layout with no range,
    /// where the push is then invalid usage and the cull reads whatever the
    /// driver leaves behind.
    pub kernel_grid: Option<super::super::m2v_cache::KernelGridRange>,
}

impl LayoutKey {
    /// Whether this layout is represented by command-buffer-local push
    /// descriptors on this device. This same answer creates the layout and
    /// chooses how every consumer writes it.
    pub(crate) fn uses_push_descriptors(
        &self,
        caps: crate::push_descriptor::PushDescriptorCaps,
    ) -> bool {
        !self.bindings.is_empty() && caps.supports_counts(self.bindings.iter().map(|b| b.count))
    }
}

pub(crate) fn canonicalize_layout_bindings(
    mut bindings: Vec<BindingSig>,
) -> Result<Vec<BindingSig>, super::DrawError> {
    bindings.sort_by_key(|binding| binding.binding);
    for pair in bindings.windows(2) {
        if pair[0].binding == pair[1].binding && pair[0] != pair[1] {
            return Err(super::DrawError::Unsupported(
                super::reason::DrawReason::DescriptorBindingConflict {
                    binding: pair[0].binding,
                    first_type: pair[0].ty,
                    first_count: pair[0].count,
                    second_type: pair[1].ty,
                    second_count: pair[1].count,
                },
            ));
        }
    }
    bindings.dedup();
    Ok(bindings)
}

/// Max secondary color attachments (MRT slot 1..): every colour slot Apple's
/// serialized render pass can carry, less the primary at slot 0.
///
/// The fourth spelling of one number, and the last one to be pinned. The wire
/// record's colour-slot array is the truth,
/// [`reims_vgpu_protocol::MAX_COLOR_ATTACHMENTS`] derives from
/// by an assertion beside itself. This one is that bound minus one.
///
/// A drift here is refused rather than lost: `execute_draw_inner` returns
/// [`super::reason::DrawReason::SecondaryAttachmentCap`] for a request past this
/// count, so a shortfall costs the whole draw and says so. That makes the
/// failure loud and still wrong — a guest sending the eighth colour slot the
/// wire format allows would have every MRT draw refused — which is what this
/// assertion is for.
pub(crate) const MAX_SECONDARY_ATTACH: usize = 7;
const _: () = assert!(1 + MAX_SECONDARY_ATTACH == reims_vgpu_protocol::MAX_COLOR_ATTACHMENTS);
const _: () = assert!(MAX_SECONDARY_ATTACH < u8::BITS as usize);

/// Vulkan color-attachment load operation selected from the semantic request.
/// Keeping all three values in the pass key prevents `DontCare` from being
/// silently converted into a clear.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub(crate) enum ColorLoadKey {
    Load,
    #[default]
    Clear,
    DontCare,
}

impl ColorLoadKey {
    fn vulkan(
        self,
        final_layout: ash::vk::ImageLayout,
    ) -> (ash::vk::AttachmentLoadOp, ash::vk::ImageLayout) {
        match self {
            Self::Load => (ash::vk::AttachmentLoadOp::LOAD, final_layout),
            Self::Clear => (
                ash::vk::AttachmentLoadOp::CLEAR,
                ash::vk::ImageLayout::UNDEFINED,
            ),
            Self::DontCare => (
                ash::vk::AttachmentLoadOp::DONT_CARE,
                ash::vk::ImageLayout::UNDEFINED,
            ),
        }
    }
}

/// A secondary MRT attachment's contribution to the render-pass / pipeline key.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
pub(crate) struct SecondaryAttachKey {
    pub format: ash::vk::Format,
    pub load: ColorLoadKey,
}

/// The depth/stencil slot's contribution to the render-pass key. `None` on
/// `PassKey` means neither aspect exists. Depth-only uses D32_SFLOAT; when
/// `stencil` is set the slot uses the device-queried combined depth-stencil
/// format (`DeviceContext::depth_stencil_format`) with a live STENCIL aspect
/// (load/store), so it must partition the pass cache. `depth` independently
/// distinguishes a stencil-only Metal attachment from a combined one.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub(crate) struct DepthAttachKey {
    /// Whether the Metal pass bound a depth aspect. A stencil-only attachment
    /// uses the same Vulkan depth/stencil slot with depth load/store disabled.
    pub depth: bool,
    pub load: ColorLoadKey,
    pub store: bool,
    /// true = combined depth-stencil native attachment. This includes both a
    /// pass-owned stencil aspect and Metal's implicit stencil value when state
    /// enables testing without such an aspect.
    pub stencil: bool,
    pub stencil_load: ColorLoadKey,
    pub stencil_store: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct PassKey {
    pub color0_load: ColorLoadKey,
    /// Slot-0 attachment format, as a format rather than a channel-order flag.
    ///
    /// This used to be `bgra: bool`, meaning `B8G8R8A8_UNORM` or
    /// `R8G8B8A8_UNORM` and nothing else, which made slot 0 the only attachment
    /// in this key that could not name a format — [`SecondaryAttachKey`] has
    /// carried a real [`ash::vk::Format`] since MRT landed. The asymmetry was
    /// not cosmetic: it is the reason a render target's resident is always
    /// eight bits per channel whatever the guest declared, because the *only*
    /// thing downstream could reconstruct from the flag was one of those two.
    ///
    /// It must stay part of the key. A render pass and a pipeline are both
    /// compiled against the attachment's format, so two draws differing only
    /// here need two of each; a key that omitted it would hand the second draw
    /// a pipeline built for the first one's format.
    pub color0_format: ash::vk::Format,
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
    /// Bit N says colour attachment N is also sampled through
    /// `VK_EXT_attachment_feedback_loop_layout`. The decoded render pass has at
    /// most eight colour attachments, so the wire-derived attachment table is
    /// the bound and one byte carries the whole set.
    pub feedback_colors: u8,
    /// Sample count of the colour attachment pipelines rasterize into.
    pub sample_count: u32,
    /// Attachment zero resolves into a single-sample attachment appended
    /// immediately after it.
    pub multisample_resolve: bool,
}

impl PassKey {
    /// Single-color-attachment pass (the pre-MRT constructor).
    pub(crate) fn single(color0_load: ColorLoadKey, color0_format: ash::vk::Format) -> Self {
        Self {
            color0_load,
            color0_format,
            secondary: [SecondaryAttachKey::default(); MAX_SECONDARY_ATTACH],
            secondary_count: 0,
            depth: None,
            color_input: false,
            feedback_colors: 0,
            sample_count: 1,
            multisample_resolve: false,
        }
    }

    /// Project a compound pass onto the cached framebuffer owned by its
    /// primary attachment.
    ///
    /// The framebuffer contains only attachment zero, but that attachment is
    /// still the multisampled image used by the compound pass. Starting again
    /// from [`Self::single`] would silently reset its sample count to one and
    /// create a framebuffer incompatible with the image it names.
    pub(crate) fn primary_attachment_only(self) -> Self {
        let mut primary = Self::single(self.color0_load, self.color0_format);
        primary.feedback_colors = self.feedback_colors & 1;
        primary.sample_count = self.sample_count;
        primary
    }

    pub(crate) fn color_feedback(self, index: usize) -> bool {
        index < u8::BITS as usize && self.feedback_colors & (1u8 << index) != 0
    }

    /// The part of a render pass that Vulkan requires to agree for pipeline,
    /// framebuffer, and in-instance compatibility.
    ///
    /// Load actions describe how a newly begun pass obtains attachment
    /// contents; they are deliberately excluded. A serialized render encoder
    /// rewrites a continuation segment to LOAD, but an uninterrupted segment
    /// remains inside the pass begun with the encoder's original action. Store
    /// actions and initial/final layouts are functions of these same fields in
    /// this backend, so there is no second compatibility spelling to normalize.
    ///
    /// `feedback_colors` is erased **exactly when it changes nothing about the
    /// pass** — that is, when [`color_feedback_layout`] and
    /// [`color0_pass_exit_layout`] are the same layout. The self-dependency is
    /// declared on every pass, so once the layouts also coincide a feedback draw
    /// and an ordinary one want a byte-identical `VkRenderPass` and there is
    /// nothing left to keep them apart. Which is the point: whether a draw samples
    /// the target it is writing is a property of the *draw*, exactly as it is in
    /// Metal, and it stops closing the render pass.
    ///
    /// The condition is not decoration. Under [`reims_vgpu_config::COLOR_GENERAL`]`=off`
    /// the resting layout admits no feedback loop, the feedback slots really are
    /// in a different layout, and erasing the field there would merge two draws
    /// whose attachment is in two different layouts — a pass naming a layout its
    /// image is not in.
    pub(crate) fn compatibility(self) -> PassCompatibilityKey {
        let mut key = self;
        key.color0_load = ColorLoadKey::Clear;
        for secondary in &mut key.secondary {
            secondary.load = ColorLoadKey::Clear;
        }
        if let Some(depth) = &mut key.depth {
            depth.load = ColorLoadKey::Clear;
            depth.store = false;
            depth.stencil_load = ColorLoadKey::Clear;
            depth.stencil_store = false;
        }
        if color_feedback_layout() == color0_pass_exit_layout() {
            key.feedback_colors = 0;
        }
        PassCompatibilityKey(key)
    }

    /// The render-pass state Vulkan uses to decide framebuffer compatibility.
    ///
    /// Attachment load actions, attachment-reference layouts, and subpass
    /// dependencies do not participate. Host accessibility changes only the
    /// primary attachment's layouts and external dependency; feedback changes
    /// layouts and a dependency. Neither requires a framebuffer to be rebuilt.
    /// Attachment formats and the subpass attachment-reference shape remain in
    /// the key.
    pub(crate) fn framebuffer_compatibility(self) -> FramebufferCompatibilityKey {
        let mut key = self.compatibility().0;
        key.feedback_colors = 0;
        FramebufferCompatibilityKey(key)
    }

    pub(crate) fn color_layout(self, index: usize) -> vk::ImageLayout {
        if self.color_feedback(index) {
            color_feedback_layout()
        } else if index == 0 && self.color_input {
            vk::ImageLayout::GENERAL
        } else {
            // The same layout the pass exits at, so an ordinary pass performs no
            // transition of its own at either end. Written as the exit rather
            // than as `COLOR_ATTACHMENT_OPTIMAL` for the reason
            // `color0_pass_exit_layout` gives: there is one spelling.
            color0_pass_exit_layout()
        }
    }

    pub(crate) fn color_final_layout(self, index: usize) -> vk::ImageLayout {
        if self.color_feedback(index) {
            color_feedback_layout()
        } else {
            color0_pass_exit_layout()
        }
    }
}

/// The layout a colour attachment a draw also samples is placed in.
///
/// The resting layout itself whenever that admits a feedback loop, which is the
/// shipping arm — so a slot the guest samples and a slot it does not are in the
/// **same** layout, the pass declares no transition for either, and there is no
/// second layout for a colour target anywhere in this device. Only the ablation
/// arm, where the resting layout is `COLOR_ATTACHMENT_OPTIMAL` and admits
/// nothing, reaches the extension's dedicated layout.
///
/// One function because four places name this: the subpass attachment reference,
/// the `finalLayout`, the sampled descriptor
/// (`super::exec::PreparedSampled::descriptor_layout`) and the registry record
/// the pass leaves behind. A descriptor naming a layout the attachment reference
/// does not is undefined behaviour, and it is not an error anywhere.
pub(crate) fn color_feedback_layout() -> vk::ImageLayout {
    let resting = color0_pass_exit_layout();
    if layout_admits_color_feedback(resting) {
        resting
    } else {
        vk::ImageLayout::ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT
    }
}

/// Whether a colour attachment resting in `layout` may also be sampled by a draw
/// inside the same render pass instance — a Vulkan *feedback loop*.
///
/// Two layouts admit one. `ATTACHMENT_FEEDBACK_LOOP_OPTIMAL` is the dedicated
/// spelling `VK_EXT_attachment_feedback_loop_layout` adds, and `GENERAL` is the
/// core one: a sampled-image descriptor may name `GENERAL`, and the core rule for
/// an attachment written earlier in the subpass permits it to be accessed "as an
/// attachment, storage image, or sampled image" by a later command. So the
/// extension layout is an *optimisation over* `GENERAL`, never a requirement.
///
/// This is the whole reason feedback is not a second layout. While
/// [`color0_pass_exit_layout`] answers `GENERAL`, a slot the guest samples and a
/// slot it does not are in the same layout, so the render pass declares no
/// transition for either and the registry's record is true for both.
///
/// It matters that this is a question about the layout and not a switch read.
/// Under [`reims_vgpu_config::COLOR_GENERAL`]`=off` the resting layout is
/// `COLOR_ATTACHMENT_OPTIMAL`, which admits no feedback loop at all, and the
/// extension layout has to come back — naming the resting layout there would be a
/// sampled read of an attachment in a layout that forbids it, which is undefined
/// behaviour rather than an error. The ablation therefore restores two layouts
/// because the contract says it must, not because a flag says so.
const fn layout_admits_color_feedback(layout: vk::ImageLayout) -> bool {
    matches!(
        layout,
        vk::ImageLayout::GENERAL | vk::ImageLayout::ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT
    )
}

/// A normalized [`PassKey`] containing exactly Vulkan render-pass
/// compatibility state. Construction is private to [`PassKey::compatibility`]
/// so a load action cannot accidentally enter a pipeline or framebuffer key.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct PassCompatibilityKey(PassKey);

/// The subset of [`PassKey`] that Vulkan requires to agree when a framebuffer
/// created against one render pass is used with another.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct FramebufferCompatibilityKey(PassKey);

impl PassCompatibilityKey {
    pub(crate) fn secondary_count(self) -> usize {
        self.0.secondary_count as usize
    }

    pub(crate) fn has_depth(self) -> bool {
        self.0.depth.is_some()
    }

    /// Which field makes two compatibility keys disagree, or `None` when they
    /// are equal.
    ///
    /// A `passdiff_compat` firing says a draw could not continue its
    /// predecessor's render pass because Vulkan would not call the two passes
    /// compatible, and on a driven Maps leg that is the dominant merge blocker
    /// once the framebuffer identity one is fixed. On its own it names no
    /// repair: this key carries nine independent things and a change in any of
    /// them lands in the same bucket. A colour format change is the guest
    /// drawing into a different target and is not repairable at all; a
    /// `sample_count` or `feedback_colors` change might be this device's own
    /// bookkeeping.
    ///
    /// The order is arbitrary — unlike [`super::pools::PassEchoField`]'s, where
    /// an earlier field makes a later one unreachable — so an answer here is
    /// *a* difference and not the only one. That is what the census needs: the
    /// question is which field to look at first, and any field that ever
    /// differs is worth a reading.
    ///
    /// The destructure is exhaustive on purpose. A tenth field added to
    /// [`PassKey`] fails this function to compile rather than joining a bucket
    /// that silently stops being a partition.
    pub(crate) fn first_difference(self, other: Self) -> Option<PassCompatField> {
        let PassKey {
            // Load actions are erased by `PassKey::compatibility`, so they are
            // equal here by construction and cannot be a difference.
            color0_load: _,
            color0_format,
            secondary,
            secondary_count,
            depth,
            color_input,
            feedback_colors,
            sample_count,
            multisample_resolve,
        } = self.0;
        let them = other.0;
        if color0_format != them.color0_format {
            return Some(PassCompatField::Color0Format);
        }
        if secondary_count != them.secondary_count {
            return Some(PassCompatField::SecondaryCount);
        }
        if secondary != them.secondary {
            return Some(PassCompatField::SecondaryFormat);
        }
        if depth != them.depth {
            return Some(PassCompatField::Depth);
        }
        if color_input != them.color_input {
            return Some(PassCompatField::ColorInput);
        }
        if feedback_colors != them.feedback_colors {
            return Some(PassCompatField::FeedbackColors);
        }
        if sample_count != them.sample_count {
            return Some(PassCompatField::SampleCount);
        }
        if multisample_resolve != them.multisample_resolve {
            return Some(PassCompatField::MultisampleResolve);
        }
        None
    }
}

/// Which field of a [`PassCompatibilityKey`] two draws disagreed about.
///
/// See [`PassCompatibilityKey::first_difference`] for why the split exists and
/// why the order carries no meaning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PassCompatField {
    Color0Format,
    SecondaryCount,
    SecondaryFormat,
    Depth,
    ColorInput,
    FeedbackColors,
    SampleCount,
    MultisampleResolve,
}

impl PassCompatField {
    pub(crate) fn route(self) -> &'static str {
        match self {
            Self::Color0Format => "passcompat_color0_format",
            Self::SecondaryCount => "passcompat_secondary_count",
            Self::SecondaryFormat => "passcompat_secondary_format",
            Self::Depth => "passcompat_depth",
            Self::ColorInput => "passcompat_color_input",
            Self::FeedbackColors => "passcompat_feedback",
            Self::SampleCount => "passcompat_sample_count",
            Self::MultisampleResolve => "passcompat_ms_resolve",
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
    pub pass: PassCompatibilityKey,
    /// Which colour attachments this draw samples while writing them.
    ///
    /// Pipeline state, not pass state. `VK_PIPELINE_CREATE_COLOR_ATTACHMENT_-
    /// FEEDBACK_LOOP_BIT_EXT` is what "feedback loop is enabled" means for the
    /// draw-time rules, and it is fixed at pipeline creation — so a feedback draw
    /// and an ordinary one need two pipelines even when they share a render pass.
    ///
    /// It lives here rather than being read back out of [`Self::pass`] because
    /// [`PassKey::compatibility`] erases it precisely when the render pass stops
    /// depending on it, which is the shipping arm. Reading it from there would
    /// silently drop the create flag off every feedback pipeline, and the result
    /// is a draw sampling an attachment it is writing with no feedback loop
    /// enabled — undefined behaviour, reported nowhere.
    pub feedback_colors: u8,
    /// Face culling. `None` (the 2D UI default) keeps the raster state at
    /// `CULL_NONE`, byte-identical to the pre-cull engine; the key still
    /// participates in hashing so a later culled draw with the same shaders gets
    /// its own pipeline rather than aliasing the no-cull one.
    pub cull_mode: CullMode,
    /// Metal front-facing winding (`true` = counter-clockwise), mapped to a
    /// Vulkan `FrontFace` by [`crate::translate::raster::vk_front_face`].
    pub front_face_ccw: bool,
    /// Metal `MTLTriangleFillMode`, mapped to a `VkPolygonMode`. In the key
    /// because Vulkan has no dynamic polygon mode below
    /// `VK_EXT_extended_dynamic_state3`: a wireframe draw and a filled draw
    /// sharing shaders need different pipelines.
    pub fill_mode: FillMode,
    /// Effective Vulkan line width. For Metal widths that suppress all line
    /// fragments this remains 1.0 and [`Self::rasterizer_discard`] carries the
    /// semantic result without binding a Vulkan-invalid sub-unit width.
    pub line_width_bits: u32,
    pub rasterizer_discard: bool,
    pub depth_bias_enable: bool,
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

/// A shader module and the words the driver compiles from it.
///
/// They travel together because a pipeline create needs the handle and the
/// crash breadcrumb needs the source, and two parameters is how a caller ends
/// up passing one shader's handle beside another's words.
#[derive(Clone, Copy)]
pub(crate) struct ShaderModuleSource<'a> {
    pub module: vk::ShaderModule,
    pub spirv: &'a [u32],
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
/// retired capacity that was previously used for pipelines.
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
    /// Last positive lookup, retained as the exact key and value. Render
    /// encoders commonly repeat one pipeline for long runs; equality against
    /// that key avoids hashing the same composite state on every draw.
    front: Option<(K, V)>,
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
            front: None,
            negative: HashMap::new(),
            negative_order: VecDeque::new(),
            negative_cap,
        }
    }

    fn get(&mut self, k: &K) -> Option<V>
    where
        V: Copy,
    {
        self.get_routed(k).map(|(value, _)| value)
    }

    /// Positive lookup and whether the one-entry front index answered it.
    fn get_routed(&mut self, k: &K) -> Option<(V, bool)>
    where
        V: Copy,
    {
        if let Some((front_key, value)) = &self.front {
            if front_key == k {
                return Some((*value, true));
            }
        }
        let value = *self.map.get(k)?;
        self.front = Some((k.clone(), value));
        Some((value, false))
    }

    fn get_negative(&self, k: &K) -> Option<DrawError> {
        // The healthy hot path has never cached a refusal. Avoid hashing the
        // full object key merely to ask an empty table; render pipeline keys in
        // particular carry attribute, attachment and descriptor arrays, and a
        // positive hit immediately hashes the same key again below.
        if self.negative.is_empty() {
            return None;
        }
        self.negative.get(k).cloned()
    }

    /// Insert. Returns the value a *replace* displaced, so the caller can
    /// destroy the Vulkan object it owned; a fresh key returns `None`. Nothing
    /// is ever displaced for capacity.
    fn insert(&mut self, k: K, v: V) -> Option<V>
    where
        V: Copy,
    {
        self.negative.remove(&k);
        let old = self.map.insert(k.clone(), v);
        self.front = Some((k, v));
        old
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
        self.front = None;
        self.map.clear();
        self.negative.clear();
        self.negative_order.clear();
    }

    fn take_all(&mut self) -> Vec<V> {
        self.front = None;
        self.negative.clear();
        self.negative_order.clear();
        self.map.drain().map(|(_, v)| v).collect()
    }
}

/// Entries the front index holds before it starts over.
///
/// Each entry is an address, a `Digest128` and an `Arc` clone — tens of bytes,
/// plus the words the `Arc` keeps alive, which the runtime owns for the
/// shader's lifetime anyway. A driven macos-13 boot binds a few hundred
/// distinct modules, so this is a ceiling with an order of magnitude of
/// headroom and `shader_digest_reset` firing is the boot saying the guest's
/// module set is not what that assumed.
const SHADER_DIGEST_ENTRIES: usize = 4096;

/// `Arc<Vec<u32>>` allocation address → the digest that module finally hashes
/// to, so a repeat bind can skip three whole-module walks.
///
/// # Why an address is a sound key
///
/// Only because the entry holds the `Arc`. While it does, the allocation cannot
/// be freed, so nothing else can be given that address and the key cannot come
/// to mean a different module. Drop the `Arc` from the entry and this becomes a
/// use-after-free dressed as a cache hit.
///
/// `usize` rather than `*const Vec<u32>` because a raw pointer is not `Send` and
/// `Caches` is held behind the engine lock and moved between threads. The
/// address is never dereferenced — it is compared, and the `Arc` beside it is
/// what keeps it meaningful.
///
/// # What it skips, and why that is safe
///
/// [`ObjectCaches::get_or_create_shader`] walks the module to verify its image
/// capability requirements and compute the digest. Both are pure functions of
/// the words, and the words behind an `Arc<Vec<u32>>` cannot change.
///
/// A hit still consults [`ObjectCaches::shaders`], positive and negative. That
/// keeps this index from depending on `ObjectCache` never evicting, which is a
/// property it happens to have and does not promise: a miss there simply falls
/// through to the full path, which recomputes and re-inserts.
#[derive(Default)]
struct ShaderDigestIndex {
    map: std::collections::HashMap<usize, (std::sync::Arc<Vec<u32>>, Digest128)>,
}

impl ShaderDigestIndex {
    /// The digest this allocation's module hashes to, if it has been walked
    /// before.
    fn get(&self, words: &std::sync::Arc<Vec<u32>>) -> Option<Digest128> {
        self.map
            .get(&(std::sync::Arc::as_ptr(words) as usize))
            .map(|(_, digest)| *digest)
    }

    /// Record what a full walk of this allocation produced.
    ///
    /// The bound is enforced here because this is the only way in: past
    /// [`SHADER_DIGEST_ENTRIES`] the whole index is dropped rather than evicting
    /// one entry, because there is no recency to evict *by* — every entry is
    /// equally cheap to rebuild, and a boot that reaches the bound is reporting
    /// something rather than asking for a policy.
    fn insert(&mut self, words: &std::sync::Arc<Vec<u32>>, digest: Digest128) {
        if self.map.len() >= SHADER_DIGEST_ENTRIES {
            reims_vgpu_observe::off(format!(
                "shader_digest_reset entries={} words={}",
                self.map.len(),
                words.len()
            ));
            self.map.clear();
        }
        self.map.insert(
            std::sync::Arc::as_ptr(words) as usize,
            (std::sync::Arc::clone(words), digest),
        );
    }

    fn clear(&mut self) {
        self.map.clear();
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

/// Per-vGPU lookup accelerators that borrow handles from shared content caches.
pub(crate) struct SessionCacheIndexes {
    /// Exact last Vulkan variant for each retained guest pipeline object.
    pipeline_objects: ObjectVariantIndex<PipelineKey, vk::Pipeline>,
    /// One guest-owned shader allocation to its content digest, so a repeat
    /// bind does not walk the same module three times.
    shader_digests: ShaderDigestIndex,
}

impl SessionCacheIndexes {
    pub(crate) fn new() -> Self {
        Self {
            pipeline_objects: ObjectVariantIndex::default(),
            shader_digests: ShaderDigestIndex::default(),
        }
    }

    pub(crate) fn clear(&mut self) {
        self.pipeline_objects.clear();
        self.shader_digests.clear();
    }
}

struct ObjectVariantIndex<K, V> {
    map: HashMap<u64, (reims_vgpu_core::ResourceLifetimeRef, K, V)>,
}

impl<K, V> Default for ObjectVariantIndex<K, V> {
    fn default() -> Self {
        Self {
            map: HashMap::new(),
        }
    }
}

impl<K: Clone + Eq, V: Copy> ObjectVariantIndex<K, V> {
    /// The variant this object last resolved to, if the object is still alive
    /// and still asks for the same one.
    ///
    /// One probe, and `strong_count` rather than `upgrade`: this answers 96 % of
    /// every draw's pipeline lookups on a driven Maps boot, so the second
    /// `HashMap::get` and the `Arc` that `upgrade` creates only to drop —
    /// two more atomics on the hottest path in the engine — were both paid per
    /// draw for nothing. A `Weak` reads `strong_count() == 0` exactly when its
    /// value has been dropped, which is the same question `upgrade().is_none()`
    /// asked.
    fn get(&mut self, identity: &reims_vgpu_core::ResourceLifetime, key: &K) -> Option<V> {
        let id = identity.id();
        let (life, held_key, pipeline) = self.map.get(&id)?;
        if !life.is_live() {
            self.map.remove(&id);
            return None;
        }
        (held_key == key).then_some(*pipeline)
    }

    fn remember(&mut self, identity: &reims_vgpu_core::ResourceLifetime, key: &K, value: V) {
        let id = identity.id();
        if !self.map.contains_key(&id) {
            // Object construction is rare. Reap identities whose runtime
            // object has gone before admitting the new one, so the index
            // follows object lifetime without a capacity or eviction policy.
            self.map.retain(|_, (life, _, _)| life.is_live());
        }
        self.map
            .insert(id, (identity.reference(), key.clone(), value));
    }

    fn clear(&mut self) {
        self.map.clear();
    }
}

/// The layout every colour attachment of every render pass this device builds is
/// left in when the pass ends.
///
/// # This is the one spelling, and the registry derives from it
///
/// [`super::pools::ResourcePools::registry_mark_ready_at`] records the layout a
/// finished pass left its target in, and it must name the same layout this
/// `finalLayout` does or every subsequent barrier is issued with the wrong
/// `oldLayout` — which is undefined behaviour, not a validation error, because
/// nothing in Vulkan re-checks it. It reads this constant rather than repeating
/// the name.
///
/// # Why it is not `TRANSFER_SRC_OPTIMAL`
///
/// It used to be, so that a present blit or a readback copy could read the
/// target without transitioning it. The trade was badly priced, and the
/// mispricing is structural rather than workload-specific: on a driven
/// macos-13 sustained-animation boot the target was read ~1 200 times a second
/// and drawn into ~24 000 times a second, and **every one of those draws paid a
/// barrier back to `COLOR_ATTACHMENT_OPTIMAL` to undo an exit that only 5 % of
/// them would ever have used.** It is charged as
/// `passmerge_outside_target_layout` and took 82 % of draws on macos-13, 37 % on
/// macos-11 and 29 % on macos-12.
///
/// On a discrete GPU that round trip is a barrier and probably little else. On a
/// GPU with framebuffer compression — every Intel iGPU (CCS), every AMD part
/// (DCC), every tiler — a transition out of `COLOR_ATTACHMENT_OPTIMAL` and back
/// is a decompress and recompress of the whole attachment, per draw.
///
/// Nothing depended on the old exit. Every consumer that reads a colour target
/// — the present blit, the readback copy, the writeback copy, the copy-on-sample
/// snapshot, and both seed copies — already issues its own barrier into
/// `TRANSFER_SRC_OPTIMAL` first, unconditionally, because the barrier is what
/// carries the *dependency* and a matching layout would not have removed the
/// need for it. So those ~1 200 reads a second gain a real transition each and
/// the ~24 000 draws lose one, which is the whole change.
///
/// # And it is `GENERAL`, because a colour target here is also a texture
///
/// The layout above is where a pass *leaves* the attachment. This one is where
/// the attachment **lives**, and it is `GENERAL` for the reason Metal has no
/// layouts at all: a `MTLTexture` a render encoder writes is the same object a
/// later fragment shader samples, and nothing in that API marks the crossing. In
/// Vulkan the crossing is an image layout, and every layout that is optimal for
/// one of the two uses is illegal for the other — so a device that picks
/// `COLOR_ATTACHMENT_OPTIMAL` has to transition on every sample, and a
/// transition is exactly what a render pass instance may not contain. That is
/// `passmerge_outside_resident_layout`, 25 344 of 176 914 pass begins on a driven
/// macos-13 Maps boot, each closing a pass worth ~100 µs of GPU.
///
/// `GENERAL` is legal for both, so the crossing disappears and the resident is
/// where the next user wants it whichever user that is. What it gives up is
/// framebuffer compression, which on this host is real hardware —
/// Intel Arrow Lake CCS.
///
/// **Measured twice, and the second chain refuted the first.** Both are
/// interleaved driven macos-13 Maps boots of one binary with the layout moved and
/// nothing else, scored by `scripts/boot-score` on `sum us/draw`, every boot at
/// `throttle_ms=0`:
///
/// ```text
///                    chain C (/tmp/wb-outC0..C5)   chain D (/tmp/wb-outD0..D5)
/// COLOR_ATTACHMENT   22.95, 22.73, 25.50          17.44, 19.07, 18.05
/// GENERAL            21.43, 22.33, 21.93          20.86, 20.38
/// ```
///
/// Chain C alone reads as a disjoint −7.7 %. Chain D reads the *other way* by a
/// similar margin, and pooled the two arms overlap completely — 21.39 mean for
/// `GENERAL` against 20.96 for `COLOR_ATTACHMENT`. (Chain D also produced one boot
/// at `sum` 52.04 with `d/frame` 225, a different workload regime, excluded from
/// both means.)
///
/// So **this layout is perf-neutral as far as this host can measure**, and the
/// three-boot disjointness in chain C was chain position, not the arm. It is worth
/// stating why the earlier reading was believed: three position-matched pairs
/// agreeing one by one looks like a controlled result and is not one, because the
/// pairs share their position in a chain whose spread is larger than the effect.
///
/// The change stays, on **correctness** rather than on speed: one resting layout
/// is what lets every spelling below be one function instead of six, and it is
/// what makes a feedback colour slot legal (see [`PassKey::color_layout`]).
/// Do not re-quote the −7.7 %.
///
/// # It is a function, and [`reims_vgpu_config::COLOR_GENERAL`] is the ablation
///
/// **Every** spelling of the layout has to move together — the pass's
/// `finalLayout`, the `initialLayout` a `LOAD` pass names, the subpass
/// reference, the registry's record of where the pass left the image, the layout
/// a sampled descriptor declares, and the comparisons
/// `super::exec::pass_exit_needs_no_barrier` and
/// [`super::pools::ResidentAccess::covered_by_pass_entry`] make. A `const` plus a
/// switch read beside it would be that second spelling, and two of them
/// disagreeing is a barrier naming an `oldLayout` the image is not in, which is
/// undefined behaviour and not an error. So there is one function and no
/// constant, and `REIMS_VGPU_COLOR_GENERAL=off` moves all of them back at once —
/// a narrowing, since it restores a transition rather than removing one.
pub(crate) fn color0_pass_exit_layout() -> vk::ImageLayout {
    if single_color_layout() {
        vk::ImageLayout::GENERAL
    } else {
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    }
}

/// Whether a colour target rests in one layout for its whole life. **Default
/// on**; `REIMS_VGPU_COLOR_GENERAL=off` is the ablation arm.
///
/// Read once. This decides the content of cached `VkRenderPass` objects and of
/// registry records that outlive them, so an answer that changed mid-boot would
/// leave both built under two layouts in one cache.
pub(crate) fn single_color_layout() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            reims_vgpu_config::read(reims_vgpu_config::COLOR_GENERAL).0,
            reims_vgpu_config::Switch::Off
        )
    })
}

/// The two `VK_SUBPASS_EXTERNAL` dependencies every render pass this device
/// builds must carry, covering **every** attachment class the pass has.
///
/// # Why this is unconditional, and what it cost to be conditional
///
/// Vulkan supplies an implicit external dependency for an attachment only *"if
/// there is no subpass dependency from `VK_SUBPASS_EXTERNAL` to the first
/// subpass that uses"* it — and the implicit one is per render pass, not per
/// attachment. So the moment a pass declares one explicit external dependency
/// for **any** reason, every attachment loses its implicit one.
///
/// This pass used to declare a pair only when it had a depth attachment, and
/// that pair named the `EARLY`/`LATE_FRAGMENT_TESTS` stages and the
/// depth-stencil accesses alone. The colour attachment silently lost the
/// synchronization it had been getting for free, so on a depth pass:
///
/// - the incoming transition into `COLOR_ATTACHMENT_OPTIMAL` was not ordered
///   against the `loadOp` clear that follows it, and
/// - the outgoing transition into `TRANSFER_SRC_OPTIMAL` was not ordered
///   against the subpass's own colour store, nor against the copy that reads
///   the target afterwards.
///
/// All three were reported by the Khronos synchronization validation layer on a
/// driven macos-11 boot, as `SYNC-HAZARD-WRITE-AFTER-WRITE` at
/// `vkCmdBeginRenderPass` and `vkCmdEndRenderPass` and
/// `SYNC-HAZARD-READ-AFTER-WRITE` at the `vkCmdCopyImage` /
/// `vkCmdCopyImageToBuffer` that follows.
///
/// Building both dependencies here, always, from the pass's own composition is
/// what makes the split unrepresentable: there is no longer an arm that adds a
/// dependency for one attachment class without stating the others.
///
/// # The outgoing `dst` scope covers every way the attachment is read next
///
/// Every slot now exits at [`color0_pass_exit_layout`], so the nearest consumer
/// is usually the next draw into the same target — which is why the attachment
/// stages and accesses are in the destination scope, and why
/// `super::exec::pass_exit_needs_no_barrier` may then drop that draw's own
/// barrier entirely. `TRANSFER` and `FRAGMENT_SHADER` stay named because a
/// readback, a present blit or a later sample can follow instead; each of those
/// issues its own transition, and this is the scope that transition orders
/// against.
///
/// # The incoming dependency is what makes the skip legal
///
/// `VK_SUBPASS_EXTERNAL` as `srcSubpass` scopes every command submitted before
/// the render pass instance, in submission order. So the incoming dependency
/// here — colour writes to attachment reads and writes — already orders the
/// previous draw's store against this pass's `loadOp`, with no barrier from the
/// draw. Weakening its source scope would silently make that skip unsound.
fn external_dependencies(
    has_depth: bool,
    color_input: bool,
    // Taken as an argument rather than read here, so both arms are reachable
    // from a test. The switch is read once, at the one call site that builds a
    // pass; see [`pass_exit_scope_narrow`].
    exit_scope_narrow: bool,
) -> [vk::SubpassDependency; 2] {
    // Colour is unconditional: every pass this device builds has slot 0.
    let mut attach_stages = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
    let mut attach_writes = vk::AccessFlags::COLOR_ATTACHMENT_WRITE;
    let mut attach_reads = vk::AccessFlags::COLOR_ATTACHMENT_READ;
    if has_depth {
        attach_stages |= vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS;
        attach_writes |= vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;
        attach_reads |= vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ;
    }
    // Framebuffer fetch reads attachment 0 through the fragment stage, so the
    // incoming transition has to be visible to that read too. The intra-subpass
    // ordering is the separate `BY_REGION` dependency; this is the entry.
    //
    // The shader stages are unconditional, and they are what
    // [`super::pools::ResidentAccess::covered_by_pass_entry`] rests on. A draw
    // inside this pass may sample a resident an *earlier* pass wrote, and with
    // one resting layout there is no transition left for that draw to record —
    // only a visibility request, which this entry then carries for every such
    // draw at once instead of each of them closing the pass to state it.
    // Weakening this to the attachment stages makes that skip a missing
    // dependency, which is a stale frame and no error.
    let (in_dst_stages, mut in_dst_access) = (
        attach_stages
            | vk::PipelineStageFlags::VERTEX_SHADER
            | vk::PipelineStageFlags::FRAGMENT_SHADER,
        attach_writes | attach_reads | vk::AccessFlags::SHADER_READ,
    );
    if color_input {
        in_dst_access |= vk::AccessFlags::INPUT_ATTACHMENT_READ;
    }
    let source_stages = attach_stages | vk::PipelineStageFlags::TRANSFER;
    let source_access = attach_writes | vk::AccessFlags::TRANSFER_WRITE;
    [
        vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            // Whatever last wrote these images: a previous pass's colour store,
            // a depth store, or the transfer that seeded a LOAD attachment.
            .src_stage_mask(source_stages)
            .src_access_mask(source_access)
            .dst_stage_mask(in_dst_stages)
            .dst_access_mask(in_dst_access),
        {
            // Narrowing this to the attachment stages alone is the probe: see
            // [`pass_exit_scope_narrow`] for what it is asking and what it must
            // not break.
            let (dst_stages, dst_access) = if exit_scope_narrow {
                (attach_stages, attach_writes | attach_reads)
            } else {
                (
                    vk::PipelineStageFlags::TRANSFER
                        | vk::PipelineStageFlags::FRAGMENT_SHADER
                        | attach_stages,
                    vk::AccessFlags::TRANSFER_READ
                        | vk::AccessFlags::SHADER_READ
                        | attach_writes
                        | attach_reads,
                )
            };
            vk::SubpassDependency::default()
                .src_subpass(0)
                .dst_subpass(vk::SUBPASS_EXTERNAL)
                .src_stage_mask(attach_stages)
                .src_access_mask(attach_writes)
                .dst_stage_mask(dst_stages)
                .dst_access_mask(dst_access)
        },
    ]
}

/// The colour write a feedback draw's own sampled read must be ordered after.
pub(crate) const COLOR_FEEDBACK_SRC: (vk::PipelineStageFlags, vk::AccessFlags) = (
    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
    vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
);

/// The sampled read a feedback draw performs of the attachment it is writing.
///
/// `FRAGMENT_SHADER` alone, and that is a rule rather than a choice. A subpass
/// self-dependency whose `srcStageMask` contains a framebuffer-space stage may
/// name only framebuffer-space stages in its `dstStageMask`
/// (`VUID-VkSubpassDependency-srcSubpass-06809`), and `VERTEX_SHADER` is not one.
/// It was never right on its own terms either: a feedback loop is a fragment
/// reading the pixel it is about to write, and `BY_REGION` below is that
/// same-pixel claim. A vertex stage reading the attachment is not a feedback loop
/// and could not be ordered by this dependency whatever it named.
pub(crate) const COLOR_FEEDBACK_DST: (vk::PipelineStageFlags, vk::AccessFlags) = (
    vk::PipelineStageFlags::FRAGMENT_SHADER,
    vk::AccessFlags::SHADER_READ,
);

/// The subpass self-dependency that orders a feedback draw's sampled read after
/// the colour writes of the draws before it in the same pass instance.
///
/// **Declared on every pass this device builds, whether or not any draw in it
/// feeds back.** A self-dependency costs nothing until a
/// `vkCmdPipelineBarrier` inside the pass invokes it — but its *presence* changes
/// `dependencyCount`, and Vulkan render-pass compatibility spares initial/final
/// layouts, attachment-reference layouts and load/store ops while sparing nothing
/// about dependencies. So a pass built with this dependency and one built without
/// it are **incompatible**, and a `VkFramebuffer` created against either cannot be
/// used with the other. Declaring it conditionally is what produced
/// `VUID-VkRenderPassBeginInfo-renderPass-00904` (`dependencyCount is
/// incompatible`) on a driven Maps boot, on both layout arms.
///
/// The general rule, which is the one to keep: **a render pass may only vary in
/// ways [`PassKey::framebuffer_compatibility`] preserves.** Anything that key
/// erases must not reach the `VkRenderPassCreateInfo`.
///
/// `dependency_flags` derives the extension bit from the attachment layout via
/// [`super::feedback_transition_dependency`], so the arm that has no feedback
/// layout cannot ask for the extension's flag.
fn color_feedback_self_dependency(color0_layout: vk::ImageLayout) -> vk::SubpassDependency {
    vk::SubpassDependency::default()
        .src_subpass(0)
        .dst_subpass(0)
        .src_stage_mask(COLOR_FEEDBACK_SRC.0)
        .src_access_mask(COLOR_FEEDBACK_SRC.1)
        .dst_stage_mask(COLOR_FEEDBACK_DST.0)
        .dst_access_mask(COLOR_FEEDBACK_DST.1)
        .dependency_flags(
            vk::DependencyFlags::BY_REGION | super::feedback_transition_dependency(color0_layout),
        )
}

/// Whether the outgoing external dependency names only the attachment stages.
///
/// **Probe, default off.** See [`reims_vgpu_config::PASS_EXIT_NARROW`] for the whole
/// argument; in one line, the shipping scope names `TRANSFER | FRAGMENT_SHADER`
/// with `TRANSFER_READ | SHADER_READ`, which asks this driver for a render-cache
/// flush and a texture-cache invalidate at **every** `vkCmdEndRenderPass`, and a
/// pass boundary is the single largest cost in this device on the iGPU pathway.
///
/// Read once. This decides the content of a cached `VkRenderPass`, so a value
/// that changed mid-boot would leave passes built under both answers in one
/// cache and make the arm unreadable.
fn pass_exit_scope_narrow() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            reims_vgpu_config::read(reims_vgpu_config::PASS_EXIT_NARROW).0,
            reims_vgpu_config::Switch::On
        )
    })
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
        // This index borrows handles owned by `pipelines`; forget those echoes
        // before destroying the authoritative objects.
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

    /// Report a driver call this device refused to repeat, and hand back the
    /// error every one of the three call sites caches negatively.
    ///
    /// One place rather than three so the three cannot drift into three
    /// different accounts of the same event — and so the line always carries
    /// both the key (which identifies the call) and what the dead process called
    /// it (which is the only human-readable thing about it).
    fn note_quarantined(
        &self,
        site: &'static str,
        hit: &super::driver_breadcrumb::quarantine::Quarantined,
    ) -> DrawError {
        let reason = super::reason::DrawReason::DriverCallQuarantined;
        reims_vgpu_observe::Emit::decline("driver_quarantine", &reason)
            .field("site", site)
            .field("key", &hit.key)
            .field("previously", &hit.previously)
            .field(
                "list",
                super::driver_breadcrumb::quarantine::list_path().display(),
            )
            .fail();
        DrawError::Unsupported(reason)
    }

    /// [`Self::get_or_create_shader`] with the three whole-module walks skipped
    /// for an allocation that has been through it before.
    ///
    /// The draw path is the caller that needs this: it binds two modules a draw
    /// at ~30 000 draws a second, from `Arc`s the runtime holds for each
    /// shader's lifetime, into a cache that on a driven macos-13 boot reports
    /// `shader_misses=0`. `pl_shader_us` was **63 ms of every second** — the
    /// largest single item inside `engine_us` — spent deriving a key for a
    /// module already in hand.
    ///
    /// The compute path calls the walking form directly and deliberately: its
    /// `spirv` is an owned `Vec` with no stable allocation to key on, and a
    /// dispatch is three orders rarer than a draw.
    pub(crate) unsafe fn get_or_create_shader_memoized(
        &mut self,
        indexes: &mut SessionCacheIndexes,
        ctx: &DeviceContext,
        words: &std::sync::Arc<Vec<u32>>,
        counters: &EngineCounters,
        pools: &mut RecordingPools<'_>,
    ) -> Result<(Digest128, vk::ShaderModule), DrawError> {
        if let Some(key) = indexes.shader_digests.get(words) {
            // Negative before positive, in the order the walking form asks them:
            // a module this device refused is refused again without being
            // rebuilt, and without the front index quietly promoting it.
            if let Some(err) = self.shaders.get_negative(&key) {
                counters.shader_misses.fetch_add(1, Ordering::Relaxed);
                return Err(err);
            }
            if let Some(module) = self.shaders.get(&key) {
                counters.shader_hits.fetch_add(1, Ordering::Relaxed);
                counters.shader_digest_hits.fetch_add(1, Ordering::Relaxed);
                return Ok((key, module));
            }
            // The module was evicted or destroyed under a digest this index
            // still names. Falling through re-walks and re-creates it, which is
            // why the index may hold a digest the cache does not.
        }
        let (key, module) = self.get_or_create_shader(ctx, words, counters, pools)?;
        indexes.shader_digests.insert(words, key);
        Ok((key, module))
    }

    pub(crate) unsafe fn get_or_create_shader(
        &mut self,
        ctx: &DeviceContext,
        words: &[u32],
        counters: &EngineCounters,
        pools: &mut RecordingPools<'_>,
    ) -> Result<(Digest128, vk::ShaderModule), DrawError> {
        // Verify that the host features enabled at device creation cover the
        // final module. metal2vulkan owns capability declaration; mutating its
        // output here would create a second translator with a weaker view of
        // the source contract.
        counters
            .shader_hash_words
            .fetch_add(words.len() as u64, Ordering::Relaxed);
        let need = crate::spirv_bind::required_image_capabilities(words);
        if need.any() {
            let missing = (need.extended_formats && !ctx.spirv_storage_extended_formats)
                || (need.write_without_format && !ctx.spirv_storage_write_without_format)
                || (need.read_without_format && !ctx.spirv_storage_read_without_format);
            if missing {
                let err = DrawError::Unsupported(super::reason::DrawReason::SpirvInvalid);
                reims_vgpu_observe::fail(format!(
                    "spirv_capability reason=host_lacks_feature words={} \
                     need_extended={} need_write={} need_read={} \
                     have_extended={} have_write={} have_read={}",
                    words.len(),
                    need.extended_formats,
                    need.write_without_format,
                    need.read_without_format,
                    ctx.spirv_storage_extended_formats,
                    ctx.spirv_storage_write_without_format,
                    ctx.spirv_storage_read_without_format,
                ));
                let key = Digest128::of_u32_words(words);
                self.shaders.insert_negative(key, err.clone());
                return Err(err);
            }
        }
        let key = Digest128::of_u32_words(words);
        if let Some(err) = self.shaders.get_negative(&key) {
            counters.shader_misses.fetch_add(1, Ordering::Relaxed);
            return Err(err);
        }
        if let Some(m) = self.shaders.get(&key) {
            counters.shader_hits.fetch_add(1, Ordering::Relaxed);
            return Ok((key, m));
        }
        counters.shader_misses.fetch_add(1, Ordering::Relaxed);
        // Last gate before the driver, and the only place every module from
        // every path passes through exactly once. An invalid module is
        // undefined behaviour inside a driver rather than an error it returns,
        // and one has been observed ending the VM process — so it becomes a
        // negative cache entry here and the guest's work is declined by name.
        // See `crate::spirv_bind::validate`.
        if let crate::spirv_bind::SpirvValidation::Rejected(why) =
            crate::spirv_bind::validate(words)
        {
            let err = DrawError::Unsupported(super::reason::DrawReason::SpirvInvalid);
            // Print what the capability derivation saw alongside the
            // validator's complaint. When the two disagree the difference is
            // the whole bug, and neither one alone says which walk is wrong.
            reims_vgpu_observe::fail(format!(
                "spirv_validate reason=module_rejected words={} need={:?} imgs={:?} detail={why}",
                words.len(),
                crate::spirv_bind::required_image_capabilities(words),
                crate::spirv_bind::image_type_census(words),
            ));
            // The complaint above names instructions by result id, which cannot
            // be read without the module they belong to. Keep it.
            super::driver_breadcrumb::keep_rejected_module(
                &format!("{:016x}{:016x}", key.a, key.b),
                words,
            );
            self.shaders.insert_negative(key, err.clone());
            return Err(err);
        }
        // The driver parses SPIR-V here, so this is one of the three calls that
        // can end the process on a module this device assembled — the other two
        // being the compute and graphics pipeline compiles below. See
        // `driver_breadcrumb` for why the words go to disk across it.
        let breadcrumb = match super::driver_breadcrumb::DriverBreadcrumb::arm(
            "create_shader_module",
            &[("module", words)],
        ) {
            Ok(breadcrumb) => breadcrumb,
            Err(hit) => {
                let err = self.note_quarantined("create_shader_module", &hit);
                self.shaders.insert_negative(key, err.clone());
                return Err(err);
            }
        };
        let created = ctx
            .device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(words), None);
        breadcrumb.disarm();
        let module = created.map_err(|e| {
            let err = DrawError::VkCall(VkCall::new(VkOp::CachesCreateShaderModule, e));
            self.shaders.insert_negative(key, err.clone());
            err
        })?;
        counters.note_create(CreateSite::ShaderModule);
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
        pools: &mut RecordingPools<'_>,
    ) -> Result<(vk::DescriptorSetLayout, vk::PipelineLayout), DrawError> {
        if let Some(err) = self.layouts.get_negative(key) {
            counters.layout_misses.fetch_add(1, Ordering::Relaxed);
            return Err(err);
        }
        if let Some((dsl, pl)) = self.layouts.get(key) {
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
                    .descriptor_count(b.count)
                    .stage_flags(vk::ShaderStageFlags::from_raw(b.stages))
            })
            .collect();
        let dsl = if bindings.is_empty() {
            vk::DescriptorSetLayout::null()
        } else {
            let binding_flags: Vec<_> = key
                .bindings
                .iter()
                .map(|binding| {
                    if binding.count > 1 {
                        vk::DescriptorBindingFlags::PARTIALLY_BOUND
                    } else {
                        vk::DescriptorBindingFlags::empty()
                    }
                })
                .collect();
            let mut flags = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
                .binding_flags(&binding_flags);
            let mut create_info = vk::DescriptorSetLayoutCreateInfo::default()
                .bindings(&bindings)
                .push_next(&mut flags);
            if key.uses_push_descriptors(ctx.caps.push_descriptor) {
                create_info =
                    create_info.flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR);
            }
            let d = ctx
                .device
                .create_descriptor_set_layout(&create_info, None)
                .map_err(|e| {
                    let err =
                        DrawError::VkCall(VkCall::new(VkOp::CachesCreateDescriptorSetLayout, e));
                    self.layouts.insert_negative(key.clone(), err.clone());
                    err
                })?;
            counters.note_create(CreateSite::DescriptorSetLayout);
            d
        };
        let layouts: Vec<vk::DescriptorSetLayout> = if dsl == vk::DescriptorSetLayout::null() {
            Vec::new()
        } else {
            vec![dsl]
        };
        let push_ranges: Vec<vk::PushConstantRange> = key
            .kernel_grid
            .into_iter()
            .map(|range| {
                vk::PushConstantRange::default()
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .offset(range.offset)
                    .size(range.size)
            })
            .collect();
        let pl = ctx
            .device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&layouts)
                    .push_constant_ranges(&push_ranges),
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
        counters.note_create(CreateSite::PipelineLayout);
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
        pools: &mut RecordingPools<'_>,
    ) -> Result<vk::RenderPass, DrawError> {
        if let Some(err) = self.passes.get_negative(&key) {
            counters.pass_misses.fetch_add(1, Ordering::Relaxed);
            return Err(err);
        }
        if let Some(rp) = self.passes.get(&key) {
            counters.pass_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(rp);
        }
        counters.pass_misses.fetch_add(1, Ordering::Relaxed);
        let target_format = key.color0_format;
        // A `LOAD` pass names the layout the *previous* pass left the attachment
        // in, so it reads the exit constant rather than respelling the layout.
        // These two agreeing is what lets `exec::pass_exit_needs_no_barrier`
        // drop the transition between consecutive draws into one target; a
        // second spelling here would make that skip a missing transition the
        // first time somebody changed one of them.
        let color0_final = key.color_final_layout(0);
        let (load_op, initial) = key.color0_load.vulkan(color0_final);
        // Slot 0 (primary) and the secondary attachments (slot 1..) now exit the
        // same way, at [`color0_pass_exit_layout`], and for the same reason: a
        // consumer's barrier is what establishes the dependency, so leaving the
        // mask at COLOR_ATTACHMENT_OPTIMAL forces that barrier to fire with a
        // colour-write source scope rather than being skipped as a no-op. The
        // registry tracks this layout.
        let mut attachments = vec![vk::AttachmentDescription::default()
            .format(target_format)
            .samples(vk_sample_count(key.sample_count))
            .load_op(load_op)
            // Vulkan render-pass splits are an implementation detail inside one
            // guest encoder. Preserve the scratch source across such a split;
            // the guest's resolve-only store action still exposes only the
            // single-sample destination.
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(initial)
            .final_layout(color0_final)];
        // Framebuffer fetch: when attachment 0 is also a subpass input, BOTH
        // references must use GENERAL (same-attachment color+input requires it);
        // the pass still transitions initial→GENERAL→final automatically.
        let color0_layout = key.color_layout(0);
        let mut color_ref = vec![vk::AttachmentReference::default()
            .attachment(0)
            .layout(color0_layout)];
        for (i, sec) in key.secondary[..key.secondary_count as usize]
            .iter()
            .enumerate()
        {
            let attachment_index = i + 1;
            let final_layout = key.color_final_layout(attachment_index);
            let (sload, sinitial) = sec.load.vulkan(final_layout);
            attachments.push(
                vk::AttachmentDescription::default()
                    .format(sec.format)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(sload)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(sinitial)
                    .final_layout(final_layout),
            );
            color_ref.push(
                vk::AttachmentReference::default()
                    .attachment(1 + i as u32)
                    .layout(key.color_layout(attachment_index)),
            );
        }
        let mut resolve_ref = Vec::new();
        if key.multisample_resolve {
            let resolve_index = attachments.len() as u32;
            attachments.push(
                vk::AttachmentDescription::default()
                    .format(target_format)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(color0_layout)
                    .final_layout(color0_final),
            );
            resolve_ref.push(
                vk::AttachmentReference::default()
                    .attachment(resolve_index)
                    .layout(color0_layout),
            );
            resolve_ref.extend((1..color_ref.len()).map(|_| {
                vk::AttachmentReference::default()
                    .attachment(vk::ATTACHMENT_UNUSED)
                    .layout(vk::ImageLayout::UNDEFINED)
            }));
        }
        // Depth attachment is appended LAST (after color + secondaries), so its
        // index is the current attachment count and color slot 0 is untouched.
        let depth_ref = key.depth.map(|d| {
            let final_layout = vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            let (dload, depth_initial) = if d.depth {
                d.load.vulkan(final_layout)
            } else {
                (vk::AttachmentLoadOp::DONT_CARE, vk::ImageLayout::UNDEFINED)
            };
            // A pass-bound stencil aspect or Metal's implicit stencil value
            // selects the combined format. Depth-only stays D32_SFLOAT with
            // DONT_CARE stencil operations.
            let (dformat, sload, sstore) = if d.stencil {
                (
                    ctx.depth_stencil_format,
                    d.stencil_load.vulkan(final_layout).0,
                    if d.stencil_store {
                        vk::AttachmentStoreOp::STORE
                    } else {
                        vk::AttachmentStoreOp::DONT_CARE
                    },
                )
            } else {
                (
                    translate::pixel::TRANSIENT_DEPTH_FORMAT,
                    vk::AttachmentLoadOp::DONT_CARE,
                    vk::AttachmentStoreOp::DONT_CARE,
                )
            };
            let dinitial = if depth_initial == final_layout
                || (d.stencil && d.stencil_load == ColorLoadKey::Load)
            {
                final_layout
            } else {
                vk::ImageLayout::UNDEFINED
            };
            let index = attachments.len() as u32;
            attachments.push(
                vk::AttachmentDescription::default()
                    .format(dformat)
                    .samples(vk_sample_count(key.sample_count))
                    .load_op(dload)
                    .store_op(if d.depth && d.store {
                        vk::AttachmentStoreOp::STORE
                    } else {
                        vk::AttachmentStoreOp::DONT_CARE
                    })
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
            .layout(key.color_layout(0))];
        let mut subpass_desc = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_ref);
        if !resolve_ref.is_empty() {
            subpass_desc = subpass_desc.resolve_attachments(&resolve_ref);
        }
        if key.color_input {
            subpass_desc = subpass_desc.input_attachments(&input_ref);
        }
        if let Some(depth_ref) = &depth_ref {
            subpass_desc = subpass_desc.depth_stencil_attachment(depth_ref);
        }
        let subpass = [subpass_desc];
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
        let mut deps: Vec<vk::SubpassDependency> = external_dependencies(
            key.depth.is_some(),
            key.color_input,
            pass_exit_scope_narrow(),
        )
        .to_vec();
        if key.color_input {
            deps.push(fetch_dep);
        }
        deps.push(color_feedback_self_dependency(key.color_layout(0)));
        let rp_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpass)
            .dependencies(&deps);
        let rp = ctx.device.create_render_pass(&rp_info, None).map_err(|e| {
            let err = DrawError::VkCall(VkCall::new(VkOp::CachesCreateRenderPass, e));
            self.passes.insert_negative(key, err.clone());
            err
        })?;
        counters.note_create(CreateSite::RenderPass);
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
        pools: &mut RecordingPools<'_>,
    ) -> Result<vk::Sampler, DrawError> {
        if let Some(err) = self.samplers.get_negative(key) {
            counters.sampler_misses.fetch_add(1, Ordering::Relaxed);
            return Err(err);
        }
        if let Some(s) = self.samplers.get(key) {
            counters.sampler_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(s);
        }
        counters.sampler_misses.fetch_add(1, Ordering::Relaxed);
        // Everything below is built from the *conformed* key, never from `key`
        // itself; `key` stays the guest's request so the cache and the negative
        // cache still index what was asked for.
        let conformed = match super::types::effective_sampler_state_key(*key) {
            Ok(k) => k,
            Err(reason) => {
                reims_vgpu_observe::Emit::decline("vk_engine_sampler", &reason)
                    .field("compare_function", format!("{:?}", key.compare_function))
                    .fail();
                let err = DrawError::Unsupported(reason);
                self.samplers.insert_negative(*key, err.clone());
                return Err(err);
            }
        };
        // One line per distinct unnormalized sampler this boot creates — the
        // cache is what bounds it, so a workload with three such samplers logs
        // three lines however many million binds it does.
        //
        // # What this is measuring, and why it is not a decline
        //
        // `VUID-vkCmdDispatch-None-08610`/`-08611` forbid an unnormalized sampler
        // being *used* by an implicit-LOD, `Proj`, `Dref`, `Bias` or `Offset`
        // sample, and `-08611` is the one violation a driven macos-11 boot under
        // the Khronos validation layer still reports after every other one here
        // was fixed. That is a property of the SPIR-V instruction, so this device
        // cannot repair it — but it can say which samplers are candidates.
        //
        // `metal2vulkan` already emulates pixel-coordinate sampling in-shader
        // rather than emitting `OpImageSample`, on three arms — a per-tap
        // fetch-and-lerp for linear, a bicubic one, and a plain fetch for
        // integer/nearest/arrayed. **All three of its predicates require
        // `min_filter == mag_filter`**, so a pixel-coordinate sampler with
        // *mixed* filters matches none of them and falls through to a real
        // `OpImageSample` against the unnormalized sampler. `min_mag_differed`
        // is therefore the field to read: a boot that logs it has the shape that
        // produces the VUID, and a boot that never does has ruled it out.
        // Logged for *every* unnormalized sampler and not only the conformed
        // ones, so a boot's line count is the population and the `conformed`
        // field splits it. Reading only the conformed ones would miss exactly
        // the samplers the translator handles correctly, which is the control.
        //
        // # The VUID is real and it is **not** what hangs this GPU
        //
        // Both halves are measured, and the second one is why this stayed a
        // census rather than becoming a repair. A driven macos-11 boot logs
        // `min_mag_differed=false` on every unnormalized sampler it creates, so
        // the mixed-filter shape above is not how this workload reaches the
        // VUID — a *dynamically* bound `MTLSamplerState` is, because
        // `metal2vulkan` intercepts only samplers whose state it knows at
        // translate time and a runtime-bound one is invisible to it.
        //
        // A probe then forced `unnormalized_coordinates` false here, which makes
        // `-08611` unreachable by construction: Maps froze anyway, with the same
        // two device recreates. So the violation is a passenger. Do not spend
        // another boot treating it as the hang, and do not read a future
        // `sampler_unnormalized` line as one — it says a sampler exists, not
        // that a frame was lost.
        //
        // Repairing it properly still needs the offset folded into the
        // coordinate, which is exact only for an unnormalized sampler and so
        // needs the normalization known at translate time. That is a
        // `metal2vulkan` input this device does not currently supply.
        if key.unnormalized_coordinates {
            reims_vgpu_observe::off(format!(
                "sampler_unnormalized min_mag_differed={} conformed={} \
                 min={:?} mag={:?} mip={:?} address_u={:?} address_v={:?} aniso={}",
                key.min_filter != key.mag_filter,
                conformed != *key,
                key.min_filter,
                key.mag_filter,
                key.mip_filter,
                key.address_mode_u,
                key.address_mode_v,
                key.max_anisotropy,
            ));
        }
        let not_mipmapped = conformed.mip_filter == super::types::SamplerMipFilter::NotMipmapped;
        let (min_lod, max_lod) = if conformed.unnormalized_coordinates || not_mipmapped {
            (0.0, 0.0)
        } else {
            (
                f32::from_bits(conformed.lod_min),
                f32::from_bits(conformed.lod_max),
            )
        };
        let unnorm = conformed.unnormalized_coordinates;
        let mag_filter = conformed.mag_filter;
        let min_filter = conformed.min_filter;
        let mip_filter = conformed.mip_filter;
        let address_mode_u = conformed.address_mode_u;
        let address_mode_v = conformed.address_mode_v;
        let max_anisotropy_req = conformed.max_anisotropy;
        let compare_function = conformed.compare_function;
        // `address_mode_w` is deliberately not clamped: `-01075` names U and V
        // only, because an unnormalized sampler may only read a 2D view.
        let address_uses_zero = [address_mode_u, address_mode_v, conformed.address_mode_w]
            .contains(&super::types::SamplerAddressMode::ClampToZero);
        if max_anisotropy_req > 1 && !ctx.sampler_anisotropy {
            let reason = super::reason::DrawReason::SamplerAnisotropyUnsupported;
            // Fail-visible here, at the check, and exactly once per sampler key:
            // the negative cache means a replay returns without reaching this
            // line, and the returned `DrawError` reaches the log only if some
            // caller happens to render it. A capability the host GPU lacks is
            // precisely the class says must
            // never surface as a silently different sampler.
            reims_vgpu_observe::Emit::decline("vk_engine_sampler", &reason)
                .field("max_anisotropy", max_anisotropy_req)
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
        //
        // Read against the conformed modes, not the requested ones: an
        // unnormalized sampler's U and V are already clamped above, so declining
        // there would refuse a mode this device is not going to bind.
        let uses_mirror_clamp = [address_mode_u, address_mode_v, conformed.address_mode_w]
            .contains(&super::types::SamplerAddressMode::MirrorClampToEdge);
        if uses_mirror_clamp && !ctx.features.mirror_clamp_to_edge.is_available() {
            let reason = super::reason::DrawReason::SamplerMirrorClampToEdgeUnsupported;
            reims_vgpu_observe::Emit::decline("vk_engine_sampler", &reason)
                .field("address_u", format!("{address_mode_u:?}"))
                .field("address_v", format!("{address_mode_v:?}"))
                .field("address_w", format!("{:?}", conformed.address_mode_w))
                .fail();
            let err = DrawError::Unsupported(reason);
            self.samplers.insert_negative(*key, err.clone());
            return Err(err);
        }
        // Not floored here: every producer of this key either writes a literal
        // 1 (the reflected static sampler) or carries a decoded
        // `SamplerDescriptor`, which `decode_sampler_descriptor` already floors.
        let max_anisotropy = (max_anisotropy_req as f32).min(ctx.max_sampler_anisotropy);
        let sampler = ctx
            .device
            .create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(crate::translate::sampler::vk_filter(mag_filter))
                    .min_filter(crate::translate::sampler::vk_filter(min_filter))
                    .mipmap_mode(crate::translate::sampler::vk_mipmap_mode(mip_filter))
                    .address_mode_u(crate::translate::sampler::vk_address_mode(address_mode_u))
                    .address_mode_v(crate::translate::sampler::vk_address_mode(address_mode_v))
                    .address_mode_w(crate::translate::sampler::vk_address_mode(
                        conformed.address_mode_w,
                    ))
                    .mip_lod_bias(0.0)
                    .anisotropy_enable(max_anisotropy_req > 1)
                    .max_anisotropy(max_anisotropy)
                    .compare_enable(compare_function != super::types::SamplerCompareFunction::Never)
                    .compare_op(crate::translate::raster::vk_compare_op(compare_function))
                    .min_lod(min_lod)
                    .max_lod(max_lod)
                    .border_color(translate::sampler::vk_border_color_with_clamp_to_zero(
                        conformed.border_color,
                        address_uses_zero,
                    ))
                    .unnormalized_coordinates(unnorm),
                None,
            )
            .map_err(|e| {
                let err = DrawError::VkCall(VkCall::new(VkOp::CachesCreateSampler, e));
                self.samplers.insert_negative(*key, err.clone());
                err
            })?;
        counters.note_create(CreateSite::Sampler);
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
        indexes: &mut SessionCacheIndexes,
        ctx: &DeviceContext,
        key: &PipelineKey,
        pipeline_lifetime: Option<&reims_vgpu_core::ResourceLifetime>,
        vert_module: vk::ShaderModule,
        vert_inputs: &VertexInputWidths,
        // The reflected-layout words `vert_module` was built from. Retained for
        // the driver breadcrumb around graphics pipeline creation.
        vert_spirv: &[u32],
        frag_module: vk::ShaderModule,
        // Read only by the driver breadcrumb: a graphics compile consumes both
        // stages and nothing outside the driver can say which one it choked on,
        // so both go to disk across the call.
        frag_spirv: &[u32],
        pipeline_layout: vk::PipelineLayout,
        render_pass: vk::RenderPass,
        counters: &EngineCounters,
        pools: &mut RecordingPools<'_>,
    ) -> Result<vk::Pipeline, DrawError> {
        if let Some(identity) = pipeline_lifetime {
            if let Some(pipeline) = indexes.pipeline_objects.get(identity, key) {
                counters.pipeline_hits.fetch_add(1, Ordering::Relaxed);
                counters
                    .pipeline_object_hits
                    .fetch_add(1, Ordering::Relaxed);
                return Ok(pipeline);
            }
        }
        if let Some(err) = self.pipelines.get_negative(key) {
            counters.pipeline_misses.fetch_add(1, Ordering::Relaxed);
            return Err(err);
        }
        if let Some((p, front)) = self.pipelines.get_routed(key) {
            counters.pipeline_hits.fetch_add(1, Ordering::Relaxed);
            if front {
                counters.pipeline_front_hits.fetch_add(1, Ordering::Relaxed);
            }
            if let Some(identity) = pipeline_lifetime {
                indexes.pipeline_objects.remember(identity, key, p);
            }
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
                reims_vgpu_observe::Emit::decline("vk_engine_pipeline", &reason).fail();
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
            reims_vgpu_observe::Emit::decline("vk_engine_pipeline", &reason).fail();
            let err = DrawError::Unsupported(reason);
            self.pipelines.insert_negative(key.clone(), err.clone());
            return Err(err);
        }
        let line_width = f32::from_bits(key.line_width_bits);
        let [line_width_min, line_width_max] = ctx.features.line_width_range;
        if !key.rasterizer_discard && line_width != 1.0 && !ctx.features.wide_lines {
            let reason = super::reason::DrawReason::WideLinesUnsupported {
                requested_bits: key.line_width_bits,
            };
            reims_vgpu_observe::Emit::decline("vk_engine_pipeline", &reason).fail();
            let err = DrawError::Unsupported(reason);
            self.pipelines.insert_negative(key.clone(), err.clone());
            return Err(err);
        }
        if !key.rasterizer_discard
            && line_width != 1.0
            && (!line_width.is_finite()
                || line_width < line_width_min
                || line_width > line_width_max)
        {
            let reason = super::reason::DrawReason::LineWidthOutOfRange {
                requested_bits: key.line_width_bits,
                min_bits: line_width_min.to_bits(),
                max_bits: line_width_max.to_bits(),
            };
            reims_vgpu_observe::Emit::decline("vk_engine_pipeline", &reason).fail();
            let err = DrawError::Unsupported(reason);
            self.pipelines.insert_negative(key.clone(), err.clone());
            return Err(err);
        }
        if key.depth_clip != DepthClipMode::default() && !ctx.features.depth_clamp {
            let reason = super::reason::DrawReason::DepthClampUnsupported;
            reims_vgpu_observe::Emit::decline("vk_engine_pipeline", &reason).fail();
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
        let mut attribute_formats = Vec::with_capacity(key.attrs.len());
        for attr in &key.attrs {
            let binding =
                match ctx
                    .vertex_formats
                    .resolve(attr.format, attr.offset, attr.stride, || {
                        vert_inputs.at(attr.location)
                    }) {
                    Ok(binding) => binding,
                    Err(translate_reason) => {
                        let err = DrawError::Unsupported(super::reason::DrawReason::VertexFormat(
                            translate_reason,
                        ));
                        reims_vgpu_observe::Emit::decline(
                            "vk_engine_vertex_format",
                            &translate_reason,
                        )
                        .fail_once(
                            (u64::from(attr.location) << 32) | u64::from(translate_reason.value()),
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
                reims_vgpu_observe::Emit::decline("vk_engine_vertex_format", &decline).fail_once(
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
        let input_asm = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(crate::translate::raster::vk_topology(key.topology));
        // Dynamic viewport/scissor so L5 key need not include extent (flip flag is static).
        // Stencil reference is dynamic (Metal's `SetStencilReferenceValue` is a
        // command distinct from the state object) so distinct references reuse
        // one pipeline; only listed for stencil pipelines.
        let mut dynamic_states = vec![vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        if key.stencil.is_some() {
            dynamic_states.push(vk::DynamicState::STENCIL_REFERENCE);
        }
        if key.depth_bias_enable {
            dynamic_states.push(vk::DynamicState::DEPTH_BIAS);
        }
        if key
            .blend
            .into_iter()
            .chain(key.secondary_blend.iter().copied().flatten())
            .any(BlendKey::uses_constants)
        {
            dynamic_states.push(vk::DynamicState::BLEND_CONSTANTS);
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
            .rasterizer_discard_enable(key.rasterizer_discard)
            .depth_bias_enable(key.depth_bias_enable)
            .line_width(line_width);
        // `rasterSampleCount` is a property of `MTLRenderPipelineDescriptor`,
        // so it reaches this device inside the render pipeline's own
        // compact-TLV block. The pass key carries that decoded count, and the
        // render-pass attachment is created with the same value; unsupported
        // count/attachment combinations are refused before either object is
        // built.
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk_sample_count(key.pass.0.sample_count));
        // One blend attachment state per color attachment; Vulkan requires the
        // count to match the render pass. Every slot uses its own decoded
        // blend, slot 0 from `key.blend` and slot n from
        // `key.secondary_blend[n-1]`.
        //
        // The secondaries used to be forced unblended here, justified by a
        // comment saying the decode side did not carry per-attachment blend
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
                    .src_color_blend_factor(crate::translate::blend::vk_factor(b.src_color))
                    .dst_color_blend_factor(crate::translate::blend::vk_factor(b.dst_color))
                    .color_blend_op(crate::translate::blend::vk_operation(b.color_op))
                    .src_alpha_blend_factor(crate::translate::blend::vk_factor(b.src_alpha))
                    .dst_alpha_blend_factor(crate::translate::blend::vk_factor(b.dst_alpha))
                    .alpha_blend_op(crate::translate::blend::vk_operation(b.alpha_op)),
                None => vk::PipelineColorBlendAttachmentState::default()
                    .color_write_mask(write)
                    .blend_enable(false),
            }
        };
        let mut blend_att = vec![attachment_blend(key.blend, key.color_write_mask[0])];
        for slot in 0..key.pass.secondary_count() {
            blend_att.push(attachment_blend(
                key.secondary_blend[slot],
                key.color_write_mask[slot + 1],
            ));
        }
        let blend = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(&blend_att)
            .blend_constants([0.0; 4]);
        // Depth-stencil state: attached ONLY when the pass carries a depth
        // attachment (Vulkan requires the pipeline's depth-stencil state to be
        // consistent with the subpass). Without it the color-only pipeline is
        // byte-identical to the pre-depth engine. Stencil is enabled only when
        // the bound state requested it (`key.stencil`); the reference field is
        // left 0 here and supplied dynamically per draw.
        let stencil_face = |ops: super::types::StencilFaceOps| {
            vk::StencilOpState::default()
                .fail_op(crate::translate::raster::vk_stencil_op(ops.fail_op))
                .pass_op(crate::translate::raster::vk_stencil_op(ops.pass_op))
                .depth_fail_op(crate::translate::raster::vk_stencil_op(ops.depth_fail_op))
                .compare_op(crate::translate::raster::vk_compare_op(ops.compare))
                .compare_mask(ops.read_mask)
                .write_mask(ops.write_mask)
                .reference(0)
        };
        let mut depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(key.depth_test)
            .depth_write_enable(key.depth_write)
            .depth_compare_op(crate::translate::raster::vk_compare_op(key.depth_compare))
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
        if key.feedback_colors != 0 {
            gpci = gpci.flags(vk::PipelineCreateFlags::COLOR_ATTACHMENT_FEEDBACK_LOOP_EXT);
        }
        if key.pass.has_depth() {
            gpci = gpci.depth_stencil_state(&depth_stencil);
        }
        // The third call that compiles a module this device assembled, and the
        // only one that compiles two at once. A macOS 15 guest's CoreAnimation
        // uber fragment shader has been observed keeping NVIDIA's compiler in
        // here for over ten minutes with the device lock held; see
        // `reims_vgpu_observe::driver_watch`, which this arming also starts.
        let breadcrumb = match super::driver_breadcrumb::DriverBreadcrumb::arm(
            &format!(
                "create_graphics_pipelines vert_words={} frag_words={}",
                vert_spirv.len(),
                frag_spirv.len()
            ),
            &[("vert", vert_spirv), ("frag", frag_spirv)],
        ) {
            Ok(breadcrumb) => breadcrumb,
            Err(hit) => {
                let err = self.note_quarantined("create_graphics_pipelines", &hit);
                self.pipelines.insert_negative(key.clone(), err.clone());
                return Err(err);
            }
        };
        let created = ctx
            .device
            .create_graphics_pipelines(ctx.pipeline_cache, &[gpci], None);
        breadcrumb.disarm();
        let pipe = created.map_err(|(_, e)| {
            let err = DrawError::VkCall(VkCall::new(VkOp::CachesCreateGraphicsPipelines, e));
            self.pipelines.insert_negative(key.clone(), err.clone());
            err
        })?[0];
        counters.note_create(CreateSite::GraphicsPipeline);
        // A fresh pipeline compile grew the VkPipelineCache — persist it so
        // the next boot warm-starts (file write is off-thread, debounced).
        ctx.persist_pipeline_cache();
        if let Some(old) = self.pipelines.insert(key.clone(), pipe) {
            pools.dispose(&ctx.device, DeferredHandle::Pipeline(old));
        }
        if let Some(identity) = pipeline_lifetime {
            indexes.pipeline_objects.remember(identity, key, pipe);
        }
        Ok(pipe)
    }

    pub(crate) unsafe fn get_or_create_compute_pipeline(
        &mut self,
        ctx: &DeviceContext,
        key: &ComputePipelineKey,
        shader: ShaderModuleSource<'_>,
        pipeline_layout: vk::PipelineLayout,
        counters: &EngineCounters,
        pools: &mut RecordingPools<'_>,
    ) -> Result<vk::Pipeline, DrawError> {
        if let Some(err) = self.compute_pipelines.get_negative(key) {
            counters
                .compute_pipeline_misses
                .fetch_add(1, Ordering::Relaxed);
            return Err(err);
        }
        if let Some(p) = self.compute_pipelines.get(key) {
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
            .module(shader.module)
            .name(&entry_c);
        let cpci = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout);
        // The other call that compiles the module, and the one an NVIDIA driver
        // has been observed dying inside on a macos-14 guest's first dispatch.
        let breadcrumb = match super::driver_breadcrumb::DriverBreadcrumb::arm(
            &format!("create_compute_pipelines entry={}", key.entry),
            &[("kernel", shader.spirv)],
        ) {
            Ok(breadcrumb) => breadcrumb,
            Err(hit) => {
                let err = self.note_quarantined("create_compute_pipelines", &hit);
                self.compute_pipelines
                    .insert_negative(key.clone(), err.clone());
                return Err(err);
            }
        };
        let created = ctx
            .device
            .create_compute_pipelines(ctx.pipeline_cache, &[cpci], None);
        breadcrumb.disarm();
        let pipe = created.map_err(|(_, e)| {
            let err = DrawError::VkCall(VkCall::new(VkOp::CachesCreateComputePipelines, e));
            self.compute_pipelines
                .insert_negative(key.clone(), err.clone());
            err
        })?[0];
        counters.note_create(CreateSite::ComputePipeline);
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

    #[derive(Clone)]
    struct CountingKey {
        value: u32,
        hashes: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl PartialEq for CountingKey {
        fn eq(&self, other: &Self) -> bool {
            self.value == other.value
        }
    }

    impl Eq for CountingKey {}

    impl std::hash::Hash for CountingKey {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            self.hashes
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.value.hash(state);
        }
    }

    #[test]
    fn an_empty_negative_cache_does_not_hash_the_object_key() {
        let hashes = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let key = CountingKey {
            value: 7,
            hashes: std::sync::Arc::clone(&hashes),
        };
        let mut cache: ObjectCache<CountingKey, u32> = ObjectCache::new();

        assert_eq!(cache.get_negative(&key), None);
        assert_eq!(hashes.load(std::sync::atomic::Ordering::Relaxed), 0);

        cache.insert_negative(
            CountingKey {
                value: 9,
                hashes: std::sync::Arc::clone(&hashes),
            },
            DrawError::VkCall(VkCall::new(
                VkOp::CachesCreateShaderModule,
                vk::Result::ERROR_UNKNOWN,
            )),
        );
        hashes.store(0, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(cache.get_negative(&key), None);
        assert_eq!(hashes.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    #[test]
    fn push_layout_selection_uses_the_device_limit_and_keeps_the_fallback() {
        let caps = crate::push_descriptor::PushDescriptorCaps {
            max_descriptors: 32,
        };
        let layout = |counts: &[u32]| LayoutKey {
            kernel_grid: None,
            bindings: counts
                .iter()
                .enumerate()
                .map(|(binding, &count)| BindingSig {
                    binding: binding as u32,
                    ty: vk::DescriptorType::STORAGE_BUFFER.as_raw() as u32,
                    stages: vk::ShaderStageFlags::COMPUTE.as_raw(),
                    count,
                })
                .collect(),
        };
        assert!(layout(&[16, 16]).uses_push_descriptors(caps));
        assert!(!layout(&[16, 17]).uses_push_descriptors(caps));
        assert!(!layout(&[]).uses_push_descriptors(caps));
    }

    /// Pipeline and live-pass compatibility exclude load actions but retain
    /// attachment formats and subpass shape.
    #[test]
    fn pass_compatibility_ignores_only_load_actions() {
        let mut clear = PassKey::single(ColorLoadKey::Clear, vk::Format::B8G8R8A8_UNORM);
        clear.secondary_count = 1;
        clear.secondary[0] = SecondaryAttachKey {
            format: vk::Format::R16G16_SFLOAT,
            load: ColorLoadKey::Clear,
        };
        clear.depth = Some(DepthAttachKey {
            depth: true,
            load: ColorLoadKey::Clear,
            stencil: true,
            ..Default::default()
        });

        let mut load = clear;
        load.color0_load = ColorLoadKey::Load;
        load.secondary[0].load = ColorLoadKey::Load;
        load.depth.as_mut().unwrap().load = ColorLoadKey::Load;
        assert_eq!(clear.compatibility(), load.compatibility());

        let mut different_format = load;
        different_format.secondary[0].format = vk::Format::R32_SFLOAT;
        assert_ne!(clear.compatibility(), different_format.compatibility());

        let mut different_subpass = load;
        different_subpass.color_input = true;
        assert_ne!(clear.compatibility(), different_subpass.compatibility());

        let mut different_depth = load;
        different_depth.depth.as_mut().unwrap().stencil = false;
        assert_ne!(clear.compatibility(), different_depth.compatibility());
    }

    #[test]
    fn color_dont_care_reaches_the_native_discard_load_operation() {
        let final_layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
        assert_eq!(
            ColorLoadKey::DontCare.vulkan(final_layout),
            (vk::AttachmentLoadOp::DONT_CARE, vk::ImageLayout::UNDEFINED,)
        );
        assert_eq!(
            ColorLoadKey::Clear.vulkan(final_layout).0,
            vk::AttachmentLoadOp::CLEAR
        );
        assert_eq!(
            ColorLoadKey::Load.vulkan(final_layout),
            (vk::AttachmentLoadOp::LOAD, final_layout)
        );
    }

    #[test]
    fn primary_attachment_projection_preserves_its_sample_count() {
        let mut compound = PassKey::single(ColorLoadKey::Load, vk::Format::B8G8R8A8_UNORM);
        compound.sample_count = 4;
        compound.multisample_resolve = true;
        compound.color_input = true;
        compound.depth = Some(DepthAttachKey {
            depth: true,
            load: ColorLoadKey::Load,
            stencil: true,
            ..Default::default()
        });
        compound.secondary_count = 1;

        let primary = compound.primary_attachment_only();
        assert_eq!(primary.sample_count, 4);
        assert_eq!(primary.color0_load, ColorLoadKey::Load);
        assert_eq!(primary.color0_format, vk::Format::B8G8R8A8_UNORM);
        assert_eq!(primary.secondary_count, 0);
        assert_eq!(primary.depth, None);
        assert!(!primary.color_input);
        assert!(!primary.multisample_resolve);
    }

    /// `first_difference` answers `None` on exactly the pairs that compare
    /// equal, which is what makes `passcompat_*` a partition of
    /// `passdiff_compat` rather than a second opinion about it.
    ///
    /// This is the property that matters, and it is not the same as "every
    /// field has a variant": a field the destructure names but the body forgets
    /// to compare would still let two unequal keys answer `None`, and the caller
    /// would then charge the *next* echo field — reporting a framebuffer change
    /// where an attachment shape moved. So the assertion is over mutations, one
    /// per field, and it is made in both directions.
    #[test]
    fn every_compatibility_difference_is_named_and_equal_keys_name_none() {
        let mut base = PassKey::single(ColorLoadKey::Clear, vk::Format::B8G8R8A8_UNORM);
        base.secondary_count = 1;
        base.secondary[0] = SecondaryAttachKey {
            format: vk::Format::R16G16_SFLOAT,
            load: ColorLoadKey::Clear,
        };
        base.depth = Some(DepthAttachKey {
            depth: true,
            load: ColorLoadKey::Clear,
            stencil: true,
            ..Default::default()
        });
        base.sample_count = 1;

        assert_eq!(
            base.compatibility().first_difference(base.compatibility()),
            None
        );

        // A load action is erased by `compatibility`, so it is not a difference
        // — the one mutation below that must answer `None`.
        /// One named mutation of a [`PassKey`] and the difference it must
        /// produce. `None` for a mutation `compatibility` erases.
        type Mutation = (&'static str, fn(&mut PassKey), Option<PassCompatField>);
        let mutations: &[Mutation] = &[
            (
                "load actions",
                |k| {
                    k.color0_load = ColorLoadKey::Load;
                    k.secondary[0].load = ColorLoadKey::Load;
                    k.depth.as_mut().unwrap().load = ColorLoadKey::Load;
                },
                None,
            ),
            (
                "color0 format",
                |k| k.color0_format = vk::Format::R8G8B8A8_UNORM,
                Some(PassCompatField::Color0Format),
            ),
            (
                "secondary count",
                |k| k.secondary_count = 2,
                Some(PassCompatField::SecondaryCount),
            ),
            (
                "secondary format",
                |k| k.secondary[0].format = vk::Format::R32_SFLOAT,
                Some(PassCompatField::SecondaryFormat),
            ),
            ("depth", |k| k.depth = None, Some(PassCompatField::Depth)),
            (
                "depth aspect",
                |k| k.depth.as_mut().unwrap().depth = false,
                Some(PassCompatField::Depth),
            ),
            (
                "color input",
                |k| k.color_input = true,
                Some(PassCompatField::ColorInput),
            ),
            // Feedback is a property of the draw, and it makes two passes
            // incompatible only on the arm where it still moves a layout. On the
            // shipping arm `compatibility` erases it, which is what stops a
            // feedback draw closing the render pass an ordinary one opened.
            (
                "feedback",
                |k| k.feedback_colors = 1,
                if color_feedback_layout() == color0_pass_exit_layout() {
                    None
                } else {
                    Some(PassCompatField::FeedbackColors)
                },
            ),
            (
                "sample count",
                |k| k.sample_count = 4,
                Some(PassCompatField::SampleCount),
            ),
            (
                "resolve",
                |k| k.multisample_resolve = true,
                Some(PassCompatField::MultisampleResolve),
            ),
        ];

        for (name, mutate, expected) in mutations {
            let mut moved = base;
            mutate(&mut moved);
            let (a, b) = (base.compatibility(), moved.compatibility());
            assert_eq!(a.first_difference(b), *expected, "{name}");
            assert_eq!(b.first_difference(a), *expected, "{name}, reversed");
            // The partition itself: `None` iff equal, on every input above.
            assert_eq!(
                a.first_difference(b).is_none(),
                a == b,
                "{name}: a named difference and key equality must agree"
            );
        }
    }

    /// Framebuffers bind attachment views, not load actions or dependencies.
    #[test]
    fn framebuffer_compatibility_ignores_load_and_dependency_state() {
        let plain = PassKey::single(ColorLoadKey::Clear, vk::Format::B8G8R8A8_UNORM);
        let mut transported = plain;
        transported.color0_load = ColorLoadKey::Load;
        transported.feedback_colors = 1;

        assert_eq!(
            plain.framebuffer_compatibility(),
            transported.framebuffer_compatibility()
        );
        let mut input = transported;
        input.color_input = true;
        assert_ne!(
            plain.framebuffer_compatibility(),
            input.framebuffer_compatibility()
        );
    }

    #[test]
    fn layout_bindings_coalesce_array_elements_and_refuse_conflicting_shapes() {
        let sig = |count| BindingSig {
            binding: 32,
            ty: vk::DescriptorType::SAMPLED_IMAGE.as_raw() as u32,
            stages: vk::ShaderStageFlags::COMPUTE.as_raw(),
            count,
        };
        assert_eq!(
            canonicalize_layout_bindings(vec![sig(8), sig(8)]),
            Ok(vec![sig(8)])
        );
        assert!(matches!(
            canonicalize_layout_bindings(vec![sig(8), sig(4)]),
            Err(super::super::DrawError::Unsupported(
                super::super::reason::DrawReason::DescriptorBindingConflict {
                    binding: 32,
                    first_count: 8,
                    second_count: 4,
                    ..
                }
            ))
        ));
    }

    #[test]
    fn vertex_format_widening_names_both_formats_and_attribute() {
        use reims_vgpu_observe::Decline as _;
        let narrow = translate::vertex::vertex_layout(VertexAttributeFormat::UChar3Normalized).vk;
        let binding = translate::VertexFormatSupport::with_unsupported(&[narrow])
            .resolve(VertexAttributeFormat::UChar3Normalized, 12, 32, || {
                crate::spirv_vertex_input::InputWidth::Components(3)
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
            reims_vgpu_observe::Emit::decline("vk_engine_vertex_format", &decline).render(),
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
            Some(0xC0FFEE),
            "the hot first entry is still served after 4095 later ones"
        );
        assert_eq!(c.get_routed(&0), Some((0xC0FFEE, true)));
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
        assert_eq!(c.get(&1), Some(20));
    }

    #[test]
    fn clearing_the_cache_forgets_the_front_value_with_the_owned_object() {
        let mut c: ObjectCache<u32, u32> = ObjectCache::new();
        c.insert(1, 20);
        assert_eq!(c.get_routed(&1), Some((20, true)));

        c.clear();

        assert_eq!(c.get_routed(&1), None);
        assert!(c.front.is_none());
    }

    #[test]
    fn retained_object_front_requires_the_same_identity_and_exact_variant() {
        let mut index: ObjectVariantIndex<u32, u32> = ObjectVariantIndex::default();
        let first = reims_vgpu_core::ResourceLifetime::new();
        let second = reims_vgpu_core::ResourceLifetime::new();

        index.remember(&first, &7, 70);
        assert_eq!(index.get(&first, &7), Some(70));
        assert_eq!(index.get(&first, &8), None, "a Vulkan-only variant differs");
        assert_eq!(
            index.get(&second, &7),
            None,
            "equal content under another guest object is not this object's front"
        );

        index.remember(&first, &8, 80);
        assert_eq!(index.get(&first, &7), None, "one exact last variant");
        assert_eq!(index.get(&first, &8), Some(80));
    }

    #[test]
    fn retained_object_front_reaps_dead_identities_without_capacity_eviction() {
        let mut index: ObjectVariantIndex<u32, u32> = ObjectVariantIndex::default();
        let first = reims_vgpu_core::ResourceLifetime::new();
        index.remember(&first, &1, 10);
        assert_eq!(index.map.len(), 1);
        drop(first);

        let second = reims_vgpu_core::ResourceLifetime::new();
        index.remember(&second, &2, 20);
        assert_eq!(index.map.len(), 1, "the expired object's weak entry went");
        assert_eq!(index.get(&second, &2), Some(20));

        index.clear();
        assert!(index.map.is_empty());
        assert_eq!(index.get(&second, &2), None);
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
        assert_eq!(c.get(&7), Some(42));
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
        assert_eq!(c.get(&key), Some(0x5EED));
    }

    /// The index is keyed on the *allocation*, not on the contents, and that is
    /// the whole of its soundness argument: it holds the `Arc`, so the address
    /// cannot be reused while the entry lives.
    ///
    /// Two `Arc`s over identical words are two allocations and therefore two
    /// entries. That is not a miss to fix — a content key is what the digest
    /// already is, and rederiving it is what this index exists to avoid.
    #[test]
    fn the_shader_digest_index_keys_the_allocation_and_not_the_contents() {
        let mut index = ShaderDigestIndex::default();
        let words = std::sync::Arc::new(vec![0x0723_0203u32, 0x0001_0000, 0x000d_000b]);
        let twin = std::sync::Arc::new((*words).clone());
        let digest = Digest128 {
            a: 0xA1,
            b: 0xB2,
            len: 3,
        };

        assert_eq!(index.get(&words), None, "nothing walked yet");
        index.insert(&words, digest);

        assert_eq!(index.get(&words), Some(digest));
        assert_eq!(
            index.get(&twin),
            None,
            "identical words in a second allocation are a second entry"
        );
        let alias = std::sync::Arc::clone(&words);
        assert_eq!(
            index.get(&alias),
            Some(digest),
            "a clone of the same Arc is the same allocation and the same entry"
        );
    }

    /// A dropped module's address may be handed to the next allocation, and the
    /// index must not answer for it. It cannot: the entry holds an `Arc`, so
    /// while it lives the allocation is not freed and the address is not
    /// available to hand out.
    ///
    /// This asserts the mechanism rather than the hazard — a test that freed an
    /// allocation and hoped for the address back would be testing the allocator.
    #[test]
    fn a_shader_digest_entry_keeps_its_words_alive() {
        let mut index = ShaderDigestIndex::default();
        let words = std::sync::Arc::new(vec![1u32, 2, 3]);
        index.insert(&words, Digest128 { a: 1, b: 2, len: 3 });
        assert_eq!(
            std::sync::Arc::strong_count(&words),
            2,
            "the index holds one, which is what makes its key an address"
        );
        index.clear();
        assert_eq!(std::sync::Arc::strong_count(&words), 1, "and releases it");
    }

    /// The bound is the container's and it starts over rather than evicting,
    /// because every entry is equally cheap to rebuild and there is no recency
    /// to evict by.
    #[test]
    fn the_shader_digest_index_starts_over_at_its_bound() {
        let mut index = ShaderDigestIndex::default();
        let held: Vec<std::sync::Arc<Vec<u32>>> = (0..SHADER_DIGEST_ENTRIES)
            .map(|i| std::sync::Arc::new(vec![i as u32]))
            .collect();
        for (i, words) in held.iter().enumerate() {
            index.insert(
                words,
                Digest128 {
                    a: i as u64,
                    b: 0,
                    len: 1,
                },
            );
        }
        assert_eq!(index.map.len(), SHADER_DIGEST_ENTRIES);
        assert!(index.get(&held[0]).is_some());

        let one_more = std::sync::Arc::new(vec![0xFFFF_FFFFu32]);
        index.insert(&one_more, Digest128 { a: 9, b: 9, len: 1 });

        assert_eq!(index.map.len(), 1, "the bound holds by starting over");
        assert_eq!(index.get(&one_more), Some(Digest128 { a: 9, b: 9, len: 1 }));
        assert!(
            index.get(&held[0]).is_none(),
            "and the reset is total, so nothing survives to be answered stale"
        );
    }

    /// Every pass shape states the colour scope on **both** external
    /// dependencies, because declaring one explicit external dependency for any
    /// reason removes the implicit one from every attachment.
    ///
    /// This is the check the previous shape could not pass. It built the pair
    /// only for a depth pass and named depth-stencil stages and accesses alone,
    /// so on `(depth, _)` the colour attachment's transitions were ordered
    /// against nothing — which the synchronization validation layer reported at
    /// both `vkCmdBeginRenderPass` and `vkCmdEndRenderPass`. Asserting the
    /// colour terms across all four shapes is what stops a future edit adding a
    /// The narrow probe removes the transfer and sampler destinations and
    /// changes nothing else — in particular it keeps the colour-attachment
    /// destination, which is the one consumer that deliberately issues no
    /// barrier of its own.
    ///
    /// `super::exec::pass_exit_needs_no_barrier` drops the next draw's barrier
    /// when it renders into the same target, and the only thing that then orders
    /// that draw's writes after this pass's is the outgoing dependency. So a
    /// narrowing that took `COLOR_ATTACHMENT_OUTPUT` out of the destination
    /// scope would be a write-after-write hazard on the most common draw in the
    /// device, and it would be silent. The source scope must not move either:
    /// what is being priced is the *visibility* request, not the ordering one.
    ///
    /// See [`reims_vgpu_config::PASS_EXIT_NARROW`] for what the probe is asking and for
    /// the validation-layer run it owes before it could ever be a default.
    #[test]
    fn the_narrow_pass_exit_keeps_the_consumer_that_issues_no_barrier() {
        for has_depth in [false, true] {
            for color_input in [false, true] {
                let wide = external_dependencies(has_depth, color_input, false)[1];
                let narrow = external_dependencies(has_depth, color_input, true)[1];
                let shape = format!("depth={has_depth} color_input={color_input}");

                assert_eq!(
                    (narrow.src_stage_mask, narrow.src_access_mask),
                    (wide.src_stage_mask, wide.src_access_mask),
                    "{shape}: the probe prices visibility and must not weaken ordering"
                );
                assert!(
                    narrow
                        .dst_stage_mask
                        .contains(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                        && narrow
                            .dst_access_mask
                            .contains(vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
                    "{shape}: the next draw into this target issues no barrier of \
                     its own, so its stage must survive the narrowing"
                );
                assert!(
                    !narrow
                        .dst_stage_mask
                        .contains(vk::PipelineStageFlags::TRANSFER)
                        && !narrow
                            .dst_stage_mask
                            .contains(vk::PipelineStageFlags::FRAGMENT_SHADER),
                    "{shape}: the probe removes exactly the two destinations whose \
                     consumers barrier for themselves"
                );
                assert!(
                    wide.dst_stage_mask
                        .contains(vk::PipelineStageFlags::TRANSFER)
                        && wide
                            .dst_stage_mask
                            .contains(vk::PipelineStageFlags::FRAGMENT_SHADER),
                    "{shape}: and the shipping arm is unchanged, or the probe is \
                     measuring against itself"
                );
                if has_depth {
                    assert!(
                        narrow
                            .dst_stage_mask
                            .contains(vk::PipelineStageFlags::LATE_FRAGMENT_TESTS),
                        "{shape}: depth is an attachment stage and stays"
                    );
                }
            }
        }
    }

    /// dependency for one attachment class and dropping the others again.
    #[test]
    fn both_external_dependencies_name_the_colour_scope_in_every_pass_shape() {
        for has_depth in [false, true] {
            for color_input in [false, true] {
                // The shipping scope. The probe arm has its own test.
                let [incoming, outgoing] = external_dependencies(has_depth, color_input, false);
                let shape = format!("depth={has_depth} color_input={color_input}");

                // Incoming: the transition into the attachment layout has to be
                // ordered against the loadOp that writes it.
                assert!(
                    incoming
                        .dst_stage_mask
                        .contains(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT),
                    "{shape}: the loadOp clear runs at COLOR_ATTACHMENT_OUTPUT"
                );
                assert!(
                    incoming
                        .dst_access_mask
                        .contains(vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
                    "{shape}: and it is a colour write"
                );

                // Outgoing: the final transition has to be ordered against the
                // subpass's own store, and the store made visible to the copy
                // that reads the target after the pass.
                assert!(
                    outgoing
                        .src_stage_mask
                        .contains(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT),
                    "{shape}: the store runs at COLOR_ATTACHMENT_OUTPUT"
                );
                assert!(
                    outgoing
                        .src_access_mask
                        .contains(vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
                    "{shape}: and it is a colour write"
                );
                assert!(
                    outgoing
                        .dst_stage_mask
                        .contains(vk::PipelineStageFlags::TRANSFER)
                        && outgoing
                            .dst_access_mask
                            .contains(vk::AccessFlags::TRANSFER_READ),
                    "{shape}: a reader still barriers slot 0 into TRANSFER_SRC_OPTIMAL \
                     for itself, and this is the scope its transition orders against"
                );

                // Depth is stated only where the pass has a depth attachment —
                // the fix is to add the missing class, not to name every class
                // on every pass.
                assert_eq!(
                    incoming
                        .dst_access_mask
                        .contains(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE),
                    has_depth,
                    "{shape}: depth terms follow the depth attachment"
                );

                // Framebuffer fetch reads attachment 0 in the fragment stage, so
                // the entry transition must be visible to that read as well.
                assert_eq!(
                    incoming
                        .dst_access_mask
                        .contains(vk::AccessFlags::INPUT_ATTACHMENT_READ),
                    color_input,
                    "{shape}: input-attachment terms follow the fetch"
                );
            }
        }
    }

    /// The pass key is the single source of truth for every place that names an
    /// attachment layout, and a feedback slot names the one
    /// [`color_feedback_layout`] answers at both ends of the pass.
    ///
    /// Asserted against that function rather than against a spelled layout, so it
    /// states the relation on both arms: under the shipping layout every slot is
    /// in the *same* layout feedback or not, and under
    /// [`reims_vgpu_config::COLOR_GENERAL`]`=off` the feedback slots separate because
    /// `COLOR_ATTACHMENT_OPTIMAL` admits no feedback loop. The failure it guards
    /// is a descriptor and a subpass reference naming one image differently,
    /// which is undefined behaviour reported nowhere.
    #[test]
    fn feedback_attachment_layout_is_derived_consistently_from_the_mask() {
        let mut key = PassKey::single(ColorLoadKey::Load, vk::Format::R8G8B8A8_UNORM);
        key.color_input = true;
        key.feedback_colors = (1 << 0) | (1 << 3);

        for index in 0..=MAX_SECONDARY_ATTACH {
            let feedback = index == 0 || index == 3;
            assert_eq!(key.color_feedback(index), feedback);
            let want = if feedback {
                color_feedback_layout()
            } else {
                color0_pass_exit_layout()
            };
            assert_eq!(key.color_layout(index), want);
            assert_eq!(key.color_final_layout(index), want);
        }
        assert!(!key.color_feedback(u8::BITS as usize));

        // Whatever the arm, the layout a feedback slot lands in must be one a
        // feedback loop is legal in. This is the check that would have caught the
        // shipping defect: with the resting layout moved to GENERAL, the slot was
        // still being placed in the extension layout while the image sat in
        // GENERAL.
        assert!(layout_admits_color_feedback(color_feedback_layout()));
    }

    /// One layout for a colour target means one for a sampled one too.
    ///
    /// The whole point of the repair: while the resting layout admits a feedback
    /// loop, a slot the guest samples and a slot it does not are in the *same*
    /// layout, so the render pass declares no transition for either and a
    /// framebuffer built for one serves the other.
    #[test]
    fn a_sampled_colour_slot_rests_where_every_other_colour_slot_rests() {
        if layout_admits_color_feedback(color0_pass_exit_layout()) {
            assert_eq!(color_feedback_layout(), color0_pass_exit_layout());
        } else {
            // The ablation arm, where the contract forces the second layout back.
            assert_eq!(
                color_feedback_layout(),
                vk::ImageLayout::ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT
            );
        }
    }

    /// A render pass may only vary in ways
    /// [`PassKey::framebuffer_compatibility`] preserves, and dependencies are not
    /// among the things Vulkan's compatibility rule spares.
    ///
    /// `feedback_colors` is erased by that key, so the feedback self-dependency
    /// must be declared on every pass rather than only on feedback ones — a
    /// conditional one made `dependencyCount` differ between two passes the key
    /// called interchangeable, which is what the validation layer reported as
    /// `VUID-VkRenderPassBeginInfo-renderPass-00904` on a driven Maps boot.
    #[test]
    fn the_dependency_count_does_not_move_with_anything_the_framebuffer_key_erases() {
        let base = PassKey::single(ColorLoadKey::Load, vk::Format::R8G8B8A8_UNORM);
        for feedback in [0u8, 1, (1 << 0) | (1 << 3)] {
            let mut key = base;
            key.feedback_colors = feedback;
            assert_eq!(
                key.framebuffer_compatibility(),
                base.framebuffer_compatibility(),
                "the framebuffer key must erase feedback"
            );
            // Same framebuffer key ⇒ the passes must agree on dependency
            // count, because a framebuffer built against one is used with the
            // other.
            assert_eq!(
                pass_dependency_count(key),
                pass_dependency_count(base),
                "feedback={feedback}"
            );
        }
    }

    /// The dependency list a pass of this shape is built with, counted without a
    /// device. Mirrors the `deps` construction in `get_or_create_render_pass`.
    fn pass_dependency_count(key: PassKey) -> usize {
        external_dependencies(
            key.depth.is_some(),
            key.color_input,
            pass_exit_scope_narrow(),
        )
        .len()
            + usize::from(key.color_input)
            + 1
    }

    /// Erasing feedback from the pass key must not erase it from the device.
    ///
    /// `compatibility()` drops `feedback_colors` on the shipping arm so a feedback
    /// draw can continue an ordinary draw's render pass. The create flag
    /// `VK_PIPELINE_CREATE_COLOR_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT` is what makes
    /// that draw legal, it is fixed at pipeline creation, and it is therefore the
    /// one thing that must **not** follow the field out of the pass key. This
    /// asserts the split: same pass compatibility, different pipeline key.
    ///
    /// Without it, the pass-merge win silently turns every feedback draw into a
    /// sampled read of an attachment it is writing with no feedback loop enabled.
    #[test]
    fn feedback_leaves_pass_compatibility_without_leaving_the_pipeline() {
        let plain = PassKey::single(ColorLoadKey::Load, vk::Format::R8G8B8A8_UNORM);
        let mut feeds = plain;
        feeds.feedback_colors = 1;

        if color_feedback_layout() == color0_pass_exit_layout() {
            assert_eq!(
                plain.compatibility(),
                feeds.compatibility(),
                "a feedback draw must be able to continue an ordinary draw's pass"
            );
        }
        // Whatever the pass key did, the draw's own answer is still reachable and
        // is what the pipeline key is built from.
        assert_eq!(plain.feedback_colors, 0);
        assert_eq!(feeds.feedback_colors, 1);
    }

    /// A subpass self-dependency sourced in a framebuffer-space stage may name
    /// only framebuffer-space stages as its destination
    /// (`VUID-VkSubpassDependency-srcSubpass-06809`). `VERTEX_SHADER` is not one,
    /// and naming it made every render pass this device created invalid.
    #[test]
    fn the_feedback_self_dependency_stays_in_framebuffer_space() {
        const FRAMEBUFFER_SPACE: vk::PipelineStageFlags = vk::PipelineStageFlags::from_raw(
            vk::PipelineStageFlags::FRAGMENT_SHADER.as_raw()
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS.as_raw()
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS.as_raw()
                | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT.as_raw(),
        );
        assert!(FRAMEBUFFER_SPACE.contains(COLOR_FEEDBACK_SRC.0));
        assert!(FRAMEBUFFER_SPACE.contains(COLOR_FEEDBACK_DST.0));

        // The in-pass barrier in `exec` is built from these same two constants,
        // which is what keeps it inside what the self-dependency declares.
        let dep = color_feedback_self_dependency(color_feedback_layout());
        assert_eq!(dep.src_stage_mask, COLOR_FEEDBACK_SRC.0);
        assert_eq!(dep.dst_stage_mask, COLOR_FEEDBACK_DST.0);
        assert!(dep
            .dependency_flags
            .contains(vk::DependencyFlags::BY_REGION));
        assert_eq!(dep.src_subpass, dep.dst_subpass);
    }

    /// An ordinary colour slot names one layout at all three points a pass can
    /// name one — the `initialLayout` a `LOAD` names, the subpass reference, and
    /// the `finalLayout` — so the pass performs no transition of its own at
    /// either end and the registry's record of where the image was left is the
    /// layout it is actually in.
    ///
    /// This is the relation, not a restatement of a constant: it holds for
    /// whatever [`color0_pass_exit_layout`] answers, including under
    /// [`reims_vgpu_config::COLOR_GENERAL`], which is the whole reason that answer is a
    /// function. It fails if any of the three grows a second spelling — which is
    /// how the MRT secondary arm in `exec` came to publish a hand-written
    /// `COLOR_ATTACHMENT_OPTIMAL` beside a feedback arm that derived.
    #[test]
    fn an_ordinary_colour_slot_enters_and_leaves_a_pass_at_one_layout() {
        let key = PassKey::single(ColorLoadKey::Load, vk::Format::B8G8R8A8_UNORM);
        for index in 0..=MAX_SECONDARY_ATTACH {
            assert_eq!(
                key.color_layout(index),
                color0_pass_exit_layout(),
                "{index}"
            );
            assert_eq!(
                key.color_final_layout(index),
                color0_pass_exit_layout(),
                "{index}"
            );
        }
    }
}
