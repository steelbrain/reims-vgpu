//! Backend-independent resolved render commands and typed results.

use crate::{ContentStamp, GatherVouch, ResourceLifetime, ResourceLifetimeRef, SamplerResource};
use reims_vgpu_memory::{GuestRunSource, GuestTargetPlan};
pub use reims_vgpu_protocol::{
    BlendFactor, BlendOp, BlendStateResource, CullMode, DepthClipMode, FillMode, IndexType,
    PrimitiveTopology, StencilOp, VertexAttributeFormat, VertexStepFunction, VisibilityResultMode,
};
use reims_vgpu_protocol::{ColorWriteMask, ImageFormat, SwizzlePlan};
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq)]
pub struct DepthState {
    pub identity: Option<crate::TargetIdentity>,
    pub test_enable: bool,
    pub write_enable: bool,
    pub compare: crate::SamplerCompareFunction,
    pub clear_value: f32,
    pub load: bool,
    pub stencil: Option<StencilState>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StencilFaceOps {
    pub compare: crate::SamplerCompareFunction,
    pub fail_op: StencilOp,
    pub depth_fail_op: StencilOp,
    pub pass_op: StencilOp,
    pub read_mask: u32,
    pub write_mask: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StencilState {
    pub front: StencilFaceOps,
    pub back: StencilFaceOps,
    pub reference_front: u32,
    pub reference_back: u32,
    pub clear_value: u32,
}

/// One executor-prepared shader stage and the semantic descriptor interface
/// the resolved module statically uses.
#[derive(Clone, Debug, Default)]
pub struct PreparedShaderStage {
    pub id: reims_vgpu_protocol::PreparedShaderId,
    pub used_descriptor_bindings: Arc<[u32]>,
}

/// The two prepared stages required by one resolved render pipeline.
#[derive(Clone, Debug, Default)]
pub struct PreparedRenderProgram {
    pub vertex: PreparedShaderStage,
    pub fragment: PreparedShaderStage,
}

/// The render encoder's line width, retained bit-for-bit from the guest.
///
/// A bitwise value keeps NaNs distinguishable until the backend decides
/// whether it can represent the requested rasterization. The default is
/// Metal's `1.0`, not `f32::default()`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct LineWidth(u32);

impl LineWidth {
    pub const ONE: Self = Self(1.0f32.to_bits());

    pub const fn from_f32(value: f32) -> Self {
        Self(value.to_bits())
    }

    pub const fn value(self) -> f32 {
        f32::from_bits(self.0)
    }

    pub const fn bits(self) -> u32 {
        self.0
    }
}

impl Default for LineWidth {
    fn default() -> Self {
        Self::ONE
    }
}

/// Per-axis raster bounds declared by a Metal render pass.
///
/// These are not attachment extents and do not narrow load/store operations.
/// A missing axis is the API default and inherits the minimum attachment
/// dimension; a present axis clips fragments to `[0, value)` independently.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RenderTargetExtent {
    pub width: Option<std::num::NonZeroU32>,
    pub height: Option<std::num::NonZeroU32>,
}

impl RenderTargetExtent {
    pub fn raster_width(self, attachment_width: u32) -> u32 {
        self.width
            .map_or(attachment_width, std::num::NonZeroU32::get)
    }

    pub fn raster_height(self, attachment_height: u32) -> u32 {
        self.height
            .map_or(attachment_height, std::num::NonZeroU32::get)
    }
}

/// Fully resolved inputs for one draw.
///
/// Guest names, wire tags, and host-native handles are absent. Resource
/// identities are generational core values; formats retain semantic layout and
/// transfer; guest-memory operands are bounded memory contracts.
#[derive(Debug, Default)]
pub struct DrawRequest {
    pub pipeline_lifetime: Option<ResourceLifetime>,
    pub program: PreparedRenderProgram,
    pub width: u32,
    pub height: u32,
    /// Pass-owned raster constraint. Attachment load/store still covers the
    /// attachment dimensions above.
    pub render_target_extent: RenderTargetExtent,
    pub vertex_count: u32,
    pub first_vertex: u32,
    pub instance_count: Option<u32>,
    pub base_instance: u32,
    pub primitive_topology: PrimitiveTopology,
    pub raster_sample_count: u32,
    pub color_sample_count: u32,
    pub multisample_resolve: bool,
    pub viewports: Vec<ViewportResource>,
    pub scissors: Vec<ScissorResource>,
    pub occlusion_query: Option<VisibilityResultMode>,
    pub indexed: Option<IndexedDrawResource>,
    pub vertex_attributes: Vec<VertexAttributeResource>,
    pub storage_buffers: Vec<StorageBufferResource>,
    pub sampled_images: Vec<SampledImageResource>,
    pub samplers: Vec<SamplerResource>,
    pub target_rgba8: Option<Arc<Vec<u8>>>,
    pub target_guest: Option<GuestTargetPlan>,
    pub target_seed_order: SeedOrder,
    pub blend: Option<BlendStateResource>,
    /// Encoder-global blend constant set by `setBlendColor…`. This is not an
    /// attachment property: every attachment using a constant factor reads the
    /// same four values.
    pub blend_constants: [f32; 4],
    pub color_write_mask: ColorWriteMask,
    pub target_identity: Option<crate::TargetIdentity>,
    pub color_attachment_format: Option<ImageFormat>,
    /// Decoded color-attachment load operation. Content placement and seed
    /// availability do not change this contract term.
    pub color_load_action: ColorLoadAction,
    pub load_from_target: bool,
    pub target_clear: [f32; 4],
    pub skip_readback: bool,
    pub seed_from_target: Option<crate::TargetIdentity>,
    pub secondary_targets: Vec<SecondaryColorTarget>,
    pub cull_mode: CullMode,
    pub front_face_ccw: bool,
    pub fill_mode: FillMode,
    pub line_width: LineWidth,
    /// `setDepthBias:slopeScale:clamp:` in that source order.
    pub depth_bias: Option<[f32; 3]>,
    pub depth_clip: DepthClipMode,
    pub depth: Option<DepthState>,
    pub color_input: bool,
    pub continues_render_pass: bool,
    pub render_pass_continues: bool,
}

impl DrawRequest {
    pub fn depth_attachment_extent(&self) -> Option<(u32, u32)> {
        self.depth.as_ref().map(|depth| {
            depth
                .identity
                .as_ref()
                .and_then(crate::TargetIdentity::geometry)
                .unwrap_or((self.width, self.height))
        })
    }

    pub fn minimum_attachment_extent(&self) -> (u32, u32) {
        let color_minimum = self
            .secondary_targets
            .iter()
            .fold((self.width, self.height), |(width, height), target| {
                (width.min(target.width), height.min(target.height))
            });
        self.depth_attachment_extent()
            .map_or(color_minimum, |(depth_width, depth_height)| {
                (
                    color_minimum.0.min(depth_width),
                    color_minimum.1.min(depth_height),
                )
            })
    }

    pub fn raster_extent(&self) -> (u32, u32) {
        let (width, height) = self.minimum_attachment_extent();
        (
            self.render_target_extent.raster_width(width),
            self.render_target_extent.raster_height(height),
        )
    }

    pub fn writes_attachment(&self, identity: &crate::TargetIdentity) -> bool {
        self.attachment_slot(identity).is_some()
    }

    pub fn color_attachment_index(&self, identity: &crate::TargetIdentity) -> Option<usize> {
        if self.target_identity.as_ref() == Some(identity) {
            Some(0)
        } else {
            self.secondary_targets
                .iter()
                .position(|target| &target.identity == identity)
                .map(|index| index + 1)
        }
    }

    pub fn attachment_slot(&self, identity: &crate::TargetIdentity) -> Option<AttachmentSlot> {
        if let Some(index) = self.color_attachment_index(identity) {
            Some(if index == 0 {
                AttachmentSlot::Primary
            } else {
                AttachmentSlot::Secondary
            })
        } else if self.depth.as_ref().and_then(|d| d.identity.as_ref()) == Some(identity) {
            Some(AttachmentSlot::Depth)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttachmentSlot {
    Primary,
    Secondary,
    Depth,
}

impl AttachmentSlot {
    pub fn sampled_self_route(self) -> &'static str {
        match self {
            Self::Primary => "sampled_self_primary",
            Self::Secondary => "sampled_self_secondary",
            Self::Depth => "sampled_self_depth",
        }
    }
}

pub fn viewport_slot_count(req: &DrawRequest) -> usize {
    req.viewports.len().max(req.scissors.len()).max(1)
}

#[derive(Debug, Clone)]
pub struct SecondaryColorTarget {
    pub identity: crate::TargetIdentity,
    /// Canonical guest allocation for this attachment when its declared
    /// layout is directly representable by the backend.
    pub target_guest: Option<reims_vgpu_memory::GuestTargetMemory>,
    pub width: u32,
    pub height: u32,
    pub format: ImageFormat,
    pub clear: [f32; 4],
    pub load_action: ColorLoadAction,
    pub blend: Option<BlendStateResource>,
    pub color_write_mask: ColorWriteMask,
}

/// Backend-independent color-attachment load operation.
///
/// This stays distinct from whether a LOAD source has already been
/// materialized. In particular, `DontCare` is not a black clear.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum ColorLoadAction {
    Load,
    #[default]
    Clear,
    DontCare,
}

#[derive(Debug, Default)]
pub struct DrawOutput {
    pub pixels: Vec<u8>,
    pub pixels_bgra: bool,
    pub occlusion_samples: Option<u64>,
    /// Exact guest pages a directly bound attachment Store was recorded to
    /// write. The executor's guest-write ledger owns physical completion until
    /// its submission fence retires; overlapping host access and the guest's
    /// completion stamp settle that ledger before observing these bytes.
    /// `None` means this draw recorded no direct guest Store.
    pub guest_store_pages: Option<reims_vgpu_memory::GuestWritePages>,
    /// Allocation-relative byte window occupied by that ordered direct Store.
    /// Carried with the result so surface publication never re-derives a
    /// possibly newer mapping layout after execution.
    pub guest_store_window: Option<std::ops::Range<u64>>,
}

#[derive(Clone, Copy, Debug)]
pub struct ViewportResource {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub min_depth: f32,
    pub max_depth: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct ScissorResource {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug)]
pub struct IndexedDrawResource {
    pub index_type: IndexType,
    pub index_count: u32,
    pub vertex_offset: i32,
    pub content: BufferContent,
}

#[derive(Debug)]
pub struct VertexAttributeResource {
    pub location: u32,
    pub binding: u32,
    pub format: VertexAttributeFormat,
    pub offset: u32,
    pub stride: u32,
    pub step_function: VertexStepFunction,
    pub step_rate: u32,
    pub content: BufferContent,
}

#[derive(Debug)]
pub struct StorageBufferResource {
    pub binding: u32,
    pub content: BufferContent,
}

#[derive(Clone, Debug)]
pub enum BufferContent {
    Bytes(Arc<Vec<u8>>),
    GuestRuns(GuestRunSource),
}

impl BufferContent {
    pub fn len(&self) -> usize {
        match self {
            Self::Bytes(bytes) => bytes.len(),
            Self::GuestRuns(source) => source.total_len as usize,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Test/instrumentation view of a potentially scattered guest source.
    pub fn cpu_bytes(&self) -> std::borrow::Cow<'_, [u8]> {
        match self {
            Self::Bytes(bytes) => std::borrow::Cow::Borrowed(bytes.as_slice()),
            Self::GuestRuns(source) => {
                let mut out = Vec::with_capacity(source.total_len as usize);
                let mut skip = source.source_offset;
                for run in source.runs.iter() {
                    let take = (source.total_len as usize).saturating_sub(out.len());
                    if take == 0 {
                        break;
                    }
                    if skip >= run.len {
                        skip -= run.len;
                        continue;
                    }
                    let within = skip as usize;
                    skip = 0;
                    let len = (run.len as usize).saturating_sub(within).min(take);
                    // SAFETY: the memory contract retains each stable host alias
                    // for the request lifetime and bounds the declared span.
                    unsafe {
                        out.extend_from_slice(std::slice::from_raw_parts(
                            (run.host_ptr as *const u8).add(within),
                            len,
                        ));
                    }
                }
                out.resize(source.total_len as usize, 0);
                std::borrow::Cow::Owned(out)
            }
        }
    }
}

impl From<Vec<u8>> for BufferContent {
    fn from(bytes: Vec<u8>) -> Self {
        Self::Bytes(Arc::new(bytes))
    }
}

#[derive(Debug)]
pub struct SampledImageResource {
    pub binding: u32,
    pub array_element: u32,
    pub descriptor_count: u32,
    pub width: u32,
    pub height: u32,
    pub layers: u32,
    pub arrayed: bool,
    pub volume: bool,
    pub cube: bool,
    pub one_dim: bool,
    pub multisampled: bool,
    pub source: SampledSource,
    pub content: Option<ContentStamp>,
    pub byte_origin: SampledByteOrigin,
    pub format: ImageFormat,
    pub identity: Option<SampledContentIdentity>,
    pub resource_lifetime: Option<ResourceLifetimeRef>,
    pub swizzle: SwizzlePlan,
}

impl SampledImageResource {
    /// Whether this resource's image planes occupy consecutive *array slices*
    /// of its guest allocation, as opposed to being a single plane or the depth
    /// slices of one 3-D image.
    ///
    /// A cube answers yes. Both APIs that meet here define a cube as six
    /// ordinary array slices in the same order — `+X, -X, +Y, -Y, +Z, -Z` — and
    /// a cube array as consecutive groups of six. So a cube needs no face
    /// permutation and no guest layout of its own: it is a `D2Array` of
    /// `6 * elements` slices that happens to be sampled by direction, which is
    /// why there is no cube variant in `GuestImageLayout` and why none is
    /// wanted.
    ///
    /// The question is answered here rather than at each consumer because
    /// `arrayed` alone is the spelling that reads correct and silently excludes
    /// cubes. That spelling cost this device every cube-sampled draw on the
    /// guest-run arm while the copying arm drew them — a topology difference the
    /// guest could see, which is the one thing placement policy may never
    /// produce.
    #[must_use]
    pub const fn planes_are_array_slices(&self) -> bool {
        self.arrayed || self.cube
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SampledByteOrigin {
    #[default]
    Synthetic,
    BufferBackedTexture,
    SerializedSurfaceView,
    SurfaceHostCache,
    SurfaceGuestFallback,
    LinearTexture,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum SeedOrder {
    #[default]
    Rgba8,
    Bgra8,
}

#[derive(Debug)]
pub enum SampledSource {
    /// A serialized texture slot containing no object.
    Null,
    Bytes(Arc<Vec<u8>>),
    Target(crate::TargetIdentity),
    Attachment {
        identity: crate::TargetIdentity,
        initial: AttachmentInitial,
    },
    /// An image view over authoritative guest storage, with an exact transfer
    /// representation for backends whose image layout cannot alias it.
    GuestImage(reims_vgpu_memory::GuestImageSource, GatherVouch),
    GuestRuns(GuestRunSource, GatherVouch),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AttachmentInitial {
    Clear([f32; 4]),
    Seed,
    DontCare,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SampledContentIdentity {
    pub key: u64,
    pub generation: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_protocol::TexelLayout;

    fn shaped(arrayed: bool, volume: bool, cube: bool) -> SampledImageResource {
        SampledImageResource {
            binding: 0,
            array_element: 0,
            descriptor_count: 1,
            width: 4,
            height: 4,
            layers: if cube { 6 } else { 1 },
            arrayed,
            volume,
            cube,
            one_dim: false,
            multisampled: false,
            source: SampledSource::Null,
            content: None,
            byte_origin: Default::default(),
            format: ImageFormat::linear(TexelLayout::Bgra8),
            identity: None,
            resource_lifetime: None,
            swizzle: Default::default(),
        }
    }

    /// The four shapes a sampled resource can declare, and which of them lay
    /// their planes out as consecutive array slices of the guest allocation.
    #[test]
    fn a_cube_lays_its_planes_out_as_array_slices_and_a_volume_does_not() {
        assert!(!shaped(false, false, false).planes_are_array_slices());
        assert!(shaped(true, false, false).planes_are_array_slices());
        assert!(!shaped(false, true, false).planes_are_array_slices());
        assert!(shaped(false, false, true).planes_are_array_slices());
    }

    fn depth(identity: crate::TargetIdentity) -> DepthState {
        DepthState {
            identity: Some(identity),
            test_enable: false,
            write_enable: false,
            compare: crate::SamplerCompareFunction::Always,
            clear_value: 1.0,
            load: false,
            stencil: None,
        }
    }

    #[test]
    fn viewport_and_attachment_relations_are_semantic() {
        let request = DrawRequest {
            target_identity: Some(crate::TargetIdentity::Gva {
                gva: 1,
                width: 4,
                height: 4,
                generation: 1,
                format: TexelLayout::Rgba8,
            }),
            ..Default::default()
        };
        assert_eq!(viewport_slot_count(&request), 1);
        assert_eq!(
            request.attachment_slot(request.target_identity.as_ref().unwrap()),
            Some(AttachmentSlot::Primary)
        );
    }

    #[test]
    fn raster_extent_is_the_explicit_bound_over_every_attachment() {
        let mut request = DrawRequest {
            width: 8,
            height: 8,
            render_target_extent: RenderTargetExtent {
                width: std::num::NonZeroU32::new(2),
                height: None,
            },
            depth: Some(depth(crate::TargetIdentity::Texture {
                ref_: 3,
                width: 5,
                height: 3,
                generation: 1,
                stencil: false,
            })),
            ..Default::default()
        };
        request.secondary_targets.push(SecondaryColorTarget {
            identity: crate::TargetIdentity::Texture {
                ref_: 2,
                width: 4,
                height: 6,
                generation: 1,
                stencil: false,
            },
            target_guest: None,
            width: 4,
            height: 6,
            format: reims_vgpu_protocol::ImageFormat::linear(TexelLayout::Rgba8),
            clear: [0.0; 4],
            load_action: ColorLoadAction::Clear,
            blend: None,
            color_write_mask: ColorWriteMask::default(),
        });
        assert_eq!(request.minimum_attachment_extent(), (4, 3));
        assert_eq!(request.raster_extent(), (2, 3));
    }

    #[test]
    fn anonymous_depth_uses_request_geometry_but_explicit_zero_geometry_stays_zero() {
        let anonymous = DrawRequest {
            width: 8,
            height: 6,
            depth: Some(depth(crate::TargetIdentity::Anonymous { slot: 1 })),
            ..Default::default()
        };
        assert_eq!(anonymous.depth_attachment_extent(), Some((8, 6)));

        let explicit = DrawRequest {
            width: 8,
            height: 6,
            depth: Some(depth(crate::TargetIdentity::Texture {
                ref_: 1,
                width: 0,
                height: 6,
                generation: 1,
                stencil: false,
            })),
            ..Default::default()
        };
        assert_eq!(explicit.depth_attachment_extent(), Some((0, 6)));
        assert_eq!(explicit.minimum_attachment_extent(), (0, 6));
    }
}

/// A resolved draw must be able to cross a thread boundary.
///
/// [`DrawRequest`] is documented as immutable, backend-neutral and fully
/// resolved, and the x86 device is drain-CPU bound with the whole encode on one
/// thread. Those two facts together make this type the seam a pipelined or
/// parallel encoder would cut on: the resolving side builds one of these and
/// hands it to whoever records it, and nothing in it may borrow the resolver's
/// state or bind it to the resolver's thread.
///
/// `Send` is that promise made checkable. A field added later that is not
/// `Send` -- a raw pointer into guest memory, an `Rc`, a handle keyed to the
/// calling thread -- would close the seam silently, because nothing else in
/// this tree would notice until someone tried to build the encoder and found
/// the type no longer crosses.
///
/// This is not a claim that the encoder exists. It is the one property the
/// encoder cannot be built without, asserted where the type is defined so it
/// cannot regress unnoticed.
///
/// # A two-stage pipeline cut here was built and measured, and it loses
///
/// `Send` is necessary and it is a long way from sufficient. A recording
/// thread that took resolved draws from an in-order queue while the resolver
/// ran ahead was built against this seam and driven on fullscreen Maps
/// (macos-13, x86/Vulkan). It classified correctly -- every draw of the driven
/// window was accepted for deferral -- and it was **3.2x slower**:
///
/// | arm | `proc_us`/draw | draws/driven-sec |
/// |---|---|---|
/// | serial | 22.01 | 44 826 |
/// | two-stage pipeline | 70.75 | 12 307 |
///
/// The cause is not the queue and not lock contention (`gap_lock_us` was
/// 0.25 us a draw). It is that the resolving side must **synchronise with the
/// recorder 6.4 times per draw**, measured directly: 6 491 deferred draws
/// against 44 730 forced drains in one census window. The resolver takes the
/// engine about seven times a draw for its own work -- resident-registry
/// reads, content-epoch stamps, read plans -- and every one of those reads
/// state the recorder mutates while recording (`registry_mark_ready_at`,
/// `registry_note_sampled_use`, the guest-write debt). Answering one of them
/// from a queue that has not drained is a stale answer, and a stale answer
/// here is a **content** defect. So the synchronisation is a real data
/// dependency at draw granularity, not a conservative guard that could be
/// relaxed.
///
/// The second number is the one that generalises past this design: each
/// synchronisation cost **7.6 us** (48.7 us a draw of added `proc_us` over 6.4
/// of them) -- an mpsc send plus a futex sleep and wake on a saturated
/// machine. Against a 22 us budget, *one* handoff a draw would eat a third of
/// it. No design in which the resolver waits for a recorder once per draw can
/// pay here, however cheap the queue is made.
///
/// What that leaves. Resolve and record share the engine's resident registry,
/// so cutting between them does not cut the state, and the seam is not where
/// the type boundary suggests. The parallelism the contract actually grants is
/// one level up: Metal command buffers are encoded independently with order
/// fixed at commit, and a driven frame carries about forty of them. Taking it
/// needs per-encoder command pools and per-command-buffer registry isolation --
/// a change to what the engine owns, not another consumer of this type.
#[cfg(test)]
mod thread_seam {
    /// See the module doc: the resolved-draw seam is only a seam if a resolved
    /// draw can leave the thread that resolved it.
    #[test]
    fn a_resolved_draw_can_leave_the_thread_that_resolved_it() {
        fn assert_send<T: Send>() {}
        assert_send::<super::DrawRequest>();
    }
}
