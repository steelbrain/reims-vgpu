//! `id<MTLRenderCommandEncoder>`: core's `RenderEncoder` over the runtime's, one runtime call per
//! `RenderCmd`, and `renderCommandEncoderWithDescriptor:`, which opens it.
//!
//! - **Every runtime refusal dies at its selector** ([`Policy::Dies`]): a refusal is the
//!   runtime's validation or a form it lacks, and Metal raises at none of the encoder's commands,
//!   going on or asserting instead.
//! - **The encoder's creation** answers nil where Metal does (a pass with no attachment and no
//!   target size), dies where Metal aborts (a pass of no attachment with a target size and no
//!   default raster sample count, a memoryless depth or stencil attachment resolving at one
//!   sample with no resolve texture, more sample positions than a multisampled pass's samples,
//!   an encoder over an open one), drops the depth and stencil resolve textures a non-resolving
//!   store action leaves unused, as Metal ignores them, and dies at the runtime's other refusals.
//!   Among them, a `renderTargetArrayLength` past an attachment's slices (`render_pass_actions`:
//!   slice 1 of a 2D array of 2, at a length of 2), which Metal runs without validation. Measured
//!   on this host (a 4x4 2D array of 2 slices, plain and linear over a buffer as core makes one,
//!   cleared and stored at slices 0 and 1 and lengths 1 to 5): Metal renders min(length, the
//!   texture's slices) layers from the attachment's slice, those past the texture into the memory
//!   after it, at its slice stride (the buffer's bytes at `0x100` for slice 2 of a linear one,
//!   whose slices are `0x80` apart), or into a heap neighbour (the runtime's measurement); a
//!   draw that writes no layer renders the base slice alone. Nothing here writes past a texture,
//!   so such a pass dies at its creation. So does a pass into a YCbCr colour attachment the
//!   runtime refuses (a format refusal of a two-plane or 4:2:2 texture), its line naming the
//!   format and what Metal does there ([`crate::gpu::ycbcr::pass_refused`]): a packed 10-bit
//!   one's, which Metal fails with an error and bytes that vary from run to run, the shapes of
//!   500, 502, 503, 505, 506 and 507 and of 540 to 545 the runtime does not store (a resolve, a
//!   texture over plane 1, a size their chroma blocks do not tile), and the other formats',
//!   their `_sRGB` twins' among them, whose stores the runtime does not hold.
//! - **Store actions** Metal goes on with: a late one equal to the descriptor's known action, or
//!   of an attachment without a texture, encodes nothing; another over a known one dies, as its
//!   effect is not measured; one the descriptor left `Unknown` and nothing decided stores nothing
//!   and resolves nothing at `endEncoding`, as on Metal, as the runtime ends the pass.
//! - **A wait for a fence the encoder updated** goes to the runtime, which waits for the update
//!   before the encoder, as Metal completes it. **A barrier after the fragment stage and before
//!   the vertex stage** encodes nothing: this GPU runs a pass's vertex stages first, and Metal's
//!   barrier does not split the pass, where the runtime's would.
//! - **A nil pipeline state** has no runtime form: it is kept here, and what next runs the
//!   encoder's pipeline state dies, as Metal crashes at a draw with no pipeline state
//!   (`display/render.rs`'s nil pipeline of an integer target), where the runtime would run the
//!   state set before: a draw, direct or indirect, a tile dispatch, and an execution of an
//!   indirect command buffer whose commands inherit the pipeline state. **A nil depth-stencil
//!   state** is Metal's default state, with no depth or stencil test, as the runtime's nil is.
//! - **A draw of an index type no `MTLIndexType` is** encodes nothing, as Metal draws nothing
//!   (`render_draw_indexed`: index type 2). **A draw of no instances**, a patch draw's too, goes
//!   to the runtime, which records nothing and checks it only as plain Metal does: a pipeline
//!   state set, of the draw's kind.
//! - **Metal's private primitive types** 5 to 8, a line loop, a triangle fan, quads and a quad
//!   strip, go to the runtime, which draws each, directly, over indices, from indirect arguments
//!   and as an indirect command buffer's command (`crate::gpu::icb`), as the list of a public
//!   type Metal draws it as ([`rt::DrawPrimitiveType`]):
//!   measured against Metal pixel for pixel at every count from 0 to 10, with flat attributes,
//!   blending and instances, and over indices and indirect arguments at either provoking vertex.
//!   Such a draw, and one the runtime draws over a list of its own at the last vertex, gives
//!   `[[base_vertex]]` and `[[vertex_id]]` as Metal's draw gives them (its non-indexed draws, of
//!   every type, give `[[base_vertex]]` as their first vertex and `[[vertex_id]]` as each
//!   vertex's: measured on the M3 Max, from 0 to `0xfffffffd`), and runs its vertex function as
//!   many times at each vertex as Metal's, so one that writes memory writes as often: on the M3
//!   Max both run it once for each primitive the vertex is in (a strip's vertices up to 3 times,
//!   a fan's first once a triangle; measured at every type, 6 to 1000 vertices, 1 to 3
//!   instances, at either provoking vertex). Quads and quad strips give a fragment function's
//!   `[[primitive_id]]` as Metal counts them, once a quad (`quads_give_metals_primitive_ids`).
//!   Where the list drawn over indices would not run as Metal's draws, the runtime draws the
//!   list's positions without indices, each once, its vertex function reading the list's index
//!   and Metal's base vertex at the list's device address: under a vertex function that writes
//!   memory, which then runs as often at each vertex as Metal's over indices and from indirect
//!   arguments too (`private_types_run_the_vertex_function_as_often_as_metals_at_every_draw`);
//!   under one that computes `[[primitive_id]]` into its varying (on a device whose fragment stage
//!   cannot declare it); and in a direct draw whose last vertex passes `i32::MAX`, which the
//!   list's base vertex cannot hold, its vertex IDs wrapping past `u32::MAX` as Metal's
//!   (`private_types_from_far_vertex_starts_draw_as_metals`). On a device without device
//!   addresses the runtime refuses such a draw (`PrivatePrimitives`), which dies.
//! - **Work of a pipeline state that can encode indirect commands** (the runtime's
//!   `can_encode_indirect_commands`) may write the records of each buffer the encoder made
//!   resident for writing, for good (`executions::gpu_writes`), through the buffer's ID, the
//!   runtime's, as a compute encoder's does ([`crate::gpu::encode_compute`]): a draw of
//!   something, a tile dispatch, and an execution of commands that inherit the state at its call
//!   ([`VulkanRenderEncoder::encodes`]); an execution whose commands name their own at the
//!   commit, which reads them.
//! - **Primitive type 9**, which Metal's GPU draws nothing of, goes to the runtime, which fails
//!   the command buffer as Metal's GPU does ([`VulkanRenderEncoder::nine`]): directly, of 3
//!   vertices or more and an instance, it hangs Metal's GPU, the hang in flight as long as
//!   Metal's and a command buffer committed on its queue meanwhile running nothing and completing
//!   without an error, as Metal's does ([`crate::gpu::command`]); of fewer it fails nothing, as on
//!   Metal. Over indices, any, none or of no instance included, Metal's GPU faults it
//!   (`MTLCommandBufferErrorPageFault`), and from indirect arguments, over indices or not, fails
//!   it as a victim of its recovery (`MTLCommandBufferErrorInternal`): its render encoder and
//!   every encoder after it run nothing, every event signal fires, and the command buffer
//!   completes no sooner than Metal's earliest, its queue's others running. Metal's GPU runs the
//!   encoders after a fault where it was idle before (the first fault after 3 s idle, and the
//!   process's first here) and loses them where it ran just before, as in a steady stream of work
//!   (the runtime's measurement on the M3 Max, macOS 27.0.1): the runtime answers the latter,
//!   which `type_9_draws_fail_as_metals` compares after hangs. A hang or a fault, not a victim,
//!   counts against its queue: from the completion of the queue's second, a command buffer
//!   committed on it fails at once (`MTLCommandBufferErrorAccessRevoked`), running nothing and
//!   signalling nothing, as Metal's (`type_9_faults_revoke_queues_as_metals`). One over an index
//!   type no `MTLIndexType` is, which nothing measured, dies ([`unmeasured_type_9`]).
//! - **Apple's private commands** compose public ones where their meaning is measured:
//!   `setDepthClipModeSPI:` is `setDepthClipMode:`; the late resolve setters set the runtime's
//!   resolve targets, `yInvert` dropped, which Metal ignores, and one the attachment cannot use
//!   encodes nothing, as Metal resolves nothing there: a colour one where the pass descriptor
//!   named no resolve texture, or any under a store action the descriptor knew and that does not
//!   resolve. A colour one of an attachment whose resolve into a texture of another format the
//!   pass was made without, as plain Metal leaves that texture untouched ([`open`]), dies, as
//!   nothing measured it, and so does a depth or stencil one in such a pass that may resolve.
//!   `setDepthCleared` and `setStencilCleared`, which change nothing Metal draws
//!   (`mutgap_dec_render_depth_stencil_cleared`), encode nothing. The rest go to the runtime,
//!   which answers each at Metal's default or refuses it.
//! - **The draw-state setters** encode nothing and mark the encoder; each draw then draws what
//!   Metal draws under them, or relays or dies where it has no exact form
//!   ([`VulkanRenderEncoder::marked`]). `setPrimitiveRestartEnabled:index:` and
//!   `setProvokingVertexMode:` go to the runtime, which draws what Metal draws under them.
//!   A draw of no primitive, patch or instance, and an execution of no command, draw nothing
//!   whatever they left, as Metal's do, and go on. Measured on Metal on this host, against a 16
//!   by 16 target:
//!   - `setPointSize:` changes nothing drawn at any size (-1 to 1000, and NaN): a point is the
//!     size its vertex function writes. It is not marked. Under a vertex function that writes
//!     no `[[point_size]]`, Metal draws a point of size 1, as one that writes 1, only in a state
//!     of the point input primitive topology class; in any other its points are undefined,
//!     whatever the setter set: most draws draw nothing, and some draw a few of the points, or a
//!     box whose size changes from one run to the next (the unspecified, line and triangle
//!     classes, 6 runs each in two processes). So a draw of points there dies, direct, over
//!     indices or from indirect arguments, but one of no vertices ([`sized()`]);
//!     and so does an execution's, at the commit, which reads its commands' records as the
//!     runtime runs them: under a state its commands inherit, or each command's own
//!     ([`crate::gpu::states::points_sized`]), or one released since. An execution of a buffer
//!     whose records the GPU may have written is asked where the runtime reads them, once the
//!     work before it completed, and its draws of points die there
//!     ([`crate::gpu::command::executions`]).
//!   - `setLineWidth:` goes to the runtime's encoder, which draws every width as Metal does:
//!     `floor(16 width) / 16` clamped to `[0, 16]`, by the device's own lines or a band of two
//!     triangles over each segment. What neither draws at the width there (triangles filled as
//!     lines and mesh draws the device's lines do not draw it at, an execution's lines, lines in
//!     a pass of a rasterization rate map) it draws by the device's lines at the nearest width
//!     they draw, 1 pixel on MoltenVK, announcing `lines-at-nearest-width`, which dies in strict
//!     mode ([`crate::answer::DIES`]).
//!   - `setProvokingVertexMode:` of 0 takes a primitive's flat inputs, and the layer and the
//!     viewport its vertex function writes (`[[render_target_array_index]]`,
//!     `[[viewport_array_index]]`), from its first vertex, Metal's default and Vulkan's; 1 from its
//!     last (a line's second vertex, a list triangle's third, strip triangle i's vertex i + 2, a
//!     private type's list primitive's last), in direct, indexed (each piece of a restarted strip
//!     its own parity), indirect and instanced draws and patches, where an execution of an indirect
//!     command buffer's commands takes the first vertex at every mode. `[[barycentric_coord]]` is
//!     in mode 0's order at mode 1 but for a strip's odd triangles, `(b2, b0, b1)` of mode 0's.
//!     Every other mode, its top bit set or not (2, 7, 255, `2^31`, `2^32 - 1`, `2^32`, `2^62`,
//!     `2^63 - 1`, `2^63`, `u64::MAX - 1`, `u64::MAX`), draws a third way: a list triangle's flat
//!     inputs from its last vertex, a patch from its first, and strips, lines, points and the layer
//!     a list triangle's vertex function writes otherwise than at 0 and at 1
//!     (`modes_past_1_draw_metals_third_way`). The runtime takes the mode as the guest set it, and
//!     draws it: mode 1 through `VK_EXT_provoking_vertex` where the device has its last vertex,
//!     elsewhere (MoltenVK) from lists of the primitives led by their last vertices, which give a
//!     triangle Metal's flat inputs, layer, viewport and weights (of one that also reads a flat
//!     input, the second and third weights but for rounding). What it draws otherwise than Metal it
//!     announces: a line drawn from its last vertex to its first (`provoking-vertex-lines`);
//!     triangles that read flat inputs and weights, and patches, without the last vertex, a layer
//!     or viewport selected in a synthesized geometry stage with it, and, at a mode from 2 up,
//!     which it draws as 1 but patches as 0, each draw whose shape Metal's third way draws
//!     otherwise as far as its state reads it: a strip of triangles under a state that reads flat
//!     inputs, weights or a selected layer or viewport, lines under one that reads flat inputs, a
//!     primitive ID or a selected layer or viewport, sized points under one that reads flat
//!     inputs or a primitive ID, list triangles under one that selects a layer or viewport, and
//!     mesh draws, not measured there (`provoking-vertex-unmatched`, announced at the draw; the
//!     setter announces nothing). Each is relayed, and dies in strict mode
//!     ([`crate::answer::DIES`]).
//!   - `setPrimitiveRestartEnabled:index:` sets the index an indexed strip restarts at, `k`,
//!     whatever `enabled` says (`{0, k}` draws as `{1, k}`, at `k` of 0, 3, 7, 15, `0xffff`
//!     and `0xffffffff`). Without it a strip restarts at its index type's all-ones index, as
//!     Vulkan's does: 16-bit `0xffff`, 32-bit `0xffffffff`, and a 32-bit `0xffff` is a vertex.
//!     After it a strip restarts at `k` alone, 16- and 32-bit alike, and draws the all-ones
//!     index as a vertex. Only line and triangle strips restart, drawn directly or from
//!     indirect arguments; points and lists draw every index as a vertex. A new encoder's
//!     strips restart at the all-ones index again. An execution of an indirect command buffer's
//!     indexed strips, 16- and 32-bit, and indexed patches draw alike with and without it.
//!     The runtime's encoder takes the setter and draws so (`set_primitive_restart_enabled`,
//!     measured on an M4): an indexed strip that restarts elsewhere than at the all-ones index
//!     draws as the list a GPU pass writes from its indices before the pass, and what that pass
//!     cannot draw as Metal does the runtime refuses, which dies at the draw. An indirect
//!     strip's list the command buffer's index pool cannot hold dies as the command buffer
//!     completes (`crate::gpu::command`). A line loop draws as without the setter, as Metal's
//!     does: its line strip's indices are this platform's ([`VulkanRenderEncoder::primitives`]).
//!     The guest's `k` is below `maxViewportCount` (core refuses the rest), and
//!     `mutgap2_rdec_restart_transform` alone draws a strip after it.
//!   - `setVertexAmplificationMode:value:` sets the amplification count that
//!     `setVertexAmplificationCount:viewMappings:` sets, and keeps the view mappings that one
//!     set; whichever of the two comes last decides, and a new encoder starts at 1. In every
//!     mode but 1 (0 to 8, 15, 16, 255, `u32::MAX` and `u64::MAX`) a value from 1 to 8 draws
//!     that many copies, amplification IDs 0 up; in mode 1 the number of the value's set bits
//!     does (1 to 9, 15 to 17, 255 to 257). Value 0 draws copies at random: 8 under a state of
//!     no amplification, and 0, 48 or 255 under one of `maxVertexAmplificationCount` 2, from
//!     one run to the next; 9 draws 1, and 16 and 32 draw 8. So the setter encodes nothing, and
//!     where it leaves a count the runtime does not draw (the last
//!     `setVertexAmplificationCount:viewMappings:`'s, or 1), or one not measured, draws and
//!     executions die until the next `setVertexAmplificationCount:viewMappings:`.
//! - **`useHeaps:count:` and `useResources:count:usage:`** reach the vertex and fragment stages,
//!   as they do on Metal's render encoder. A nil entry of `useResources:count:usage:` or of
//!   `useResources:count:usage:stages:` is skipped, as Metal goes on past one
//!   (`interplay_decodewalk_6`: an entry naming the render pipeline is nil, and the walk runs on;
//!   `interplay_lifetime_emptied_resource_render`: a ref past the object list is nil to the
//!   stages form, and the triangle after it draws as without it). A nil entry of
//!   `useHeaps:count:` or `useHeaps:count:stages:` is skipped too: Metal's encoder goes on past
//!   one and the draw after it runs (measured on an M3 Max under macOS 27.0.1, without API
//!   validation, which asserts at it).
//! - **Mesh draws** (`drawMeshThreadgroups:`, `drawMeshThreads:` and their indirect form, of a
//!   state of `newRenderPipelineStateWithMeshDescriptor:`) and the object and mesh stages'
//!   setters go to the runtime's encoder, which draws them as Metal does, on a device without
//!   `VK_EXT_mesh_shader` (MoltenVK) as compute stages whose records a vertex stage draws
//!   (`emulated-mesh-draws`, relayed). A grid with a zero axis draws nothing on either, and its
//!   draw takes nothing else here. A draw of an object or mesh threadgroup larger than the
//!   state's maximum for its stage, or of another size than the one the state requires, dies at
//!   the draw: Metal's validation stops it, and plain Metal's draw there is not measured
//!   ([`threads_held`]). What a mesh draw may write through the buffers and textures bound to its
//!   object and mesh stages is accounted for as a vertex or tile draw's is ([`Reach`]), a store
//!   into a buffer on a surface whose GPU writes never reach its memory as a tile function's,
//!   which nothing measured. `useHeaps:count:` and `useResources:count:usage:` reach the vertex
//!   and fragment stages only, as below: what they reach of a mesh draw's stages on Metal is not
//!   measured.
//! - **Executing an indirect command buffer** runs its draws, with the encoder's render state, as
//!   their records stand when the command buffer is committed, or, for a buffer the GPU may have
//!   written (a blit wrote it, or an encoder made it resident for writing), once the work before
//!   the execution completed (the runtime's host-side step), where Metal's GPU reads them as it
//!   reaches them. The runtime's refusal of the records dies naming the execution, at the commit
//!   or as the command buffer completes ([`crate::gpu::command`]); a refusal at the call (a range
//!   past the buffer's size, which Metal runs without validation) dies there. An empty range
//!   within the size runs nothing, as Metal's does under any pipeline state (`render_draw_icb`:
//!   one made without `supportIndirectCommandBuffers` too), as the runtime runs it.
//! - **The pass's rate map** is the runtime's, which draws the pass by the map as the GPU runs
//!   it, as Metal's does: a reset after the encoder is made encodes the pass again by the map as
//!   reset, before its command buffer's submission (`crate::gpu::rate_map`). What the encoder
//!   decides by the map is decided by its class at the encoder's creation
//!   (`crate::gpu::rate_map::Class`), and a reset to a map that decision does not hold over, while
//!   the pass may run, dies. A pass of a mutable map not yet reset rasterizes nothing, as on
//!   Metal, the runtime's too: its vertex functions run and no fragment does. A tile dispatch in
//!   it dies: only Metal's draws were measured there. A pass of a map of no layers loads, clears
//!   and stores as Metal's does, but Metal's draws in it rasterize by what earlier maps left,
//!   where the runtime's draw nothing: a draw and an execution of indirect commands in it die,
//!   but one of no primitive or command, which draws nothing on either (the encoder's
//!   `layered`). A tile dispatch in it dies too, as nothing measured Metal's there (the
//!   encoder's `tiled`). A pass of a map whose rates vary (`rasterization-rate-map`,
//!   [`Class::Varying`]) loads, clears and stores as Metal's does too, but the runtime's draws in
//!   it rasterize by regions whose lines are the map's tiles' exact lines, where Metal's are its
//!   fixed-point ones: a draw and an execution of indirect commands that is not empty relay it
//!   there, and so does a tile dispatch, each dying in strict mode. A pass over a map made of a
//!   `Mutable` descriptor, reset or not, that no draw of the guest's took the place of ends with
//!   a draw of no vertices the guest did not encode, as the Metal platform's does, without which
//!   the runtime fails the command buffer of one that nothing drew in as a GPU hang, as Metal's
//!   GPU does ([`crate::gpu::rate_map::no_draw`]). A draw takes its place where the Metal
//!   platform's does: one of at least one whole triangle of a triangle list or strip (3 vertices
//!   or more), once or more, that the runtime took, which it counts as the pass's draw call
//!   ([`VulkanRenderEncoder::drew`]).
//! - **What the runtime has no form for dies**: an index or a count past `u32`, a raw value no
//!   Metal enumerator is, a slice shorter than its range, a nil resource (but in
//!   `useResources:count:usage:`), and a command after `endEncoding`. A rotated texture is the
//!   runtime's, as an attachment, a resolve texture and bound to a stage
//!   ([`crate::gpu::resource::TextureObject::texels`]), and so are its refusals.

use std::cell::{RefCell, RefMut};
use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use metal2vulkan::metal::{
    BarrierScope, CullMode, DepthClipMode, IndexType, PixelFormat, PrimitiveTopologyClass,
    PrimitiveType, RenderStages, ResourceUsage, TextureType, TriangleFillMode, Winding,
};
use metal2vulkan::pipeline::ColorAttachmentDescriptor;
use metal2vulkan_runtime as rt;
use reims_vgpu_core::platform::Handle as _;
use reims_vgpu_core::platform::blit::ResourceRef;
use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder, VertexDraw, encode_each};
use reims_vgpu_core::platform::resource::RenderPipeline as _;
use reims_vgpu_core::platform::types::{MtlScissorRect, MtlSize, MtlViewport, NsRange};
use reims_vgpu_core::raise::{OnFault, RaiseResult};
use reims_vgpu_core::sync::{Mutex, ReentrantMutex};

use crate::answer::{Policy, decide, get, metal_aborts, plain, plain_at, refused, unrepresentable};
use crate::gpu::command::VulkanCommandBuffer;
use crate::gpu::command::executions::{Execution, Points};
use crate::gpu::desc::{Over, Source, Sources, VulkanRenderPassDescriptor};
use crate::gpu::encode_blit::EncoderObject;
use crate::gpu::icb::{Inherits, VulkanIcb};
use crate::gpu::rate_map::Class;
use crate::gpu::resource::{MeshAnswers, RenderDescriptor, VulkanBuffer, VulkanTexture};
use crate::gpu::writes::{
    Bound, DRAWING_STAGES, FRAGMENT, Kept, MESH, OBJECT, RENDER_STAGES, Reflected, Reflection,
    Slot, TILE, VERTEX, Whether,
};
use crate::gpu::{listed, metal_value, region, size};
use crate::platform::{Obj, VulkanPlatform};

/// The first of a drawing descriptor's colour attachments, `colors`, that the state's fragment
/// function writes (`written`, the runtime's `fragment_color_outputs`), a render or a mesh render
/// pipeline state's alike, and its format, where a `half` colour output stores a
/// value further from Metal's than the bound the runtime's rounding of it is relayed within (one
/// code of a unorm, snorm or packed float channel, one unit in
/// the last place of a 16-bit float channel, a 32-bit float channel's own value). The runtime
/// rounds such an output to `half` before the blend and the format's conversion
/// (`HalfOutputPrecision`, a module's without fast math; the runtime notes a fast-math module's
/// `HalfOutputFastMathBlend`, relayed), where Metal may carry it at `float`
/// precision into the blend. Measured against Metal on the M3 Max under fast math
/// (`half_outputs_store_within_the_bound_or_die_named`: a `half4` converted from `float`, over
/// two sweeps from -2 to 2, onto loaded texels; a `float4` output under the same state stores
/// Metal's bytes in every case), the attachments past the bound are:
/// - `R16Float`, `RG16Float` and `RGBA16Float`, blended by other than `Min` or `Max`: `One` and
///   `One` 384 `half` units in the last place apart in `R16Float` and 3072 in `RG16Float`, `Zero`
///   and `OneMinusSourceAlpha` 3468 in `RGBA16Float`, `DestinationColor` and
///   `OneMinusDestinationColor` 2213, where the blend's terms cancel to less than the source;
///   and `One` and `Zero`, added, which stored Metal's bits over the finite texels measured, but
///   where Metal's blend of a module without fast math keeps the zero factor's product (a NaN
///   over an infinite or NaN texel, `+0` of a `-0` output over a texel of `+0` or above, the
///   runtime's EV FBL-zf1-m3), and the device's blend, which the runtime then runs, answers its
///   own there, as the runtime notes (`HalfOutputPrecision`, a `float` output's too);
/// - `R32Float`, `RG32Float` and `RGBA32Float`, blended by a factor that reads the source beside
///   a factor neither `Zero` nor `One`: in `RGBA32Float`, `SourceAlpha` and `OneMinusSourceAlpha`
///   1 float unit in the last place apart, `OneMinusSourceColor` twice 8.6e8 (`R32Float` and
///   `RG32Float` stored Metal's bits at the blends measured, and take the class's rule);
/// - the 8-bit `_sRGB` formats, blended by `DestinationColor` and `OneMinusDestinationColor`: 2
///   codes.
///
/// Every other format and blend measured stores within the bound: the 8-, 10- and 16-bit unorm
/// formats, `RG11B10Float` and `RGB9E5Float` at every blend measured, the `_sRGB` formats at the
/// other blends, and the float formats unblended or otherwise blended (premultiplied `One` and
/// `OneMinusSourceAlpha` stores Metal's bits in `RGBA16Float`, `RG16Float` and `RGBA32Float`, but
/// not in `R16Float`). The extended-range formats die of `extended-range-blend-source` first when
/// blended. A factor left to a specialization counts as one that reads the source.
fn half_apart(
    colors: &[Option<ColorAttachmentDescriptor>],
    written: &[u32],
) -> Option<(usize, PixelFormat)> {
    use metal2vulkan::metal::{BlendFactor as F, BlendOperation as O};
    let reads_the_source = |f: F| {
        matches!(
            f,
            F::SourceColor
                | F::OneMinusSourceColor
                | F::SourceAlpha
                | F::OneMinusSourceAlpha
                | F::SourceAlphaSaturated
                | F::Source1Color
                | F::OneMinusSource1Color
                | F::Source1Alpha
                | F::OneMinusSource1Alpha
                | F::Unspecialized
        )
    };
    let constant = |f: F| matches!(f, F::Zero | F::One);
    written.iter().find_map(|&i| {
        let i = usize::try_from(i).ok()?;
        let c = colors.get(i)?.as_ref()?;
        let channels = [
            (
                c.source_rgb_blend_factor,
                c.destination_rgb_blend_factor,
                c.rgb_blend_operation,
            ),
            (
                c.source_alpha_blend_factor,
                c.destination_alpha_blend_factor,
                c.alpha_blend_operation,
            ),
        ];
        let blended = channels
            .into_iter()
            .filter(|&(.., op)| c.blending_enabled && !matches!(op, O::Min | O::Max));
        let apart = match c.pixel_format {
            PixelFormat::R16Float | PixelFormat::RG16Float | PixelFormat::RGBA16Float => {
                { blended }.next().is_some()
            }
            PixelFormat::R32Float | PixelFormat::RG32Float | PixelFormat::RGBA32Float => {
                { blended }.any(|(s, t, _)| {
                    (reads_the_source(s) && !constant(t)) || (reads_the_source(t) && !constant(s))
                })
            }
            PixelFormat::R8UnormSRGB
            | PixelFormat::RG8UnormSRGB
            | PixelFormat::RGBA8UnormSRGB
            | PixelFormat::BGRA8UnormSRGB => { blended }
                .any(|ch| matches!(ch, (F::DestinationColor, F::OneMinusDestinationColor, _))),
            _ => false,
        };
        apart.then_some((i, c.pixel_format))
    })
}

/// Relays at `sel`, or dies there in strict mode ([`crate::answer::inexact`]), a draw at which
/// the runtime announced a `half` colour output of the state's fragment function
/// (`HalfOutputPrecision`), whose colour attachment `index` of `format` stores it past the bound
/// the rounding is relayed within ([`half_apart`]).
fn half_output_apart(sel: &'static str, (index, format): (usize, PixelFormat)) {
    crate::answer::inexact(sel, crate::answer::Inexact::HalfOutputBlend, || {
        format!(
            "a fragment function's half colour output into colour attachment {index}, \
             {format:?}, as blended or converted there: the runtime rounds it to half first, \
             where Metal carries it at float precision, and the stored value lies past the \
             bound that rounding is relayed within ({})",
            rt::Degradation::HalfOutputPrecision.code()
        )
    });
}

/// An open runtime render encoder, and what Metal's encoder holds that the runtime's has no form
/// for.
#[derive(Debug)]
pub struct RenderEncoderObject {
    /// The runtime's encoder, until it ends: a handle of at most 128 bytes, its lowering boxed by
    /// the runtime, held in place.
    pub encoder: EncoderObject<rt::RenderCommandEncoder>,
    /// What Metal's encoder holds of its draws that the runtime's has no form for, behind one
    /// lock: a draw reads it once, and a command that sets it takes the lock once ([`Drawing`]).
    /// Only the thread that encodes takes it, as [`EncoderObject::raw`], and a run of commands
    /// holds it ([`RenderEncoder::encoding`]), so each command's take inside the run is this
    /// thread's again, which no atomic operation takes.
    drawing: ReentrantMutex<RefCell<Drawing>>,
    /// The pass the runtime's encoder was made of: the descriptor as core set it, with the rate
    /// map [`open`] took, without the colour resolves [`untouched_resolves`] dropped and the
    /// unused depth and stencil resolve textures [`without_unused_resolves`] removes. Its
    /// attachments' store actions and resolve textures decide what a late setter does.
    pass: rt::RenderPassDescriptor,
    /// The colour attachments whose resolves [`untouched_resolves`] dropped: a late resolve
    /// setter of one dies, and a late store action is compared with the descriptor's own; where
    /// there is one, a late depth or stencil resolve texture that may resolve dies
    /// ([`VulkanRenderEncoder::resolves_beside_dropped`]).
    dropped: Vec<Dropped>,
    /// The attachments whose store action the descriptor left `Unknown` and no late setter has
    /// decided yet, each with why a resolve into the resolve texture it names now, the
    /// descriptor's or a late resolve setter's, is never sent ([`Unsent`]): a late store action
    /// that resolves into such a one dies ([`VulkanRenderEncoder::late_store`]).
    undecided: Mutex<Vec<(Attachment, Option<Unsent>)>>,
    /// The attachments of [`Self::undecided`] whose texture lies over read-only memory the
    /// runtime copies: a late store action that keeps such a one fails the command buffer as
    /// Metal's GPU faults it ([`VulkanRenderEncoder::read_only_stores`]).
    read_only_undecided: Mutex<Vec<Attachment>>,
    /// The store action a late setter gave each attachment whose store action the descriptor
    /// left `Unknown`, the last one set: whether a late depth or stencil resolve texture beside a
    /// dropped colour resolve may resolve ([`VulkanRenderEncoder::resolves_beside_dropped`]).
    late_stores: Mutex<Vec<(Attachment, rt::StoreAction)>>,
    /// Whether the pass rasterizes: one of a mutable rate map not yet reset does not, as the
    /// runtime's draws there do not (`tiled`).
    rasterizes: bool,
    /// Whether the pass's rate map has no layers, whose draws Metal rasterizes by what earlier
    /// maps left (`layered`).
    no_layers: bool,
    /// Whether the pass's rate map is one whose rates vary, whose regions the runtime's draws
    /// rasterize by its tiles' exact lines, where Metal's are its fixed-point ones (`layered`).
    varying_rates: bool,
    /// Whether the pass ends with the platform's draw of no vertices
    /// ([`crate::gpu::rate_map::no_draw`]): set where the pass's rate map was made of a `Mutable`
    /// descriptor, until a draw of the guest's takes its place ([`VulkanRenderEncoder::drew`]).
    undrawn: AtomicBool,
    /// Whether the reach of `drawing` holds nothing a draw asks of ([`Reach::quiet`]), so a draw, a tile
    /// dispatch or an execution asks nothing of it ([`VulkanRenderEncoder::issues`]): set at
    /// each bind.
    quiet: AtomicBool,
    /// The pass's bfloat colour attachment it loads and stores, where it has one: a draw, an
    /// execution of render commands or a tile dispatch in the pass dies ([`bfloat_loaded`]).
    bfloat_loaded: Option<String>,
}

/// What a render pipeline state tells the encoder that sets it of how its draws draw, read from
/// the runtime's state and the guest's descriptor, neither of which changes, at the state's
/// first `setRenderPipelineState:` ([`crate::gpu::resource::RenderPipelineObject::drawn`]): a
/// set reads none of it again.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StateFacts {
    /// Whether its vertex function writes any memory, its fragment, tile and object functions
    /// write textures, and its mesh function writes any memory (the runtime's
    /// `vertex_writes_memory`, `fragment_writes_textures`, `tile_writes_textures`,
    /// `object_writes_textures`, and `vertex_writes_memory` again, which answers a mesh render
    /// pipeline state's mesh function's), in [`VERTEX`] to [`MESH`] order ([`Reach::set`]). A
    /// stage the state has no function of writes nothing.
    writes: [bool; RENDER_STAGES],
    /// [`Marks::points_sized`].
    points_sized: bool,
    /// Whether it can encode indirect commands (the runtime's `can_encode_indirect_commands`).
    encodes: bool,
    /// [`Marks::half_apart`].
    half_apart: Option<(usize, PixelFormat)>,
    /// [`Marks::mesh`].
    mesh: Option<MeshAnswers>,
}

impl StateFacts {
    /// What `state` tells.
    fn of(state: &crate::gpu::resource::RenderPipelineObject) -> StateFacts {
        let raw = &state.raw;
        let colors = match state.reflection.descriptor() {
            Some(Reflected::Render(RenderDescriptor::Render(d))) => Some(&d.color_attachments),
            Some(Reflected::Render(RenderDescriptor::Mesh(m))) => {
                Some(&m.descriptor.color_attachments)
            }
            _ => None,
        };
        let apart = colors.and_then(|c| half_apart(c, &plain(|| raw.fragment_color_outputs())));
        let mesh = state.mesh_answers().copied();
        plain(|| StateFacts {
            writes: [
                mesh.is_none() && raw.vertex_writes_memory(),
                raw.fragment_writes_textures(),
                raw.tile_writes_textures(),
                raw.object_writes_textures(),
                mesh.is_some() && raw.vertex_writes_memory(),
            ],
            points_sized: raw.writes_point_size()
                || state.input_primitive_topology == PrimitiveTopologyClass::Point,
            encodes: raw.can_encode_indirect_commands(),
            half_apart: apart,
            mesh,
        })
    }
}

/// An attachment of a pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Attachment {
    /// The colour attachment at an index.
    Color(u32),
    /// The depth attachment.
    Depth,
    /// The stencil attachment.
    Stencil,
}

/// Whether `attachment` is a colour attachment past the eighth.
fn past_the_eighth(attachment: Attachment) -> bool {
    matches!(attachment, Attachment::Color(i)
        if usize::try_from(i).map_or(true, |i| i >= rt::MAX_COLOR_ATTACHMENTS))
}

/// What Metal's render encoder holds of its draws that the runtime's has no form for
/// ([`RenderEncoderObject::drawing`]).
#[derive(Debug, Default)]
struct Drawing {
    /// `setRenderPipelineState:` of nil since the last state set: the next draw dies.
    nil_pipeline: bool,
    /// What Metal's encoder holds of how its draws draw ([`Marks`]).
    marks: Marks,
    /// What its draws may write through what it binds ([`Reach`]).
    reach: Reach,
}

/// What the private setters left of how the encoder's draws draw, and what of the pipeline state
/// and fill mode decides whether a draw under them is exact ([`VulkanRenderEncoder::marked`]).
#[derive(Debug, Default, Clone, Copy)]
struct Marks {
    /// The arguments of the last `setPrimitiveRestartEnabled:index:`, which the runtime's
    /// encoder takes too: Metal restarts an indexed strip at its index alone after it, but not
    /// the line strip a line loop draws as here ([`VulkanRenderEncoder::primitives`]).
    restart: Option<(bool, u64)>,
    /// A pipeline state was set.
    pipeline: bool,
    /// The pipeline state set sizes its points as Metal does: its vertex function writes
    /// `[[point_size]]`, or its input primitive topology is the point class (the module docs).
    points_sized: bool,
    /// The amplification count the last `setVertexAmplificationCount:viewMappings:` set, which
    /// the runtime draws; `None` for none, Metal's 1.
    amplification: Option<u64>,
    /// `setVertexAmplificationMode:value:` left a count other than the runtime's, or one not
    /// measured ([`amplification_count`]).
    amplified: bool,
    /// The colour attachment of the pipeline state set, and its format, where a `half` colour
    /// output of its fragment function stores a value past the bound ([`half_apart`]).
    half_apart: Option<(usize, PixelFormat)>,
    /// What the pipeline state set, a mesh render pipeline state, holds its draws' threadgroups
    /// to ([`threads_held`]); `None` for any other state.
    mesh: Option<MeshAnswers>,
}

/// The amplification count `setVertexAmplificationMode:value:` sets, where it was measured (the
/// module docs): `value` in every mode but 1, the number of its set bits in mode 1, from 1 to 8;
/// `None` elsewhere: 0, which Metal draws copies of at random, and counts past 8.
fn amplification_count(mode: u64, value: u64) -> Option<u64> {
    let count = if mode == 1 {
        u64::from(value.count_ones())
    } else {
        value
    };
    (1..=8).contains(&count).then_some(count)
}

/// What a draw draws, for what the private setters left ([`VulkanRenderEncoder::marked`]).
#[derive(Clone, Copy, Debug)]
enum Shape {
    /// Primitives of a type, public or private, over the vertices in order, over indices or from
    /// indirect arguments.
    Primitives(rt::DrawPrimitiveType),
    /// Patches, over a control-point index buffer or not.
    Patches,
    /// A mesh draw, of its threads per object and per mesh threadgroup, directly or from
    /// indirect arguments.
    Mesh {
        /// Its threads per object threadgroup.
        object: MtlSize,
        /// Its threads per mesh threadgroup.
        mesh: MtlSize,
    },
}

/// Whether a store action resolves, or may (`Unknown`, which the encoder decides).
fn may_resolve(store: rt::StoreAction) -> bool {
    matches!(
        store,
        rt::StoreAction::MultisampleResolve
            | rt::StoreAction::StoreAndMultisampleResolve
            | rt::StoreAction::Unknown
    )
}

/// `d` without the depth and stencil resolve textures of an attachment without a texture whose
/// store action leaves them unused: Metal resolves nothing into them (`render_pass_memoryless`'s
/// unused resolve textures on depth and stencil), where the runtime refuses a resolve texture of
/// an attachment without a texture as unmeasured. One beside a texture goes to the runtime,
/// which leaves it untouched and stores the attachment as its action says, as Metal does
/// (`unused_depth_and_stencil_resolves_leave_their_textures`), and so does a colour
/// attachment's. Nothing else of the pass reads them: what Metal aborts at reads a depth or
/// stencil resolve texture only under a store action that resolves ([`metal_aborts_at`]).
fn without_unused_resolves(mut d: rt::RenderPassDescriptor) -> rt::RenderPassDescriptor {
    let unused = |a: &Option<rt::Texture>, store| a.is_none() && !may_resolve(store);
    if unused(&d.depth_attachment.texture, d.depth_attachment.store_action) {
        d.depth_attachment.resolve_texture = None;
    }
    if unused(
        &d.stencil_attachment.texture,
        d.stencil_attachment.store_action,
    ) {
        d.stencil_attachment.resolve_texture = None;
    }
    d
}

/// Whether the colour resolve of an attachment of `from` into a texture of `into` is the
/// runtime's to decide, as it was before any resolve was dropped here: the same format; an sRGB
/// pair (`R8Unorm`, `RG8Unorm`, `RGBA8Unorm`, `BGRA8Unorm`, `BGR10_XR` and `BGRA10_XR`, each with
/// its `_sRGB` twin: the colour-renderable pairs among the runtime's twins), which the runtime
/// resolves from a multisampled attachment in the attachment's encoding, the extended-range ones
/// by its own resolve of their codes after the pass (`extended-range-resolve`, of
/// [`crate::answer::DIES`], where Metal averages values finer than the codes); and the 8-bit
/// four-channel family across its channel orders (`RGBA8Unorm` and `BGRA8Unorm`, and their
/// twins), which it does not build. The runtime refuses every other pair (`ResolveFormat`),
/// which [`fate`] takes.
fn runtime_resolves(from: PixelFormat, into: PixelFormat) -> bool {
    use PixelFormat as P;
    const TWINS: [(PixelFormat, PixelFormat); 6] = [
        (P::R8Unorm, P::R8UnormSRGB),
        (P::RG8Unorm, P::RG8UnormSRGB),
        (P::RGBA8Unorm, P::RGBA8UnormSRGB),
        (P::BGRA8Unorm, P::BGRA8UnormSRGB),
        (P::BGR10XR, P::BGR10XRSRGB),
        (P::BGRA10XR, P::BGRA10XRSRGB),
    ];
    let eight_bit = |f| {
        matches!(
            f,
            P::RGBA8Unorm | P::RGBA8UnormSRGB | P::BGRA8Unorm | P::BGRA8UnormSRGB
        )
    };
    from == into
        || (eight_bit(from) && eight_bit(into))
        || TWINS.contains(&(from, into))
        || TWINS.contains(&(into, from))
}

/// What plain Metal was measured to do with the resolve of a colour attachment of `t` into a
/// texture of `r`, by their formats and the attachment's sample count alone, on the M3 Max
/// (`resolves_into_other_formats_are_metals_or_die_named`): a 4-sample `BGRA8Unorm` or
/// `BGRA8Unorm_sRGB` attachment's into `RGBA16Float`, `BGR10_XR` or `BGR10_XR_sRGB`, the formats
/// a drawable takes beside the 8-bit ones, leaves the texture untouched in the shapes
/// [`unmeasured_shape`] states; into `RGB10A2Unorm`, `BGR10A2Unorm`, `BGRA10_XR` or
/// `BGRA10_XR_sRGB` it writes into the texture, measured from a Private 2D multisample texture of
/// its own into a Shared single-level 2D texture of its own, both 16 by 16. `None` for any other
/// pair.
fn measured_pair(t: &rt::Texture, r: &rt::Texture) -> Option<Pair> {
    use PixelFormat as P;
    if !matches!(t.pixel_format(), P::BGRA8Unorm | P::BGRA8UnormSRGB) || t.sample_count() != 4 {
        return None;
    }
    match r.pixel_format() {
        P::RGBA16Float | P::BGR10XR | P::BGR10XRSRGB => Some(Pair::Untouched),
        P::RGB10A2Unorm | P::BGR10A2Unorm | P::BGRA10XR | P::BGRA10XRSRGB => Some(Pair::Written),
        _ => None,
    }
}

/// What plain Metal was measured to do with a resolve into a texture of another format
/// ([`measured_pair`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pair {
    /// It leaves the texture untouched.
    Untouched,
    /// It writes into the texture.
    Written,
}

/// What of the pass `d` lies outside the shapes plain Metal was measured to leave the resolve
/// texture `r` of colour attachment `index`, of texture `t`, untouched in ([`Pair::Untouched`]),
/// if anything: `source` holds the attachment's texture and resolve texture as core set them
/// ([`Over`]), and `mapped` is the rate map the pass takes ([`PassMap`]). Metal's validation
/// stops such a pass ("does not match Resolve PixelFormat"); a guest runs without it, and macOS
/// 26.6.2's Maps resolves its `BGRA8Unorm_sRGB` samples into a texture of another format, its
/// resolving passes loading them by `Load` or `DontCare`. Measured on the M3 Max
/// (`resolves_into_other_formats_are_metals_or_die_named`), Metal writes nothing into the
/// texture, `StoreAndMultisampleResolve` stores the samples as `Store` does, and
/// `MultisampleResolve` leaves them as they were before the pass, as `DontCare` does, for:
/// - an attachment of a Private 2D multisample texture of its own, compressed where Metal may
///   (`allowGPUOptimizedContents`) and not, of usage `RenderTarget`, alone or with `ShaderRead`
///   or `PixelFormatView`, at its level 0, slice 0 and depth plane 0, without store action
///   options;
/// - cleared by the pass, or loaded by `Load` or `DontCare` after a pass that cleared and stored
///   it, and drawn, or not: with no draw call, with only the state a draw takes, or with a draw
///   call of no vertices, of no instances, of two vertices of a triangle list, or indirect of no
///   vertices;
/// - resolved by `MultisampleResolve` or `StoreAndMultisampleResolve` into level 0, slice 0 and
///   depth plane 0 of an unrotated single-level, single-sample 2D texture of its size (16 by 16,
///   and 96 by 37; 1920 by 1080 and 1023 by 771 into a Shared one of its own, drawn and with no
///   draw call): one of its own (Shared, Managed, or Private compressed where Metal may and
///   not) of usage `ShaderRead` and `RenderTarget` (Shared and Private ones also of
///   `RenderTarget` alone, and with `ShaderWrite`, `PixelFormatView` or both), a Private
///   shareable one or one opened from its handle, or one over an IOSurface without planes
///   (Shared or Managed, each also presented as a framebuffer-only drawable);
/// - in a pass beside no other attachment, a depth one (`Depth32Float`, memoryless and not
///   stored or Private and stored; `Depth32Float_Stencil8` as both depth and stencil attachment,
///   memoryless, or Private cleared or not loaded, and not stored), or another 4-sample colour
///   attachment (`RGBA8Unorm` at index 0 or 1 or `BGRA8Unorm` at 2, resolving into a texture of
///   its format, or `R32Float` at 2, stored), or with a rate map of one sample of 1 on each axis.
///
/// A pass that draws nothing is no exception: Metal leaves the samples of one that clears the
/// attachment and resolves it by `MultisampleResolve` with no draw call as they were, compressed
/// or not, where a resolve into the attachment's own format leaves the clear in a compressed
/// one's (alike here). Measured on 2026-10-10, each as this platform's
/// samples and texture: 216 passes with no draw call (each load action, both actions, the
/// attachment compressed and not, into each format, 16 by 16 and 96 by 37 into a Shared texture
/// and 16 by 16 into a Private one); 112 of 120 with only the state a draw takes or a draw call
/// that draws nothing (into `RGBA16Float`, 16 by 16; the other 8 below); and 144 at 1920 by 1080
/// and 1023 by 771 (each load action, both actions, into each format, drawn and with no draw
/// call), of which the test holds 22. As Metal's `MultisampleResolve` stores none of the pass's
/// tiles into the samples, drawn or not, and its `StoreAndMultisampleResolve` stores them as
/// `Store` does, what writes the tiles (draws of any kind, an execution of render commands, a
/// tile dispatch) does not decide the drop: the pass made without the resolve leaves or stores
/// the same tiles, but where the runtime splits it, storing each part but the last, its
/// `DontCareStore`, relayed as for any `DontCare` store ([`crate::answer`]). Where the pass
/// loads by `DontCare`, stores by `StoreAndMultisampleResolve` and its draw calls draw nothing
/// on the GPU (two vertices of a triangle list, an indirect draw of no vertices), Metal keeps
/// the samples where the runtime's `Store` clears their tiles to zero: the runtime's
/// `DontCareLoadTiles`, announced and relayed as for any pass that loads by `DontCare` and
/// stores, not the drop's.
///
/// The drop takes those shapes at any size, of any usage, whatever the pass draws, beside any
/// depth and stencil attachment that does not resolve and any other colour attachment whose
/// resolve is the runtime's ([`runtime_resolves`]), and with any rate map the runtime takes as
/// the identity inside the screen ([`PassMap::Identity`]). Any other shape, which the runtime
/// may refuse for its own reasons too (a memoryless resolve texture, a level or slice past it),
/// is not measured: it is named. So is a depth or stencil resolve texture set late in such a
/// pass where it may resolve ([`VulkanRenderEncoder::resolves_beside_dropped`]).
fn unmeasured_shape(
    d: &rt::RenderPassDescriptor,
    index: usize,
    (t, r): (&rt::Texture, &rt::Texture),
    source: &[Source; 2],
    mapped: PassMap,
) -> Option<String> {
    use rt::StorageMode as S;
    let a = &d.color_attachments[index];
    let [texture, resolve] = source;
    if t.texture_type() != TextureType::Type2DMultisample {
        return Some(format!("the attachment is of {:?}", t.texture_type()));
    }
    if t.storage_mode() != S::Private {
        return Some(format!("the attachment is {:?}", t.storage_mode()));
    }
    if texture.over() != Some(Over::Own) {
        return Some(format!("the attachment is {}", Over::name(texture.over())));
    }
    if (a.level, a.slice, a.depth_plane) != (0, 0, 0) {
        return Some(format!(
            "the attachment is at level {}, slice {} and depth plane {}",
            a.level, a.slice, a.depth_plane
        ));
    }
    if a.store_action_options != rt::StoreActionOptions::NONE {
        return Some(format!(
            "the attachment's store action options are {:?}",
            a.store_action_options
        ));
    }
    let kind = (r.texture_type(), r.sample_count(), r.mipmap_level_count());
    if kind != (TextureType::Type2D, 1, 1) {
        return Some(format!(
            "the resolve texture is of {:?}, {} samples and {} levels",
            kind.0, kind.1, kind.2
        ));
    }
    if r.rotation() != 0 {
        return Some(format!(
            "the resolve texture is of rotation {}",
            r.rotation()
        ));
    }
    if (r.width(), r.height()) != (t.width(), t.height()) {
        return Some(format!(
            "the resolve texture is {} by {}, the attachment {} by {}",
            r.width(),
            r.height(),
            t.width(),
            t.height()
        ));
    }
    if (a.resolve_level, a.resolve_slice, a.resolve_depth_plane) != (0, 0, 0) {
        return Some(format!(
            "the resolve is into level {}, slice {} and depth plane {}",
            a.resolve_level, a.resolve_slice, a.resolve_depth_plane
        ));
    }
    match (resolve.over(), r.storage_mode()) {
        (Some(Over::Own), S::Shared | S::Managed | S::Private)
        | (Some(Over::Shareable), S::Private)
        | (Some(Over::Surface), S::Shared | S::Managed) => {}
        (over, storage) => {
            return Some(format!(
                "the resolve texture is {} of storage mode {storage:?}",
                Over::name(over)
            ));
        }
    }
    match mapped {
        PassMap::Unmapped | PassMap::Identity => {}
        PassMap::Varying => {
            return Some("the pass has a rasterization rate map whose rates vary".to_owned());
        }
        PassMap::Unset => {
            return Some(
                "the pass has a mutable rasterization rate map not yet reset, and rasterizes \
                 nothing"
                    .to_owned(),
            );
        }
    }
    let (depth, stencil) = (&d.depth_attachment, &d.stencil_attachment);
    let depth_resolves = [
        (&depth.texture, &depth.resolve_texture, depth.store_action),
        (
            &stencil.texture,
            &stencil.resolve_texture,
            stencil.store_action,
        ),
    ]
    .into_iter()
    .any(|(t, r, store)| t.is_some() && r.is_some() && may_resolve(store));
    if depth_resolves {
        return Some("the pass's depth or stencil attachment resolves".to_owned());
    }
    let other = (d.color_attachments.iter().enumerate()).find(|&(i, o)| {
        i != index
            && match (&o.texture, &o.resolve_texture) {
                (Some(t), Some(r)) => {
                    may_resolve(o.store_action)
                        && !runtime_resolves(t.pixel_format(), r.pixel_format())
                }
                _ => false,
            }
    });
    other.map(|(i, _)| {
        format!(
            "colour attachment {i} resolves into a texture of another format too, which the \
             runtime does not resolve into"
        )
    })
}

/// The rasterization rate map a pass takes, as far as its colour resolves tell maps apart
/// ([`unmeasured_shape`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PassMap {
    /// None.
    Unmapped,
    /// A map the runtime takes as the identity inside the screen ([`Class::Identity`]), or one
    /// of no layers.
    Identity,
    /// A map whose rates vary ([`Class::Varying`]).
    Varying,
    /// A mutable map not yet reset, whose pass rasterizes nothing.
    Unset,
}

/// What becomes of the resolve of a colour attachment ([`fate`]).
#[derive(Debug)]
enum Fate {
    /// The runtime's to resolve or refuse: there is none, its store action does not resolve,
    /// or the runtime decides the pair of formats ([`runtime_resolves`]).
    Runtimes,
    /// Dropped, as plain Metal leaves the texture untouched ([`untouched_resolves`]).
    Dropped,
    /// The pass dies at its creation, named so.
    Dies(String),
}

/// What the death of a resolve plain Metal writes into ([`Pair::Written`]) says Metal was
/// measured to do.
const WRITTEN: &str = "plain Metal writes into such a texture, measured from a Private 2D \
                       multisample texture of its own into a Shared single-level 2D texture of \
                       its own, both 16 by 16";

/// What becomes of the resolve of colour attachment `index` of the pass `d`, `sources` holding
/// the textures as core set them and `mapped` being the rate map the pass takes ([`PassMap`]):
/// the runtime's ([`Fate::Runtimes`]); dropped where plain Metal was measured to leave the
/// texture untouched ([`measured_pair`], [`unmeasured_shape`]); else, by a store action that
/// resolves or is left `Unknown` into a texture of another format the runtime refuses
/// (`ResolveFormat`), the pass dies, named with what plain
/// Metal was measured to do there. A store action left `Unknown` resolves by its late store
/// action, which nothing measured here, and the runtime counts it as resolving: it refuses such
/// a pass as it is made.
fn fate(d: &rt::RenderPassDescriptor, index: usize, sources: &Sources, mapped: PassMap) -> Fate {
    let a = &d.color_attachments[index];
    let (Some(t), Some(r)) = (&a.texture, &a.resolve_texture) else {
        return Fate::Runtimes;
    };
    let (from, into) = (t.pixel_format(), r.pixel_format());
    if !may_resolve(a.store_action) || runtime_resolves(from, into) {
        return Fate::Runtimes;
    }
    let metal = match (a.store_action, measured_pair(t, r)) {
        (rt::StoreAction::Unknown, _) => {
            "its store action left Unknown, whose late resolve plain Metal is not measured at"
                .to_owned()
        }
        (_, Some(Pair::Written)) => WRITTEN.to_owned(),
        (_, Some(Pair::Untouched)) => {
            let source = &sources.color[index];
            match unmeasured_shape(d, index, (t, r), source, mapped) {
                None => return Fate::Dropped,
                Some(what) => format!(
                    "plain Metal leaves such a texture untouched, but {what}, a shape it is not \
                     measured at"
                ),
            }
        }
        (_, None) => "a pair plain Metal is not measured at".to_owned(),
    };
    Fate::Dies(format!(
        "a resolve of colour attachment {index} ({}, {} samples) by {:?} into a texture of {} \
         ({} samples), another format, which the runtime does not resolve into: {metal}",
        from.name(),
        t.sample_count(),
        a.store_action,
        into.name(),
        r.sample_count()
    ))
}

/// A colour attachment whose resolve [`untouched_resolves`] dropped, as plain Metal leaves its
/// resolve texture untouched.
#[derive(Debug, Clone, Copy)]
struct Dropped {
    /// The attachment's index.
    index: u32,
    /// The store action the pass descriptor gave it, which a late store action is compared with
    /// ([`VulkanRenderEncoder::late_store`]).
    store: rt::StoreAction,
    /// The resolve texture's format.
    into: PixelFormat,
}

/// `d` without the colour resolves plain Metal leaves untouched ([`Fate::Dropped`], `sources`
/// holding the textures as core set them and `mapped` being the rate map the pass takes), and
/// those attachments: each stores as its action does without the resolve, as plain Metal was
/// measured to (`MultisampleResolve` as `DontCare`, which leaves the samples as they were, the
/// pass drawn or not, and `StoreAndMultisampleResolve` as `Store`), and the resolve texture
/// keeps its texels ([`unmeasured_shape`]). A late resolve setter of such an attachment dies, a
/// late store action is compared with the descriptor's own ([`Dropped`]), and a late depth or
/// stencil resolve texture of the pass that may resolve dies
/// ([`VulkanRenderEncoder::resolves_beside_dropped`]). Each attachment's fate is decided on `d`
/// as it is given, before any resolve is dropped.
fn untouched_resolves(
    mut d: rt::RenderPassDescriptor,
    sources: &Sources,
    mapped: PassMap,
) -> (rt::RenderPassDescriptor, Vec<Dropped>) {
    let drops: Vec<bool> = (0..d.color_attachments.len())
        .map(|i| matches!(fate(&d, i, sources, mapped), Fate::Dropped))
        .collect();
    let mut dropped = Vec::new();
    for ((index, a), drop) in (0u32..).zip(&mut d.color_attachments).zip(drops) {
        if !drop {
            continue;
        }
        let Some(r) = a.resolve_texture.take() else {
            continue;
        };
        dropped.push(Dropped {
            index,
            store: a.store_action,
            into: r.pixel_format(),
        });
        a.store_action = match a.store_action {
            rt::StoreAction::StoreAndMultisampleResolve => rt::StoreAction::Store,
            _ => rt::StoreAction::DontCare,
        };
    }
    (d, dropped)
}

/// The first colour resolve of the pass `d` that dies ([`Fate::Dies`]), named, if any, `sources`
/// holding the textures as core set them and `mapped` being the rate map the pass takes.
fn refused_resolve_format(
    d: &rt::RenderPassDescriptor,
    sources: &Sources,
    mapped: PassMap,
) -> Option<String> {
    (0..d.color_attachments.len()).find_map(|i| match fate(d, i, sources, mapped) {
        Fate::Dies(what) => Some(what),
        Fate::Runtimes | Fate::Dropped => None,
    })
}

/// The attachments of `d` with a texture whose store action is `Unknown`, each with `None`,
/// which [`VulkanRenderEncoder::undecided_dropped`] sets where a resolve into its resolve texture
/// is never sent ([`Unsent`]).
fn undecided(d: &rt::RenderPassDescriptor) -> Vec<(Attachment, Option<Unsent>)> {
    let unknown = |t: &Option<rt::Texture>, store| t.is_some() && store == rt::StoreAction::Unknown;
    let colors = (0u32..).zip(&d.color_attachments);
    let mut out: Vec<_> = colors
        .filter(|(_, c)| unknown(&c.texture, c.store_action))
        .map(|(i, _)| (Attachment::Color(i), None))
        .collect();
    if unknown(&d.depth_attachment.texture, d.depth_attachment.store_action) {
        out.push((Attachment::Depth, None));
    }
    if unknown(
        &d.stencil_attachment.texture,
        d.stencil_attachment.store_action,
    ) {
        out.push((Attachment::Stencil, None));
    }
    out
}

/// Why a pass's write into a texture is never sent: its memory is read-only, which the runtime
/// imports through a writable twin of a surface's pages, where the write would land, or copies,
/// where what Metal does with a resolve into it is unmeasured
/// ([`crate::gpu::writes::kept_write`]), or it lies on a surface whose GPU writes never reach its
/// memory ([`crate::os::dropped_write`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Unsent {
    /// Over read-only memory whose writes this platform does not answer as Metal's GPU does.
    Kept,
    /// On a surface whose GPU writes never reach its memory.
    Dropped,
}

impl Unsent {
    /// Why a write into `t` is never sent, if it is not: the twin first, which the write would
    /// reach.
    fn of(t: &VulkanTexture) -> Option<Unsent> {
        if crate::gpu::writes::texture_kept(t) {
            Some(Unsent::Kept)
        } else if t.gpu_read_only() {
            Some(Unsent::Dropped)
        } else {
            None
        }
    }

    /// Dies at `sel`, `what` naming a write into a texture, which the death names it lies in.
    fn dies(self, sel: &'static str, what: &str) -> ! {
        match self {
            Unsent::Kept => crate::gpu::writes::kept_write(sel, &format!("{what} over")),
            Unsent::Dropped => crate::os::dropped_write(sel, &format!("{what} on")),
        }
    }
}

/// What a pass of `d` may write in read-only memory whose writes this platform does not answer
/// as Metal's GPU does ([`crate::gpu::writes::kept_write`]), as `sources` says of its
/// attachments: an attachment's texture over a surface's memory the runtime twins, whatever the
/// pass's load and store actions, as the runtime's GPU may write the texels of one it loads and
/// stores unchanged, or whose store it does not care for, which Vulkan leaves undefined; one over
/// read-only memory the runtime copies that the pass keeps by `CustomSampleDepthStore`, which
/// nothing measured; or a resolve texture over either that its known store action resolves
/// into, which nothing measured either for read-only memory Metal keeps the GPU's writes from.
/// `None` for none. Another attachment over memory the runtime copies is answered by its store
/// action ([`VulkanRenderEncoder::read_only_stores`]). A resolve texture of an attachment whose
/// store action is `Unknown` is written only where a late store action resolves into it
/// ([`VulkanRenderEncoder::late_store`]), and a visibility result buffer only where a mode is set
/// ([`Reach::visibility`]).
fn kept_attachment(d: &rt::RenderPassDescriptor, sources: &Sources) -> Option<&'static str> {
    let (depth, stencil) = (&d.depth_attachment, &d.stencil_attachment);
    let colors = d.color_attachments.iter().map(|a| a.store_action);
    let stores = colors.chain([depth.store_action, stencil.store_action]);
    let held = (sources.color.iter()).chain([&sources.depth, &sources.stencil]);
    for (store, [texture, resolve]) in stores.zip(held) {
        if texture.twinned() {
            return Some(
                "a pass, which the runtime's GPU may write whatever its load and store actions, \
                 whose attachment lies over",
            );
        }
        if store == rt::StoreAction::CustomSampleDepthStore && texture.writes_kept() {
            return Some("a pass that keeps by CustomSampleDepthStore an attachment over");
        }
        if store.resolves() && resolve.writes_kept() {
            return Some("a pass that resolves into a texture over");
        }
    }
    None
}

/// Whether a pass's store action `store` keeps an attachment's texels in its memory: `Store`, and
/// `StoreAndMultisampleResolve`, which keeps a multisample primary, at which Metal's GPU faults
/// the command buffer where the memory is read-only (`crate::gpu::writes`'s measurements).
/// `CustomSampleDepthStore`, which nothing measured there, dies before it is asked
/// ([`kept_attachment`]).
fn stores_out(store: rt::StoreAction) -> bool {
    matches!(
        store,
        rt::StoreAction::Store | rt::StoreAction::StoreAndMultisampleResolve
    )
}

/// Whether `texture`, an attachment's, lies in read-only memory the runtime copies, dropping the
/// GPU's writes: read-only and not a surface's twin ([`crate::gpu::writes::Kept::Until`]).
fn over_read_only_copy(texture: &Source) -> bool {
    texture.writes_kept() && !texture.twinned()
}

/// What [`metal_aborts`] says of a pass without attachments whose `defaultRasterSampleCount` is
/// `count`, which the runtime refuses as plain Metal stops it
/// (`RenderPassProblem::DefaultRasterSampleCount`): one whose low 8 bits are
/// 8 to 15 or 128 to 255, or 0 in a count of 256 or more, where plain Metal ends the process
/// making the encoder with `abort()`, nothing on stderr (on this host's M3 Max under macOS 26.7,
/// at every such count the runtime's sample-count probe tried there: 8 to 15, 128, 255 and 256;
/// on an M4 under macOS 27.0.1, at every count to 256 and the larger ones the runtime measured).
/// A count of 0 is [`metal_aborts_at`]'s.
fn stopped_default_samples(count: u64) -> String {
    format!(
        "a pass without attachments of defaultRasterSampleCount {count}, a count plain Metal ends \
         the process at"
    )
}

/// What Metal aborts at in a pass of `d`, which the runtime would refuse or make: a pass of no
/// attachment with a target size and no default raster sample count (`render_pass_no_target`:
/// Metal aborts once the encoder is made; the runtime refuses its count of 0, and
/// the death is this one), a memoryless depth or stencil attachment stored with
/// `MultisampleResolve` at one sample with no resolve texture (`render_pass_memoryless`,
/// `render_pass_memoryless_stencil_resolve`), and more programmable sample positions than a
/// multisampled pass's samples (`render_pass_msaa`: 8 on a 4-sample pass).
fn metal_aborts_at(d: &rt::RenderPassDescriptor) -> Option<&'static str> {
    let textures: Vec<&rt::Texture> = d
        .color_attachments
        .iter()
        .filter_map(|c| c.texture.as_ref())
        .chain(d.depth_attachment.texture.as_ref())
        .chain(d.stencil_attachment.texture.as_ref())
        .collect();
    if textures.is_empty()
        && d.render_target_width != 0
        && d.render_target_height != 0
        && d.default_raster_sample_count == 0
    {
        return Some(
            "a pass of no attachment with a target size and no default raster sample count",
        );
    }
    let resolves_from_memory = |t: &Option<rt::Texture>, store, resolve: &Option<rt::Texture>| {
        t.as_ref().is_some_and(|t| {
            t.storage_mode() == rt::StorageMode::Memoryless
                && t.sample_count() == 1
                && store == rt::StoreAction::MultisampleResolve
                && resolve.is_none()
        })
    };
    let (depth, stencil) = (&d.depth_attachment, &d.stencil_attachment);
    if resolves_from_memory(&depth.texture, depth.store_action, &depth.resolve_texture)
        || resolves_from_memory(
            &stencil.texture,
            stencil.store_action,
            &stencil.resolve_texture,
        )
    {
        return Some(
            "a memoryless depth or stencil attachment resolving at one sample with no resolve texture",
        );
    }
    let samples = textures
        .first()
        .map_or(d.default_raster_sample_count, |t| t.sample_count());
    let positions = u64::try_from(d.sample_positions.len()).unwrap_or(u64::MAX);
    if samples > 1 && positions > samples {
        return Some("more programmable sample positions than the pass's samples");
    }
    None
}

/// What a pass of `d` writes in the memory of a surface whose GPU writes never reach it
/// ([`crate::os::dropped_write`]), as `sources` says of its attachments: an attachment's
/// texture, a resolve texture its known store action resolves into, or its visibility result
/// buffer; `None` for none. A resolve texture of an attachment whose store action is `Unknown`
/// is written only where a late store action resolves into it
/// ([`VulkanRenderEncoder::late_store`]).
fn dropped_attachment(d: &rt::RenderPassDescriptor, sources: &Sources) -> Option<&'static str> {
    let (depth, stencil) = (&d.depth_attachment, &d.stencil_attachment);
    let colors = d.color_attachments.iter().map(|a| a.store_action);
    let stores = colors.chain([depth.store_action, stencil.store_action]);
    let held = (sources.color.iter()).chain([&sources.depth, &sources.stencil]);
    for (store, [texture, resolve]) in stores.zip(held) {
        if texture.gpu_read_only() {
            return Some("a pass whose attachment lies on");
        }
        if store.resolves() && resolve.gpu_read_only() {
            return Some("a pass that resolves into a texture on");
        }
    }
    let visibility = sources.visibility.as_ref();
    visibility
        .is_some_and(|b| b.gpu_read_only())
        .then_some("a pass whose visibility result buffer lies on")
}

/// `id<MTLRenderCommandEncoder>`.
pub type VulkanRenderEncoder = Obj<RenderEncoderObject>;

/// `renderCommandEncoderWithDescriptor:` of `descriptor` on `command_buffer`: nil for a pass with
/// no attachment and no target size, as Metal answers; the runtime's other refusals (Metal's
/// validation's, an encoder open, a committed buffer) die at the send, as Metal raises at none
/// of them: it goes on or asserts (`render_pass_actions`: slice 1 of a two-slice array at an
/// array length of 2, which the runtime refuses, completes on Metal). A pass without attachments
/// takes Metal's answer to its `defaultRasterSampleCount` by the count's low 8 bits (this host's
/// answers in a run of the runtime's probe): 1 to 7 run, 0, 8 to
/// 15 and 128 to 255 die as Metal ends the process there ([`metal_aborts_at`],
/// [`stopped_default_samples`]), and 16 to 127 make the encoder and fail the command buffer
/// with Metal's code ([`crate::gpu::command`]). A pass of a rotated
/// texture is the runtime's, which stores its texels turned and loads them as Metal does, but
/// for the 32 by 32 tiles that none of the draws reach of a pass that loads a texture of
/// rotation 4 and draws, which Metal keeps in place and the runtime lands mirrored as the others
/// (`rotated-load-tiles`, [`crate::gpu::desc`]), so such a pass dies at `endEncoding`, naming
/// the kind; a pass that draws nothing is Metal's. One whose attachment lies over read-only
/// memory the runtime twins, or whose resolve texture it resolves into lies over read-only
/// memory, dies first ([`kept_attachment`]); one that stores an attachment over read-only memory
/// the runtime copies fails its command buffer as Metal's GPU faults it
/// ([`VulkanRenderEncoder::read_only_stores`]).
/// One whose attachment, resolve texture or visibility result buffer lies on a surface whose GPU
/// writes never reach its memory dies ([`crate::os::dropped_write`]): Metal faults the command
/// buffer of such a pass that stores what it cleared or drew, and completes one that stores
/// nothing, or draws nothing. A clear to a NaN is the runtime's, which stores Metal's bits, and
/// so is its resolve (`rgbx`'s `nan_clears_are_metals`).
/// A colour attachment that resolves into a texture of another format the runtime refuses is
/// made without the resolve where plain Metal was measured to leave the texture untouched, in
/// the shapes measured ([`untouched_resolves`]), and dies naming both formats and what plain
/// Metal was measured to do otherwise ([`refused_resolve_format`]), both decided by the rate map
/// the pass takes, as it takes it; the pairs the runtime decides itself are its own
/// ([`runtime_resolves`]). A resolve texture on
/// a surface whose GPU writes never reach its memory dies as a dropped write before that, as no
/// such pass was measured.
pub(crate) fn open(
    command_buffer: &VulkanCommandBuffer,
    _on: OnFault,
    descriptor: &VulkanRenderPassDescriptor,
) -> RaiseResult<Option<VulkanRenderEncoder>> {
    command_buffer.protected_creation("renderCommandEncoderWithDescriptor:");
    if let Some((attachment, setter, raw)) = descriptor.raw_value() {
        crate::answer::metal_aborts(
            "renderCommandEncoderWithDescriptor:",
            &format!(
                "a pass whose {attachment} holds {raw:#x} from {setter}, which Metal's validation \
                 refuses"
            ),
        );
    }
    let (d, map) = descriptor.pass();
    let sources = descriptor.sources();
    if crate::os::TWINS_READ_ONLY
        && let Some(what) = kept_attachment(&d, &sources)
    {
        crate::gpu::writes::kept_write("renderCommandEncoderWithDescriptor:", what);
    }
    if let Some(what) = dropped_attachment(&d, &sources) {
        crate::os::dropped_write("renderCommandEncoderWithDescriptor:", what);
    }
    // The colour resolves into a texture of another format, decided by the rate map the pass
    // takes, as it takes it.
    let settled = |d: rt::RenderPassDescriptor, mapped: PassMap| {
        if let Some(what) = refused_resolve_format(&d, &sources, mapped) {
            unrepresentable("renderCommandEncoderWithDescriptor:", &what);
        }
        untouched_resolves(d, &sources, mapped)
    };
    let made = match map {
        None => {
            let (d, dropped) = settled(d, PassMap::Unmapped);
            make(command_buffer, d, dropped, Taken::UNMAPPED)
        }
        Some(map) => map.for_pass(command_buffer, |raw, class| {
            let mapped = match class {
                Class::Unset => PassMap::Unset,
                Class::NoLayers | Class::Identity => PassMap::Identity,
                Class::Varying => PassMap::Varying,
            };
            let (mut d, dropped) = settled(d, mapped);
            d.rasterization_rate_map = Some(raw);
            let taken = Taken {
                class: Some(class),
                mutable: map.is_mutable(),
            };
            make(command_buffer, d, dropped, taken)
        }),
    };
    if let Ok(Some(e)) = &made {
        e.undecided_dropped(&sources);
        e.read_only_stores(&sources);
        let visibility = (sources.visibility.as_ref())
            .map_or(Kept::No, |b| crate::gpu::writes::buffer_bound(b).kept);
        plain(|| e.drawing.lock().borrow_mut().reach.visibility = visibility);
    }
    made
}

/// The colour attachment of the pass `d` of a bfloat format (`R16Bfloat`, `RG16Bfloat`,
/// `RGBA16Bfloat`) that the pass loads (`Load`) and stores (any store action but `DontCare`), if
/// any, named: a draw (of anything, or of nothing), an execution of render commands or a tile
/// dispatch in the pass dies ([`VulkanRenderEncoder::bfloat_drawn`]), where Metal's draws. The
/// runtime draws into a bfloat attachment, and the pass keeps the codes its draws do not write,
/// the device's own load and store, where Metal's tile store writes each code of the tiles a
/// draw's primitives reach back through `half` (NaNs to `0x7fc0`; the runtime's measurement on an
/// M4 and an M3 Max), announcing `extended-range-bits-kept`.
///
/// That kind is relayed for the extended-range formats on their guest reach ([`crate::answer`]),
/// where only bytes no Metal write leaves tell them apart. That ground does not hold for bfloat:
/// a kernel's write or a fill leaves codes `half` does not hold, and no guest is known to draw
/// into a bfloat attachment. The runtime announces the kind once a device, not at the pass that
/// meets it, so every such pass dies, those whose codes are Metal's among them. Each such death
/// is strict mode's: by default the draw relays it ([`crate::answer::Inexact::BfloatLoaded`]).
fn bfloat_loaded(d: &rt::RenderPassDescriptor) -> Option<String> {
    d.color_attachments
        .iter()
        .enumerate()
        .find_map(|(index, a)| {
            let format = a.texture.as_ref()?.pixel_format();
            let bfloat = matches!(
                format,
                PixelFormat::R16Bfloat | PixelFormat::RG16Bfloat | PixelFormat::RGBA16Bfloat
            );
            let kept = a.load_action == rt::LoadAction::Load
                && a.store_action != rt::StoreAction::DontCare;
            (bfloat && kept).then(|| format!("colour attachment {index} ({})", format.name()))
        })
}

/// Whether the runtime answers nil for the pass `d`, as Metal does: no attachment has a texture
/// and the target width or height is 0 (`RenderPassProblem::NoTargets`). A pass whose tile
/// size, per-sample storage, threadgroup memory or tile memory is past the device's is nil too
/// ([`make`]).
fn answers_nil(d: &rt::RenderPassDescriptor) -> bool {
    d.color_attachments.iter().all(|a| a.texture.is_none())
        && d.depth_attachment.texture.is_none()
        && d.stencil_attachment.texture.is_none()
        && (d.render_target_width == 0 || d.render_target_height == 0)
}

/// What a render command binds or makes resident, by the host memory of each object it names,
/// which a draw may write through ([`Reach`]): a slot's object read as it is bound
/// ([`crate::gpu::writes::buffer_bind`], [`crate::gpu::writes::texture_bind`]).
enum Bind<'a> {
    /// A buffer bound to a stage ([`VERTEX`] to [`MESH`]) at a slot.
    Buffer(usize, u64, Option<&'a VulkanBuffer>),
    /// A texture bound to a stage at a slot.
    Texture(usize, u64, Option<&'a VulkanTexture>),
    /// Buffers bound to a stage from a slot.
    Buffers(usize, u64, &'a [Option<&'a VulkanBuffer>]),
    /// Textures bound to a stage from a slot.
    Textures(usize, u64, &'a [Option<&'a VulkanTexture>]),
    /// The textures among resources made resident, the buffers among them, and whether they
    /// were made resident with write usage, with any of them lying in any memory.
    Used(Vec<Bound>, Vec<Bound>, bool),
}

/// What `cmd` binds or makes resident ([`Bind`]); `None` for any other command.
fn binds<'a>(cmd: &RenderCmd<'a, VulkanPlatform>) -> Option<Bind<'a>> {
    use crate::gpu::writes::{buffer_bound, texture_bound};
    Some(match *cmd {
        RenderCmd::SetFragmentBuffer {
            buffer: b, index, ..
        } => Bind::Buffer(FRAGMENT, index, b),
        RenderCmd::SetVertexBuffer {
            buffer: b, index, ..
        } => Bind::Buffer(VERTEX, index, b),
        RenderCmd::SetFragmentBuffers {
            buffers: list,
            range,
            ..
        } => Bind::Buffers(FRAGMENT, range.location, list),
        RenderCmd::SetObjectBuffers {
            buffers: list,
            range,
            ..
        } => Bind::Buffers(OBJECT, range.location, list),
        RenderCmd::SetMeshBuffers {
            buffers: list,
            range,
            ..
        } => Bind::Buffers(MESH, range.location, list),
        RenderCmd::SetTileBuffers {
            buffers: list,
            range,
            ..
        } => Bind::Buffers(TILE, range.location, list),
        RenderCmd::SetVertexBuffers {
            buffers: list,
            range,
            ..
        }
        | RenderCmd::SetVertexBuffersAttributeStrides {
            buffers: list,
            range,
            ..
        } => Bind::Buffers(VERTEX, range.location, list),
        RenderCmd::SetFragmentTexture { texture: t, index } => Bind::Texture(FRAGMENT, index, t),
        RenderCmd::SetFragmentTextures {
            textures: list,
            range,
        } => Bind::Textures(FRAGMENT, range.location, list),
        RenderCmd::SetObjectTextures {
            textures: list,
            range,
        } => Bind::Textures(OBJECT, range.location, list),
        RenderCmd::SetMeshTextures {
            textures: list,
            range,
        } => Bind::Textures(MESH, range.location, list),
        RenderCmd::SetTileTextures {
            textures: list,
            range,
        } => Bind::Textures(TILE, range.location, list),
        RenderCmd::SetVertexTextures {
            textures: list,
            range,
        } => Bind::Textures(VERTEX, range.location, list),
        RenderCmd::UseResources { resources, usage }
        | RenderCmd::UseResourcesStages {
            resources, usage, ..
        } => {
            let write = usage & u64::from(ResourceUsage::WRITE.as_raw()) != 0;
            let textures: Vec<_> = (resources.iter().flatten())
                .filter_map(|r| match r {
                    ResourceRef::Texture(t) => Some(texture_bound(t)),
                    _ => None,
                })
                .collect();
            let buffers: Vec<_> = (resources.iter().flatten())
                .filter_map(|r| match r {
                    ResourceRef::Buffer(b) => Some(buffer_bound(b)),
                    _ => None,
                })
                .collect();
            let any = textures.iter().chain(&buffers).any(|m| m.meets);
            Bind::Used(textures, buffers, write && any)
        }
        _ => return None,
    })
}

/// What a pass's draws may write through what it binds and made resident: the host memory of
/// the buffers and textures bound at each stage's slots, of the textures made resident with
/// write usage, and the pipeline state's facts of which slots its functions write. A draw, a
/// tile dispatch or an execution reads it ([`Reach::unsent`]). It holds only the binds that can
/// decide what a draw asks ([`matters`]), each distinct one made resident once, so a draw asks
/// nothing of the objects of plain memory every pass binds, and a pass that makes the same
/// resources resident before each draw holds them once.
#[derive(Debug, Default)]
struct Reach {
    /// Each stage's buffers by slot, in [`VERTEX`] to [`MESH`] order.
    buffers: [BTreeMap<u64, Bound>; RENDER_STAGES],
    /// Each stage's textures by slot.
    textures: [BTreeMap<u64, Bound>; RENDER_STAGES],
    /// The textures made resident with write usage, each distinct one once, in the order first
    /// made resident.
    used_textures: Vec<Bound>,
    /// The buffers made resident with write usage, likewise.
    used_buffers: Vec<Bound>,
    /// Whether the pass's visibility result buffer lies over read-only memory the runtime imports
    /// through a writable twin of its pages (a surface's): a mode other than `Disabled` dies, as
    /// the pass writes a result at its offset, a draw after it or not
    /// ([`crate::gpu::writes::kept_write`]). One over read-only memory the runtime copies keeps
    /// nothing: the result is dropped, as Metal drops it.
    visibility: Kept,
    /// The reflection of the pipeline state set, which this holds as Metal's encoder holds the
    /// state, and whether its functions write textures, which tells which texture slots they
    /// may write where its slot writes cannot: the vertex and mesh functions' where they write
    /// any memory, and the fragment, tile and object functions' where they write textures.
    pipeline: Option<(Arc<Reflection>, [bool; RENDER_STAGES])>,
}

/// Whether a bind's `m` can decide what a draw asks of it ([`Reach::kept`], [`Reach::unsent`]):
/// it lies over read-only memory whose writes this platform does not answer as Metal's GPU does
/// (a surface's the runtime twins, or a texture over memory the runtime copies), on a surface
/// whose GPU writes never reach its memory, or is a texture whose writes the runtime refuses. A
/// draw asks nothing of any other.
fn matters(m: &Bound) -> bool {
    !matches!(m.kept, Kept::No) || m.dropped || m.refused.is_some()
}

/// Whether a slot's bind of `m` is held by the reach ([`Reach::bind_one`]): it lies in memory
/// and decides something ([`matters`]). A slot bound to anything else is empty.
fn slot_decides(m: &Bound) -> bool {
    m.meets && matters(m)
}

/// Whether `bind` decides something in a reach that holds nothing ([`Reach::quiet`]): an object
/// it binds at a slot does ([`slot_decides`]), or it makes resident for writing one that matters
/// ([`matters`]). A buffer [`buffer_bind`] reads nothing of decides nothing, and its default
/// [`Bound`] is not made.
///
/// [`buffer_bind`]: crate::gpu::writes::buffer_bind
fn quiet_bind_decides(bind: &Bind<'_>) -> bool {
    use crate::gpu::writes::{buffer_bind_reads, buffer_bound, texture_bind};
    let buffer = |b: Option<&VulkanBuffer>| {
        b.is_some_and(|b| buffer_bind_reads(b) && slot_decides(&buffer_bound(b)))
    };
    let texture = |t: Option<&VulkanTexture>| t.is_some_and(|t| slot_decides(&texture_bind(t)));
    match bind {
        Bind::Buffer(_, _, b) => buffer(*b),
        Bind::Texture(_, _, t) => texture(*t),
        Bind::Buffers(_, _, list) => list.iter().any(|&b| buffer(b)),
        Bind::Textures(_, _, list) => list.iter().any(|&t| texture(t)),
        Bind::Used(textures, buffers, write) => {
            *write && textures.iter().chain(buffers).any(matters)
        }
    }
}

/// Whether a draw's write through `m` is never sent where it lies on a surface whose GPU writes
/// never reach its memory, or into a texture whose writes the runtime refuses
/// ([`VulkanRenderEncoder::issues`]).
fn unsendable(m: &Bound) -> bool {
    m.dropped || m.refused.is_some()
}

impl Reach {
    /// `m`, of one object, bound to `stage`'s buffer slot (`buffers`) or texture slot `slot`. A
    /// slot that holds an equal `m` already is left as it is, as each draw binds its objects anew;
    /// one of no memory, or that decides nothing ([`matters`]), empties the slot.
    fn bind_one(&mut self, stage: usize, buffers: bool, slot: u64, m: Bound) {
        let slots = if buffers {
            &mut self.buffers[stage]
        } else {
            &mut self.textures[stage]
        };
        if !slot_decides(&m) {
            slots.remove(&slot);
        } else if slots.get(&slot) != Some(&m) {
            slots.insert(slot, m);
        }
    }

    /// `textures` and `buffers` made resident with write usage: those that decide something
    /// ([`matters`]), each not held already.
    fn used(&mut self, textures: Vec<Bound>, buffers: Vec<Bound>) {
        for (list, held) in [
            (textures, &mut self.used_textures),
            (buffers, &mut self.used_buffers),
        ] {
            for m in list.into_iter().filter(matters) {
                if !held.contains(&m) {
                    held.push(m);
                }
            }
        }
    }

    /// The pipeline state of `reflection` set, whose functions write as `writes` says
    /// ([`StateFacts::writes`]).
    fn set(&mut self, reflection: &Arc<Reflection>, writes: [bool; RENDER_STAGES]) {
        self.pipeline = Some((Arc::clone(reflection), writes));
    }

    /// Whether it holds nothing a draw asks of: no bind at a slot and nothing made resident that
    /// decides anything ([`matters`]), so [`Self::kept`] and [`Self::unsent_dropped`] find
    /// nothing, whatever the pipeline state. Most passes bind only plain memory, and stay so.
    fn quiet(&self) -> bool {
        (self.buffers.iter().chain(&self.textures)).all(BTreeMap::is_empty)
            && self.used_textures.is_empty()
            && self.used_buffers.is_empty()
    }

    /// Whether a texture bound or made resident for writing, or a buffer bound to the tile,
    /// object or mesh stage, lies on a surface whose GPU writes never reach its memory
    /// ([`crate::os::gpu_read_only`]), or is a texture whose writes the runtime refuses. A vertex
    /// or fragment function's stores into a buffer on one are dropped, as on Metal, and are sent
    /// ([`crate::os::dropped_write`]); an object or mesh function's, which nothing measured, are
    /// not, as a tile function's.
    fn unsent_dropped(&self) -> bool {
        let unsent_buffers = [TILE, OBJECT, MESH].map(|stage| &self.buffers[stage]);
        let bound = (self.textures.iter().chain(unsent_buffers)).flat_map(BTreeMap::values);
        bound.chain(&self.used_textures).any(unsendable)
    }

    /// What a draw, a tile dispatch, or an execution whose commands take `inherits` of the
    /// encoder, may write that is never sent where it lies on a surface whose GPU writes never
    /// reach its memory, or into a texture whose writes the runtime refuses ([`unsendable`],
    /// [`VulkanRenderEncoder::issues`]): the textures made resident with write usage, then the
    /// textures bound at the slots the pipeline state's functions write, and, but for an
    /// execution, whose commands run no tile, object or mesh function, the buffers bound at the
    /// slots those functions write.
    fn unsent(&self, inherits: Option<Inherits>) -> Vec<(Whether, Bound)> {
        let used = (self.used_textures.iter()).filter(|m| unsendable(m));
        let mut written: Vec<_> = used.map(|m| (Whether::Writes, m.clone())).collect();
        let own = inherits.is_none();
        let buffers = [false, false, own, own, own];
        written.extend(self.at_slots(inherits, buffers, unsendable));
        written
    }

    /// How a draw, a tile dispatch, or an execution whose commands take `inherits` of the encoder,
    /// writes through an object over read-only memory whose writes this platform does not answer
    /// as Metal's GPU does ([`crate::gpu::writes::kept_write`]): a surface's the runtime twins, or,
    /// for a texture, memory the runtime copies, the first found: a texture or buffer
    /// made resident with write usage, and, as [`Self::at_slots`] reads them, a texture bound at
    /// a slot a function writes and a buffer bound at one, of every stage, but for an execution,
    /// whose commands run no tile, object or mesh function and take the encoder's buffers only
    /// where they inherit them. `None` where it writes none.
    fn kept(&self, inherits: Option<Inherits>) -> Option<Whether> {
        let mut used = (self.used_textures.iter()).chain(&self.used_buffers);
        if used.any(|m| m.kept.holds()) {
            return Some(Whether::Writes);
        }
        let (buffers, own) = (inherits.is_none_or(|i| i.buffers), inherits.is_none());
        let slots = self.at_slots(inherits, [buffers, buffers, own, own, own], |m| {
            m.kept.holds()
        });
        slots.into_iter().map(|(w, _)| w).find(Whether::holds)
    }

    /// What a draw, or an execution whose commands take `inherits` of the encoder, may write at the
    /// stages' slots of the objects `wanted` picks, one entry an object, with whether it does: the
    /// textures bound at the slots the pipeline state's functions may store through, by its slot
    /// writes, and the buffers too at the stages `buffers` names, in [`VERTEX`] to [`MESH`] order,
    /// of the stages the state has functions of ([`Reflection::stages`]: a mesh render pipeline
    /// state's object, mesh and fragment stages, any other's vertex, fragment and tile stages).
    /// Only a slot they answer the function cannot store through is unwritten: a read-only buffer's
    /// bind at one draws, whatever else the function declares. Where they cannot tell, for the
    /// state or for a slot an undetermined argument may cover, a function may write the buffer
    /// slot, as a kernel may, a vertex or mesh function that writes memory the texture slot, and a
    /// fragment, tile or object function that writes textures the texture slot. A vertex or mesh
    /// function that writes no memory, by the runtime's translation of it (its
    /// `vertex_writes_memory`), writes no slot of its stage, whatever its slot writes answer of an
    /// argument: they answer that a `device` pointer that is not `const` may be stored through,
    /// stored through or not. An execution whose commands run pipeline states of their own, which
    /// no reflection here tells, writes every bound buffer it takes, and no bound texture: Metal
    /// makes no state that supports indirect command buffers of a function with a texture argument,
    /// and such commands reach a texture only through an argument buffer, whose textures are made
    /// resident. Measured on the M3 Max (macOS 26.7): `supportIndirectCommandBuffers` with a
    /// vertex, fragment or kernel function taking a texture to read, sample or write answers no
    /// state ("Fragment shader cannot be used with indirect command buffers"), with one taking
    /// buffers or an argument buffer of a texture a state; an execution that inherits a state made
    /// without it, of such a function, faults its command buffer (`PageFault`), the texture bound
    /// unwritten.
    fn at_slots(
        &self,
        inherits: Option<Inherits>,
        buffers: [bool; RENDER_STAGES],
        wanted: fn(&Bound) -> bool,
    ) -> Vec<(Whether, Bound)> {
        let mut written = Vec::new();
        let state = match inherits {
            Some(i) if !i.pipeline => None,
            _ => match &self.pipeline {
                Some(p) => Some(p),
                // No state for the draw, which the runtime refuses.
                None => return written,
            },
        };
        let stages = state.map_or(DRAWING_STAGES, |(reflection, _)| reflection.stages());
        for &stage in stages {
            if matches!(stage, VERTEX | MESH) && state.is_some_and(|(_, facts)| !facts[stage]) {
                continue;
            }
            let at = |texture: bool, slot: u64| match state {
                Some((reflection, facts)) => Whether::Slot(
                    Arc::clone(reflection),
                    Slot {
                        stage,
                        texture,
                        slot,
                        untold: !texture || facts[stage],
                    },
                ),
                None => Whether::Writes,
            };
            let held = [
                (false, &self.buffers[stage], buffers[stage]),
                (true, &self.textures[stage], state.is_some()),
            ];
            for (texture, slots, _) in held.into_iter().filter(|&(.., on)| on) {
                for (&slot, m) in slots.iter().filter(|(_, m)| wanted(m)) {
                    written.push((at(texture, slot), m.clone()));
                }
            }
        }
        written
    }
}

/// What a pass took of its rate map as [`open`] took it ([`crate::gpu::rate_map::Class`]).
#[derive(Debug, Clone, Copy)]
struct Taken {
    /// The map's class as the pass took it; `None` for a pass of no map.
    class: Option<Class>,
    /// Whether the map was made of a `Mutable` descriptor.
    mutable: bool,
}

impl Taken {
    /// A pass of no rate map.
    const UNMAPPED: Taken = Taken {
        class: None,
        mutable: false,
    };
}

/// The encoder of the pass `d` on `command_buffer`, with its rate map as [`open`] took it,
/// `taken`, and without the colour resolves it `dropped` ([`untouched_resolves`]). One made is
/// work for Metal's tiler ([`crate::gpu::command::CommandBufferObject::tiler_work`]).
fn make(
    command_buffer: &VulkanCommandBuffer,
    d: rt::RenderPassDescriptor,
    dropped: Vec<Dropped>,
    taken: Taken,
) -> RaiseResult<Option<VulkanRenderEncoder>> {
    let sel = "renderCommandEncoderWithDescriptor:";
    let d = without_unused_resolves(d);
    match plain_at(sel, || command_buffer.raw.render_command_encoder(&d)) {
        Ok(raw) => {
            command_buffer.tiler_work(sel, crate::gpu::metal_space::FirstUse::RenderEncoder);
            opened(command_buffer, raw, (d, dropped), taken).map(Some)
        }
        Err(rt::Error::Nil(n))
            if n.problem == rt::Problem::RenderPass(rt::RenderPassProblem::NoTargets) =>
        {
            debug_assert!(answers_nil(&d), "a nil pass answers_nil missed");
            Ok(None)
        }
        Err(rt::Error::Invalid(i)) if i.problem == rt::Problem::EncoderOpen => metal_aborts(
            sel,
            "a command buffer with an open encoder (\"with uncommitted encoder\")",
        ),
        Err(e) => match metal_aborts_at(&d) {
            Some(what) => metal_aborts(sel, what),
            None => match e {
                // What plain Metal answers its encoder nil for: a tile size, per-sample storage,
                // threadgroup memory or tile memory past the device's.
                rt::Error::Nil(_) => Ok(None),
                rt::Error::Invalid(i)
                    if matches!(
                        i.problem,
                        rt::Problem::RenderPass(
                            rt::RenderPassProblem::DefaultRasterSampleCount { .. }
                        )
                    ) =>
                {
                    metal_aborts(sel, &stopped_default_samples(d.default_raster_sample_count))
                }
                rt::Error::Invalid(i)
                    if matches!(
                        i.problem,
                        rt::Problem::RenderPass(
                            rt::RenderPassProblem::ArrayLengthPastSlices { .. }
                        )
                    ) =>
                {
                    unrepresentable(
                        sel,
                        &format!(
                            "{:?}: a renderTargetArrayLength past an attachment's slices, where \
                             Metal renders min(length, the texture's slices) layers from the \
                             attachment's slice, those past the texture into the memory after \
                             it at its slice stride (a buffer's bytes, for a texture over one) \
                             or into a heap neighbour, and a draw that writes no layer renders \
                             the base slice alone; nothing here writes past a texture",
                            i.problem
                        ),
                    )
                }
                e => {
                    if crate::gpu::desc::is_format_refusal(&e) {
                        crate::gpu::ycbcr::pass_refused(sel, &d, &e);
                    }
                    decide(sel, Policy::Dies, e)
                }
            },
        },
    }
}

/// The encoder of the runtime's `raw`, just made on `command_buffer` of the pass `d`, without its
/// unused resolves ([`without_unused_resolves`]) and the colour resolves `dropped` lists
/// ([`untouched_resolves`]). What Metal aborts at in the pass dies then.
fn opened(
    command_buffer: &VulkanCommandBuffer,
    raw: rt::RenderCommandEncoder,
    (d, dropped): (rt::RenderPassDescriptor, Vec<Dropped>),
    taken: Taken,
) -> RaiseResult<VulkanRenderEncoder> {
    let sel = "renderCommandEncoderWithDescriptor:";
    let aborts = metal_aborts_at(&d);
    let e = Obj::new(RenderEncoderObject {
        encoder: EncoderObject::new(raw, command_buffer),
        drawing: ReentrantMutex::new(RefCell::new(Drawing::default())),
        undecided: Mutex::new(undecided(&d)),
        read_only_undecided: Mutex::new(Vec::new()),
        late_stores: Mutex::new(Vec::new()),
        no_layers: taken.class == Some(Class::NoLayers),
        dropped,
        rasterizes: taken.class != Some(Class::Unset),
        varying_rates: taken.class == Some(Class::Varying),
        undrawn: AtomicBool::new(taken.mutable),
        quiet: AtomicBool::new(true),
        bfloat_loaded: bfloat_loaded(&d),
        pass: d,
    });
    if let Some(what) = aborts {
        metal_aborts(sel, what);
    }
    Ok(e)
}

/// A binding index or a count core passes as `NSUInteger`, which the runtime takes as `u32`.
fn small(sel: &'static str, value: u64) -> u32 {
    u32::try_from(value).unwrap_or_else(|_| unrepresentable(sel, &format!("{value} as a u32")))
}

/// The first index of `range`, and the first `range.length` elements of `slice`: a slice shorter
/// than its range dies, as Metal's reads past it.
fn ranged<'a, T>(sel: &'static str, range: NsRange, slice: &'a [T]) -> (u32, &'a [T]) {
    let n = usize::try_from(range.length)
        .ok()
        .filter(|&n| n <= slice.len())
        .unwrap_or_else(|| {
            unrepresentable(sel, &format!("{} elements for {range:?}", slice.len()))
        });
    (small(sel, range.location), &slice[..n])
}

/// Runs `call` with core's buffers as the runtime's ([`listed`]).
fn buffers<'a, R>(
    list: &[Option<&'a VulkanBuffer>],
    call: impl FnOnce(&[Option<&'a rt::Buffer>]) -> R,
) -> R {
    listed(list, |b| b.map(|b| &b.raw), call)
}

/// Runs `call` with core's textures as the runtime's, bound by `sel`
/// ([`crate::gpu::resource::TextureObject::texels`], [`listed`]).
fn textures<'a, R>(
    sel: &'static str,
    list: &[Option<&'a VulkanTexture>],
    call: impl FnOnce(&[Option<&'a rt::Texture>]) -> R,
) -> R {
    listed(list, |t| t.map(|t| t.texels(sel)), call)
}

/// Runs `call` with core's sampler states as the runtime's ([`listed`]).
fn samplers<'a, R>(
    list: &[Option<&'a crate::gpu::resource::VulkanSamplerState>],
    call: impl FnOnce(&[Option<&'a rt::SamplerState>]) -> R,
) -> R {
    listed(list, |s| s.map(|s| &**s), call)
}

/// The runtime's form of core's resources, the nil ones skipped.
fn present<'a>(list: &'a [Option<ResourceRef<'a, VulkanPlatform>>]) -> Vec<rt::ResourceRef<'a>> {
    list.iter().flatten().map(resource).collect()
}

/// The runtime's form of a resource of core's.
fn resource<'a>(r: &'a ResourceRef<'a, VulkanPlatform>) -> rt::ResourceRef<'a> {
    match r {
        ResourceRef::Buffer(b) => rt::ResourceRef::Buffer(&b.raw),
        ResourceRef::Texture(t) => rt::ResourceRef::Texture(&t.raw),
        ResourceRef::Icb(icb) => rt::ResourceRef::IndirectCommandBuffer(&icb.raw),
    }
}

/// Core's heaps, as the runtime's: a nil one is skipped, as Metal's encoder skips it.
fn heaps<'a>(list: &[Option<&'a crate::gpu::resource::VulkanHeap>]) -> Vec<&'a rt::Heap> {
    list.iter().flatten().map(|h| &h.raw).collect()
}

/// The stages `useHeaps:count:` and `useResources:count:usage:` reach on a render encoder.
const VERTEX_AND_FRAGMENT: RenderStages = RenderStages::VERTEX.union(RenderStages::FRAGMENT);

/// Core's `MTLViewport`.
fn viewport(v: MtlViewport) -> rt::Viewport {
    rt::Viewport::new(v.origin_x, v.origin_y, v.width, v.height, v.znear, v.zfar)
}

/// Core's `MTLScissorRect`.
fn scissor(r: MtlScissorRect) -> rt::ScissorRect {
    rt::ScissorRect::new(r.x, r.y, r.width, r.height)
}

/// The line of a draw's death in a pass of a rate map whose rates vary
/// ([`Class::Varying`]).
const VARYING_RATES_DRAW: &str = "a draw in a pass of a rasterization rate map whose rates vary, \
     which the runtime rasterizes by regions whose lines are the map's tiles' exact lines, where \
     Metal's are its fixed-point ones (rasterization-rate-map)";

/// The line of a tile dispatch's death in a pass of a rate map whose rates vary
/// ([`Class::Varying`]).
const VARYING_RATES_TILES: &str = "a tile dispatch in a pass of a rasterization rate map whose \
     rates vary, which Metal is not measured at (rasterization-rate-map)";

/// The line of a draw's death, direct or an execution's command, of points under a state whose
/// points Metal sizes none of ([`sized()`]).
pub(crate) const UNSIZED_POINTS: &str = "points under a vertex function that writes no \
     [[point_size]], outside the point input primitive topology class, whose points Metal does \
     not size: most such draws draw nothing, and some a box whose size changes from one run to \
     the next";

/// A raw store action, sent with `sel`.
fn store_action(sel: &'static str, raw: u64) -> rt::StoreAction {
    rt::StoreAction::from_raw(raw)
        .unwrap_or_else(|_| unrepresentable(sel, &format!("store action {raw}")))
}

/// A raw store action option set, sent with `sel`.
fn store_action_options(sel: &'static str, raw: u64) -> rt::StoreActionOptions {
    rt::StoreActionOptions::from_raw(raw)
        .unwrap_or_else(|_| unrepresentable(sel, &format!("store action options {raw}")))
}

/// A late resolve target of a texture at its level, slice and depth plane; `None` for nil.
fn resolve_target(
    texture: Option<&rt::Texture>,
    (slice, depth_plane, level): (u64, u64, u64),
) -> Option<rt::ResolveTarget> {
    texture.map(|t| {
        let mut target = rt::ResolveTarget::new(t.clone());
        target.slice = slice;
        target.depth_plane = depth_plane;
        target.level = level;
        target
    })
}

/// Whether a draw of `count` vertices or indices of `t` makes no primitive, which Metal draws
/// nothing of whatever the private setters left ([`VulkanRenderEncoder::marked`]): fewer than
/// one of its primitives takes, a private type's as the list Metal draws it as
/// ([`rt::DrawPrimitiveType`]).
fn draws_none(t: rt::DrawPrimitiveType, count: u32) -> bool {
    use rt::DrawPrimitiveType as T;
    let least = match t {
        T::Public(PrimitiveType::Point) => 1,
        T::Public(PrimitiveType::Line | PrimitiveType::LineStrip) | T::LineLoop => 2,
        T::Public(PrimitiveType::Triangle | PrimitiveType::TriangleStrip) | T::TriangleFan => 3,
        _ => 4,
    };
    count < least
}

/// A draw's raw primitive type: a public one, one of the private ones Metal draws (5 to 8), or
/// 9, which Metal's GPU fails the command buffer at ([`VulkanRenderEncoder::nine`]), each of
/// which the runtime draws as Metal does ([`rt::DrawPrimitiveType`]); a type past it no
/// `MTLPrimitiveType` is, which dies.
fn draw_type(sel: &'static str, raw: u64) -> rt::DrawPrimitiveType {
    match raw {
        5..=9 => rt::DrawPrimitiveType::from_raw(raw).unwrap_or_else(|e| refused(sel, &e)),
        _ => rt::DrawPrimitiveType::Public(metal_value(sel, "primitive type", raw)),
    }
}

/// The primitive and index type of an indexed draw: `None` for an index type no `MTLIndexType`
/// is, which Metal draws nothing for, where the primitive type is not 9; of type 9 it dies
/// there ([`unmeasured_type_9`]).
fn indexed_kinds(
    sel: &'static str,
    primitive_type: u64,
    raw_index_type: u64,
) -> Option<(rt::DrawPrimitiveType, IndexType)> {
    let t = draw_type(sel, primitive_type);
    let i = index_type(raw_index_type);
    if t == rt::DrawPrimitiveType::Nine && i.is_none() {
        unmeasured_type_9(sel);
    }
    Some((t, i?))
}

/// Dies at `sel`, a draw of primitive type 9 over an index type no `MTLIndexType` is: Metal
/// draws nothing of a draw of a public type over one, and faults its command buffer at a draw of
/// type 9 over any of its index types, and nothing measured which it does at both.
fn unmeasured_type_9(sel: &'static str) -> ! {
    unrepresentable(
        sel,
        "primitive type 9 over an index type no MTLIndexType is, which nothing measured: Metal \
         draws nothing of a public type over one, and faults at type 9 over its index types",
    )
}

/// A draw's raw index type: `None` for one no `MTLIndexType` is, which Metal draws nothing for.
fn index_type(raw: u64) -> Option<IndexType> {
    u32::try_from(raw)
        .ok()
        .and_then(|raw| IndexType::try_from(raw).ok())
}

/// The range of an `NSRange`, for `sel`.
fn range_of(sel: &'static str, r: NsRange) -> Range<u64> {
    let end = r
        .location
        .checked_add(r.length)
        .unwrap_or_else(|| unrepresentable(sel, &format!("{r:?}")));
    r.location..end
}

/// The selectors of the commands [`VulkanRenderEncoder::encode`] takes by paths of their own,
/// as [`RenderCmd::selector`] names them.
const SET_VERTEX_BUFFER: &str = "setVertexBuffer:offset:atIndex:";
const DRAW_PRIMITIVES: &str = "drawPrimitives:vertexStart:vertexCount:";

/// A command on the open runtime encoder.
type Call<'c> = &'c mut dyn FnMut(&mut rt::RenderCommandEncoder) -> Result<(), rt::Error>;

/// The runtime's encoder `raw` holds, borrowed for a call of `sel`, which dies after `endEncoding`
/// as every command does ([`EncoderObject::with`]).
fn borrowed<'r>(
    raw: &'r RefCell<Option<rt::RenderCommandEncoder>>,
    sel: &'static str,
) -> RefMut<'r, rt::RenderCommandEncoder> {
    RefMut::map(raw.borrow_mut(), |e| {
        e.as_mut()
            .unwrap_or_else(|| unrepresentable(sel, "a command after endEncoding"))
    })
}

/// The runtime's `setVertexBuffer:offset:atIndex:` on `e`, as each path sends it.
#[inline]
fn set_vertex_buffer(
    e: &mut rt::RenderCommandEncoder,
    buffer: Option<&VulkanBuffer>,
    offset: u64,
    index: u32,
) -> Result<(), rt::Error> {
    #[cfg(test)]
    tests::sent(SET_VERTEX_BUFFER, || {
        let buffer = buffer.map(|b| std::ptr::from_ref(&b.raw));
        format!("{buffer:?} {offset} {index}")
    });
    e.set_vertex_buffer(buffer.map(|b| &b.raw), offset, index)
}

/// The runtime's draw of `count` vertices of `t` from `start`, `instances` times from instance
/// `base`, on `e`, as each path sends it.
#[inline]
fn draw_primitives(
    e: &mut rt::RenderCommandEncoder,
    t: rt::DrawPrimitiveType,
    (start, count): (u32, u32),
    (instances, base): (u32, u32),
) -> Result<(), rt::Error> {
    #[cfg(test)]
    tests::sent(DRAW_PRIMITIVES, || {
        format!("{t:?} {start} {count} {instances} {base}")
    });
    e.draw_primitives(t, start, count, instances, base)
}

impl VulkanRenderEncoder {
    /// `bind`, of a command: what a draw may write through what it binds or makes resident is
    /// read at the draw ([`Reach`]).
    fn bound(&self, bind: Bind<'_>) {
        // A bind into a reach that holds nothing ([`Reach::quiet`]) of objects that decide
        // nothing there leaves it as it is: `bind_one` would empty slots already empty, and
        // `used` keeps none of them. Most binds are such, of plain memory, a draw's among them,
        // and they take no lock. The check reaches the runtime only through `buffer_bind` and
        // `texture_bind`'s own calls, as the bind does; one whose objects decide something binds
        // as before.
        if self.quiet.load(Ordering::Acquire) && !quiet_bind_decides(&bind) {
            return;
        }
        self.bind_reach(bind);
    }

    /// `bind` into the reach, the rest of [`Self::bound`]: out of line, as most binds end
    /// before it.
    #[inline(never)]
    fn bind_reach(&self, bind: Bind<'_>) {
        use crate::gpu::writes::{buffer_bind, texture_bind};
        let buffer = |b: Option<&VulkanBuffer>| b.map(|b| buffer_bind(b)).unwrap_or_default();
        let texture = |t: Option<&VulkanTexture>| t.map(|t| texture_bind(t)).unwrap_or_default();
        plain(|| {
            let locked = self.drawing.lock();
            let mut drawing = locked.borrow_mut();
            let reach = &mut drawing.reach;
            match bind {
                Bind::Buffer(stage, slot, b) => reach.bind_one(stage, true, slot, buffer(b)),
                Bind::Texture(stage, slot, t) => reach.bind_one(stage, false, slot, texture(t)),
                // A list binds from `first` to the last slot a `u64` names: a slot past `u32`,
                // which the runtime cannot hold, dies at the command's selector after it
                // ([`small`]).
                Bind::Buffers(stage, first, list) => {
                    for (slot, &b) in (first..=u64::MAX).zip(list) {
                        reach.bind_one(stage, true, slot, buffer(b));
                    }
                }
                Bind::Textures(stage, first, list) => {
                    for (slot, &t) in (first..=u64::MAX).zip(list) {
                        reach.bind_one(stage, false, slot, texture(t));
                    }
                }
                Bind::Used(textures, buffers, true) => reach.used(textures, buffers),
                Bind::Used(_, _, false) => {}
            }
            self.quiet.store(reach.quiet(), Ordering::Release);
        });
    }

    /// Dies at `sel`, a draw, a tile dispatch or an execution the encoder issues, `inherits` what
    /// its commands take of the encoder, where it may write the memory of a surface whose GPU
    /// writes never reach it through a texture, one the pass binds at a slot a function writes,
    /// by its slot writes, or one made resident for writing, or through a buffer the tile function
    /// writes, which nothing measured ([`Reach::unsent`], [`crate::os::dropped_write`]), naming
    /// how ([`Whether::through`]). A vertex or fragment function's stores into a buffer on such a
    /// surface are sent, and dropped as on Metal. Before that, it dies where it may write through
    /// an object over read-only memory whose writes this platform does not answer as Metal's GPU
    /// does ([`Reach::kept`], [`crate::gpu::writes::kept_write`]); a write through a buffer over
    /// read-only memory the runtime copies is sent, and dropped as on Metal.
    /// One that issues nothing, which Metal writes nothing of, is not asked. A pass that holds
    /// nothing a draw asks of ([`Reach::quiet`]) finds nothing.
    fn issues(&self, sel: &'static str, inherits: Option<Inherits>) {
        if self.quiet.load(Ordering::Acquire) {
            return;
        }
        let (kept, written) = plain(|| {
            let locked = self.drawing.lock();
            let drawing = locked.borrow();
            let reach = &drawing.reach;
            let kept = (crate::os::TWINS_READ_ONLY)
                .then(|| reach.kept(inherits))
                .flatten();
            let written =
                (kept.is_none() && reach.unsent_dropped()).then(|| reach.unsent(inherits));
            (kept, written)
        });
        if let Some(whether) = kept {
            let what = format!("a draw's write through {}, into", whether.through());
            crate::gpu::writes::kept_write(sel, &what);
        }
        let hit = (written.iter().flatten()).find(|(w, m)| unsendable(m) && w.holds());
        if let Some((whether, m)) = hit {
            let what = format!("a draw's write through {}", whether.through());
            match m.refused.filter(|_| !m.dropped) {
                None => crate::os::dropped_write(sel, &format!("{what}, on")),
                Some(pf) => crate::gpu::ycbcr::refused_write(sel, &format!("{what}, into"), pf),
            }
        }
    }

    /// Runs `call` on the open encoder for `sel`: a refusal dies at the send, naming the
    /// runtime's error, as Metal raises at none of these commands.
    fn run(&self, sel: &'static str, call: Call<'_>) -> RaiseResult<()> {
        let mut refusal = None;
        self.sent(sel, call, &mut refusal);
        refusal.map_or(Ok(()), |e| decide(sel, Policy::Dies, e))
    }

    /// Runs `call` on the open encoder for `sel` as [`crate::answer::send`] runs it (a
    /// degradation of [`crate::answer::DIES`] announced inside it dies at `sel`), and leaves its
    /// refusal, where it has one, in `refusal`, for the caller to decide as `send` does next. A
    /// command the runtime takes moves no result out through the calls around it: the runtime's
    /// error is 112 bytes, and moving its `Result` out of each call around the runtime's took
    /// most of their time (measured on msi's RTX 5080 under `draws.10k`: 61% of the samples in
    /// `inside`, 70% in `hearing` and 82% in `send` were those moves).
    fn sent(&self, sel: &'static str, call: Call<'_>, refusal: &mut Option<rt::Error>) {
        plain_at(sel, || {
            self.encoder.with(sel, |e| {
                if let Err(e) = call(e) {
                    *refusal = Some(e);
                }
            });
        });
    }

    /// Dies at `sel`, a draw or an execution of indirect commands that is not empty, in a pass
    /// of a rate map of no layers: Metal's rasterizes it by what earlier maps left, which no call
    /// tells, where the runtime's draws nothing ([`crate::gpu::rate_map`]). In a pass of a map
    /// whose rates vary, whose regions the runtime's rasterizes by the tiles' exact lines, where
    /// Metal's are their fixed-point ones ([`Class::Varying`]), it draws, relayed, and dies in
    /// strict mode ([`crate::answer::Inexact::RateMap`]). The pass's loads, clears and stores are
    /// Metal's, so one that draws nothing runs. A tile dispatch there dies by its own line
    /// ([`Self::tiled`]).
    fn layered(&self, sel: &'static str) {
        if self.no_layers {
            unrepresentable(
                sel,
                "a draw in a pass of a rasterization rate map of no layers, which Metal \
                 rasterizes by what earlier maps left, where the runtime's draws nothing",
            );
        }
        if self.varying_rates {
            crate::answer::inexact(sel, crate::answer::Inexact::RateMap, || {
                VARYING_RATES_DRAW.into()
            });
        }
    }

    /// A draw `sel` of `shape` with the pipeline state set: one after a nil state dies, as Metal
    /// crashes there ([`Self::pipelined`]), and so does one in a pass that loads and stores a
    /// bfloat colour attachment, empty or not ([`Self::bfloat_drawn`]). So do one of points
    /// Metal sizes none of ([`sized()`]) and one the private setters leave no exact form of
    /// ([`Self::marked`]), unless it is `empty`, of no primitive, patch or instance, which
    /// Metal draws nothing of whatever they left; `None` for one Metal completes without
    /// drawing, which encodes nothing.
    ///
    /// What the draw asks of the encoder's state ([`Held`]) is read once, at its start: none of
    /// its checks changes it.
    fn draw<K: Copy>(
        &self,
        sel: &'static str,
        (drawn, empty): (Option<(Shape, K)>, bool),
        mut call: impl FnMut(&mut rt::RenderCommandEncoder, K) -> Result<(), rt::Error>,
    ) -> RaiseResult<()> {
        // Reading the encoder's state reaches no runtime, so nothing the runtime announces can
        // fall inside it.
        let held = self.held();
        nil_pipeline_drawn(sel, held.nil_pipeline);
        crate::gpu::timing::count_draw();
        match drawn {
            Some((shape, k)) => {
                self.drawn(sel, shape, empty, &held);
                if !empty {
                    marked_by(sel, held.marks);
                }
                self.binds(sel, &mut |e| call(e, k))
            }
            None => self.encoder.nothing(sel),
        }
    }

    /// `setVertexBuffer:offset:atIndex:`, as [`Self::encode`] runs a bind: what the draws may
    /// reach through the buffer first ([`Self::bound`]), then the command.
    fn vertex_buffer(
        &self,
        buffer: Option<&VulkanBuffer>,
        offset: u64,
        index: u64,
    ) -> RaiseResult<()> {
        let sel = SET_VERTEX_BUFFER;
        if self.encoder.has_command_buffer() {
            self.bound(Bind::Buffer(VERTEX, index, buffer));
        }
        let index = small(sel, index);
        self.run(sel, &mut |e| set_vertex_buffer(e, buffer, offset, index))
    }

    /// What a draw reads of the encoder's state ([`Held`]).
    fn held(&self) -> Held {
        let (nil_pipeline, marks) = {
            let locked = self.drawing.lock();
            let drawing = locked.borrow();
            (drawing.nil_pipeline, drawing.marks)
        };
        Held {
            nil_pipeline,
            marks,
            encodes: self.encoder.encodes_icbs.load(Ordering::SeqCst),
        }
    }

    /// What a draw `sel` of `shape` asks of what the encoder holds, `held`, after its pipeline
    /// state ([`Self::pipelined`]): in a pass that loads and stores a bfloat colour attachment,
    /// empty or not ([`Self::bfloat_drawn`]), and unless it is `empty`, of no primitive, patch,
    /// instance or threadgroup, the pass's rate map ([`Self::layered`]), its points ([`sized`]),
    /// a mesh draw's threadgroups ([`threads_held`]), what it may write ([`Self::issues`]) and
    /// the indirect commands its state may encode ([`Self::encodes`]). A draw over indices of the
    /// platform's own asks it too ([`Self::primitives`]).
    fn drawn(&self, sel: &'static str, shape: Shape, empty: bool, held: &Held) {
        self.bfloat_drawn(sel);
        if !empty {
            self.layered(sel);
            sized(sel, shape, held.marks);
            threads_held(sel, shape, held.marks);
            self.issues(sel, None);
            if held.encodes {
                self.icbs_written(sel);
            }
        }
    }

    /// Runs `call`, a draw `sel` that binds the pipeline state set, which the runtime announces
    /// a `half` colour output of the state's fragment function at (`HalfOutputPrecision`): the
    /// draw relays it, or in strict mode dies, where an attachment the function writes stores it
    /// past the bound ([`half_apart`]). An execution's commands announce it at the commit,
    /// outside the call.
    fn binds(&self, sel: &'static str, call: Call<'_>) -> RaiseResult<()> {
        let mut refusal = None;
        let ((), heard) = crate::answer::hearing(|| self.sent(sel, call, &mut refusal));
        let drawn = refusal.map_or(Ok(()), |e| decide(sel, Policy::Dies, e));
        if heard.contains(&rt::Degradation::HalfOutputPrecision)
            && let Some(apart) = plain(|| self.drawing.lock().borrow().marks.half_apart)
        {
            half_output_apart(sel, apart);
        }
        drawn
    }

    /// Relays at `sel`, or dies there in strict mode ([`crate::answer::inexact`]), a draw (of
    /// anything, or of nothing), an execution of render commands or a tile dispatch, in a pass
    /// that loads and stores a bfloat colour attachment ([`bfloat_loaded`]): the tiles the pass
    /// leaves of it are not Metal's.
    fn bfloat_drawn(&self, sel: &'static str) {
        if let Some(attachment) = &self.bfloat_loaded {
            crate::answer::inexact(sel, crate::answer::Inexact::BfloatLoaded, || {
                format!(
                    "a draw in a pass that loads and stores the bfloat {attachment}, whose codes \
                     the pass keeps where its draws do not write them (extended-range-bits-kept), \
                     where Metal's tile store moves those of the tiles a draw reaches through half"
                )
            });
        }
    }

    /// Dies at `sel`, which runs the encoder's pipeline state, after `setRenderPipelineState:`
    /// of nil ([`nil_pipeline_drawn`]).
    fn pipelined(&self, sel: &'static str) {
        nil_pipeline_drawn(sel, plain(|| self.drawing.lock().borrow().nil_pipeline));
    }

    /// A draw `sel` of primitive type 9 ([`rt::DrawPrimitiveType::Nine`]), which goes to the
    /// runtime: Metal's GPU draws nothing of it, and `fails` its command buffer from its render
    /// encoder on, its event signals firing, as the runtime fails it: directly, of 3 vertices or
    /// more and an instance, with `MTLCommandBufferErrorTimeout` (`GpuHang`), of fewer vertices or
    /// no instance with nothing (`None`); over indices with `MTLCommandBufferErrorPageFault`
    /// (`PageFault`), of any indices or instances; from indirect arguments with
    /// `MTLCommandBufferErrorInternal` (`GpuRecoveryVictim`) (measured on the M3 Max, macOS
    /// 27.0.1, `type_9_draws_fail_as_metals`). One that fails it tells the command buffer, whose
    /// commit dies where it fails another way too
    /// ([`crate::gpu::command::CommandBufferObject::fails_at_type_9`]), and asks the pass's rate
    /// map as a draw does ([`Self::layered`]): Metal's failure was measured in no pass over a rate
    /// map. One that fails nothing is a draw the runtime records nothing of. After
    /// `setRenderPipelineState:` of nil it dies as every draw does ([`Self::pipelined`]), and with
    /// no pipeline state set, which nothing measured, it dies.
    fn nine(
        &self,
        sel: &'static str,
        fails: Option<rt::CommandBufferError>,
        call: Call<'_>,
    ) -> RaiseResult<()> {
        self.pipelined(sel);
        if !plain(|| self.drawing.lock().borrow().marks.pipeline) {
            unrepresentable(
                sel,
                "primitive type 9 with no pipeline state set, which nothing measured",
            );
        }
        crate::gpu::timing::count_draw();
        if let Some(error) = fails {
            self.layered(sel);
            self.encoder.fails_at_type_9(error.metal_code());
        }
        self.run(sel, call)
    }

    /// An indirect command buffer's execution `sel`, of `icb`'s commands `range` (`None`: a range
    /// a buffer holds): it draws, so it dies as a draw does where the private setters leave it no
    /// exact form ([`Self::marked`]), unless it is `empty`, of no command, and after a nil
    /// pipeline state where its commands inherit the encoder's ([`Self::pipelined`]). One that is
    /// not empty dies where it may write a surface whose GPU writes never reach its memory
    /// ([`Self::issues`]). Where its commands inherit the encoder's state, that state decides
    /// what a draw's would ([`Self::encodes`]); its commit reads its commands where they name
    /// their own, or inherit a state whose points Metal sizes none of
    /// ([`crate::gpu::command::executions`]).
    fn executable(
        &self,
        sel: &'static str,
        (icb, range): (&VulkanIcb, Option<Range<u64>>),
        empty: bool,
    ) -> Option<Execution> {
        if icb.inherits.pipeline {
            self.pipelined(sel);
        }
        self.bfloat_drawn(sel);
        if !empty {
            self.layered(sel);
            self.issues(sel, Some(icb.inherits));
        }
        if empty {
            return None;
        }
        self.marked(sel);
        let inherits_pipeline = icb.inherits.pipeline;
        let execution = if inherits_pipeline {
            self.encodes(sel);
            let m = plain(|| self.drawing.lock().borrow().marks);
            Execution {
                sel,
                icb: icb.clone_ref(),
                range,
                points: if m.pipeline && !m.points_sized {
                    Points::Unsized
                } else {
                    Points::Unasked
                },
                writable: Vec::new(),
            }
        } else {
            Execution {
                sel,
                icb: icb.clone_ref(),
                range,
                points: Points::Own,
                writable: plain(|| {
                    let writable = self.encoder.writable_icbs.lock();
                    writable.iter().map(|icb| icb.clone_ref()).collect()
                }),
            }
        };
        execution.asks().then_some(execution)
    }

    /// `execution`, which the runtime took, is checked at its command buffer's commit
    /// ([`crate::gpu::command::CommandBufferObject::checked_at_commit`]).
    fn checked_at_commit(&self, execution: Option<Execution>) {
        if let Some(execution) = execution
            && let Some(cb) = self.encoder.command_buffer()
        {
            cb.checked_at_commit(execution);
        }
    }

    /// Work `sel` of the pipeline state the encoder set last: where that state can encode indirect
    /// commands, its functions may write the records of each buffer the encoder made resident for
    /// writing ([`crate::gpu::command::executions::gpu_writes`]).
    fn encodes(&self, sel: &'static str) {
        if plain(|| self.encoder.encodes_icbs.load(Ordering::SeqCst)) {
            self.icbs_written(sel);
        }
    }

    /// [`Self::encodes`] of a state that can encode indirect commands.
    fn icbs_written(&self, sel: &'static str) {
        let writable: Vec<VulkanIcb> = plain(|| {
            let writable = self.encoder.writable_icbs.lock();
            writable.iter().map(|icb| icb.clone_ref()).collect()
        });
        for icb in &writable {
            crate::gpu::command::executions::gpu_writes(sel, icb);
        }
    }

    /// `useResource` forms made `icbs` resident for writing: the encoder's functions whose state
    /// can encode indirect commands may write their records ([`Self::encodes`]).
    fn writable(&self, icbs: &[VulkanIcb]) {
        plain(|| {
            let mut writable = self.encoder.writable_icbs.lock();
            writable.extend(icbs.iter().map(|icb| icb.clone_ref()));
        });
    }

    /// A tile dispatch `sel`: one after a nil pipeline state dies ([`Self::pipelined`]). One in a
    /// pass of a rate map of no layers, where what Metal's does is not measured (only its draws
    /// were, [`Self::layered`]), and one in a pass of a map whose rates vary, where it is not
    /// measured either ([`Class::Varying`]), run, relayed, and die in strict mode
    /// ([`crate::answer::Inexact::RateMap`]). One in a pass that rasterizes nothing, whose
    /// empty scissor rectangles the runtime's tile dispatch does not read, so its tile function
    /// would run over every tile, where what Metal's does there is not measured either.
    fn tiled(&self, sel: &'static str) {
        self.pipelined(sel);
        self.bfloat_drawn(sel);
        self.encodes(sel);
        if self.no_layers {
            crate::answer::inexact(sel, crate::answer::Inexact::RateMap, || {
                "a tile dispatch in a pass of a rasterization rate map of no layers, which Metal \
                 is not measured at (only its draws were)"
                    .into()
            });
        }
        if self.varying_rates {
            crate::answer::inexact(sel, crate::answer::Inexact::RateMap, || {
                VARYING_RATES_TILES.into()
            });
        }
        self.issues(sel, None);
        if !self.rasterizes {
            unrepresentable(
                sel,
                "a tile dispatch in a pass of a mutable rate map not yet reset, which Metal is \
                 not measured at",
            );
        }
    }

    /// Dies at a draw or execution `sel` after `setVertexAmplificationMode:value:` left a count
    /// the runtime does not draw, or one not measured, in either mode ([`Marks`]).
    fn marked(&self, sel: &'static str) {
        marked_by(sel, plain(|| self.drawing.lock().borrow().marks));
    }
}

/// What a draw reads of its encoder's state, once ([`VulkanRenderEncoder::draw`]).
#[derive(Debug, Clone, Copy)]
struct Held {
    /// `setRenderPipelineState:` of nil since the last state set ([`nil_pipeline_drawn`]).
    nil_pipeline: bool,
    /// What the private setters and the pipeline state left ([`Marks`]).
    marks: Marks,
    /// Whether the pipeline state set can encode indirect commands
    /// ([`VulkanRenderEncoder::encodes`]).
    encodes: bool,
}

/// Dies at `sel`, which runs the encoder's pipeline state, after `setRenderPipelineState:` of nil
/// (`nil_pipeline`): Metal crashes at a draw with no pipeline state, where the runtime would run
/// the state set before the nil (a draw, direct or indirect, a tile dispatch, and an indirect
/// command buffer's execution whose commands inherit the encoder's state).
fn nil_pipeline_drawn(sel: &'static str, nil_pipeline: bool) {
    if nil_pipeline {
        unrepresentable(
            sel,
            "a draw after a nil pipeline state, which Metal crashes at",
        );
    }
}

/// Dies at a draw `sel` of points, `shape`, under a pipeline state whose points Metal sizes none
/// of, by the encoder's marks `m`: its vertex function writes no `[[point_size]]`, and its input
/// primitive topology is not the point class (the module docs).
fn sized(sel: &'static str, shape: Shape, m: Marks) {
    let points = matches!(
        shape,
        Shape::Primitives(rt::DrawPrimitiveType::Public(PrimitiveType::Point))
    );
    if points && m.pipeline && !m.points_sized {
        unrepresentable(sel, UNSIZED_POINTS);
    }
}

/// Dies at a mesh draw `sel` of `shape` whose threadgroups the mesh render pipeline state set
/// does not hold to, by the encoder's marks `m` ([`MeshAnswers::held`]): a threadgroup of more
/// threads than the maximum the state's descriptor set for its stage (Metal's validation stops
/// it: "threadsPerMeshThreadgroup (1025 x 1 x 1 = 1025) cannot exceed
/// maxTotalThreadsPerMeshThreadgroup (1024)", the runtime's measurement), or of another size
/// than the one the descriptor requires, which plain Metal's draws of are unmeasured, and the
/// runtime, whose descriptor holds neither, would draw as within them. The object stage's are
/// asked only of a state with an object function: Metal takes the draw's threads per object
/// threadgroup only then. A threadgroup of no thread draws nothing, as does one past Metal's
/// 1024, which the runtime draws nothing of as plain Metal does (`render/mesh.rs`): neither is
/// asked.
fn threads_held(sel: &'static str, shape: Shape, m: Marks) {
    let (Shape::Mesh { object, mesh }, Some(answers)) = (shape, m.mesh) else {
        return;
    };
    let stages = [
        ("object", object, answers.held[0]),
        ("mesh", mesh, answers.held[1]),
    ];
    for (stage, threads, held) in stages {
        if stage == "object" && !answers.object {
            continue;
        }
        let total = (threads.width)
            .saturating_mul(threads.height)
            .saturating_mul(threads.depth);
        if total == 0 || total > METAL_MESH_THREADGROUP {
            continue;
        }
        let MtlSize {
            width,
            height,
            depth,
        } = threads;
        if held.max != 0 && total > held.max {
            unrepresentable(
                sel,
                &format!(
                    "a {stage} threadgroup of {width} x {height} x {depth} threads, past the \
                     {} its pipeline state's descriptor sets as its maximum, which Metal's \
                     validation stops and plain Metal's draw of is unmeasured",
                    held.max
                ),
            );
        }
        if let Some(r) = held.required.filter(|r| *r != threads) {
            unrepresentable(
                sel,
                &format!(
                    "a {stage} threadgroup of {width} x {height} x {depth} threads, where its \
                     pipeline state's descriptor requires {} x {} x {}, which plain Metal's draw \
                     of is unmeasured",
                    r.width, r.height, r.depth
                ),
            );
        }
    }
}

/// Metal's most threads of an object or mesh threadgroup: past it plain Metal draws nothing, as
/// the runtime does (`render/mesh.rs`, measured).
const METAL_MESH_THREADGROUP: u64 = 1024;

/// Whether a mesh draw's grid, of threadgroups or of threads, `grid`, has an axis of none, which
/// draws nothing on Metal, and on the runtime, with any state (`render/mesh.rs`, measured).
fn empty_grid(grid: MtlSize) -> bool {
    grid.width == 0 || grid.height == 0 || grid.depth == 0
}

/// [`VulkanRenderEncoder::marked`], by the encoder's marks `m`.
fn marked_by(sel: &'static str, m: Marks) {
    if m.amplified {
        unrepresentable(
            sel,
            "a draw after setVertexAmplificationMode:value: left an amplification count the \
             runtime does not draw",
        );
    }
}

impl RenderEncoder<VulkanPlatform> for VulkanRenderEncoder {
    #[expect(clippy::too_many_lines, reason = "one arm per command")]
    fn encode(&self, _on: OnFault, cmd: RenderCmd<'_, VulkanPlatform>) -> RaiseResult<()> {
        // The two commands of nearly every draw take paths of their own, which run what the
        // arms below run for them, in their order, without the dispatch on every command first.
        match cmd {
            RenderCmd::SetVertexBuffer {
                buffer,
                offset,
                index,
            } => return self.vertex_buffer(buffer, offset, index),
            RenderCmd::DrawPrimitives {
                primitive_type,
                vertex_start,
                vertex_count,
            } => {
                let sel = DRAW_PRIMITIVES;
                return self.primitives(sel, primitive_type, (vertex_start, vertex_count, 1, 0));
            }
            _ => {}
        }
        let sel = cmd.selector();
        let encoded = |call: Call<'_>| self.run(sel, call);
        if let Some(bind) = binds(&cmd)
            && self.encoder.has_command_buffer()
        {
            self.bound(bind);
        }
        match cmd {
            RenderCmd::DispatchThreadsPerTile { threads_per_tile } => {
                self.tiled(sel);
                encoded(&mut |e| e.dispatch_threads_per_tile(size(threads_per_tile)))
            }
            RenderCmd::DispatchThreadsPerTileInRegion {
                threads_per_tile,
                region: r,
            } => {
                self.tiled(sel);
                encoded(&mut |e| {
                    e.dispatch_threads_per_tile_in_region(size(threads_per_tile), region(r), None)
                })
            }
            RenderCmd::DispatchThreadsPerTileInRegionIndex {
                threads_per_tile,
                region: r,
                render_target_array_index,
            } => {
                self.tiled(sel);
                encoded(&mut |e| {
                    e.dispatch_threads_per_tile_in_region(
                        size(threads_per_tile),
                        region(r),
                        Some(render_target_array_index),
                    )
                })
            }
            RenderCmd::DrawIndexedPatchesIndirect {
                number_of_patch_control_points,
                patch_index_buffer,
                patch_index_buffer_offset,
                control_point_index_buffer,
                control_point_index_buffer_offset,
                indirect_buffer,
                indirect_buffer_offset,
            } => {
                let points = small(sel, number_of_patch_control_points);
                let shape = Shape::Patches;
                self.draw(sel, (Some((shape, ())), false), |e, ()| {
                    e.draw_indexed_patches_indirect(
                        points,
                        patch_index_buffer.map(|b| &b.raw),
                        patch_index_buffer_offset,
                        &control_point_index_buffer.raw,
                        control_point_index_buffer_offset,
                        &indirect_buffer.raw,
                        indirect_buffer_offset,
                    )
                })
            }
            RenderCmd::DrawIndexedPatches {
                number_of_patch_control_points,
                patch_start,
                patch_count,
                patch_index_buffer,
                patch_index_buffer_offset,
                control_point_index_buffer,
                control_point_index_buffer_offset,
                instance_count,
                base_instance,
            } => {
                let points = small(sel, number_of_patch_control_points);
                let (start, count) = (small(sel, patch_start), small(sel, patch_count));
                let (instances, base) = (small(sel, instance_count), small(sel, base_instance));
                let shape = Shape::Patches;
                let empty = count == 0 || instances == 0;
                self.draw(sel, (Some((shape, ())), empty), |e, ()| {
                    e.draw_indexed_patches(
                        points,
                        start,
                        count,
                        patch_index_buffer.map(|b| &b.raw),
                        patch_index_buffer_offset,
                        &control_point_index_buffer.raw,
                        control_point_index_buffer_offset,
                        instances,
                        base,
                    )
                })
            }
            RenderCmd::DrawIndexedPrimitives {
                primitive_type,
                index_count,
                index_type: raw_index_type,
                index_buffer,
                index_buffer_offset,
            } => self.indexed(
                sel,
                (primitive_type, raw_index_type),
                (index_count, 1, 0, 0),
                (index_buffer, index_buffer_offset),
            ),
            RenderCmd::DrawIndexedPrimitivesInstances {
                primitive_type,
                index_count,
                index_type: raw_index_type,
                index_buffer,
                index_buffer_offset,
                instance_count,
            } => self.indexed(
                sel,
                (primitive_type, raw_index_type),
                (index_count, instance_count, 0, 0),
                (index_buffer, index_buffer_offset),
            ),
            RenderCmd::DrawIndexedPrimitivesBase {
                primitive_type,
                index_count,
                index_type: raw_index_type,
                index_buffer,
                index_buffer_offset,
                instance_count,
                base_vertex,
                base_instance,
            } => self.indexed(
                sel,
                (primitive_type, raw_index_type),
                (index_count, instance_count, base_vertex, base_instance),
                (index_buffer, index_buffer_offset),
            ),
            RenderCmd::DrawIndexedPrimitivesIndirect {
                primitive_type,
                index_type: raw_index_type,
                index_buffer,
                index_buffer_offset,
                indirect_buffer,
                indirect_buffer_offset,
            } => {
                let kinds = indexed_kinds(sel, primitive_type, raw_index_type);
                let call = |e: &mut rt::RenderCommandEncoder, (t, i)| {
                    e.draw_indexed_primitives_indirect(
                        t,
                        i,
                        &index_buffer.raw,
                        index_buffer_offset,
                        &indirect_buffer.raw,
                        indirect_buffer_offset,
                    )
                };
                if let Some((t @ rt::DrawPrimitiveType::Nine, i)) = kinds {
                    let victim = Some(rt::CommandBufferError::GpuRecoveryVictim);
                    return self.nine(sel, victim, &mut |e| call(e, (t, i)));
                }
                let drawn = kinds.map(|(t, i)| (Shape::Primitives(t), (t, i)));
                self.draw(sel, (drawn, false), call)
            }
            RenderCmd::DrawPatchesIndirect {
                number_of_patch_control_points,
                patch_index_buffer,
                patch_index_buffer_offset,
                indirect_buffer,
                indirect_buffer_offset,
            } => {
                let points = small(sel, number_of_patch_control_points);
                let shape = Shape::Patches;
                self.draw(sel, (Some((shape, ())), false), |e, ()| {
                    e.draw_patches_indirect(
                        points,
                        patch_index_buffer.map(|b| &b.raw),
                        patch_index_buffer_offset,
                        &indirect_buffer.raw,
                        indirect_buffer_offset,
                    )
                })
            }
            RenderCmd::DrawPatches {
                number_of_patch_control_points,
                patch_start,
                patch_count,
                patch_index_buffer,
                patch_index_buffer_offset,
                instance_count,
                base_instance,
            } => {
                let points = small(sel, number_of_patch_control_points);
                let (start, count) = (small(sel, patch_start), small(sel, patch_count));
                let (instances, base) = (small(sel, instance_count), small(sel, base_instance));
                let shape = Shape::Patches;
                let empty = count == 0 || instances == 0;
                self.draw(sel, (Some((shape, ())), empty), |e, ()| {
                    e.draw_patches(
                        points,
                        start,
                        count,
                        patch_index_buffer.map(|b| &b.raw),
                        patch_index_buffer_offset,
                        instances,
                        base,
                    )
                })
            }
            RenderCmd::DrawPrimitivesIndirect {
                primitive_type,
                indirect_buffer,
                indirect_buffer_offset,
            } => {
                let t = draw_type(sel, primitive_type);
                let call = |e: &mut rt::RenderCommandEncoder, t| {
                    e.draw_primitives_indirect(t, &indirect_buffer.raw, indirect_buffer_offset)
                };
                if t == rt::DrawPrimitiveType::Nine {
                    let victim = Some(rt::CommandBufferError::GpuRecoveryVictim);
                    return self.nine(sel, victim, &mut |e| call(e, t));
                }
                self.draw(sel, (Some((Shape::Primitives(t), t)), false), call)
            }
            RenderCmd::SetVertexBuffer { .. } | RenderCmd::DrawPrimitives { .. } => {
                unreachable!("{sel} takes its own path above")
            }
            RenderCmd::DrawPrimitivesInstances {
                primitive_type,
                vertex_start,
                vertex_count,
                instance_count,
            } => self.primitives(
                sel,
                primitive_type,
                (vertex_start, vertex_count, instance_count, 0),
            ),
            RenderCmd::DrawPrimitivesBase {
                primitive_type,
                vertex_start,
                vertex_count,
                instance_count,
                base_instance,
            } => self.primitives(
                sel,
                primitive_type,
                (vertex_start, vertex_count, instance_count, base_instance),
            ),
            RenderCmd::EndEncoding => {
                // A store action left `Unknown` stores nothing and resolves nothing, as on Metal
                // (`render_pass_unknown_store`; `mutgap2_rdec_depth_stencil_resolve`: a depth
                // left `Unknown` over a resolve texture resolves nothing), as the runtime's end
                // settles it (`unknown_stores_over_resolves_resolve_nothing`).
                // The runtime decides the pass's resolves as it ends, and announces a colour
                // resolve the device averages and rounds there (`DeviceRoundedResolve`), which a
                // native device relays ([`crate::answer::native::NATIVE_RELAYED`]).
                // A pass over a mutable map that no draw of the guest's took the place of ends
                // with a draw of no vertices, which keeps the runtime from failing it as a GPU
                // hang where nothing drew in it, as the Metal platform's draw keeps Metal's GPU
                // from hanging (`rate_map::no_draw`). It is the pass's last command, so its
                // pipeline is bound for nothing the guest encodes. Its pipeline is made here,
                // once per device; where the command buffer is gone, nothing encoded here runs,
                // and the pass needs no draw.
                if self.undrawn.load(Ordering::Acquire)
                    && let Some(cb) = self.encoder.command_buffer()
                {
                    let no_draw = crate::gpu::rate_map::no_draw(sel, &cb.device);
                    self.run(sel, &mut |e| {
                        e.set_render_pipeline_state(&no_draw)?;
                        e.draw_primitives(PrimitiveType::Triangle, 0, 0, 1, 0)
                    })?;
                }
                self.encoder.end(sel, |e| e.end_encoding())
            }
            RenderCmd::ExecuteCommandsIndirect {
                indirect_commandbuffer,
                indirect_range_buffer,
                indirect_buffer_offset,
            } => {
                let checked = self.executable(sel, (indirect_commandbuffer, None), false);
                let mut check = checked.as_ref().and_then(Execution::host_step_check);
                let progress = self.encoder.progress(sel);
                indirect_commandbuffer.executed(sel, None, progress.as_ref());
                encoded(&mut |e| {
                    let (icb, buffer) = (&indirect_commandbuffer.raw, &indirect_range_buffer.raw);
                    match check.take() {
                        Some((check, _)) => e.execute_commands_in_buffer_indirect_checked(
                            icb,
                            buffer,
                            indirect_buffer_offset,
                            check,
                            rt::IndirectCommandCheckNeeds::ALL,
                        ),
                        None => e.execute_commands_in_buffer_indirect(
                            icb,
                            buffer,
                            indirect_buffer_offset,
                        ),
                    }
                })?;
                self.checked_at_commit(checked);
                if let Some(progress) = progress {
                    progress.executes(sel, indirect_commandbuffer.clone_ref());
                }
                Ok(())
            }
            RenderCmd::ExecuteCommandsInBuffer {
                indirect_command_buffer,
                execution_range,
            } => {
                let range = range_of(sel, execution_range);
                let checked = self.executable(
                    sel,
                    (indirect_command_buffer, Some(range.clone())),
                    range.is_empty(),
                );
                let mut check = checked.as_ref().and_then(Execution::host_step_check);
                let progress = self.encoder.progress(sel);
                indirect_command_buffer.executed(sel, Some(range.clone()), progress.as_ref());
                // An empty range runs nothing, as Metal's does under any pipeline state
                // (`render_draw_icb`), as the runtime runs it.
                encoded(&mut |e| {
                    let icb = &indirect_command_buffer.raw;
                    match check.take() {
                        Some((check, needs)) => {
                            e.execute_commands_in_buffer_checked(icb, range.clone(), check, needs)
                        }
                        None => e.execute_commands_in_buffer(icb, range.clone()),
                    }
                })?;
                self.checked_at_commit(checked);
                if let Some(progress) = progress {
                    progress.executes(sel, indirect_command_buffer.clone_ref());
                }
                Ok(())
            }
            RenderCmd::MemoryBarrierWithScope {
                scope,
                after,
                before,
            } => {
                let scope: BarrierScope = metal_value(sel, "barrier scope", scope);
                let after: RenderStages = metal_value(sel, "render stages", after);
                let before: RenderStages = metal_value(sel, "render stages", before);
                // A barrier after the fragment stage and before the vertex stage orders nothing
                // on this GPU, which runs a pass's vertex stages first: it does not split the
                // pass (`mutgap_dec_render_fragment_vertex_barrier`: the later draw's vertices
                // see none of the earlier draw's fragment writes), where the runtime splits it.
                if after == RenderStages::FRAGMENT && before == RenderStages::VERTEX {
                    return self.encoder.nothing(sel);
                }
                encoded(&mut |e| {
                    e.memory_barrier_with_scope(scope, after, before);
                    Ok(())
                })
            }
            RenderCmd::SetAlphaTestReferenceValue { reference_value } => {
                encoded(&mut |e| e.set_alpha_test_reference_value(reference_value))
            }
            RenderCmd::SetBlendColorRed {
                red,
                green,
                blue,
                alpha,
            } => encoded(&mut |e| {
                e.set_blend_color(red, green, blue, alpha);
                Ok(())
            }),
            RenderCmd::SetColorResolveTexture {
                texture,
                slice,
                depth_plane,
                level,
                y_invert: _,
                index,
            } => {
                let at = usize::try_from(index)
                    .ok()
                    .filter(|&i| i < rt::MAX_COLOR_ATTACHMENTS)
                    .unwrap_or_else(|| unrepresentable(sel, &format!("colour attachment {index}")));
                let index = small(sel, index);
                if let Some(d) = self.dropped_resolve(index) {
                    // Relayed, it stays dropped: the texture is left untouched, as plain Metal
                    // leaves the descriptor's.
                    crate::answer::inexact(sel, crate::answer::Inexact::LateResolve, || {
                        format!(
                            "a late resolve texture of colour attachment {index}, whose resolve \
                             by {:?} into {} the pass was made without, as plain Metal leaves \
                             that texture untouched; a late resolve texture there is not measured",
                            d.store,
                            d.into.name()
                        )
                    });
                    return self.encoder.nothing(sel);
                }
                if self.pass.color_attachments[at].resolve_texture.is_none()
                    || self.resolve_unused(Attachment::Color(index))
                {
                    return self.encoder.nothing(sel);
                }
                let at = (slice, depth_plane, level);
                let target = self.late_resolve(sel, Attachment::Color(index), texture, at);
                encoded(&mut |e| e.set_color_resolve_target(target.clone(), index))
            }
            RenderCmd::SetColorStoreAction {
                store_action: raw,
                color_attachment_index,
            } => {
                let (action, index) = (store_action(sel, raw), small(sel, color_attachment_index));
                self.late_store(sel, Attachment::Color(index), action, &mut |e| {
                    e.set_color_store_action(action, index)
                })
            }
            RenderCmd::SetColorStoreActionOptions {
                store_action_options: raw,
                color_attachment_index,
            } => {
                let options = store_action_options(sel, raw);
                let index = small(sel, color_attachment_index);
                self.late_options(sel, Attachment::Color(index), options, &mut |e| {
                    e.set_color_store_action_options(options, index)
                })
            }
            RenderCmd::SetCullMode { cull_mode } => {
                let mode: CullMode = metal_value(sel, "cull mode", cull_mode);
                encoded(&mut |e| {
                    e.set_cull_mode(mode);
                    Ok(())
                })
            }
            RenderCmd::SetDepthBias {
                depth_bias,
                slope_scale,
                clamp,
            } => encoded(&mut |e| e.set_depth_bias(depth_bias, slope_scale, clamp)),
            // The cleared hints change nothing drawn on this host, over loaded values too
            // (`mutgap_dec_render_depth_stencil_cleared`): they encode nothing.
            RenderCmd::SetDepthCleared | RenderCmd::SetStencilCleared => self.encoder.nothing(sel),
            RenderCmd::SetDepthClipModeSpi { mode } => {
                let mode: DepthClipMode = metal_value(sel, "depth clip mode", mode);
                encoded(&mut |e| e.set_depth_clip_mode(mode))
            }
            RenderCmd::SetDepthResolveTexture {
                texture,
                slice,
                depth_plane,
                level,
                y_invert: _,
            } => {
                if self.resolve_unused(Attachment::Depth) {
                    return self.encoder.nothing(sel);
                }
                if texture.is_some() {
                    self.resolves_beside_dropped(sel, Attachment::Depth);
                }
                let at = (slice, depth_plane, level);
                let target = self.late_resolve(sel, Attachment::Depth, texture, at);
                encoded(&mut |e| e.set_depth_resolve_target(target.clone()))
            }
            RenderCmd::SetDepthStencilState {
                depth_stencil_state,
            } => encoded(&mut |e| e.set_depth_stencil_state(depth_stencil_state.map(|s| &**s))),
            RenderCmd::SetDepthStoreAction { store_action: raw } => {
                let action = store_action(sel, raw);
                self.late_store(sel, Attachment::Depth, action, &mut |e| {
                    e.set_depth_store_action(action)
                })
            }
            RenderCmd::SetFragmentBuffer {
                buffer,
                offset,
                index,
            } => {
                let index = small(sel, index);
                self.run(sel, &mut |e| {
                    e.set_fragment_buffer(buffer.map(|b| &b.raw), offset, index)
                })
            }
            RenderCmd::SetFragmentBufferOffset { offset, index } => {
                let index = small(sel, index);
                encoded(&mut |e| e.set_fragment_buffer_offset(offset, index))
            }
            RenderCmd::SetFragmentBuffers {
                buffers: list,
                offsets,
                range,
            } => {
                let (at, list) = ranged(sel, range, list);
                let (_, offsets) = ranged(sel, range, offsets);
                buffers(list, |list| {
                    encoded(&mut |e| e.set_fragment_buffers(list, offsets, at))
                })
            }
            RenderCmd::SetFragmentSamplerStatesLodClamps {
                samplers: list,
                lod_min_clamps,
                lod_max_clamps,
                range,
            } => {
                let (at, list) = ranged(sel, range, list);
                let (_, min) = ranged(sel, range, lod_min_clamps);
                let (_, max) = ranged(sel, range, lod_max_clamps);
                samplers(list, |list| {
                    encoded(&mut |e| e.set_fragment_sampler_states(list, Some((min, max)), at))
                })
            }
            RenderCmd::SetFragmentSamplerStates {
                samplers: list,
                range,
            } => {
                let (at, list) = ranged(sel, range, list);
                samplers(list, |list| {
                    encoded(&mut |e| e.set_fragment_sampler_states(list, None, at))
                })
            }
            RenderCmd::SetFragmentTexture { texture, index } => {
                let index = small(sel, index);
                let texture = texture.map(|t| t.texels(sel));
                self.run(sel, &mut |e| e.set_fragment_texture(texture, index))
            }
            RenderCmd::SetFragmentTextures {
                textures: list,
                range,
            } => {
                let (at, list) = ranged(sel, range, list);
                textures(sel, list, |list| {
                    encoded(&mut |e| e.set_fragment_textures(list, at))
                })
            }
            RenderCmd::SetFrontFacingWinding {
                front_facing_winding,
            } => {
                let winding: Winding = metal_value(sel, "winding", front_facing_winding);
                encoded(&mut |e| {
                    e.set_front_facing_winding(winding);
                    Ok(())
                })
            }
            // The runtime's encoder draws the width as Metal does (the module docs).
            RenderCmd::SetLineWidth { line_width } => {
                encoded(&mut |e| e.set_line_width(line_width))
            }
            // Metal draws alike at every point size on this host, and a vertex function's
            // `[[point_size]]` decides a point's: it encodes nothing.
            RenderCmd::SetPointSize { .. } => self.encoder.nothing(sel),
            // A mesh draw draws as a draw of primitives does ([`Self::draw`]), its threadgroups
            // held to its state's ([`threads_held`]); one of a grid with an axis of none draws
            // nothing, as on Metal. The runtime draws it as Metal does, or refuses it, which dies
            // (`render/mesh.rs`): with a state that is not a mesh render pipeline's, among them,
            // which stops Metal.
            RenderCmd::DrawMeshThreadgroups {
                threadgroups_per_grid,
                threads_per_object_threadgroup: object,
                threads_per_mesh_threadgroup: mesh,
            } => {
                let shape = Shape::Mesh { object, mesh };
                let empty = empty_grid(threadgroups_per_grid);
                let grid = size(threadgroups_per_grid);
                self.draw(sel, (Some((shape, ())), empty), |e, ()| {
                    e.draw_mesh_threadgroups(grid, size(object), size(mesh))
                })
            }
            RenderCmd::DrawMeshThreads {
                threads_per_grid,
                threads_per_object_threadgroup: object,
                threads_per_mesh_threadgroup: mesh,
            } => {
                let shape = Shape::Mesh { object, mesh };
                let empty = empty_grid(threads_per_grid);
                let grid = size(threads_per_grid);
                self.draw(sel, (Some((shape, ())), empty), |e, ()| {
                    e.draw_mesh_threads(grid, size(object), size(mesh))
                })
            }
            RenderCmd::DrawMeshThreadgroupsIndirect {
                indirect_buffer,
                indirect_buffer_offset,
                threads_per_object_threadgroup: object,
                threads_per_mesh_threadgroup: mesh,
            } => {
                let shape = Shape::Mesh { object, mesh };
                self.draw(sel, (Some((shape, ())), false), |e, ()| {
                    e.draw_mesh_threadgroups_indirect(
                        &indirect_buffer.raw,
                        indirect_buffer_offset,
                        size(object),
                        size(mesh),
                    )
                })
            }
            RenderCmd::SetMeshBufferOffset { offset, index } => {
                let index = small(sel, index);
                encoded(&mut |e| e.set_mesh_buffer_offset(offset, index))
            }
            RenderCmd::SetMeshBuffers {
                buffers: list,
                offsets,
                range,
            } => {
                let (at, list) = ranged(sel, range, list);
                let (_, offsets) = ranged(sel, range, offsets);
                buffers(list, |list| {
                    encoded(&mut |e| e.set_mesh_buffers(list, offsets, at))
                })
            }
            RenderCmd::SetMeshSamplerStatesLodClamps {
                samplers: list,
                lod_min_clamps,
                lod_max_clamps,
                range,
            } => {
                let (at, list) = ranged(sel, range, list);
                let (_, min) = ranged(sel, range, lod_min_clamps);
                let (_, max) = ranged(sel, range, lod_max_clamps);
                samplers(list, |list| {
                    encoded(&mut |e| e.set_mesh_sampler_states(list, Some((min, max)), at))
                })
            }
            RenderCmd::SetMeshSamplerStates {
                samplers: list,
                range,
            } => {
                let (at, list) = ranged(sel, range, list);
                samplers(list, |list| {
                    encoded(&mut |e| e.set_mesh_sampler_states(list, None, at))
                })
            }
            RenderCmd::SetMeshTextures {
                textures: list,
                range,
            } => {
                let (at, list) = ranged(sel, range, list);
                textures(sel, list, |list| {
                    encoded(&mut |e| e.set_mesh_textures(list, at))
                })
            }
            RenderCmd::SetObjectBufferOffset { offset, index } => {
                let index = small(sel, index);
                encoded(&mut |e| e.set_object_buffer_offset(offset, index))
            }
            RenderCmd::SetObjectBuffers {
                buffers: list,
                offsets,
                range,
            } => {
                let (at, list) = ranged(sel, range, list);
                let (_, offsets) = ranged(sel, range, offsets);
                buffers(list, |list| {
                    encoded(&mut |e| e.set_object_buffers(list, offsets, at))
                })
            }
            RenderCmd::SetObjectSamplerStatesLodClamps {
                samplers: list,
                lod_min_clamps,
                lod_max_clamps,
                range,
            } => {
                let (at, list) = ranged(sel, range, list);
                let (_, min) = ranged(sel, range, lod_min_clamps);
                let (_, max) = ranged(sel, range, lod_max_clamps);
                samplers(list, |list| {
                    encoded(&mut |e| e.set_object_sampler_states(list, Some((min, max)), at))
                })
            }
            RenderCmd::SetObjectSamplerStates {
                samplers: list,
                range,
            } => {
                let (at, list) = ranged(sel, range, list);
                samplers(list, |list| {
                    encoded(&mut |e| e.set_object_sampler_states(list, None, at))
                })
            }
            RenderCmd::SetObjectTextures {
                textures: list,
                range,
            } => {
                let (at, list) = ranged(sel, range, list);
                textures(sel, list, |list| {
                    encoded(&mut |e| e.set_object_textures(list, at))
                })
            }
            // The length the object function's `[[threadgroup(index)]]` argument takes in the
            // encoder's later mesh draws, which the runtime checks at each draw against the
            // device's workgroup memory for the stage.
            RenderCmd::SetObjectThreadgroupMemoryLength { length, index } => {
                let (length, index) = (small(sel, length), small(sel, index));
                encoded(&mut |e| e.set_object_threadgroup_memory_length(length, index))
            }
            // Metal's indexed strips restart at the index set, until the encoder ends, and
            // nothing else changes (the module docs), as the runtime's do.
            RenderCmd::SetPrimitiveRestartEnabled { enabled, index } => {
                plain(|| self.drawing.lock().borrow_mut().marks.restart = Some((enabled, index)));
                encoded(&mut |e| {
                    e.set_primitive_restart_enabled(enabled, index);
                    Ok(())
                })
            }
            // The runtime draws the encoder's draws after it as Metal does at the mode, or
            // announces what it draws otherwise (the module docs).
            RenderCmd::SetProvokingVertexMode { mode } => {
                encoded(&mut |e| e.set_provoking_vertex_mode(mode))
            }
            RenderCmd::SetRenderPipelineState { pipeline_state } => match pipeline_state {
                None => {
                    self.encoder.nothing(sel)?;
                    plain(|| self.drawing.lock().borrow_mut().nil_pipeline = true);
                    self.encoder.pipeline.store(0, Ordering::Relaxed);
                    self.encoder.encodes_icbs.store(false, Ordering::SeqCst);
                    Ok(())
                }
                Some(state) => {
                    self.run(sel, &mut |e| e.set_render_pipeline_state(&state.raw))?;
                    let f = *state.drawn.get_or_init(|| StateFacts::of(state));
                    self.encoder
                        .pipeline
                        .store(state.unique_identifier(), Ordering::Relaxed);
                    self.encoder.encodes_icbs.store(f.encodes, Ordering::SeqCst);
                    plain(|| {
                        let locked = self.drawing.lock();
                        let mut drawing = locked.borrow_mut();
                        drawing.nil_pipeline = false;
                        drawing.reach.set(&state.reflection, f.writes);
                        let m = &mut drawing.marks;
                        m.pipeline = true;
                        m.points_sized = f.points_sized;
                        m.half_apart = f.half_apart;
                        m.mesh = f.mesh;
                    });
                    Ok(())
                }
            },
            RenderCmd::SetScissorRect { rect } => {
                let rect = scissor(rect);
                encoded(&mut |e| e.set_scissor_rect(rect))
            }
            RenderCmd::SetScissorRects { scissor_rects } => {
                let rects: Vec<_> = scissor_rects.iter().copied().map(scissor).collect();
                encoded(&mut |e| e.set_scissor_rects(&rects))
            }
            RenderCmd::SetStencilFrontReferenceValue {
                front_reference_value,
                back_reference_value,
            } => encoded(&mut |e| {
                e.set_stencil_reference_values(front_reference_value, back_reference_value);
                Ok(())
            }),
            RenderCmd::SetStencilResolveTexture {
                texture,
                slice,
                depth_plane,
                level,
                y_invert: _,
            } => {
                if self.resolve_unused(Attachment::Stencil) {
                    return self.encoder.nothing(sel);
                }
                if texture.is_some() {
                    self.resolves_beside_dropped(sel, Attachment::Stencil);
                }
                let at = (slice, depth_plane, level);
                let target = self.late_resolve(sel, Attachment::Stencil, texture, at);
                encoded(&mut |e| e.set_stencil_resolve_target(target.clone()))
            }
            RenderCmd::SetStencilStoreAction { store_action: raw } => {
                let action = store_action(sel, raw);
                self.late_store(sel, Attachment::Stencil, action, &mut |e| {
                    e.set_stencil_store_action(action)
                })
            }
            RenderCmd::SetStencilStoreActionOptions {
                store_action_options: raw,
            } => {
                let options = store_action_options(sel, raw);
                self.late_options(sel, Attachment::Stencil, options, &mut |e| {
                    e.set_stencil_store_action_options(options)
                })
            }
            RenderCmd::SetDepthStoreActionOptions {
                store_action_options: raw,
            } => {
                let options = store_action_options(sel, raw);
                self.late_options(sel, Attachment::Depth, options, &mut |e| {
                    e.set_depth_store_action_options(options)
                })
            }
            RenderCmd::SetTessellationFactorBuffer {
                buffer,
                offset,
                instance_stride,
            } => encoded(&mut |e| {
                e.set_tessellation_factor_buffer(buffer.map(|b| &b.raw), offset, instance_stride)
            }),
            RenderCmd::SetTessellationFactorScale { scale } => encoded(&mut |e| {
                e.set_tessellation_factor_scale(scale);
                Ok(())
            }),
            RenderCmd::SetThreadgroupMemoryLength {
                length,
                offset,
                index,
            } => {
                let index = small(sel, index);
                encoded(&mut |e| e.set_threadgroup_memory_length(length, offset, index))
            }
            RenderCmd::SetTileBufferOffset { offset, index } => {
                let index = small(sel, index);
                encoded(&mut |e| e.set_tile_buffer_offset(offset, index))
            }
            RenderCmd::SetTileBuffers {
                buffers: list,
                offsets,
                range,
            } => {
                let (at, list) = ranged(sel, range, list);
                let (_, offsets) = ranged(sel, range, offsets);
                buffers(list, |list| {
                    encoded(&mut |e| e.set_tile_buffers(list, offsets, at))
                })
            }
            RenderCmd::SetTileSamplerStatesLodClamps {
                samplers: list,
                lod_min_clamps,
                lod_max_clamps,
                range,
            } => {
                let (at, list) = ranged(sel, range, list);
                let (_, min) = ranged(sel, range, lod_min_clamps);
                let (_, max) = ranged(sel, range, lod_max_clamps);
                samplers(list, |list| {
                    encoded(&mut |e| e.set_tile_sampler_states(list, Some((min, max)), at))
                })
            }
            RenderCmd::SetTileSamplerStates {
                samplers: list,
                range,
            } => {
                let (at, list) = ranged(sel, range, list);
                samplers(list, |list| {
                    encoded(&mut |e| e.set_tile_sampler_states(list, None, at))
                })
            }
            RenderCmd::SetTileTextures {
                textures: list,
                range,
            } => {
                let (at, list) = ranged(sel, range, list);
                textures(sel, list, |list| {
                    encoded(&mut |e| e.set_tile_textures(list, at))
                })
            }
            RenderCmd::SetTriangleFillMode { fill_mode } => {
                let mode: TriangleFillMode = metal_value(sel, "triangle fill mode", fill_mode);
                encoded(&mut |e| e.set_triangle_fill_mode(mode))
            }
            RenderCmd::SetVertexAmplificationCount { view_mappings } => {
                let count = u64::try_from(view_mappings.len()).unwrap_or_else(|_| {
                    unrepresentable(sel, &format!("{} view mappings", view_mappings.len()))
                });
                let mappings: Vec<_> = view_mappings
                    .iter()
                    .map(|m| {
                        rt::VertexAmplificationViewMapping::new(
                            m.viewport_array_index_offset,
                            m.render_target_array_index_offset,
                        )
                    })
                    .collect();
                let mappings = (!mappings.is_empty()).then_some(mappings.as_slice());
                encoded(&mut |e| e.set_vertex_amplification_count(count, mappings))?;
                plain(|| {
                    let locked = self.drawing.lock();
                    let mut drawing = locked.borrow_mut();
                    let m = &mut drawing.marks;
                    m.amplification = Some(count);
                    m.amplified = false;
                });
                Ok(())
            }
            // The count it sets, which keeps the view mappings set before, is the runtime's
            // where it equals the runtime's (the module docs).
            RenderCmd::SetVertexAmplificationMode { mode, value } => {
                plain(|| {
                    let locked = self.drawing.lock();
                    let mut drawing = locked.borrow_mut();
                    let m = &mut drawing.marks;
                    let runtime = m.amplification.unwrap_or(1);
                    m.amplified = amplification_count(mode, value) != Some(runtime);
                });
                self.encoder.nothing(sel)
            }
            RenderCmd::SetVertexBufferOffset { offset, index } => {
                let index = small(sel, index);
                encoded(&mut |e| e.set_vertex_buffer_offset(offset, index))
            }
            RenderCmd::SetVertexBufferOffsetAttributeStride {
                offset,
                stride,
                index,
            } => {
                let index = small(sel, index);
                encoded(&mut |e| e.set_vertex_buffer_offset_with_stride(offset, stride, index))
            }
            RenderCmd::SetVertexBuffersAttributeStrides {
                buffers: list,
                offsets,
                strides,
                range,
            } => {
                let (at, list) = ranged(sel, range, list);
                let (_, offsets) = ranged(sel, range, offsets);
                let (_, strides) = ranged(sel, range, strides);
                buffers(list, |list| {
                    encoded(&mut |e| e.set_vertex_buffers_with_strides(list, offsets, strides, at))
                })
            }
            RenderCmd::SetVertexBuffers {
                buffers: list,
                offsets,
                range,
            } => {
                let (at, list) = ranged(sel, range, list);
                let (_, offsets) = ranged(sel, range, offsets);
                buffers(list, |list| {
                    encoded(&mut |e| e.set_vertex_buffers(list, offsets, at))
                })
            }
            RenderCmd::SetVertexSamplerStatesLodClamps {
                samplers: list,
                lod_min_clamps,
                lod_max_clamps,
                range,
            } => {
                let (at, list) = ranged(sel, range, list);
                let (_, min) = ranged(sel, range, lod_min_clamps);
                let (_, max) = ranged(sel, range, lod_max_clamps);
                samplers(list, |list| {
                    encoded(&mut |e| e.set_vertex_sampler_states(list, Some((min, max)), at))
                })
            }
            RenderCmd::SetVertexSamplerStates {
                samplers: list,
                range,
            } => {
                let (at, list) = ranged(sel, range, list);
                samplers(list, |list| {
                    encoded(&mut |e| e.set_vertex_sampler_states(list, None, at))
                })
            }
            RenderCmd::SetVertexTextures {
                textures: list,
                range,
            } => {
                let (at, list) = ranged(sel, range, list);
                textures(sel, list, |list| {
                    encoded(&mut |e| e.set_vertex_textures(list, at))
                })
            }
            RenderCmd::SetViewport { viewport: v } => {
                let v = viewport(v);
                self.run(sel, &mut |e| e.set_viewport(v))
            }
            RenderCmd::SetViewportTransformEnabled { enabled } => {
                encoded(&mut |e| e.set_viewport_transform_enabled(enabled))
            }
            RenderCmd::SetViewports { viewports } => {
                let list: Vec<_> = viewports.iter().copied().map(viewport).collect();
                encoded(&mut |e| e.set_viewports(&list))
            }
            RenderCmd::SetVisibilityResultMode { mode, offset } => {
                let mode = rt::VisibilityResultMode::from_raw(mode).unwrap_or_else(|_| {
                    unrepresentable(sel, &format!("visibility result mode {mode}"))
                });
                if mode != rt::VisibilityResultMode::Disabled
                    && plain(|| self.drawing.lock().borrow().reach.visibility.holds())
                {
                    crate::gpu::writes::kept_write(
                        sel,
                        "a visibility result mode whose result the pass writes into a buffer over",
                    );
                }
                encoded(&mut |e| e.set_visibility_result_mode(mode, offset))
            }
            RenderCmd::UpdateFence { fence, stages } => {
                let stages: RenderStages = metal_value(sel, "render stages", stages);
                encoded(&mut |e| e.update_fence(fence, stages))
            }
            RenderCmd::UseHeaps { heaps: list } => {
                let list = heaps(list);
                encoded(&mut |e| e.use_heaps(&list, VERTEX_AND_FRAGMENT))
            }
            RenderCmd::UseHeapsStages {
                heaps: list,
                stages,
            } => {
                let list = heaps(list);
                let stages: RenderStages = metal_value(sel, "render stages", stages);
                encoded(&mut |e| e.use_heaps(&list, stages))
            }
            RenderCmd::UseResources {
                resources: list,
                usage,
            } => {
                let some = present(list);
                let usage: ResourceUsage = metal_value(sel, "resource usage", usage);
                let icbs = crate::gpu::icb::made_resident(list);
                if !icbs.is_empty() {
                    // A function of the encoder's may copy any of their commands.
                    let progress = self.encoder.progress(sel);
                    for icb in &icbs {
                        icb.resident(sel, progress.as_ref(), usage.contains(ResourceUsage::WRITE));
                    }
                }
                encoded(&mut |e| e.use_resources(&some, usage, VERTEX_AND_FRAGMENT))?;
                if usage.contains(ResourceUsage::WRITE) {
                    // A function of the encoder's may write their records.
                    self.writable(&icbs);
                }
                if !icbs.is_empty() {
                    self.encoder.hold(sel, icbs);
                }
                Ok(())
            }
            RenderCmd::UseResourcesStages {
                resources: list,
                usage,
                stages,
            } => {
                let some = present(list);
                let usage: ResourceUsage = metal_value(sel, "resource usage", usage);
                let stages: RenderStages = metal_value(sel, "render stages", stages);
                let icbs = crate::gpu::icb::made_resident(list);
                if !icbs.is_empty() {
                    // A function of the encoder's may copy any of their commands.
                    let progress = self.encoder.progress(sel);
                    for icb in &icbs {
                        icb.resident(sel, progress.as_ref(), usage.contains(ResourceUsage::WRITE));
                    }
                }
                encoded(&mut |e| e.use_resources(&some, usage, stages))?;
                if usage.contains(ResourceUsage::WRITE) {
                    // A function of the encoder's may write their records.
                    self.writable(&icbs);
                }
                if !icbs.is_empty() {
                    self.encoder.hold(sel, icbs);
                }
                Ok(())
            }
            RenderCmd::WaitForFence { fence, stages } => {
                let stages: RenderStages = metal_value(sel, "render stages", stages);
                // A wait for a fence this encoder updated before completes on Metal
                // (`render_objects_fence`'s update then wait), as the runtime's waits for the
                // update before the encoder.
                encoded(&mut |e| e.wait_for_fence(fence, stages))
            }
        }
    }

    /// A run of draws in a pass whose draws ask nothing of what the encoder holds
    /// ([`Self::draws_quietly`]) takes each command by what is left of its path there
    /// ([`Self::quiet_draws`]); any other, each command as [`Self::encode`] takes it.
    fn encode_draws(
        &self,
        on: OnFault,
        draws: &[VertexDraw<'_, VulkanPlatform>],
    ) -> RaiseResult<()> {
        self.encoding(|| match self.draws_quietly() {
            Some(held) => self.quiet_draws(&held, draws),
            None => encode_each(self, on, draws),
        })
    }

    /// The run holds what the encoder holds of its draws ([`RenderEncoderObject::drawing`]) and
    /// the runtime's encoder ([`EncoderObject::running`]), which each command inside it then
    /// takes again on this thread.
    fn encoding<T>(&self, run: impl FnOnce() -> T) -> T {
        let _drawing = self.drawing.lock();
        self.encoder.running(run)
    }

    fn tile_height(&self) -> u64 {
        let sel = "tileHeight";
        get(sel, || self.encoder.with(sel, |e| e.tile_height()))
    }

    fn tile_width(&self) -> u64 {
        let sel = "tileWidth";
        get(sel, || self.encoder.with(sel, |e| e.tile_width()))
    }
}

impl VulkanRenderEncoder {
    /// The descriptor's store action and options of `attachment`, as core set them (a dropped
    /// resolve's action too, [`Dropped`]); `None` for one without a texture, or past the eighth
    /// colour attachment.
    fn stored(&self, attachment: Attachment) -> Option<(rt::StoreAction, rt::StoreActionOptions)> {
        let p = &self.pass;
        let (texture, action, options) = match attachment {
            Attachment::Color(i) => {
                let c = p.color_attachments.get(usize::try_from(i).ok()?)?;
                let action = self.dropped_resolve(i).map_or(c.store_action, |d| d.store);
                (&c.texture, action, c.store_action_options)
            }
            Attachment::Depth => {
                let a = &p.depth_attachment;
                (&a.texture, a.store_action, a.store_action_options)
            }
            Attachment::Stencil => {
                let a = &p.stencil_attachment;
                (&a.texture, a.store_action, a.store_action_options)
            }
        };
        texture.as_ref().map(|_| (action, options))
    }

    /// The colour attachment at `index` whose resolve [`untouched_resolves`] dropped, if it is
    /// one.
    fn dropped_resolve(&self, index: u32) -> Option<Dropped> {
        self.dropped.iter().find(|d| d.index == index).copied()
    }

    /// Fails the command buffer from this pass on with Metal's page fault where the pass keeps an
    /// attachment whose texture lies over read-only memory the runtime copies
    /// ([`over_read_only_copy`], [`stores_out`]), as `sources` holds the pass descriptor's: Metal's
    /// GPU faults the command buffer there, where the runtime's copy would take the store
    /// ([`crate::gpu::command::CommandBufferObject::faults_at_read_only_store`]). One whose store
    /// action is `Unknown` waits for its late store action ([`Self::late_store`]). One that does
    /// not store (`DontCare`, `MultisampleResolve`) is sent, as Metal completes it, its writes
    /// dropped by the copy.
    fn read_only_stores(&self, sources: &Sources) {
        let colors = (0u32..).zip(&sources.color);
        let attachments = (colors.map(|(i, [t, _])| (Attachment::Color(i), t)))
            .chain([(Attachment::Depth, &sources.depth[0])])
            .chain([(Attachment::Stencil, &sources.stencil[0])]);
        let mut faults = false;
        for (attachment, texture) in attachments {
            if !plain(|| over_read_only_copy(texture)) {
                continue;
            }
            match self.stored(attachment).map(|(action, _)| action) {
                Some(rt::StoreAction::Unknown) => {
                    plain(|| self.read_only_undecided.lock().push(attachment));
                }
                Some(action) => faults |= stores_out(action),
                None => {}
            }
        }
        if faults {
            self.encoder.faults_at_read_only_store();
        }
    }

    /// Marks each undecided attachment ([`Self::late_store`]) with why a resolve into its resolve
    /// texture, as `sources` holds the pass descriptor's, is never sent ([`Unsent`]).
    fn undecided_dropped(&self, sources: &Sources) {
        plain(|| {
            for (a, unsent) in self.undecided.lock().iter_mut() {
                let resolve = match *a {
                    Attachment::Color(i) => usize::try_from(i)
                        .ok()
                        .and_then(|i| sources.color.get(i))
                        .map(|[_, r]| r),
                    Attachment::Depth => Some(&sources.depth[1]),
                    Attachment::Stencil => Some(&sources.stencil[1]),
                };
                *unsent = resolve.and_then(Source::texture).and_then(Unsent::of);
            }
        });
    }

    /// Relays at `sel`, or dies there in strict mode ([`crate::answer::Inexact::LateResolve`]), a
    /// late resolve texture of the depth or stencil `attachment`, whose store
    /// action resolves or is left `Unknown` ([`Self::resolve_unused`]), in a pass made without a
    /// colour resolve plain Metal leaves untouched ([`Dropped`]): the attachment may then resolve,
    /// a shape the drop is not measured beside, at which the pass's creation dies where the
    /// descriptor names the resolve texture ([`unmeasured_shape`]). One of a pass without that
    /// attachment goes on, and so does one whose action a late setter decided and that does not
    /// resolve ([`Self::late_store`]): nothing resolves beside the drop, so the setter goes to
    /// the runtime, as in a pass without a dropped resolve (which refuses a resolve texture under
    /// an action that does not resolve, `ResolveStoreAction`, and that refusal dies there).
    fn resolves_beside_dropped(&self, sel: &'static str, attachment: Attachment) {
        let Some(d) = self.dropped.first() else {
            return;
        };
        if self.stored(attachment).is_none() {
            return;
        }
        let late = plain(|| {
            let late = self.late_stores.lock();
            late.iter()
                .find(|&&(a, _)| a == attachment)
                .map(|&(_, action)| action)
        });
        if late.is_some_and(|action| !may_resolve(action)) {
            return;
        }
        let name = match attachment {
            Attachment::Depth => "depth",
            _ => "stencil",
        };
        crate::answer::inexact(sel, crate::answer::Inexact::LateResolve, || {
            format!(
                "a late {name} resolve texture in a pass made without the resolve of colour \
                 attachment {} by {:?} into {}, as plain Metal leaves that texture untouched: \
                 the pass's depth or stencil attachment resolves, a shape that drop is not \
                 measured beside",
                d.index,
                d.store,
                d.into.name()
            )
        });
    }

    /// Whether `attachment` resolves nowhere whatever its resolve texture: the descriptor knew
    /// its store action, and the action does not resolve, so Metal ignores a late resolve setter
    /// there as it ignores the descriptor's resolve texture ([`without_unused_resolves`]).
    fn resolve_unused(&self, attachment: Attachment) -> bool {
        matches!(self.stored(attachment), Some((action, _))
            if action != rt::StoreAction::Unknown && !may_resolve(action))
    }

    /// A late store action `action` of `attachment`, set by `call`. Where the descriptor knew
    /// the action, or named no texture, Metal goes on (`render_pass_eight`): the same action
    /// again, and any action of an attachment without a texture, encode nothing; another action
    /// over a known one dies, as what Metal does with it is not measured. The known action is the
    /// descriptor's as core set it, a dropped resolve's too ([`Dropped`]): its resolving action
    /// again encodes nothing, as plain Metal resolves nothing into the texture
    /// (`resolves_into_other_formats_are_metals_or_die_named`). A colour index past the eighth is
    /// the runtime's to refuse. One that resolves an undecided attachment into a resolve texture
    /// whose GPU writes never reach its memory dies ([`crate::os::dropped_write`]). One that keeps
    /// an undecided attachment over read-only memory the runtime copies fails the command buffer
    /// as Metal's GPU faults it, and one that keeps it by `CustomSampleDepthStore` dies
    /// ([`Self::read_only_stores`]).
    fn late_store(
        &self,
        sel: &'static str,
        attachment: Attachment,
        action: rt::StoreAction,
        call: Call<'_>,
    ) -> RaiseResult<()> {
        if past_the_eighth(attachment) {
            return self.run(sel, call);
        }
        match self.stored(attachment) {
            Some((rt::StoreAction::Unknown, _)) => {
                let named = plain(|| {
                    let undecided = self.undecided.lock();
                    undecided
                        .iter()
                        .find(|(a, _)| *a == attachment)
                        .and_then(|&(_, unsent)| unsent)
                });
                if let Some(unsent) = named
                    && action.resolves()
                {
                    unsent.dies(sel, "a store action that resolves into a texture");
                }
                let read_only = plain(|| {
                    let mut undecided = self.read_only_undecided.lock();
                    let at = undecided.iter().position(|&a| a == attachment);
                    at.map(|at| undecided.remove(at)).is_some()
                });
                if read_only && action == rt::StoreAction::CustomSampleDepthStore {
                    crate::gpu::writes::kept_write(
                        sel,
                        "a store action CustomSampleDepthStore that keeps an attachment over",
                    );
                }
                if read_only && stores_out(action) {
                    self.encoder.faults_at_read_only_store();
                }
                self.run(sel, call)?;
                plain(|| {
                    self.undecided.lock().retain(|&(a, _)| a != attachment);
                    let mut late = self.late_stores.lock();
                    late.retain(|&(a, _)| a != attachment);
                    late.push((attachment, action));
                });
                Ok(())
            }
            Some((known, _)) if known != action => unrepresentable(
                sel,
                &format!("a late store action {action:?} over the known {known:?}"),
            ),
            _ => self.encoder.nothing(sel),
        }
    }

    /// A late resolve target of `texture` at `at`, its slice, depth plane and level, for
    /// `attachment`, which `sel` sets. Where the attachment's store action is undecided, whether
    /// the texture's GPU writes never reach its memory is kept for the late store action that
    /// decides whether it resolves ([`Self::late_store`]); elsewhere the action resolves, so one
    /// whose GPU writes never reach its memory dies ([`crate::os::dropped_write`]).
    fn late_resolve(
        &self,
        sel: &'static str,
        attachment: Attachment,
        texture: Option<&VulkanTexture>,
        at: (u64, u64, u64),
    ) -> Option<rt::ResolveTarget> {
        let unsent = texture.and_then(Unsent::of);
        let kept = plain(|| {
            let mut undecided = self.undecided.lock();
            let named = undecided.iter_mut().find(|(a, _)| *a == attachment);
            named.map(|(_, never)| *never = unsent)
        });
        if kept.is_none()
            && let Some(unsent) = unsent
        {
            unsent.dies(sel, "a resolve into a texture");
        }
        match kept {
            Some(()) => resolve_target(texture.map(|t| &t.raw), at),
            None => resolve_target(texture.map(|t| t.texels(sel)), at),
        }
    }

    /// Late store action options `options` of `attachment`, set by `call`: as
    /// [`late_store`](Self::late_store), the descriptor's own options again encoding nothing.
    fn late_options(
        &self,
        sel: &'static str,
        attachment: Attachment,
        options: rt::StoreActionOptions,
        call: Call<'_>,
    ) -> RaiseResult<()> {
        if past_the_eighth(attachment) {
            return self.run(sel, call);
        }
        match self.stored(attachment) {
            Some((rt::StoreAction::Unknown, _)) => self.run(sel, call),
            Some((_, known)) if known != options => unrepresentable(
                sel,
                &format!("late store action options {options:?} over the known {known:?}"),
            ),
            _ => self.encoder.nothing(sel),
        }
    }

    /// A non-indexed draw of a raw primitive type: `(start, count, instances, base instance)`; one
    /// of no instances is an empty draw, which the runtime records nothing of (the module docs).
    /// The runtime draws a private type as the list Metal draws it as
    /// ([`rt::DrawPrimitiveType`]), and type 9 fails the command buffer as Metal's GPU does
    /// ([`Self::nine`]).
    fn primitives(
        &self,
        sel: &'static str,
        primitive_type: u64,
        (start, count, instances, base): (u64, u64, u64, u64),
    ) -> RaiseResult<()> {
        let (start, count) = (small(sel, start), small(sel, count));
        let (instances, base) = (small(sel, instances), small(sel, base));
        let t = draw_type(sel, primitive_type);
        let call = |e: &mut rt::RenderCommandEncoder, t| {
            draw_primitives(e, t, (start, count), (instances, base))
        };
        if t == rt::DrawPrimitiveType::Nine {
            let hang = (count >= 3 && instances > 0).then_some(rt::CommandBufferError::GpuHang);
            return self.nine(sel, hang, &mut |e| call(e, t));
        }
        let empty = instances == 0 || draws_none(t, count);
        let drawn = self.draw(sel, (Some((Shape::Primitives(t), t)), empty), call);
        self.drew(t, (count, instances), &drawn);
        drawn
    }

    /// After a guest's draw of `count` vertices of `t`, `instances` times, answered `drawn`: one
    /// of at least one whole triangle of a triangle list or strip (3 vertices or more), once or
    /// more, that the runtime took, takes the place of the platform's draw of no vertices at the
    /// pass's end ([`crate::gpu::rate_map::no_draw`]), as the Metal platform's does (its
    /// `gpu::rate_map::drew`). The runtime counts every draw it takes as the pass's draw call
    /// (`rate_mapped_pass.rs`), so it no longer fails the pass as a GPU hang. Other draws leave
    /// the platform's in place, as the Metal platform's do: what Metal's GPU takes as the pass's
    /// draw was measured for this shape alone.
    fn drew(
        &self,
        t: rt::DrawPrimitiveType,
        (count, instances): (u32, u32),
        drawn: &RaiseResult<()>,
    ) {
        let whole = matches!(
            t,
            rt::DrawPrimitiveType::Public(PrimitiveType::Triangle | PrimitiveType::TriangleStrip)
        ) && count >= 3
            && instances > 0;
        if whole && drawn.is_ok() {
            self.undrawn.store(false, Ordering::Release);
        }
    }

    /// What the encoder holds for its draws ([`Held`]), where none of a draw's checks that read it
    /// can find anything, nor any that reads the thread or the log: no nil pipeline state
    /// ([`nil_pipeline_drawn`]), none that encodes indirect commands ([`Self::icbs_written`]), a
    /// pass that loads no bfloat attachment ([`Self::bfloat_drawn`]) and has no rate map of no
    /// layers or of varying rates ([`Self::layered`]), a command buffer to bind for
    /// ([`Self::bound`]), and no timing lines
    /// ([`crate::gpu::timing::encoder_call`]). A command changes none of them but a pipeline
    /// state's set, a private setter or the encoder's end, none of which a run of draws holds;
    /// whether the pass holds anything a draw asks of ([`Reach::quiet`]) is read at each.
    fn draws_quietly(&self) -> Option<Held> {
        let held = self.held();
        let quietly = !held.nil_pipeline
            && !held.encodes
            && self.bfloat_loaded.is_none()
            && !self.no_layers
            && !self.varying_rates
            && self.encoder.has_command_buffer()
            && !crate::gpu::timing::on();
        quietly.then_some(held)
    }

    /// `draws`, where [`Self::draws_quietly`] answered `held`: each command does what
    /// [`Self::encode`] does for it, in its order, by what is left of its path once the run's
    /// answers are known, and takes that path where the pass holds something a draw asks of
    /// ([`Reach::quiet`]), or where its bind or draw needs more. The run takes the runtime's
    /// encoder once, and borrows it for each call ([`EncoderObject::with`]), so a command that
    /// takes its path takes it as it does alone; and this thread's calls into the runtime and the
    /// hearing of the draws are looked up once ([`crate::answer::run`]).
    fn quiet_draws(
        &self,
        held: &Held,
        draws: &[VertexDraw<'_, VulkanPlatform>],
    ) -> RaiseResult<()> {
        #[cfg(test)]
        tests::QUIET_RUNS.set(tests::QUIET_RUNS.get() + 1);
        let raw = self.encoder.raw.lock();
        crate::answer::run(|run| {
            for draw in draws {
                self.quiet_vertex_buffer(run, &raw, draw)?;
                self.quiet_primitives(run, &raw, held, draw)?;
            }
            Ok(())
        })
    }

    /// `draw`'s `setVertexBuffer:offset:atIndex:` in [`Self::quiet_draws`]: [`Self::vertex_buffer`]
    /// where its bind is read ([`Self::bound`]), else its index ([`small`]) and the call.
    fn quiet_vertex_buffer(
        &self,
        run: &crate::answer::Run<'_>,
        raw: &RefCell<Option<rt::RenderCommandEncoder>>,
        draw: &VertexDraw<'_, VulkanPlatform>,
    ) -> RaiseResult<()> {
        let sel = SET_VERTEX_BUFFER;
        let (buffer, offset) = (draw.buffer, draw.offset);
        let bind = Bind::Buffer(VERTEX, draw.index, buffer);
        if !self.quiet.load(Ordering::Acquire) || quiet_bind_decides(&bind) {
            return self.vertex_buffer(buffer, offset, draw.index);
        }
        let index = small(sel, draw.index);
        let mut refusal = None;
        run.at(sel, || {
            if let Err(e) = set_vertex_buffer(&mut borrowed(raw, sel), buffer, offset, index) {
                refusal = Some(e);
            }
        });
        refusal.map_or(Ok(()), |e| decide(sel, Policy::Dies, e))
    }

    /// `draw`'s `drawPrimitives:vertexStart:vertexCount:` in [`Self::quiet_draws`], under `held`:
    /// [`Self::primitives`] where the pass holds something a draw asks of and for type 9
    /// ([`Self::nine`]), else what [`Self::draw`] checks that can find anything there (its points,
    /// [`sized`], and the private setters' marks, [`marked_by`]), the call, the `half` outputs the
    /// runtime announced inside it ([`Self::binds`]), and what it drew ([`Self::drew`]).
    fn quiet_primitives(
        &self,
        run: &crate::answer::Run<'_>,
        raw: &RefCell<Option<rt::RenderCommandEncoder>>,
        held: &Held,
        draw: &VertexDraw<'_, VulkanPlatform>,
    ) -> RaiseResult<()> {
        let sel = DRAW_PRIMITIVES;
        let shape = (draw.vertex_start, draw.vertex_count, 1, 0);
        if !self.quiet.load(Ordering::Acquire) {
            return self.primitives(sel, draw.primitive_type, shape);
        }
        let (start, count) = (small(sel, draw.vertex_start), small(sel, draw.vertex_count));
        let t = draw_type(sel, draw.primitive_type);
        if t == rt::DrawPrimitiveType::Nine {
            return self.primitives(sel, draw.primitive_type, shape);
        }
        crate::gpu::timing::count_draw();
        if !draws_none(t, count) {
            sized(sel, Shape::Primitives(t), held.marks);
            marked_by(sel, held.marks);
        }
        let at = run.heard();
        let mut refusal = None;
        run.at(sel, || {
            if let Err(e) = draw_primitives(&mut borrowed(raw, sel), t, (start, count), (1, 0)) {
                refusal = Some(e);
            }
        });
        let drawn = refusal.map_or(Ok(()), |e| decide(sel, Policy::Dies, e));
        if run.heard_since(at, rt::Degradation::HalfOutputPrecision)
            && let Some(apart) = held.marks.half_apart
        {
            half_output_apart(sel, apart);
        }
        self.drew(t, (count, 1), &drawn);
        drawn
    }

    /// An indexed draw of a raw primitive and index type: `(count, instances, base vertex, base
    /// instance)` from the index buffer at its offset; one of no instances is an empty draw,
    /// which the runtime records nothing of (the module docs). Of type 9, of any count and
    /// instances, Metal's GPU faults the command buffer, as the runtime does ([`Self::nine`]).
    fn indexed(
        &self,
        sel: &'static str,
        (primitive_type, raw_index_type): (u64, u64),
        (count, instances, base_vertex, base_instance): (u64, u64, i64, u64),
        (buffer, offset): (&VulkanBuffer, u64),
    ) -> RaiseResult<()> {
        let kinds = indexed_kinds(sel, primitive_type, raw_index_type);
        let (count, instances) = (small(sel, count), small(sel, instances));
        let base_vertex = i32::try_from(base_vertex).unwrap_or_else(|_| {
            unrepresentable(sel, &format!("base vertex {base_vertex} as an i32"))
        });
        let base_instance = small(sel, base_instance);
        let call = |e: &mut rt::RenderCommandEncoder, (t, i)| {
            e.draw_indexed_primitives(
                t,
                count,
                i,
                &buffer.raw,
                offset,
                instances,
                base_vertex,
                base_instance,
            )
        };
        if let Some((t @ rt::DrawPrimitiveType::Nine, i)) = kinds {
            let fault = Some(rt::CommandBufferError::PageFault);
            return self.nine(sel, fault, &mut |e| call(e, (t, i)));
        }
        let empty = instances == 0 || kinds.is_some_and(|(t, _)| draws_none(t, count));
        let drawn = kinds.map(|(t, i)| (Shape::Primitives(t), (t, i)));
        self.draw(sel, (drawn, empty), call)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    /// The commands `encode` takes by paths of their own die at the selectors the rest name.
    #[test]
    fn the_own_paths_selectors_are_the_commands() {
        use super::{DRAW_PRIMITIVES, SET_VERTEX_BUFFER};
        let set = RenderCmd::<crate::platform::VulkanPlatform>::SetVertexBuffer {
            buffer: None,
            offset: 0,
            index: 0,
        };
        assert_eq!(set.selector(), SET_VERTEX_BUFFER);
        let draw = RenderCmd::<crate::platform::VulkanPlatform>::DrawPrimitives {
            primitive_type: 0,
            vertex_start: 0,
            vertex_count: 0,
        };
        assert_eq!(draw.selector(), DRAW_PRIMITIVES);
    }

    /// What a test records of the runtime calls of draws' two commands ([`sent`]).
    struct Recording {
        /// Each call: its selector, then its arguments.
        calls: Vec<String>,
        /// What runs inside each call, by its place among them, as the runtime's own work there
        /// would.
        inside: Box<dyn FnMut(usize)>,
    }

    thread_local! {
        /// What this thread's test records ([`recorded`]).
        static RECORDING: std::cell::RefCell<Option<Recording>> =
            const { std::cell::RefCell::new(None) };
        /// How many runs of draws this thread's encoders took by [`VulkanRenderEncoder::quiet_draws`].
        pub(super) static QUIET_RUNS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    /// Records the runtime call of `sel` this thread makes, `args` its arguments, where its test
    /// records them ([`recorded`]), and runs inside it what the test runs there.
    pub(super) fn sent(sel: &'static str, args: impl FnOnce() -> String) {
        RECORDING.with(|r| {
            if let Some(r) = r.borrow_mut().as_mut() {
                r.calls.push(format!("{sel} {}", args()));
                (r.inside)(r.calls.len() - 1);
            }
        });
    }

    /// What `run` does, `inside` run inside each runtime call of draws' commands by its place:
    /// each call, then how `run` ended (its answer, or the site of its death and the death's
    /// line).
    fn recorded(
        inside: Box<dyn FnMut(usize)>,
        run: impl FnOnce() -> RaiseResult<()>,
    ) -> (Vec<String>, String) {
        let calls = Vec::new();
        RECORDING.with(|r| *r.borrow_mut() = Some(Recording { calls, inside }));
        *crate::answer::LAST_LINE.lock() = None;
        let ended = match catch_death(run) {
            Ok(answer) => format!("{answer:?}"),
            Err(site) => format!("{site:?}: {:?}", crate::answer::LAST_LINE.lock().take()),
        };
        let recording = RECORDING.with(|r| r.borrow_mut().take());
        (recording.map(|r| r.calls).unwrap_or_default(), ended)
    }

    /// Draws sent together ([`VulkanRenderEncoder::encode_draws`]) make the runtime calls each
    /// makes alone, with the same arguments, in the same order, and end as they do alone: five
    /// draws of a pass quiet throughout draw; a notice of [`crate::answer::DIES`] the runtime
    /// announces inside the third draw's `setVertexBuffer:offset:atIndex:` (the copy of
    /// `answer`'s `a_call_that_unwinds_lets_its_notice_go`, run inside the call) dies at that
    /// call; so do an offset past the buffer the runtime refuses there and an index past `u32`;
    /// a third draw of points the state sizes none of, or of no primitive type, dies at its draw;
    /// and one of type 9, an empty one and one of no buffer take their own paths and go on
    #[test]
    fn draws_sent_together_make_the_calls_each_makes_alone() {
        let d = dev();
        let state = fixture_state::<VulkanPlatform>(&d, "vert:tri_vert", "frag:tri_frag");
        let vertices = buffer::<VulkanPlatform>(&d, 4096, 0, &[0; 4096]);
        let plain = VertexDraw::<VulkanPlatform> {
            buffer: Some(&vertices),
            offset: 32,
            index: 0,
            primitive_type: 3,
            vertex_start: 6,
            vertex_count: 3,
        };
        /// Five draws like `plain`, the third `third`.
        fn five<'a>(
            plain: VertexDraw<'a, VulkanPlatform>,
            third: VertexDraw<'a, VulkanPlatform>,
        ) -> [VertexDraw<'a, VulkanPlatform>; 5] {
            let at = |k: u64| VertexDraw {
                offset: 16 * k,
                vertex_start: 3 * k,
                ..plain
            };
            [at(0), at(1), third, at(3), at(4)]
        }
        // The runtime's notice inside the call at `place`: the third draw's bind is the fifth.
        let notice_at = |place: usize| -> Box<dyn FnMut(usize)> {
            let desc = platform().texture_descriptor(&TextureDesc {
                pixel_format: 550,
                width: 16,
                height: 4,
                usage: 0x3,
                ..TextureDesc::default()
            });
            let t = d.new_texture(OnFault::Return, &desc).unwrap().unwrap();
            let queue = d.new_command_queue().unwrap();
            let cb = queue.raw.command_buffer().unwrap();
            let mut blit = cb.blit_command_encoder().unwrap();
            Box::new(move |at| {
                if at == place {
                    let texels = t.copied("the copy");
                    let copied = blit.copy_from_texture_to_texture(
                        texels,
                        0,
                        0,
                        rt::Origin::new(0, 0, 0),
                        rt::Size::new(4, 1, 1),
                        texels,
                        0,
                        0,
                        rt::Origin::new(8, 0, 0),
                    );
                    assert!(copied.is_ok(), "{copied:?}");
                }
                let _ = (&queue, &cb);
            })
        };
        let nothing = || -> Box<dyn FnMut(usize)> { Box::new(|_| {}) };
        let cases = [
            ("a quiet pass", plain, None, None),
            ("a notice", plain, Some(4), Some(SET_VERTEX_BUFFER)),
            (
                "an offset past the buffer",
                VertexDraw {
                    offset: 8192,
                    ..plain
                },
                None,
                Some(SET_VERTEX_BUFFER),
            ),
            (
                "an index past u32",
                VertexDraw {
                    index: 1 << 32,
                    ..plain
                },
                None,
                Some(SET_VERTEX_BUFFER),
            ),
            (
                "points",
                VertexDraw {
                    primitive_type: 0,
                    ..plain
                },
                None,
                Some(DRAW_PRIMITIVES),
            ),
            (
                "no type",
                VertexDraw {
                    primitive_type: 99,
                    ..plain
                },
                None,
                Some(DRAW_PRIMITIVES),
            ),
            (
                "type 9",
                VertexDraw {
                    primitive_type: 9,
                    ..plain
                },
                None,
                None,
            ),
            (
                "an empty draw",
                VertexDraw {
                    vertex_count: 0,
                    ..plain
                },
                None,
                None,
            ),
            (
                "no buffer",
                VertexDraw {
                    buffer: None,
                    ..plain
                },
                None,
                None,
            ),
        ];
        for (case, third, notice, dies) in cases {
            let draws = five(plain, third);
            let mut ends = Vec::new();
            for together in [false, true] {
                // The command buffer stays, as the walk's does: an encoder whose command buffer
                // went binds nothing for its draws, and takes no draws together.
                let (p, d) = (platform(), dev());
                let (_dst, pass) = pass_of(&p, &d, (4, 4));
                let q = d.new_command_queue().unwrap();
                let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
                let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
                let set = RenderCmd::SetRenderPipelineState {
                    pipeline_state: Some(&state),
                };
                e.encode(OnFault::Die, set).unwrap();
                let inside = notice.map_or_else(nothing, notice_at);
                let run = || match together {
                    true => e.encoding(|| e.encode_draws(OnFault::Return, &draws)),
                    false => e.encoding(|| encode_each(&e, OnFault::Return, &draws)),
                };
                let quiet = QUIET_RUNS.get();
                ends.push(recorded(inside, run));
                assert_eq!(QUIET_RUNS.get() - quiet, usize::from(together), "{case}");
                let _ = catch_death(|| e.encode(OnFault::Die, RenderCmd::EndEncoding));
                drop((e, cb));
            }
            assert_eq!(ends[0], ends[1], "{case}");
            let (calls, ended) = &ends[0];
            match dies {
                Some(sel) => {
                    assert!(ended.contains(&format!("file: {sel:?}")), "{case}: {ended}");
                    let sets = (calls.iter()).filter(|c| c.starts_with(SET_VERTEX_BUFFER));
                    assert!(sets.count() >= 2, "{case}: {calls:?}");
                }
                None => {
                    assert_eq!(ended, "Ok(())", "{case}");
                    assert!(calls.len() >= 9, "{case}: {calls:?}");
                }
            }
        }
    }

    use reims_vgpu_core::display::pipeline::{DisplayPipelineCache, PipelineKey};
    use reims_vgpu_core::platform::blit::ResourceRef;
    use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder};
    use reims_vgpu_core::platform::command::{CommandBuffer, CommandQueue};
    use reims_vgpu_core::platform::descriptor::RenderPassDescriptor as _;
    use reims_vgpu_core::platform::device::GpuDevice;
    use reims_vgpu_core::platform::resource::{Buffer, Texture};
    use reims_vgpu_core::platform::types::{
        ClearColor, MtlOrigin, MtlRegion, MtlScissorRect, MtlSize, NsRange,
    };
    use reims_vgpu_core::platform::values::{ColorAttachmentSet, TextureDesc};
    use reims_vgpu_core::platform::{Handle, HostRange, Platform};
    use reims_vgpu_core::testkit::capture::catch_death;

    use super::*;
    use crate::gpu::device::tests::dev;
    use crate::gpu::resource::VulkanTexture;
    use crate::platform::tests::platform;

    const BGRA8: u64 = 80;
    const RGBA16F: u64 = 115;
    /// `MTLTextureUsageShaderRead` and `RenderTarget`.
    const READ: u64 = 1;
    const TARGET: u64 = 4;
    /// `MTLLoadActionClear`, `MTLStoreActionStore`, `MTLPrimitiveTypeTriangleStrip`.
    const CLEAR: u64 = 2;
    const STORE: u64 = 1;
    const STRIP: u64 = 4;

    /// The native-endian bytes of `floats`.
    fn bytes(floats: &[f32]) -> Vec<u8> {
        floats.iter().flat_map(|f| f.to_ne_bytes()).collect()
    }

    /// A buffer of `length` bytes of `options`, holding `contents` from its start.
    fn buffer<P: Platform>(d: &P::Device, length: u64, options: u64, contents: &[u8]) -> P::Buffer {
        let b = d
            .new_buffer_of_length(OnFault::Die, length, options)
            .unwrap()
            .unwrap();
        // SAFETY: a CPU-visible buffer of at least `contents.len()` bytes, which no command uses
        // yet.
        unsafe {
            std::ptr::copy_nonoverlapping(
                contents.as_ptr(),
                b.contents().as_ptr::<u8>(),
                contents.len(),
            )
        };
        b
    }

    /// A 2D texture of `format`, `usage`, `width` × `height`, holding `texels` (rows of
    /// `bytes_per_row`).
    fn texture<P: Platform>(
        p: &P,
        d: &P::Device,
        format: u64,
        usage: u64,
        (width, height): (u64, u64),
        texels: Option<(&[u8], u64)>,
    ) -> P::Texture {
        let desc = p.texture_descriptor(&TextureDesc {
            pixel_format: format,
            width,
            height,
            usage,
            ..TextureDesc::default()
        });
        let t = d.new_texture(OnFault::Die, &desc).unwrap().unwrap();
        if let Some((texels, row)) = texels {
            let region = MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(width, height, 1));
            t.replace_region(region, 0, 0, HostRange::of_slice(texels), row, 0)
                .unwrap();
        }
        t
    }

    /// Sixteen vertices at scattered positions inside clip space, as the present pass's
    /// position (float4) and texture-coordinate (float2) buffers hold them.
    fn scattered() -> (Vec<f32>, Vec<f32>) {
        let mut state = 7919u32;
        let mut next = move || {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
            f32::from(u16::try_from((state >> 8) & 0xffff).unwrap()) / 65536.0
        };
        let (mut pos, mut tex) = (Vec::new(), Vec::new());
        for _ in 0..16 {
            pos.extend([next() * 1.9 - 0.95, next() * 1.9 - 0.95, 0.0, 1.0]);
            tex.extend([next(), next()]);
        }
        (pos, tex)
    }
    /// `MTLIndexTypeUInt16`.
    const U16: u64 = 0;

    /// A pass holds only the binds that can decide what a draw asks of what it may write
    /// (`matters`), and each distinct one made resident for writing once, however often it is
    /// made resident: a draw asks the same of them as of all it bound
    #[test]
    fn a_pass_holds_only_the_binds_a_draw_asks_of() {
        let plain = Bound {
            meets: true,
            ..Bound::default()
        };
        let dropped = Bound {
            dropped: true,
            ..plain.clone()
        };
        let kept = Bound {
            kept: Kept::Always,
            ..plain.clone()
        };
        let mut r = Reach::default();
        for slot in 0..8 {
            r.bind_one(FRAGMENT, false, slot, plain.clone());
        }
        r.bind_one(VERTEX, true, 3, plain.clone());
        assert!((r.textures.iter().chain(&r.buffers)).all(BTreeMap::is_empty));
        assert!(!r.unsent_dropped() && r.kept(None).is_none());
        for _ in 0..1000 {
            let textures = vec![plain.clone(), dropped.clone()];
            r.used(textures, vec![plain.clone(), kept.clone()]);
        }
        assert_eq!((r.used_textures.len(), r.used_buffers.len()), (1, 1));
        assert!(r.unsent_dropped());
        assert!(matches!(r.kept(None), Some(Whether::Writes)));
        assert!(matches!(&r.unsent(None)[..], [(Whether::Writes, m)] if *m == dropped));
        r.bind_one(FRAGMENT, false, 2, dropped.clone());
        assert_eq!(r.textures[FRAGMENT].get(&2), Some(&dropped));
        r.bind_one(FRAGMENT, false, 2, plain);
        assert!(r.textures[FRAGMENT].is_empty());
    }

    /// A pass is quiet while it holds nothing a draw asks of, as a pass that binds only plain
    /// memory does: its draws then ask nothing ([`VulkanRenderEncoder::issues`]). A bind or a
    /// resident use that decides something ends it, and a bind of plain memory in that slot
    /// brings it back
    #[test]
    fn a_pass_is_quiet_while_it_holds_nothing_a_draw_asks_of() {
        let plain = Bound {
            meets: true,
            ..Bound::default()
        };
        let dropped = Bound {
            dropped: true,
            ..plain.clone()
        };
        let mut r = Reach::default();
        assert!(r.quiet());
        r.bind_one(FRAGMENT, false, 0, plain.clone());
        r.bind_one(VERTEX, true, 1, plain.clone());
        assert!(r.quiet());
        r.bind_one(FRAGMENT, false, 0, dropped.clone());
        assert!(!r.quiet() && r.unsent_dropped());
        r.bind_one(FRAGMENT, false, 0, plain);
        assert!(r.quiet());
        r.used(vec![dropped], Vec::new());
        assert!(!r.quiet() && r.unsent_dropped());
    }

    /// An open encoder on a target, on this platform.
    fn open_encoder() -> (VulkanRenderEncoder, VulkanTexture) {
        let (p, d) = (platform(), dev());
        let (dst, pass) = pass_of(&p, &d, (4, 4));
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        (e, dst)
    }

    /// What the runtime has no form for, and what it refuses where Metal goes on, dies at the
    /// selector
    #[test]
    fn render_refusals_die() {
        let (e, dst) = open_encoder();
        let die = |cmd: RenderCmd<'_, VulkanPlatform>| {
            catch_death(|| e.encode(OnFault::Return, cmd))
                .unwrap_err()
                .file
        };
        assert_eq!(
            die(RenderCmd::DrawPrimitives {
                primitive_type: 9,
                vertex_start: 0,
                vertex_count: 3,
            }),
            "drawPrimitives:vertexStart:vertexCount:"
        );
        assert_eq!(
            die(RenderCmd::SetFragmentTextures {
                textures: &[Some(&dst)],
                range: NsRange::new(0, 2),
            }),
            "setFragmentTextures:withRange:"
        );
        assert_eq!(
            die(RenderCmd::SetColorResolveTexture {
                texture: None,
                slice: 0,
                depth_plane: 0,
                level: 0,
                y_invert: false,
                index: 8,
            }),
            "setColorResolveTexture:slice:depthPlane:level:yInvert:atIndex:"
        );
        assert_eq!(
            die(RenderCmd::SetFragmentBufferOffset {
                offset: 0,
                index: 99,
            }),
            "setFragmentBufferOffset:atIndex:"
        );
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
    }

    /// The tests' render functions that read `[[barycentric_coord]]` (`barycentric.metal`).
    const BARYCENTRIC: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/barycentric.metallib"));

    /// The tests' render functions that read `[[vertex_id]]` (`vertex_ids.metal`).
    const VERTEX_IDS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/vertex_ids.metallib"));

    /// The tests' functions that use SIMD-group operations in each stage (`simd_stages.metal`).
    const SIMD_STAGES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/simd_stages.metallib"));

    /// A function of the suite's fixtures on `d`: `name` of `fixtures/shaders/<file>.metallib`,
    /// given as `file:name`, or as `name` where the two are one; of the files `barycentric`,
    /// `vertex_ids` and `simd_stages`, the crate's own [`BARYCENTRIC`], [`VERTEX_IDS`] and
    /// [`SIMD_STAGES`].
    fn fixture<P: Platform>(d: &P::Device, function: &str) -> P::Function {
        use reims_vgpu_core::platform::resource::Library as _;
        let (file, name) = function.split_once(':').unwrap_or((function, function));
        let bytes = if file == "barycentric" {
            BARYCENTRIC.to_vec()
        } else if file == "vertex_ids" {
            VERTEX_IDS.to_vec()
        } else if file == "simd_stages" {
            SIMD_STAGES.to_vec()
        } else {
            let path =
                crate::gpu::device::tests::fixture_shaders().join(format!("{file}.metallib"));
            std::fs::read(path).unwrap()
        };
        let library = d.new_library(&bytes).unwrap().unwrap().unwrap();
        library.new_function(OnFault::Die, name).unwrap().unwrap()
    }

    /// The state of the fixtures `vertex` and `fragment` on `d`, for a BGRA8 target.
    fn fixture_state<P: Platform>(
        d: &P::Device,
        vertex: &str,
        fragment: &str,
    ) -> P::RenderPipeline {
        use reims_vgpu_core::platform::values::RenderPipelineDesc;
        let (v, f) = (fixture::<P>(d, vertex), fixture::<P>(d, fragment));
        d.new_render_pipeline_state(&RenderPipelineDesc {
            label: None,
            vertex: &v,
            fragment: &f,
            color0_pixel_format: BGRA8,
            resource_index: None,
        })
        .unwrap()
        .unwrap()
    }

    /// The colours of [`flat_pass`]'s vertices: sixteen, each its own.
    fn vertex_colours() -> Vec<f32> {
        (0..16u8)
            .flat_map(|i| {
                let c = |k: u8| f32::from(k) / 3.0;
                [c(i % 4), c((i / 4) % 4), c((i * 3) % 4), 1.0]
            })
            .collect()
    }

    /// [`scattered`]'s positions in a buffer, then [`vertex_colours`] in another, padded to `n`
    /// vertices, each past the sixteenth white and off the target.
    fn flat_vertices<P: Platform>(d: &P::Device, n: usize) -> (P::Buffer, P::Buffer) {
        let (mut pos, _) = scattered();
        let mut colours = vertex_colours();
        pos.resize(n * 4, 0.0);
        colours.resize(n * 4, 1.0);
        for vertex in pos.as_chunks_mut::<4>().0.iter_mut().skip(16) {
            *vertex = [3.0, 3.0, 0.0, 1.0];
        }
        let length = |data: &[f32]| u64::try_from(data.len() * 4).unwrap();
        (
            buffer::<P>(d, length(&pos), 0, &bytes(&pos)),
            buffer::<P>(d, length(&colours), 0, &bytes(&colours)),
        )
    }

    /// The pixels of a 16 by 16 BGRA8 target, cleared to black, after `cmds` and `EndEncoding`
    /// in a render encoder of it.
    fn pass_pixels<P: Platform>(p: &P, d: &P::Device, cmds: Vec<RenderCmd<'_, P>>) -> Vec<[u8; 4]> {
        let (w, h) = (16, 16);
        let (dst, pass) = pass_of(p, d, (w, h));
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        for cmd in cmds.into_iter().chain([RenderCmd::EndEncoding]) {
            e.encode(OnFault::Die, cmd).unwrap();
        }
        let out = buffer::<P>(d, w * h * 4, 0, &[]);
        let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
        blit.encode(
            OnFault::Die,
            BlitCmd::CopyTextureToBuffer {
                source_texture: &dst,
                source_slice: 0,
                source_level: 0,
                source_origin: MtlOrigin::new(0, 0, 0),
                source_size: MtlSize::new(w, h, 1),
                destination_buffer: &out,
                destination_offset: 0,
                destination_bytes_per_row: w * 4,
                destination_bytes_per_image: w * h * 4,
            },
        )
        .unwrap();
        blit.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert!(cb.error().is_none(), "{:?}", cb.error());
        let n = usize::try_from(w * h * 4).unwrap();
        // SAFETY: a Shared buffer's bytes, read after the blit completed.
        let texels = unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), n) };
        texels.as_chunks::<4>().0.to_vec()
    }

    /// A texture of protection options 0x1, as a macOS 26 guest's WindowServer on the x86 device
    /// asks for once protected video is on screen (`BGRA8Unorm` and `RGBA16Float`, usage 0x5,
    /// `Private`), is made by `newTextureWithDescriptor:` and `newSharedTextureWithDescriptor:`
    /// as without them, each relayed and counted, where Metal answers nil
    /// ([`crate::gpu::desc::TextureFields::made_unprotected`]): it answers protection options 0,
    /// and a pass clears it and a blit copies it out. In strict mode each creator dies there
    #[test]
    fn a_protected_texture_is_made_unprotected() {
        let (p, d) = (platform(), dev());
        let (w, h) = (64, 16);
        let relayed = || relayed_of(crate::answer::Inexact::ProtectedContent);
        for (format, texel) in [(BGRA8, 4), (RGBA16F, 8)] {
            let desc = p.texture_descriptor(&TextureDesc {
                pixel_format: format,
                width: w,
                height: h,
                usage: READ | TARGET,
                resource_options: Some(0x20),
                protection_options: Some(0x1),
                ..TextureDesc::default()
            });
            type Made = RaiseResult<Option<VulkanTexture>>;
            let creators: [(&str, &dyn Fn() -> Made); 2] = [
                ("newTextureWithDescriptor:", &|| {
                    d.new_texture(OnFault::Return, &desc)
                }),
                ("newSharedTextureWithDescriptor:", &|| {
                    d.new_shared_texture(&desc)
                }),
            ];
            let mut made = Vec::new();
            for (sel, make) in creators {
                *crate::answer::LAST_LINE.lock() = None;
                let site = catch_death(|| drop(make())).unwrap_err();
                let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
                assert_eq!(site.file, sel, "{line}");
                assert!(line.contains("protection options 0x1"), "{line}");
                let before = relayed();
                let t = crate::answer::relaxed(make).unwrap();
                let t = t.unwrap_or_else(|| panic!("{format}: {sel} made nothing"));
                assert!(relayed() > before, "{format}: {sel}");
                assert_eq!(t.protection_options(), 0, "{format}: {sel}");
                made.push(t);
            }
            let pass = p.render_pass_descriptor();
            pass.set_color_attachment(
                OnFault::Die,
                0,
                &ColorAttachmentSet {
                    texture: Some(Some(&made[0])),
                    load_action: Some(CLEAR),
                    store_action: Some(STORE),
                    clear_color: Some(ClearColor {
                        red: 0.25,
                        green: 0.5,
                        blue: 0.75,
                        alpha: 1.0,
                    }),
                    ..ColorAttachmentSet::default()
                },
            )
            .unwrap();
            let q = d.new_command_queue().unwrap();
            let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
            let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
            e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
            let out = buffer::<VulkanPlatform>(&d, w * h * texel, 0, &[]);
            let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
            blit.encode(
                OnFault::Die,
                BlitCmd::CopyTextureToBuffer {
                    source_texture: &made[0],
                    source_slice: 0,
                    source_level: 0,
                    source_origin: MtlOrigin::new(0, 0, 0),
                    source_size: MtlSize::new(w, h, 1),
                    destination_buffer: &out,
                    destination_offset: 0,
                    destination_bytes_per_row: w * texel,
                    destination_bytes_per_image: w * h * texel,
                },
            )
            .unwrap();
            blit.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
            cb.commit(OnFault::Die).unwrap();
            cb.wait_until_completed(OnFault::Die).unwrap();
            assert!(cb.error().is_none(), "{format}: {:?}", cb.error());
            let n = usize::try_from(w * h * texel).unwrap();
            // SAFETY: a Shared buffer's bytes, read after the blit completed.
            let bytes = unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), n) };
            // 0.25, 0.5, 0.75 and 1.0: BGRA8Unorm's bytes, and RGBA16Float's halves. BGRA8's
            // 0.5 is a tie of its codes (127.5), which a device not held to Metal's codes may
            // store as either neighbour ([`ties_as_metals`]).
            let cleared: &[u8] = match format {
                BGRA8 => &[0xbf, 0x80, 0x40, 0xff],
                _ => &[0x00, 0x34, 0x00, 0x38, 0x00, 0x3a, 0x00, 0x3c],
            };
            let tie: &[u8] = match format {
                BGRA8 if !ties_as_metals(&d) => &[0xbf, 0x7f, 0x40, 0xff],
                _ => cleared,
            };
            assert!(
                bytes
                    .chunks(cleared.len())
                    .all(|t| t == cleared || t == tie),
                "{format}"
            );
        }
    }

    /// The fixtures of a state whose colour is each vertex's, flat.
    const FLAT: (&str, &str) = ("rdec_flat_v", "rdec_flat_f");

    /// The functions of a state that sizes its points, writing `[[point_size]]`, and reads a
    /// flat input; its vertex function reads no buffer.
    const SIZED: (&str, &str) = ("vertex_ids:sized_vertex", "vertex_ids:ids_fragment");

    /// A draw of primitive type 9 encodes nothing, and the command buffer fails as Metal's GPU
    /// fails it: with `MTLCommandBufferErrorTimeout` (status 5), the blit before the pass run,
    /// neither the pass's clear nor the blit after it, an event signal after them firing. The
    /// commands after the draw in its encoder are taken
    #[test]
    fn a_type_9_draw_fails_its_command_buffer_as_metals_gpu_does() {
        use reims_vgpu_core::platform::event::EventRef;
        use reims_vgpu_core::platform::types::NsRange;

        let (p, d) = (platform(), dev());
        let (dst, pass) = pass_of(&p, &d, (4, 4));
        crate::gpu::resource::tests::fill(&dst, &[0x11; 64]);
        let state = fixture_state::<VulkanPlatform>(&d, "vert:tri_vert", "frag:tri_frag");
        let (before, after) = (
            buffer::<VulkanPlatform>(&d, 16, 0, &[0; 16]),
            buffer::<VulkanPlatform>(&d, 16, 0, &[0; 16]),
        );
        let event = d.new_shared_event().unwrap();
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let fill = |b, value| {
            let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
            let range = NsRange::new(0, 16);
            let fill = BlitCmd::FillBuffer {
                buffer: b,
                range,
                value,
            };
            blit.encode(OnFault::Die, fill).unwrap();
            blit.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
        };
        fill(&before, 0x5a);
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        send_all(
            &e,
            vec![
                RenderCmd::SetRenderPipelineState {
                    pipeline_state: Some(&state),
                },
                three(9, None),
                three(3, None),
                RenderCmd::EndEncoding,
            ],
        );
        fill(&after, 0x6b);
        cb.encode_signal(Some(EventRef::Shared(&event)), 5).unwrap();
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert_eq!(cb.status().0, 5);
        assert_eq!(cb.error().map(|e| e.code), Some(2));
        let bytes = |b: &VulkanBuffer| {
            // SAFETY: a Shared buffer's bytes, read after its command buffer completed.
            unsafe { std::slice::from_raw_parts(b.contents().as_ptr::<u8>(), 16) }.to_vec()
        };
        assert_eq!(
            (bytes(&before), bytes(&after)),
            (vec![0x5a; 16], vec![0; 16])
        );
        assert_eq!(crate::gpu::resource::tests::read_back(&dst), vec![0x11; 64]);
        assert_eq!(event.signaled_value(), 5);
    }

    /// What a command buffer of a new queue shows on `p`'s device `d` whose render pass, over a
    /// 4 by 4 target filled with 0x11, binds a memoryless 4 by 4 `RGBA8Unorm` texture and a
    /// sampler at fragment slot 0 and draws a triangle over the target with the fixtures'
    /// `tri_vert` and `fragment`, a blit before it filling a buffer with 0x5a and one after it
    /// another with 0x6b, then an event signal: its status and error, the first byte of each
    /// buffer, whether the signal fired, and the target's first texel, read through a new queue.
    pub(crate) fn memoryless_slot_pass<P: Platform>(
        p: &P,
        d: &P::Device,
        fragment: &str,
    ) -> String {
        use reims_vgpu_core::platform::event::{EventRef, SharedEvent as _};
        use reims_vgpu_core::platform::render::RenderEncoder as _;

        /// `MTLResourceStorageModeMemoryless`, and `MTLTextureUsageShaderWrite`.
        const MEMORYLESS: u64 = 0x30;
        const WRITE: u64 = 2;
        let (dst, pass) = pass_of(p, d, (4, 4));
        let region = MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(4, 4, 1));
        dst.replace_region(region, 0, 0, HostRange::of_slice(&[0x11; 64]), 16, 0)
            .unwrap();
        let desc = p.texture_descriptor(&TextureDesc {
            pixel_format: 70,
            width: 4,
            height: 4,
            usage: TARGET | READ | WRITE,
            resource_options: Some(MEMORYLESS),
            ..TextureDesc::default()
        });
        let memoryless = d.new_texture(OnFault::Die, &desc).unwrap().unwrap();
        let sampler = d
            .new_sampler_state(&reims_vgpu_core::platform::values::SamplerDesc::default())
            .unwrap()
            .unwrap();
        let state = fixture_state::<P>(d, "vert:tri_vert", fragment);
        let (before, after) = (
            buffer::<P>(d, 16, 0, &[0; 16]),
            buffer::<P>(d, 16, 0, &[0; 16]),
        );
        let event = d.new_shared_event().unwrap();
        let cb = (d.new_command_queue().unwrap())
            .command_buffer(OnFault::Die)
            .unwrap()
            .unwrap();
        let fill = |b: &P::Buffer, value| {
            let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
            let range = NsRange::new(0, 16);
            let fill = BlitCmd::FillBuffer {
                buffer: b,
                range,
                value,
            };
            blit.encode(OnFault::Die, fill).unwrap();
            blit.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
        };
        fill(&before, 0x5a);
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        for cmd in [
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&state),
            },
            RenderCmd::SetFragmentTextures {
                textures: &[Some(&memoryless)],
                range: NsRange::new(0, 1),
            },
            RenderCmd::SetFragmentSamplerStates {
                samplers: &[Some(&sampler)],
                range: NsRange::new(0, 1),
            },
            RenderCmd::DrawPrimitives {
                primitive_type: 3,
                vertex_start: 0,
                vertex_count: 3,
            },
            RenderCmd::EndEncoding,
        ] {
            e.encode(OnFault::Die, cmd).unwrap();
        }
        fill(&after, 0x6b);
        cb.encode_signal(Some(EventRef::Shared(&event)), 5).unwrap();
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        let out = buffer::<P>(d, 4, 0, &[]);
        let read = (d.new_command_queue().unwrap())
            .command_buffer(OnFault::Die)
            .unwrap()
            .unwrap();
        let blit = read.blit_encoder(OnFault::Die).unwrap().unwrap();
        for cmd in [
            BlitCmd::CopyTextureToBuffer {
                source_texture: &dst,
                source_slice: 0,
                source_level: 0,
                source_origin: MtlOrigin::new(0, 0, 0),
                source_size: MtlSize::new(1, 1, 1),
                destination_buffer: &out,
                destination_offset: 0,
                destination_bytes_per_row: 4,
                destination_bytes_per_image: 4,
            },
            BlitCmd::EndEncoding,
        ] {
            blit.encode(OnFault::Die, cmd).unwrap();
        }
        read.commit(OnFault::Die).unwrap();
        read.wait_until_completed(OnFault::Die).unwrap();
        // SAFETY: Shared buffers' bytes, read after their command buffers completed.
        let first = |b: &P::Buffer| unsafe { *b.contents().as_ptr::<u8>() };
        // SAFETY: as above, the 4 bytes of `out`.
        let texel = unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), 4) };
        format!(
            "status {} error {:?} before {:#x} after {:#x} signal {} texel {texel:x?}",
            cb.status().0,
            cb.error().map(|e| e.code),
            first(&before),
            first(&after),
            event.wait_until_signaled_value(5, 1),
        )
    }

    /// Whether `d` stores Metal's code at a tie of a unorm format's codes, as every Mac does
    /// (measured: M3 Max, AMD 5300M, Intel 630, in `BGRA8Unorm` and `RGBA8Unorm`): a device the
    /// runtime holds to Metal's codes (`VkProperty::MetalColorWrites`, MoltenVK on Apple's
    /// GPUs). Any other stores either neighbour by its own conversion (0x7f for 0.5 on an RTX
    /// 5080), which the runtime records as `device-color-conversion`
    /// (`PrecisionNote::DeviceColorConversion`)
    pub(crate) fn ties_as_metals(d: &crate::gpu::device::VulkanDevice) -> bool {
        (d.raw.compile_device().profile().enabled().properties)
            .contains(&metal2vulkan::device::VkProperty::MetalColorWrites)
    }

    /// [`memoryless_slot_pass`]'s lines on Metal, by its fragment function (measured on the M3
    /// Max, macOS 27.0.1, `memoryless_slots_fail_as_metals`): `tri_frag`, which uses no texture,
    /// draws and completes; `render_ftex`, which samples slot 0, fails the command buffer with
    /// `MTLCommandBufferErrorInvalidResource` (status 5, code 9), the blit before the pass run and
    /// neither the pass's clear nor the blit after it, the signal not firing.
    pub(crate) const METALS_MEMORYLESS_SLOT: [(&str, &str); 2] = [
        (
            "frag:tri_frag",
            "status 4 error None before 0x5a after 0x6b signal true texel [40, 80, ff, ff]",
        ),
        (
            "render_ftex",
            "status 5 error Some(9) before 0x5a after 0x0 signal false texel [11, 11, 11, 11]",
        ),
    ];

    /// A memoryless texture bound at a fragment slot is taken, as plain Metal takes it, and fails
    /// the command buffer only where the draw's function uses the slot, as Metal's
    /// ([`METALS_MEMORYLESS_SLOT`])
    ///
    /// `tri_frag`'s green, 0.5, is a tie of `BGRA8Unorm`'s codes, which a device not held to
    /// Metal's codes may store as 0x7f ([`ties_as_metals`])
    #[test]
    fn a_memoryless_texture_at_a_fragment_slot_fails_only_a_draw_that_uses_it() {
        let (p, d) = (platform(), dev());
        for (fragment, metals) in METALS_MEMORYLESS_SLOT {
            let got = memoryless_slot_pass(&p, &d, fragment);
            let tie = metals.replace("texel [40, 80,", "texel [40, 7f,");
            assert!(
                got == metals || (!ties_as_metals(&d) && got == tie),
                "{fragment}: {got}"
            );
        }
    }

    /// A command buffer of a draw of primitive type 9 is in flight as long as Metal's: its signal
    /// after the hang fires, and it completes, no sooner than [`HANG`] after its commit. A
    /// command buffer committed on its queue meanwhile runs nothing and completes without an
    /// error, its event signal firing, as Metal's; one that encoded an event wait the hang's end
    /// had not met runs, as Metal's; one of another queue runs then, and one of its queue once it
    /// completed (`metal_runs_after_a_type_9_hang_here` measures Metal's,
    /// `later_command_buffers_of_a_hang_run_as_metals` compares). The command buffers of its
    /// queue are made before the hang's commit and committed right after it, within [`HANG`]
    /// however long the other queue's takes to run under load
    #[test]
    fn a_type_9_hang_is_in_flight_as_long_as_metals() {
        use reims_vgpu_core::platform::event::EventRef;
        use reims_vgpu_core::platform::os::Clock as _;
        use reims_vgpu_core::platform::types::NsRange;

        let (p, d) = (platform(), dev());
        let (_dst, pass) = pass_of(&p, &d, (4, 4));
        let state = fixture_state::<VulkanPlatform>(&d, "vert:tri_vert", "frag:tri_frag");
        let (q, other) = (
            d.new_command_queue().unwrap(),
            d.new_command_queue().unwrap(),
        );
        let event = d.new_shared_event().unwrap();
        let hang = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = hang.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        send_all(
            &e,
            vec![
                RenderCmd::SetRenderPipelineState {
                    pipeline_state: Some(&state),
                },
                three(9, None),
                RenderCmd::EndEncoding,
            ],
        );
        hang.encode_signal(Some(EventRef::Shared(&event)), 1)
            .unwrap();
        // A command buffer of `q` that fills a buffer with 0x5a, and the buffer.
        let filling = |q: &crate::gpu::command::VulkanCommandQueue| {
            let b = buffer::<VulkanPlatform>(&d, 16, 0, &[0; 16]);
            let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
            let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
            let fill = BlitCmd::FillBuffer {
                buffer: &b,
                range: NsRange::new(0, 16),
                value: 0x5a,
            };
            blit.encode(OnFault::Die, fill).unwrap();
            blit.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
            (cb, b)
        };
        let runs = |q| {
            let (cb, b) = filling(q);
            cb.commit(OnFault::Die).unwrap();
            cb.wait_until_completed(OnFault::Die).unwrap();
            assert_eq!(cb.status().0, 4);
            assert!(cb.error().is_none(), "{:?}", cb.error());
            // SAFETY: a Shared buffer's bytes, read after its command buffer completed.
            let first = unsafe { *b.contents().as_ptr::<u8>() };
            assert_eq!(first, 0x5a);
        };
        let clock = crate::os::OsClock::new();
        let since = |start| clock.now_mono().saturating_duration_since(start);
        let (behind, b) = filling(&q);
        let signalled = d.new_shared_event().unwrap();
        behind
            .encode_signal(Some(EventRef::Shared(&signalled)), 3)
            .unwrap();
        let (waits, waited) = filling(&q);
        waits
            .encode_wait(Some(EventRef::Shared(&signalled)), 1)
            .unwrap();
        let started = clock.now_mono();
        hang.commit(OnFault::Die).unwrap();
        behind.commit(OnFault::Die).unwrap();
        waits.commit(OnFault::Die).unwrap();
        runs(&other);
        assert!(event.wait_until_signaled_value(1, std::time::Duration::from_secs(5)));
        assert!(since(started) >= HANG);
        hang.wait_until_completed(OnFault::Die).unwrap();
        assert!(since(started) >= HANG);
        assert_eq!(
            (hang.status().0, hang.error().map(|e| e.code)),
            (5, Some(2))
        );
        behind.wait_until_completed(OnFault::Die).unwrap();
        assert_eq!(behind.status().0, 4);
        assert!(behind.error().is_none(), "{:?}", behind.error());
        // SAFETY: a Shared buffer's bytes, read after its command buffer completed.
        let first = unsafe { *b.contents().as_ptr::<u8>() };
        assert_eq!(first, 0, "it ran");
        assert_eq!(signalled.signaled_value(), 3);
        waits.wait_until_completed(OnFault::Die).unwrap();
        assert_eq!(waits.status().0, 4);
        assert!(waits.error().is_none(), "{:?}", waits.error());
        // SAFETY: a Shared buffer's bytes, read after its command buffer completed.
        let first = unsafe { *waited.contents().as_ptr::<u8>() };
        assert_eq!(first, 0x5a, "it ran nothing");
        runs(&q);
    }

    /// A command buffer of a draw of primitive type 9 over indices, at which Metal's GPU faults,
    /// is in flight as long as Metal's: it completes no sooner than [`FAULT`] after its commit,
    /// with `MTLCommandBufferErrorPageFault`, its signal after the fault firing. A command buffer
    /// committed on its queue right after it runs and completes after it, and one of another
    /// queue runs meanwhile, as Metal's (the runtime's measurement, `command_buffer/hang.rs`)
    #[test]
    fn a_type_9_fault_is_in_flight_as_long_as_metals() {
        use reims_vgpu_core::platform::event::EventRef;
        use reims_vgpu_core::platform::os::Clock as _;
        use reims_vgpu_core::platform::types::NsRange;

        let (p, d) = (platform(), dev());
        let (_dst, pass) = pass_of(&p, &d, (4, 4));
        let state = fixture_state::<VulkanPlatform>(&d, "vert:tri_vert", "frag:tri_frag");
        let indices = buffer::<VulkanPlatform>(&d, 16, 0, &u32s(&[0, 1, 2]));
        let (q, other) = (
            d.new_command_queue().unwrap(),
            d.new_command_queue().unwrap(),
        );
        let event = d.new_shared_event().unwrap();
        let fault = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = fault.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        send_all(
            &e,
            vec![
                RenderCmd::SetRenderPipelineState {
                    pipeline_state: Some(&state),
                },
                three(9, Some(&indices)),
                RenderCmd::EndEncoding,
            ],
        );
        fault
            .encode_signal(Some(EventRef::Shared(&event)), 1)
            .unwrap();
        // A command buffer of `q` that fills a buffer with 0x5a, and the buffer.
        let filling = |q: &crate::gpu::command::VulkanCommandQueue| {
            let b = buffer::<VulkanPlatform>(&d, 16, 0, &[0; 16]);
            let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
            let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
            let fill = BlitCmd::FillBuffer {
                buffer: &b,
                range: NsRange::new(0, 16),
                value: 0x5a,
            };
            blit.encode(OnFault::Die, fill).unwrap();
            blit.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
            (cb, b)
        };
        let ran = |cb: &crate::gpu::command::VulkanCommandBuffer, b: &VulkanBuffer| {
            assert_eq!(cb.status().0, 4);
            assert!(cb.error().is_none(), "{:?}", cb.error());
            // SAFETY: a Shared buffer's bytes, read after its command buffer completed.
            let first = unsafe { *b.contents().as_ptr::<u8>() };
            assert_eq!(first, 0x5a);
        };
        let clock = crate::os::OsClock::new();
        let (behind, b) = filling(&q);
        let started = clock.now_mono();
        fault.commit(OnFault::Die).unwrap();
        behind.commit(OnFault::Die).unwrap();
        let (meanwhile, m) = filling(&other);
        meanwhile.commit(OnFault::Die).unwrap();
        meanwhile.wait_until_completed(OnFault::Die).unwrap();
        ran(&meanwhile, &m);
        behind.wait_until_completed(OnFault::Die).unwrap();
        assert_eq!(fault.status().0, 5, "it completed before the fault");
        ran(&behind, &b);
        fault.wait_until_completed(OnFault::Die).unwrap();
        assert!(clock.now_mono().saturating_duration_since(started) >= FAULT);
        assert_eq!(fault.error().map(|e| e.code), Some(3));
        assert_eq!(event.signaled_value(), 1);
    }

    /// The earliest after it read `Scheduled`, which follows its commit, that Metal completed a
    /// command buffer its GPU faulted at or failed a victim of its recovery at, at a draw of
    /// primitive type 9 or an execution under a state that cannot inherit: 6.86 ms, of 164 such
    /// command buffers on the M3 Max, macOS 27.0.1, which completed 6.86 to 10.3 ms after it (the
    /// runtime's measurement, `command_buffer/hang.rs`).
    const FAULT: std::time::Duration = std::time::Duration::from_micros(6_860);

    /// The earliest after its commit that Metal failed a command buffer its GPU hung at, at a
    /// draw of primitive type 9: 28.3 ms, of 33 hangs on the M3 Max, macOS 27.0.1, queues warmed
    /// and not, which failed 28.3 to 46.2 ms after their commits (the runtime's measurement,
    /// `command_buffer/hang.rs`).
    const HANG: std::time::Duration = std::time::Duration::from_micros(28_300);

    /// A command buffer Metal's GPU would fail in two ways, a draw of primitive type 9 in a
    /// render encoder made under protection options, dies at its commit, as nothing measured it
    #[test]
    fn a_type_9_draw_under_protection_options_dies_at_its_commit() {
        let (p, d) = (platform(), dev());
        let (_dst, pass) = pass_of(&p, &d, (4, 4));
        let state = fixture_state::<VulkanPlatform>(&d, "vert:tri_vert", "frag:tri_frag");
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        cb.set_protection_options(OnFault::Die, 2).unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        send_all(
            &e,
            vec![
                RenderCmd::SetRenderPipelineState {
                    pipeline_state: Some(&state),
                },
                three(9, None),
                RenderCmd::EndEncoding,
            ],
        );
        let site = catch_death(|| cb.commit(OnFault::Die)).unwrap_err();
        assert_eq!(site.file, "commit");
    }

    /// A draw of primitive type 9 over an index type no `MTLIndexType` is, directly and from
    /// indirect arguments, and one of each form with no pipeline state set, which nothing
    /// measured, die
    #[test]
    fn unmeasured_type_9_draws_die() {
        const INDEXED: &str =
            "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:";
        const INDEXED_INSTANCES: &str = "drawIndexedPrimitives:indexCount:indexType:\
             indexBuffer:indexBufferOffset:instanceCount:";
        const INDIRECT: &str = "drawPrimitives:indirectBuffer:indirectBufferOffset:";
        const INDEXED_INDIRECT: &str = "drawIndexedPrimitives:indexType:indexBuffer:\
             indexBufferOffset:indirectBuffer:indirectBufferOffset:";
        let (e, indices) = drawing_encoder("vert:tri_vert", "frag:tri_frag");
        let instanced =
            |index_count, instance_count, index_type| RenderCmd::DrawIndexedPrimitivesInstances {
                primitive_type: 9,
                index_count,
                index_type,
                index_buffer: &indices,
                index_buffer_offset: 0,
                instance_count,
            };
        let indirect = || RenderCmd::DrawPrimitivesIndirect {
            primitive_type: 9,
            indirect_buffer: &indices,
            indirect_buffer_offset: 0,
        };
        let indexed_indirect = |index_type| RenderCmd::DrawIndexedPrimitivesIndirect {
            primitive_type: 9,
            index_type,
            index_buffer: &indices,
            index_buffer_offset: 0,
            indirect_buffer: &indices,
            indirect_buffer_offset: 0,
        };
        for (index_count, instance_count) in [(3, 1), (0, 1), (3, 0)] {
            assert_eq!(
                dies_at(&e, instanced(index_count, instance_count, 2)),
                INDEXED_INSTANCES
            );
        }
        assert_eq!(dies_at(&e, indexed_indirect(2)), INDEXED_INDIRECT);
        let (e, _dst) = open_encoder();
        assert_eq!(
            dies_at(&e, three(9, None)),
            "drawPrimitives:vertexStart:vertexCount:"
        );
        assert_eq!(dies_at(&e, three(9, Some(&indices))), INDEXED);
        assert_eq!(dies_at(&e, indirect()), INDIRECT);
        assert_eq!(dies_at(&e, indexed_indirect(U16)), INDEXED_INDIRECT);
    }

    /// The sweep library, `rounding.metal`, compiled with fast math, the compiler's default.
    #[cfg(target_os = "macos")]
    pub(crate) const ROUNDING: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/rounding.metallib"));

    /// A pass that resolves `RGBA16Float` and one that resolves `RG11B10Float` (92), each at 4
    /// samples, in strict mode, on the platform's device, which resolves by the device's own
    /// average: each ends. MoltenVK averages as Metal does (the runtime's
    /// `ColorResolveAverageIsMetals`) and announces no colour resolve the device averages and
    /// rounds (`DeviceRoundedResolve`); a native device averages and rounds them, which it relays
    /// ([`crate::answer::native::NATIVE_RELAYED`]), counted, so a strict run there goes on past it
    #[test]
    fn a_device_resolve_ends_its_pass() {
        use crate::answer::native::{self, Placement};

        /// `MTLTextureType2DMultisample` and `MTLStoreActionMultisampleResolve`.
        const MULTISAMPLE: u64 = 4;
        const RESOLVE: u64 = 2;
        assert!(crate::answer::strict());
        let p = platform();
        let end = |d: &crate::gpu::device::VulkanDevice, format| {
            let samples = d
                .new_texture(
                    OnFault::Die,
                    &p.texture_descriptor(&TextureDesc {
                        texture_type: MULTISAMPLE,
                        pixel_format: format,
                        width: 16,
                        height: 16,
                        sample_count: 4,
                        usage: TARGET,
                        resource_options: Some(0x20),
                        ..TextureDesc::default()
                    }),
                )
                .unwrap()
                .unwrap();
            let target = texture(&p, d, format, READ | TARGET, (16, 16), None);
            let pass = p.render_pass_descriptor();
            let attachment = ColorAttachmentSet {
                texture: Some(Some(&samples)),
                resolve_texture: Some(Some(&target)),
                load_action: Some(CLEAR),
                store_action: Some(RESOLVE),
                ..ColorAttachmentSet::default()
            };
            pass.set_color_attachment(OnFault::Die, 0, &attachment)
                .unwrap();
            let q = d.new_command_queue().unwrap();
            let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
            let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
            e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
        };
        let rounded = || {
            (native::counts().into_iter())
                .find(|&(k, _)| k == rt::Degradation::DeviceRoundedResolve)
                .map_or(0, |(_, n)| n)
        };
        let before = rounded();
        let d = dev();
        for format in [RGBA16F, 92] {
            end(&d, format);
        }
        if native::placement() == Placement::Native {
            assert!(rounded() > before);
        } else {
            assert_eq!(rounded(), before);
        }
    }

    /// An open encoder on a BGRA8 target with the state of the fixtures `vertex` and `fragment`
    /// set, and an index buffer.
    fn drawing_encoder(vertex: &str, fragment: &str) -> (VulkanRenderEncoder, VulkanBuffer) {
        open_drawing(&fixture_state::<VulkanPlatform>(&dev(), vertex, fragment))
    }

    /// An open encoder on a BGRA8 target with `state` set, and an index buffer.
    fn open_drawing(
        state: &crate::gpu::resource::VulkanRenderPipeline,
    ) -> (VulkanRenderEncoder, VulkanBuffer) {
        let (e, _dst) = open_encoder();
        e.encode(
            OnFault::Die,
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(state),
            },
        )
        .unwrap();
        let indices = buffer::<VulkanPlatform>(&dev(), 64, 0, &[0; 64]);
        (e, indices)
    }

    /// Sends each of `cmds` to `e`.
    fn send_all(e: &VulkanRenderEncoder, cmds: Vec<RenderCmd<'_, VulkanPlatform>>) {
        for cmd in cmds {
            let sel = cmd.selector();
            e.encode(OnFault::Return, cmd)
                .unwrap_or_else(|r| panic!("{sel}: {r:?}"));
        }
    }

    /// How many times this process relayed `kind` ([`crate::answer::inexact_counts`]).
    fn relayed_of(kind: crate::answer::Inexact) -> u64 {
        (crate::answer::inexact_counts().into_iter())
            .find(|&(k, _)| k == kind)
            .map_or(0, |(_, n)| n)
    }

    /// A draw of `primitive_type`, three vertices or indices of `indices`.
    fn three(primitive_type: u64, indices: Option<&VulkanBuffer>) -> RenderCmd<'_, VulkanPlatform> {
        match indices {
            None => RenderCmd::DrawPrimitives {
                primitive_type,
                vertex_start: 0,
                vertex_count: 3,
            },
            Some(index_buffer) => RenderCmd::DrawIndexedPrimitives {
                primitive_type,
                index_count: 3,
                index_type: U16,
                index_buffer,
                index_buffer_offset: 0,
            },
        }
    }

    /// The selector a death of `cmd` names.
    fn dies_at(e: &VulkanRenderEncoder, cmd: RenderCmd<'_, VulkanPlatform>) -> &'static str {
        catch_death(|| e.encode(OnFault::Return, cmd))
            .unwrap_err()
            .file
    }

    /// A draw of `read_only_beside_nested_vertex` and `read_only_beside_nested_fragment`
    /// ([`crate::gpu::writes::tests::PROBE`]), whose argument buffers' type the runtime refuses
    /// the reflection of, through a buffer on a surface whose GPU writes never reach its memory
    /// bound at both functions' buffer slot 1, which they declare `const` and cannot store
    /// through, draws, indexed: the vertex function writes its words of slot 2 and the fragment
    /// function red. Bound at the vertex function's slot 2 too, a `device` pointer, the draw dies
    /// where it is encoded: on Linux the runtime imports a writable twin of the surface's
    /// read-only alias, which the write would reach ([`crate::gpu::writes::kept_write`]). The
    /// guest's page stays zeros
    #[cfg(target_os = "linux")]
    #[test]
    fn a_draw_through_a_read_only_surface_at_slots_it_cannot_write_draws_on_linux() {
        use reims_vgpu_core::platform::os::vm::HostVm as _;
        use reims_vgpu_core::platform::resource::Library as _;
        use reims_vgpu_core::platform::values::RenderPipelineDesc;

        use crate::gpu::resource::VulkanRenderPipeline;
        use crate::gpu::resource::tests::on_read_only_surface;
        use crate::os::{OsVm, PAGE};

        /// The state, `nested` at both functions' slot 0, `read_only` at their slot 1 and
        /// `written` at the vertex function's slot 2, then an indexed triangle of `indices`.
        fn draw<'a>(
            state: &'a VulkanRenderPipeline,
            [nested, read_only, written, indices]: [&'a VulkanBuffer; 4],
        ) -> Vec<RenderCmd<'a, VulkanPlatform>> {
            let at = |buffer: &'a VulkanBuffer, index| RenderCmd::SetVertexBuffer {
                buffer: Some(buffer),
                offset: 0,
                index,
            };
            let fragment_at = |buffer: &'a VulkanBuffer, index| RenderCmd::SetFragmentBuffer {
                buffer: Some(buffer),
                offset: 0,
                index,
            };
            vec![
                RenderCmd::SetRenderPipelineState {
                    pipeline_state: Some(state),
                },
                at(nested, 0),
                at(read_only, 1),
                at(written, 2),
                fragment_at(nested, 0),
                fragment_at(read_only, 1),
                three(3, Some(indices)),
            ]
        }

        let (p, d) = (platform(), dev());
        let library = (d.new_library(crate::gpu::writes::tests::PROBE))
            .unwrap()
            .unwrap()
            .unwrap();
        let function = |name| library.new_function(OnFault::Die, name).unwrap().unwrap();
        let (vertex, fragment) = (
            function("read_only_beside_nested_vertex"),
            function("read_only_beside_nested_fragment"),
        );
        let state = d
            .new_render_pipeline_state(&RenderPipelineDesc {
                label: None,
                vertex: &vertex,
                fragment: &fragment,
                color0_pixel_format: BGRA8,
                resource_index: None,
            })
            .unwrap()
            .unwrap();
        let (read_only, pages) = on_read_only_surface();
        let nested = buffer::<VulkanPlatform>(&d, 16, 0, &[0; 16]);
        let written = buffer::<VulkanPlatform>(&d, 16, 0, &[0; 16]);
        let indices = buffer::<VulkanPlatform>(&d, 8, 0, &[0, 0, 1, 0, 2, 0, 0, 0]);
        let pixels = pass_pixels(
            &p,
            &d,
            draw(&state, [&nested, &read_only, &written, &indices]),
        );
        assert!(
            pixels.iter().all(|&t| t == [0, 0, 0xff, 0xff]),
            "{pixels:?}"
        );
        // SAFETY: a Shared buffer's words, read after the draw completed.
        let words = unsafe { std::slice::from_raw_parts(written.contents().as_ptr::<u32>(), 4) };
        assert_eq!(words, [1, 1, 1, 0]);
        let (_dst, pass) = pass_of(&p, &d, (16, 16));
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        let mut cmds = draw(&state, [&nested, &read_only, &read_only, &indices]);
        let issue = cmds.pop().unwrap();
        send_all(&e, cmds);
        assert_eq!(
            dies_at(&e, issue),
            "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:"
        );
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert!(cb.error().is_none(), "{:?}", cb.error());
        // SAFETY: the page, mapped readable, which no GPU work uses now.
        let first = unsafe { pages.as_ptr::<u32>().read_volatile() };
        assert_eq!(first, 0);
        drop((cb, e, state, read_only));
        assert!(OsVm::new().deallocate(pages, PAGE).is_success());
    }

    /// `MTLLoadActionDontCare`, `MTLLoadActionLoad`, `MTLStoreActionDontCare`,
    /// `MTLStoreActionMultisampleResolve` and `MTLStoreActionUnknown`.
    #[cfg(target_os = "linux")]
    const LOAD_DONT_CARE: u64 = 0;
    #[cfg(target_os = "linux")]
    const LOAD: u64 = 1;
    #[cfg(target_os = "linux")]
    const DONT_CARE: u64 = 0;
    #[cfg(target_os = "linux")]
    const MULTISAMPLE_RESOLVE: u64 = 2;
    #[cfg(target_os = "linux")]
    const UNKNOWN: u64 = 4;

    /// A pass whose colour attachment is a 16 × 16 `BGRA8Unorm` texture over a guest's read-only
    /// buffer (made with bit 24, `resource::tests::read_only_buffer`), loaded by `load` and
    /// stored by `store`, or by `late`'s store action set in the pass where given, drawing
    /// nothing, committed and completed: its command buffer's status and error code, and whether
    /// the pages stayed zeros.
    #[cfg(target_os = "linux")]
    fn read_only_pass(load: u64, store: u64, late: Option<u64>) -> (u64, Option<i64>, bool) {
        use reims_vgpu_core::platform::os::vm::HostVm as _;

        use crate::gpu::resource::tests::{bgra_over, host_bytes_at, read_only_buffer};
        use crate::os::{OsVm, PAGE};

        let (p, d) = (platform(), dev());
        let (ro, pages) = read_only_buffer(1);
        let target = bgra_over(&ro, READ | TARGET);
        let pass = p.render_pass_descriptor();
        let attachment = ColorAttachmentSet {
            texture: Some(Some(&target)),
            load_action: Some(load),
            store_action: Some(store),
            clear_color: Some(ClearColor {
                red: 1.0,
                green: 0.5,
                blue: 0.25,
                alpha: 1.0,
            }),
            ..ColorAttachmentSet::default()
        };
        pass.set_color_attachment(OnFault::Die, 0, &attachment)
            .unwrap();
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        if let Some(store_action) = late {
            let set = RenderCmd::SetColorStoreAction {
                store_action,
                color_attachment_index: 0,
            };
            e.encode(OnFault::Die, set).unwrap();
        }
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        let zeros = host_bytes_at(pages, 16 * 64).iter().all(|&b| b == 0);
        let answer = (cb.status().0, cb.error().map(|e| e.code), zeros);
        drop((cb, e, pass, target, ro));
        assert!(OsVm::new().deallocate(pages, PAGE).is_success());
        answer
    }

    /// Commits an empty command buffer of a fresh queue: the selector it dies at, if it dies.
    #[cfg(target_os = "linux")]
    fn later_commit() -> Option<&'static str> {
        let q = dev().new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        match catch_death(|| cb.commit(OnFault::Die)) {
            Ok(_) => {
                cb.wait_until_completed(OnFault::Die).unwrap();
                None
            }
            Err(site) => Some(site.file),
        }
    }

    /// A pass that stores a colour attachment over a guest's read-only memory (`Store`) fails its
    /// command buffer with Metal's page fault (`MTLCommandBufferErrorPageFault`, 3), as Metal's
    /// GPU faults it, the pages staying zeros, where the runtime's copy of them would take the
    /// store; the commit of every later command buffer of the process then dies, as Metal fails
    /// each with a code nothing recorded (`crate::gpu::command`'s docs; one process per test, as
    /// nextest runs them). Linux's: on macOS the pass goes to Metal, whose fault this host's GPU
    /// takes no test of
    #[cfg(target_os = "linux")]
    #[test]
    fn a_store_into_read_only_memory_fails_its_command_buffer() {
        assert_eq!(later_commit(), None, "before the fault");
        assert_eq!(read_only_pass(CLEAR, STORE, None), (5, Some(3), true));
        assert_eq!(later_commit(), Some("commit"), "after it");
    }

    /// A late store action that stores the colour attachment over a guest's read-only memory,
    /// of a pass that left it `Unknown`, fails the command buffer as `Store` given at once does
    /// (`a_store_into_read_only_memory_fails_its_command_buffer`)
    #[cfg(target_os = "linux")]
    #[test]
    fn a_late_store_into_read_only_memory_fails_its_command_buffer() {
        assert_eq!(
            read_only_pass(CLEAR, UNKNOWN, Some(STORE)),
            (5, Some(3), true)
        );
        assert_eq!(later_commit(), Some("commit"));
    }

    /// A pass over a colour attachment over a guest's read-only memory that does not store it
    /// completes with no error, as Metal's does, the pages staying zeros: `DontCare`, given at
    /// once or late, after a clear, a load or no load; and every later command buffer runs
    #[cfg(target_os = "linux")]
    #[test]
    fn a_pass_that_does_not_store_read_only_memory_completes() {
        for (load, store, late) in [
            (CLEAR, DONT_CARE, None),
            (LOAD, DONT_CARE, None),
            (LOAD_DONT_CARE, DONT_CARE, None),
            (CLEAR, UNKNOWN, Some(DONT_CARE)),
        ] {
            let answer = read_only_pass(load, store, late);
            assert_eq!(answer, (4, None, true), "{load} {store} {late:?}");
        }
        assert_eq!(later_commit(), None);
    }

    /// A pass that resolves a multisample attachment of the device's memory into a texture over
    /// a guest's read-only memory dies where it is made: what Metal does there is unknown (no
    /// verdict over 520 processes, `crate::gpu::writes`'s docs). Linux's: on macOS it goes to
    /// Metal
    #[cfg(target_os = "linux")]
    #[test]
    fn a_resolve_into_read_only_memory_dies() {
        use reims_vgpu_core::platform::os::vm::HostVm as _;

        use crate::gpu::resource::tests::{bgra_over, read_only_buffer};
        use crate::os::{OsVm, PAGE};

        /// `MTLTextureType2DMultisample`.
        const MULTISAMPLE: u64 = 4;
        let (p, d) = (platform(), dev());
        let (ro, pages) = read_only_buffer(1);
        let target = bgra_over(&ro, READ | TARGET);
        let samples = d
            .new_texture(
                OnFault::Die,
                &p.texture_descriptor(&TextureDesc {
                    texture_type: MULTISAMPLE,
                    pixel_format: BGRA8,
                    width: 16,
                    height: 16,
                    sample_count: 4,
                    usage: TARGET,
                    resource_options: Some(0x20),
                    ..TextureDesc::default()
                }),
            )
            .unwrap()
            .unwrap();
        let pass = p.render_pass_descriptor();
        let attachment = ColorAttachmentSet {
            texture: Some(Some(&samples)),
            resolve_texture: Some(Some(&target)),
            load_action: Some(CLEAR),
            store_action: Some(MULTISAMPLE_RESOLVE),
            ..ColorAttachmentSet::default()
        };
        pass.set_color_attachment(OnFault::Die, 0, &attachment)
            .unwrap();
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let site = catch_death(|| cb.render_encoder(OnFault::Die, &pass)).unwrap_err();
        assert_eq!(site.file, "renderCommandEncoderWithDescriptor:");
        drop((cb, pass, samples, target, ro));
        assert!(OsVm::new().deallocate(pages, PAGE).is_success());
    }

    /// An indirect strip of 200,000 16- or 32-bit indices that restarts at the setter's index
    /// ([`long_restarted_strips`]) draws as the same strip drawn directly: the command buffer's
    /// index pool, a sixty-fourth of the device's largest device-local heap, holds the list the
    /// runtime rewrites it into before the pass. A list past the pool still dies as the command
    /// buffer's failure is read ([`crate::gpu::command`])
    #[test]
    fn a_long_restarted_strip_draws_within_the_index_pool() {
        let (p, d) = (platform(), dev());
        for (index_type, indices) in long_restarted_strips() {
            let drawn = (STRIP, index_type, &indices[..]);
            let restart = Some((true, 3));
            let indirect = restart_draw(&p, &d, drawn, restart, (true, 0));
            let direct = restart_draw(&p, &d, drawn, restart, (false, 0));
            assert_eq!(indirect, direct, "{index_type}");
        }
    }

    /// Strips of 200,000 indices, 16-bit and 32-bit, each its type's: 0, 1, 2 and 3 over and
    /// over, so a strip that restarts at 3 draws the triangle (0, 1, 2) again after each.
    fn long_restarted_strips() -> [(u64, Vec<u8>); 2] {
        let pattern: Vec<u32> = [0, 1, 2, 3].into_iter().cycle().take(200_000).collect();
        let narrow: Vec<u16> = pattern.iter().map(|&i| u16::try_from(i).unwrap()).collect();
        [(U16, u16s(&narrow)), (U32, u32s(&pattern))]
    }

    /// Little-endian `u16` indices.
    fn u16s(indices: &[u16]) -> Vec<u8> {
        indices.iter().flat_map(|i| i.to_le_bytes()).collect()
    }

    /// Little-endian `u32` indices.
    fn u32s(indices: &[u32]) -> Vec<u8> {
        indices.iter().flat_map(|i| i.to_le_bytes()).collect()
    }

    /// `MTLIndexTypeUInt32`.
    const U32: u64 = 1;

    /// The pixels of [`pass_pixels`] after a draw of `primitive_type` over `indices` of
    /// `index_type`, under [`FLAT`]'s state over [`flat_vertices`] padded to 65536, so every
    /// 16-bit index reads a vertex: drawn directly, or from indirect arguments where
    /// `indirect`, after `restart`'s `setPrimitiveRestartEnabled:index:` where it has one. The
    /// indices lie `offset` bytes into their buffer, after bytes of index 5 (a 16-bit one where
    /// `offset` is 2), which a draw reading them would show.
    fn restart_draw<P: Platform>(
        p: &P,
        d: &P::Device,
        (primitive_type, index_type, indices): (u64, u64, &[u8]),
        restart: Option<(bool, u64)>,
        (indirect, offset): (bool, u64),
    ) -> Vec<[u8; 4]> {
        let state = fixture_state::<P>(d, FLAT.0, FLAT.1);
        let (pos, colours) = flat_vertices::<P>(d, 0x10000);
        let length = u64::try_from(indices.len()).unwrap();
        let mut bytes: Vec<u8> = [5u8, 0, 0, 0]
            .into_iter()
            .cycle()
            .take(usize::try_from(offset).unwrap())
            .collect();
        bytes.extend_from_slice(indices);
        let index_buffer = buffer::<P>(d, offset + length, 0, &bytes);
        let index_count = length / if index_type == U16 { 2 } else { 4 };
        // `MTLDrawIndexedPrimitivesIndirectArguments`: the count, one instance, and zeros.
        let arguments: Vec<u8> = [u32::try_from(index_count).unwrap(), 1, 0, 0, 0]
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        let arguments = buffer::<P>(d, 20, 0, &arguments);
        let mut cmds = vec![
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&state),
            },
            RenderCmd::SetVertexBuffer {
                buffer: Some(&pos),
                offset: 0,
                index: 0,
            },
            RenderCmd::SetVertexBuffer {
                buffer: Some(&colours),
                offset: 0,
                index: 1,
            },
        ];
        cmds.extend(
            restart
                .map(|(enabled, index)| RenderCmd::SetPrimitiveRestartEnabled { enabled, index }),
        );
        cmds.push(match indirect {
            false => RenderCmd::DrawIndexedPrimitives {
                primitive_type,
                index_count,
                index_type,
                index_buffer: &index_buffer,
                index_buffer_offset: offset,
            },
            true => RenderCmd::DrawIndexedPrimitivesIndirect {
                primitive_type,
                index_type,
                index_buffer: &index_buffer,
                index_buffer_offset: offset,
                indirect_buffer: &arguments,
                indirect_buffer_offset: 0,
            },
        });
        pass_pixels(p, d, cmds)
    }

    /// After `setLineWidth:` of any width, lines, line strips, points and filled triangles draw,
    /// relaxed, as a device's own wide lines may differ from Metal's and say so
    /// (`device-wide-lines`); a width of 1 again draws as before. What the widths draw is the
    /// runtime's, as Metal draws them (`metal_tests::lines_draw_at_metals_widths`)
    #[test]
    fn draws_go_on_after_a_line_width() {
        let points = points_reach_metals();
        let (e, indices) = if points {
            drawing_encoder(SIZED.0, SIZED.1)
        } else {
            drawing_encoder("vert:tri_vert", "frag:tri_frag")
        };
        for width in [2.0, 0.0, -1.0, f32::NAN, 1.0] {
            send_all(&e, vec![RenderCmd::SetLineWidth { line_width: width }]);
            crate::answer::relaxed(|| {
                send_all(
                    &e,
                    vec![
                        three(1, None),
                        three(2, None),
                        three(3, None),
                        three(STRIP, Some(&indices)),
                    ],
                );
            });
            if points {
                send_all(&e, vec![three(0, None)]);
            }
        }
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
    }

    /// Whether the device's points reach Metal's largest, 511 pixels (its `pointSizeRange`).
    /// Elsewhere (255 on ANV) every draw of points by a state that sizes them dies in strict mode
    /// (`device-point-size`, [`crate::answer::DIES`]), which a draw of [`SIZED`]'s pins here, so
    /// no such state draws a point there, and the test says so
    /// ([`a_state_that_sizes_its_points_draws_lines_and_triangles_silently`] draws its others)
    fn points_reach_metals() -> bool {
        let reaches = dev().raw.properties().description.limits.max_point_size >= 511;
        if !reaches {
            let (e, _) = drawing_encoder(SIZED.0, SIZED.1);
            let kind = rt::Degradation::DevicePointSize;
            let site = crate::answer::dies_of(kind, || e.encode(OnFault::Return, three(0, None)));
            assert_eq!(site.file, "drawPrimitives:vertexStart:vertexCount:");
            e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
            eprintln!("the device's points end below Metal's 511 pixels: no point is drawn here");
        }
        reaches
    }

    /// A state that sizes its points ([`SIZED`]) draws lines, line strips, triangles and
    /// triangle strips, indexed or not, in strict mode, with nothing that dies announced, on every
    /// device, one whose points end below Metal's 511 pixels too: the runtime announces
    /// `device-point-size` at a draw of points alone ([`points_reach_metals`])
    #[test]
    fn a_state_that_sizes_its_points_draws_lines_and_triangles_silently() {
        let (e, indices) = drawing_encoder(SIZED.0, SIZED.1);
        let draws = [1, 2, 3, STRIP].into_iter();
        let cmds = draws.flat_map(|p| [three(p, None), three(p, Some(&indices))]);
        send_all(&e, cmds.collect());
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
    }

    /// Whether the runtime draws the last provoking vertex through the device's own
    /// (`VK_EXT_provoking_vertex`'s `provokingVertexLast`, with
    /// `provokingVertexModePerPipeline`), where it draws lists led by each primitive's last
    /// vertex elsewhere (MoltenVK among them).
    pub(crate) fn last_vertex_native() -> bool {
        use metal2vulkan::device::{VkExtension, VkFeature};
        let d = dev();
        let desc = &d.raw.properties().description;
        desc.extensions.contains(&VkExtension::ExtProvokingVertex)
            && desc.features.contains(&VkFeature::ProvokingVertexLast)
            && desc.limits.provoking_vertex_mode_per_pipeline
    }

    /// How many times this process relayed `kind`, of [`crate::answer::DIES`].
    pub(crate) fn relayed_kind(kind: rt::Degradation) -> u64 {
        (crate::answer::relayed_counts().into_iter())
            .find(|&(k, _)| k == kind)
            .map_or(0, |(_, n)| n)
    }

    /// After `setProvokingVertexMode:` of 1, every draw of a state that reads no flat input draws,
    /// and so do the triangles of one that does, direct, over indices and from indirect
    /// arguments, and its points. Its lines and its line loop, which the runtime draws from their
    /// last vertex to their first without the device's last vertex, die there in strict mode
    /// (`provoking-vertex-lines`) and by default draw, relayed and counted. A mode from 2 up, its
    /// top bit set or not, sets silently; under it the list and fan triangles of a state that
    /// reads a flat input draw, and its strip and lines, which Metal draws a third way, die at
    /// the draw in strict mode (`provoking-vertex-unmatched`) and by default draw, relayed and
    /// counted; mode 0 again draws all
    #[test]
    fn the_last_vertex_relays_only_what_the_runtime_announces() {
        let (e, indices) = drawing_encoder("vert:tri_vert", "frag:tri_frag");
        let indirect = |primitive_type| RenderCmd::DrawPrimitivesIndirect {
            primitive_type,
            indirect_buffer: &indices,
            indirect_buffer_offset: 0,
        };
        send_all(
            &e,
            vec![
                RenderCmd::SetProvokingVertexMode { mode: 1 },
                three(3, Some(&indices)),
                three(STRIP, Some(&indices)),
                three(1, None),
                three(6, None),
                indirect(3),
            ],
        );
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
        let (e, indices) = drawing_encoder(FLAT.0, FLAT.1);
        let indirect = |primitive_type| RenderCmd::DrawPrimitivesIndirect {
            primitive_type,
            indirect_buffer: &indices,
            indirect_buffer_offset: 0,
        };
        send_all(
            &e,
            vec![
                RenderCmd::SetProvokingVertexMode { mode: 1 },
                three(3, None),
                three(STRIP, None),
                three(3, Some(&indices)),
                three(STRIP, Some(&indices)),
                three(6, None),
                indirect(3),
                indirect(STRIP),
            ],
        );
        // Points, under a state that reads a flat input and sizes its points, where the device's
        // points reach Metal's.
        if points_reach_metals() {
            let (sized, sized_indices) = drawing_encoder(SIZED.0, SIZED.1);
            send_all(
                &sized,
                vec![
                    RenderCmd::SetProvokingVertexMode { mode: 1 },
                    three(0, None),
                    three(0, Some(&sized_indices)),
                    RenderCmd::EndEncoding,
                ],
            );
        }
        let draw = "drawPrimitives:vertexStart:vertexCount:";
        let indexed = "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:";
        let lines = || vec![three(1, None), three(2, Some(&indices)), three(5, None)];
        if last_vertex_native() {
            send_all(&e, lines());
        } else {
            let kind = rt::Degradation::ProvokingVertexLines;
            for (cmd, at) in lines().into_iter().zip([draw, indexed, draw]) {
                assert_eq!(
                    crate::answer::dies_of(kind, || e.encode(OnFault::Return, cmd)).file,
                    at
                );
            }
            let before = relayed_kind(kind);
            crate::answer::relaxed(|| send_all(&e, lines()));
            assert_eq!(relayed_kind(kind), before + 3);
        }
        let unmatched = rt::Degradation::ProvokingVertexUnmatched;
        for mode in [2, 7, 1 << 32, 1 << 63, u64::MAX] {
            send_all(
                &e,
                vec![
                    RenderCmd::SetProvokingVertexMode { mode },
                    three(3, None),
                    three(3, Some(&indices)),
                    three(6, None),
                ],
            );
            let strip = three(STRIP, None);
            let site = crate::answer::dies_of(unmatched, || e.encode(OnFault::Return, strip));
            assert_eq!(site.file, draw, "{mode:#x}");
            let before = relayed_kind(unmatched);
            crate::answer::relaxed(|| send_all(&e, vec![three(STRIP, None), three(1, None)]));
            assert_eq!(relayed_kind(unmatched), before + 2, "{mode:#x}");
        }
        send_all(
            &e,
            vec![
                RenderCmd::SetProvokingVertexMode { mode: 0 },
                three(3, Some(&indices)),
                three(1, None),
                three(6, None),
            ],
        );
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
    }

    /// A draw of no primitive draws nothing whatever the private setters left, as on Metal: of
    /// too few vertices or indices for one, and an execution of no command, after a restart, a
    /// line width of 2, triangles filled as lines and the last vertex provoking under a state
    /// that reads a flat input. The runtime announces nothing at such a draw, so a draw of lines
    /// of one vertex or index goes on in strict mode too, where the device lacks the last vertex.
    /// What a draw of a primitive draws after them is the runtime's
    /// (`metal_tests::lines_draw_at_metals_widths`,
    /// `metal_tests::lines_take_metals_flat_inputs_at_the_last_vertex`)
    #[test]
    fn empty_draws_go_on_after_the_private_setters() {
        use reims_vgpu_core::platform::values::IcbDesc;

        let (e, indices) = drawing_encoder(FLAT.0, FLAT.1);
        let icb = dev()
            .new_indirect_command_buffer(
                &IcbDesc {
                    command_types: 1,
                    ..IcbDesc::default()
                },
                1,
                0,
            )
            .unwrap()
            .unwrap();
        let drawn = |primitive_type, vertex_count| RenderCmd::DrawPrimitives {
            primitive_type,
            vertex_start: 0,
            vertex_count,
        };
        let indexed = |primitive_type, index_count| RenderCmd::DrawIndexedPrimitives {
            primitive_type,
            index_count,
            index_type: U16,
            index_buffer: &indices,
            index_buffer_offset: 0,
        };
        send_all(
            &e,
            vec![
                RenderCmd::SetPrimitiveRestartEnabled {
                    enabled: true,
                    index: 0xffff,
                },
                RenderCmd::SetLineWidth { line_width: 2.0 },
                RenderCmd::SetTriangleFillMode { fill_mode: 1 },
                RenderCmd::SetProvokingVertexMode { mode: 1 },
            ],
        );
        let mut empty = vec![drawn(0, 0), indexed(0, 0)];
        for (primitive_type, least) in [(1, 2), (2, 2), (3, 3), (STRIP, 3)] {
            for count in 0..least {
                empty.extend([drawn(primitive_type, count), indexed(primitive_type, count)]);
            }
        }
        empty.push(RenderCmd::ExecuteCommandsInBuffer {
            indirect_command_buffer: &icb,
            execution_range: NsRange::new(0, 0),
        });
        send_all(&e, empty);
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
    }

    /// A draw of no instances draws nothing whatever the private setters left, plain and
    /// indexed, as on Metal, and the encoder goes on; one with no pipeline state set dies, where
    /// plain Metal stops the process (the runtime's probe `zero_instances`, an M4)
    #[test]
    fn draws_of_no_instances_draw_nothing_as_plain_metal_does() {
        let (e, indices) = drawing_encoder(FLAT.0, FLAT.1);
        send_all(
            &e,
            vec![
                RenderCmd::SetLineWidth { line_width: 2.0 },
                RenderCmd::SetTriangleFillMode { fill_mode: 1 },
                RenderCmd::SetProvokingVertexMode { mode: 1 },
            ],
        );
        let plain = |primitive_type| RenderCmd::DrawPrimitivesInstances {
            primitive_type,
            vertex_start: 0,
            vertex_count: 3,
            instance_count: 0,
        };
        let indexed = |primitive_type| RenderCmd::DrawIndexedPrimitivesInstances {
            primitive_type,
            index_count: 3,
            index_type: U16,
            index_buffer: &indices,
            index_buffer_offset: 0,
            instance_count: 0,
        };
        let mut empty = Vec::new();
        for primitive_type in [0, 1, 2, 3, STRIP] {
            empty.extend([plain(primitive_type), indexed(primitive_type)]);
        }
        send_all(&e, empty);
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
        let (e, _dst) = open_encoder();
        assert_eq!(
            dies_at(&e, plain(3)),
            "drawPrimitives:vertexStart:vertexCount:instanceCount:"
        );
    }

    /// `setVertexAmplificationMode:value:` encodes nothing; draws go on where it leaves the
    /// runtime's count, and where it leaves another, or 0, which Metal draws copies of at random,
    /// draws and executions die until `setVertexAmplificationCount:viewMappings:` sets one, as
    /// `render_state_misc` sends them
    #[test]
    fn an_amplification_mode_kills_draws_only_where_it_changes_the_count() {
        use reims_vgpu_core::platform::types::VertexAmplificationViewMapping;
        use reims_vgpu_core::platform::values::IcbDesc;

        let (e, _indices) = drawing_encoder("vert:tri_vert", "frag:tri_frag");
        let icb = dev()
            .new_indirect_command_buffer(
                &IcbDesc {
                    command_types: 1,
                    ..IcbDesc::default()
                },
                1,
                0,
            )
            .unwrap()
            .unwrap();
        let mode = |mode, value| RenderCmd::SetVertexAmplificationMode { mode, value };
        let none = VertexAmplificationViewMapping {
            viewport_array_index_offset: 0,
            render_target_array_index_offset: 0,
        };
        let one = [none];
        let counted = RenderCmd::SetVertexAmplificationCount {
            view_mappings: &one,
        };
        send_all(
            &e,
            vec![
                mode(0, 1),
                mode(1, 4),
                mode(7, 1),
                three(3, None),
                mode(0, 0),
                counted,
                three(3, None),
            ],
        );
        let draw = "drawPrimitives:vertexStart:vertexCount:";
        for (m, value) in [(0, 0), (0, 2), (1, 3), (1, 0), (2, 9), (u64::MAX, 0)] {
            send_all(&e, vec![mode(m, value)]);
            assert_eq!(dies_at(&e, three(3, None)), draw, "{m} {value}");
        }
        assert_eq!(
            dies_at(
                &e,
                RenderCmd::ExecuteCommandsInBuffer {
                    indirect_command_buffer: &icb,
                    execution_range: NsRange::new(0, 1),
                }
            ),
            "executeCommandsInBuffer:withRange:"
        );
        let empty = RenderCmd::DrawPrimitives {
            primitive_type: 3,
            vertex_start: 0,
            vertex_count: 0,
        };
        let counted = RenderCmd::SetVertexAmplificationCount {
            view_mappings: &one,
        };
        send_all(
            &e,
            vec![empty, counted, three(3, None), RenderCmd::EndEncoding],
        );
    }

    /// The state commands the runtime answers are taken
    #[test]
    fn state_commands_are_taken() {
        let (e, dst) = open_encoder();
        let d = dev();
        let sampler = d
            .new_sampler_state(&reims_vgpu_core::platform::values::SamplerDesc::default())
            .unwrap()
            .unwrap();
        let buf = d
            .new_buffer_of_length(OnFault::Die, 256, 0)
            .unwrap()
            .unwrap();
        let (two, one) = ([Some(&buf), None], [Some(&buf)]);
        let (texture, sampled) = ([Some(&dst)], [Some(&sampler)]);
        let used = [Some(ResourceRef::Buffer(&buf))];
        let cmds: Vec<RenderCmd<'_, VulkanPlatform>> = vec![
            RenderCmd::SetBlendColorRed {
                red: 0.5,
                green: 0.25,
                blue: 0.0,
                alpha: 1.0,
            },
            RenderCmd::SetCullMode { cull_mode: 2 },
            RenderCmd::SetFrontFacingWinding {
                front_facing_winding: 1,
            },
            RenderCmd::SetTriangleFillMode { fill_mode: 1 },
            RenderCmd::SetDepthBias {
                depth_bias: 1.0,
                slope_scale: 0.5,
                clamp: 0.0,
            },
            RenderCmd::SetDepthClipModeSpi { mode: 1 },
            RenderCmd::SetStencilFrontReferenceValue {
                front_reference_value: 3,
                back_reference_value: 4,
            },
            RenderCmd::SetScissorRects {
                scissor_rects: &[MtlScissorRect {
                    x: 0,
                    y: 0,
                    width: 2,
                    height: 2,
                }],
            },
            RenderCmd::SetVertexBuffers {
                buffers: &two,
                offsets: &[0, 0],
                range: NsRange::new(2, 2),
            },
            RenderCmd::SetVertexBufferOffset {
                offset: 16,
                index: 2,
            },
            RenderCmd::SetFragmentBuffers {
                buffers: &one,
                offsets: &[32],
                range: NsRange::new(1, 1),
            },
            RenderCmd::SetVertexTextures {
                textures: &texture,
                range: NsRange::new(0, 1),
            },
            RenderCmd::SetFragmentSamplerStates {
                samplers: &sampled,
                range: NsRange::new(0, 1),
            },
            RenderCmd::SetVertexSamplerStatesLodClamps {
                samplers: &sampled,
                lod_min_clamps: &[0.0],
                lod_max_clamps: &[4.0],
                range: NsRange::new(1, 1),
            },
            RenderCmd::SetVisibilityResultMode { mode: 0, offset: 0 },
            RenderCmd::UseResources {
                resources: &used,
                usage: 1,
            },
            RenderCmd::MemoryBarrierWithScope {
                scope: 1,
                after: 2,
                before: 1,
            },
            RenderCmd::SetDepthCleared,
            RenderCmd::SetStencilCleared,
            RenderCmd::SetProvokingVertexMode { mode: 0 },
            RenderCmd::SetPrimitiveRestartEnabled {
                enabled: true,
                index: 0,
            },
        ];
        for cmd in cmds {
            let sel = cmd.selector();
            e.encode(OnFault::Return, cmd)
                .unwrap_or_else(|r| panic!("{sel}: {r:?}"));
        }
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
    }

    /// A late resolve setter under a store action the descriptor knew and that does not
    /// resolve encodes nothing, as Metal ignores the resolve texture there: colour, depth and
    /// stencil, over a pass that named resolve textures for each
    #[test]
    fn late_resolves_under_known_stores_encode_nothing() {
        /// `MTLTextureType2DMultisample`, `Depth32Float_Stencil8`.
        const MULTISAMPLE: u64 = 4;
        const DEPTH32F_S8: u64 = 260;
        let (p, d) = (platform(), dev());
        let made = |texture_type, format, sample_count| {
            let desc = p.texture_descriptor(&TextureDesc {
                texture_type,
                pixel_format: format,
                width: 4,
                height: 4,
                sample_count,
                usage: TARGET,
                resource_options: Some(0x20),
                ..TextureDesc::default()
            });
            d.new_texture(OnFault::Die, &desc).unwrap().unwrap()
        };
        let (color, color_resolve) = (made(MULTISAMPLE, BGRA8, 4), made(2, BGRA8, 1));
        let (depth, depth_resolve) = (made(MULTISAMPLE, DEPTH32F_S8, 4), made(2, DEPTH32F_S8, 1));
        let pass = p.render_pass_descriptor();
        pass.set_color_attachment(
            OnFault::Die,
            0,
            &ColorAttachmentSet {
                texture: Some(Some(&color)),
                resolve_texture: Some(Some(&color_resolve)),
                load_action: Some(CLEAR),
                store_action: Some(STORE),
                ..ColorAttachmentSet::default()
            },
        )
        .unwrap();
        pass.set_depth_attachment(&reims_vgpu_core::platform::values::DepthAttachmentSet {
            texture: Some(Some(&depth)),
            resolve_texture: Some(Some(&depth_resolve)),
            load_action: Some(CLEAR),
            store_action: Some(STORE),
            ..Default::default()
        });
        pass.set_stencil_attachment(&reims_vgpu_core::platform::values::StencilAttachmentSet {
            texture: Some(Some(&depth)),
            resolve_texture: Some(Some(&depth_resolve)),
            load_action: Some(CLEAR),
            store_action: Some(STORE),
            ..Default::default()
        });
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        for cmd in [
            RenderCmd::SetColorResolveTexture {
                texture: Some(&color_resolve),
                slice: 0,
                depth_plane: 0,
                level: 0,
                y_invert: false,
                index: 0,
            },
            RenderCmd::SetDepthResolveTexture {
                texture: Some(&depth_resolve),
                slice: 0,
                depth_plane: 0,
                level: 0,
                y_invert: false,
            },
            RenderCmd::SetStencilResolveTexture {
                texture: Some(&depth_resolve),
                slice: 0,
                depth_plane: 0,
                level: 0,
                y_invert: false,
            },
            RenderCmd::EndEncoding,
        ] {
            e.encode(OnFault::Return, cmd).unwrap();
        }
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert!(cb.error().is_none(), "{:?}", cb.error());
    }

    /// A target of `w` × `h` cleared and stored, and a pass of it.
    pub(crate) fn pass_of<P: Platform>(
        p: &P,
        d: &P::Device,
        (w, h): (u64, u64),
    ) -> (P::Texture, P::RenderPassDescriptor) {
        let dst = texture(p, d, BGRA8, TARGET | READ, (w, h), None);
        let pass = p.render_pass_descriptor();
        pass.set_color_attachment(
            OnFault::Die,
            0,
            &ColorAttachmentSet {
                texture: Some(Some(&dst)),
                load_action: Some(CLEAR),
                store_action: Some(STORE),
                ..ColorAttachmentSet::default()
            },
        )
        .unwrap();
        (dst, pass)
    }

    /// Each set reaches the runtime's descriptor, and the attachment states answer what was set
    #[test]
    fn pass_sets_reach_the_runtime_descriptor() {
        let p = platform();
        let d = dev();
        let (dst, pass) = pass_of(&p, &d, (4, 4));
        pass.set_color_attachment(
            OnFault::Die,
            2,
            &ColorAttachmentSet {
                texture: Some(Some(&dst)),
                load_action: Some(1),
                store_action: Some(0),
                clear_color: Some(ClearColor {
                    red: 0.25,
                    green: 0.5,
                    blue: 0.75,
                    alpha: 1.0,
                }),
                ..ColorAttachmentSet::default()
            },
        )
        .unwrap();
        pass.set_pass(&reims_vgpu_core::platform::values::PassSet {
            render_target_width: Some(16),
            render_target_height: Some(8),
            default_raster_sample_count: Some(1),
            ..Default::default()
        });
        let r = pass.descriptor();
        let c2 = &r.color_attachments[2];
        assert!(c2.texture.is_some());
        assert_eq!(
            (c2.load_action, c2.store_action),
            (rt::LoadAction::Load, rt::StoreAction::DontCare)
        );
        assert_eq!(
            (
                c2.clear_color.red,
                c2.clear_color.green,
                c2.clear_color.blue
            ),
            (0.25, 0.5, 0.75)
        );
        assert_eq!(
            (
                r.render_target_width,
                r.render_target_height,
                r.default_raster_sample_count
            ),
            (16, 8, 1)
        );
        assert!(r.color_attachments[1].texture.is_none());
        let state = pass.color_attachment_state(2).unwrap();
        assert_eq!((state.load_action, state.store_action), (1, 0));
        let state = pass.color_attachment_state(0).unwrap();
        assert_eq!((state.load_action, state.store_action), (CLEAR, STORE));
    }

    /// A colour attachment past the eighth is Metal's raise; a default colour sample count, which
    /// changes nothing on this host, is taken
    #[test]
    fn pass_refusals_raise() {
        let p = platform();
        let pass = p.render_pass_descriptor();
        let r = pass.set_color_attachment(OnFault::Return, 8, &ColorAttachmentSet::default());
        assert!(r.is_err());
        assert!(pass.color_attachment_state(8).is_err());
        pass.set_pass(&reims_vgpu_core::platform::values::PassSet {
            default_color_sample_count: Some(4),
            ..Default::default()
        });
    }

    /// Every creation of a render pipeline takes a new nonzero identifier, even for an equal
    /// request the runtime answers its cached state for; every reference answers its handle's
    #[test]
    fn every_render_pipeline_creation_takes_a_new_identifier() {
        use reims_vgpu_core::platform::resource::RenderPipeline as _;
        let d = dev();
        let key = PipelineKey {
            format: BGRA8,
            gamma: false,
        };
        let first = DisplayPipelineCache::<VulkanPlatform>::new(d.clone_ref()).unwrap();
        let second = DisplayPipelineCache::<VulkanPlatform>::new(d.clone_ref()).unwrap();
        let a = first.pipeline_for(key).unwrap();
        let again = first.pipeline_for(key).unwrap();
        let b = second.pipeline_for(key).unwrap();
        assert_ne!(a.unique_identifier(), 0);
        assert_eq!(again.unique_identifier(), a.unique_identifier());
        assert_ne!(b.unique_identifier(), a.unique_identifier());
    }

    /// A bind at the last slot a `u64` names, or of a range from it, dies at its selector as one
    /// past `u32`, which the runtime cannot hold: the slots are taken up to that last one, none
    /// past it
    #[test]
    fn binds_at_the_last_slot_die_at_their_selector() {
        use reims_vgpu_core::raise::death::DeathKind;

        let (p, d) = (platform(), dev());
        let b = crate::gpu::resource::tests::shared(64);
        let cases: [RenderCmd<'_, VulkanPlatform>; 4] = [
            RenderCmd::SetFragmentTexture {
                texture: None,
                index: u64::MAX,
            },
            RenderCmd::SetFragmentTextures {
                textures: &[None, None],
                range: NsRange::new(u64::MAX, 2),
            },
            RenderCmd::SetVertexTextures {
                textures: &[None],
                range: NsRange::new(u64::MAX, 1),
            },
            RenderCmd::SetFragmentBuffers {
                buffers: &[Some(&b), Some(&b)],
                offsets: &[0, 0],
                range: NsRange::new(u64::MAX - 1, 2),
            },
        ];
        for cmd in cases {
            let sel = cmd.selector();
            let (_target, pass) = pass_of(&p, &d, (4, 4));
            let q = d.new_command_queue().unwrap();
            let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
            let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
            let site = catch_death(|| e.encode(OnFault::Die, cmd)).unwrap_err();
            assert_eq!((site.kind, site.file), (DeathKind::Abort, sel));
        }
    }

    /// A pass of no attachment and no target size has no encoder, as Metal answers nil
    #[test]
    fn a_pass_without_targets_has_no_encoder() {
        let p = platform();
        let d = dev();
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let pass = p.render_pass_descriptor();
        assert!(cb.render_encoder(OnFault::Die, &pass).unwrap().is_none());
    }

    /// A pass without attachments of a 4x4 target answers this host's Metal at each default
    /// raster sample count the runtime's probe ran on it (the M3 Max under macOS 26.7): 0, 8 to
    /// 15, 128, 255 and 256 die at the encoder's creation, as Metal ends the process there; 16
    /// and 127 make the encoder and fail the command buffer with `MTLCommandBufferErrorInternal`
    /// (code 1); 1, 4 and 7 complete it
    #[test]
    fn a_pass_without_attachments_answers_metals_default_sample_counts() {
        use reims_vgpu_core::platform::types::CommandBufferStatus;
        use reims_vgpu_core::platform::values::PassSet;
        use reims_vgpu_core::raise::death::DeathKind;

        let p = platform();
        let q = dev().new_command_queue().unwrap();
        let pass = |count| {
            let pass = p.render_pass_descriptor();
            pass.set_pass(&PassSet {
                render_target_width: Some(4),
                render_target_height: Some(4),
                default_raster_sample_count: Some(count),
                ..Default::default()
            });
            pass
        };
        let ended = |count| {
            let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
            let e = cb
                .render_encoder(OnFault::Die, &pass(count))
                .unwrap()
                .unwrap();
            e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
            cb.commit(OnFault::Die).unwrap();
            cb.wait_until_completed(OnFault::Die).unwrap();
            (cb.status(), cb.error().map(|e| e.code))
        };
        for count in [0, 8, 9, 10, 11, 12, 13, 14, 15, 128, 255, 256] {
            let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
            *crate::answer::LAST_LINE.lock() = None;
            let site =
                catch_death(|| cb.render_encoder(OnFault::Return, &pass(count))).unwrap_err();
            let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
            assert_eq!(
                (site.kind, site.file),
                (DeathKind::Abort, "renderCommandEncoderWithDescriptor:"),
                "{count}: {line}"
            );
            assert!(line.starts_with("Metal aborts at "), "{count}: {line}");
        }
        for count in [16, 127] {
            assert_eq!(
                ended(count),
                (CommandBufferStatus::ERROR, Some(1)),
                "{count}"
            );
        }
        for count in [1, 4, 7] {
            assert_eq!(
                ended(count),
                (CommandBufferStatus::COMPLETED, None),
                "{count}"
            );
        }
    }

    /// A nil pipeline state is taken; what runs the encoder's pipeline state after it dies, as
    /// Metal crashes at a draw with no pipeline state: a draw, direct or indirect, a tile
    /// dispatch, and an execution of an indirect command buffer whose commands inherit the
    /// pipeline state. The execution of one whose commands set their own is taken
    #[test]
    fn what_runs_the_pipeline_after_a_nil_one_dies() {
        use reims_vgpu_core::platform::values::IcbDesc;

        let p = platform();
        let d = dev();
        let (_dst, pass) = pass_of(&p, &d, (4, 4));
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        let args = buffer::<VulkanPlatform>(&d, 64, 0, &[]);
        let icb = |inherit_pipeline_state| {
            d.new_indirect_command_buffer(
                &IcbDesc {
                    command_types: 1,
                    inherit_pipeline_state,
                    ..IcbDesc::default()
                },
                1,
                0,
            )
            .unwrap()
            .unwrap()
        };
        let (inheriting, own) = (icb(true), icb(false));
        e.encode(
            OnFault::Die,
            RenderCmd::SetRenderPipelineState {
                pipeline_state: None,
            },
        )
        .unwrap();
        let dies = |cmd: RenderCmd<'_, VulkanPlatform>| {
            let sel = cmd.selector();
            let site = catch_death(|| e.encode(OnFault::Die, cmd)).unwrap_err();
            assert_eq!(site.file, sel);
        };
        dies(RenderCmd::DrawPrimitives {
            primitive_type: STRIP,
            vertex_start: 0,
            vertex_count: 4,
        });
        dies(RenderCmd::DrawPrimitivesIndirect {
            primitive_type: STRIP,
            indirect_buffer: &args,
            indirect_buffer_offset: 0,
        });
        dies(RenderCmd::DispatchThreadsPerTile {
            threads_per_tile: MtlSize::new(1, 1, 1),
        });
        dies(RenderCmd::ExecuteCommandsInBuffer {
            indirect_command_buffer: &inheriting,
            execution_range: NsRange::new(0, 1),
        });
        dies(RenderCmd::ExecuteCommandsIndirect {
            indirect_commandbuffer: &inheriting,
            indirect_range_buffer: &args,
            indirect_buffer_offset: 0,
        });
        e.encode(
            OnFault::Die,
            RenderCmd::ExecuteCommandsInBuffer {
                indirect_command_buffer: &own,
                execution_range: NsRange::new(0, 1),
            },
        )
        .unwrap();
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
    }

    /// Runs, on a new device, a pass that executes a command of an indirect command buffer whose
    /// pipeline state draws to RGBA16Float in a pass of a BGRA8 target, which the runtime refuses,
    /// the buffer made resident with `usage`.
    fn execute_a_mismatched_draw(usage: u64) {
        use metal2vulkan::metal::PixelFormat;
        use metal2vulkan::pipeline::{
            ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource,
        };
        use reims_vgpu_core::platform::icb::{
            IcbRenderCmd, IndirectCommandBuffer as _, IndirectRenderCommand as _,
        };
        use reims_vgpu_core::platform::values::IcbDesc;

        const VERT: &[u8] = include_bytes!("../../../../tests/shaders/vert.metallib");
        const FRAG: &[u8] = include_bytes!("../../../../tests/shaders/frag.metallib");
        let p = platform();
        let d = dev();
        let function = |bytes, name| {
            d.raw
                .new_library_with_data(bytes)
                .unwrap()
                .new_function_with_name(name)
                .unwrap()
        };
        let mut desc = RenderPipelineDescriptor::new(StageSource::Function(
            function(VERT, "tri_vert").function_descriptor(),
        ));
        desc.fragment = Some(StageSource::Function(
            function(FRAG, "tri_frag").function_descriptor(),
        ));
        desc.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float));
        desc.support_indirect_command_buffers = true;
        let pipeline = crate::gpu::resource::render_pipeline(
            d.raw
                .new_render_pipeline_state(&desc, &rt::PipelineOptions::default())
                .unwrap(),
        );
        let icb = d
            .new_indirect_command_buffer(
                &IcbDesc {
                    command_types: 1,
                    ..IcbDesc::default()
                },
                1,
                0,
            )
            .unwrap()
            .unwrap();
        let c = icb.render_command(0).unwrap().unwrap();
        for cmd in [
            IcbRenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&pipeline),
            },
            IcbRenderCmd::DrawPrimitives {
                primitive_type: 3,
                vertex_start: 0,
                vertex_count: 3,
                instance_count: 1,
                base_instance: 0,
            },
        ] {
            c.encode(cmd).unwrap();
        }
        let (_dst, pass) = pass_of(&p, &d, (4, 4));
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        for cmd in [
            RenderCmd::UseResources {
                resources: &[Some(ResourceRef::Icb(&icb))],
                usage,
            },
            RenderCmd::ExecuteCommandsInBuffer {
                indirect_command_buffer: &icb,
                execution_range: NsRange::new(0, 1),
            },
            RenderCmd::EndEncoding,
        ] {
            e.encode(OnFault::Die, cmd).unwrap();
        }
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        panic!("ran: {:?}", (cb.status(), cb.error()));
    }

    /// The records of an indirect command buffer that the runtime refuses die naming the
    /// execution: at the commit where only the CPU wrote them, and as the command buffer ends
    /// where an encoder made the buffer resident for writing. The runtime holds that refusal for
    /// the command buffer's host step, as a command buffer of another queue that this one waits
    /// for through an event, encoded after the commit, may still write the records before the
    /// work runs, and Metal runs what it wrote; the step records the work from the records as
    /// they stand once the work before it completed, and fails the command buffer where they are
    /// still refused, which dies at the execution's selector, as no handler of core's learns of it
    #[test]
    fn refused_icb_records_die_naming_the_execution() {
        use reims_vgpu_core::raise::death::{DeathKind, DeathSite};
        use reims_vgpu_core::testkit::capture::expect_death;

        let site = catch_death(|| execute_a_mismatched_draw(1)).unwrap_err();
        assert_eq!((site.kind, site.file), (DeathKind::Abort, "commit"));
        let site = DeathSite {
            kind: DeathKind::Abort,
            file: "executeCommandsInBuffer:withRange:",
            line: 0,
        };
        expect_death(site, || execute_a_mismatched_draw(3));
    }

    /// The texels of a 4 by 4 BGRA8 target, cleared to black, after `setters` and then each of
    /// `draws`, `(indexed, patch_count, instance_count)`: a patch draw of `render_tess`'s
    /// triangle, which covers the target in `tri_frag`'s colour, plain or over the 32-bit
    /// control point indices 0, 1 and 2, under a state of its kind.
    fn patch_texels(
        setters: Vec<RenderCmd<'_, VulkanPlatform>>,
        draws: &[(bool, u64, u64)],
    ) -> Vec<u8> {
        use metal2vulkan::metal::{PixelFormat, TessellationControlPointIndexType};
        use metal2vulkan::pipeline::{
            ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource,
            TessellationDescriptor,
        };

        const TESS: &[u8] = include_bytes!("../../../../tests/shaders/render_tess.metallib");
        const FRAG: &[u8] = include_bytes!("../../../../tests/shaders/frag.metallib");
        let p = platform();
        let d = dev();
        let function = |bytes, name| {
            d.raw
                .new_library_with_data(bytes)
                .unwrap()
                .new_function_with_name(name)
                .unwrap()
        };
        let pipeline = |indices| {
            let mut desc = RenderPipelineDescriptor::new(StageSource::Function(
                function(TESS, "render_tess").function_descriptor(),
            ));
            desc.fragment = Some(StageSource::Function(
                function(FRAG, "tri_frag").function_descriptor(),
            ));
            desc.color_attachments[0] =
                Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm));
            let mut tessellation = TessellationDescriptor::default();
            tessellation.control_point_index_type = indices;
            desc.tessellation = Some(tessellation);
            crate::gpu::resource::render_pipeline(
                d.raw
                    .new_render_pipeline_state(&desc, &rt::PipelineOptions::default())
                    .unwrap(),
            )
        };
        let (plain, indexed) = (
            pipeline(TessellationControlPointIndexType::None),
            pipeline(TessellationControlPointIndexType::UInt32),
        );
        let points =
            buffer::<VulkanPlatform>(&d, 64, 0, &bytes(&[-3.0, -3.0, 3.0, -3.0, 0.0, 3.0]));
        let factors = buffer::<VulkanPlatform>(&d, 64, 0, &[0x00, 0x3c].repeat(4));
        let indices = buffer::<VulkanPlatform>(&d, 64, 0, &u32s(&[0, 1, 2]));
        let (dst, pass) = pass_of(&p, &d, (4, 4));
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        let mut cmds = setters;
        for &(by_index, patch_count, instance_count) in draws {
            cmds.extend([
                RenderCmd::SetRenderPipelineState {
                    pipeline_state: Some(if by_index { &indexed } else { &plain }),
                },
                RenderCmd::SetVertexBuffer {
                    buffer: Some(&points),
                    offset: 0,
                    index: 0,
                },
                RenderCmd::SetTessellationFactorBuffer {
                    buffer: Some(&factors),
                    offset: 0,
                    instance_stride: 0,
                },
            ]);
            cmds.push(if by_index {
                RenderCmd::DrawIndexedPatches {
                    number_of_patch_control_points: 3,
                    patch_start: 0,
                    patch_count,
                    patch_index_buffer: None,
                    patch_index_buffer_offset: 0,
                    control_point_index_buffer: &indices,
                    control_point_index_buffer_offset: 0,
                    instance_count,
                    base_instance: 0,
                }
            } else {
                RenderCmd::DrawPatches {
                    number_of_patch_control_points: 3,
                    patch_start: 0,
                    patch_count,
                    patch_index_buffer: None,
                    patch_index_buffer_offset: 0,
                    instance_count,
                    base_instance: 0,
                }
            });
        }
        cmds.push(RenderCmd::EndEncoding);
        for cmd in cmds {
            e.encode(OnFault::Die, cmd).unwrap();
        }
        let out = buffer::<VulkanPlatform>(&d, 64, 0, &[]);
        let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
        for cmd in [
            BlitCmd::CopyTextureToBuffer {
                source_texture: &dst,
                source_slice: 0,
                source_level: 0,
                source_origin: MtlOrigin::new(0, 0, 0),
                source_size: MtlSize::new(4, 4, 1),
                destination_buffer: &out,
                destination_offset: 0,
                destination_bytes_per_row: 16,
                destination_bytes_per_image: 64,
            },
            BlitCmd::EndEncoding,
        ] {
            blit.encode(OnFault::Die, cmd).unwrap();
        }
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert_eq!(cb.error(), None);
        // SAFETY: the Shared buffer's 64 bytes, read after the blit completed.
        unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), 64) }.to_vec()
    }

    /// The selector of [`patch_texels`]'s plain patch draw.
    const DRAW_PATCHES: &str = "drawPatches:patchStart:patchCount:patchIndexBuffer:\
                                patchIndexBufferOffset:instanceCount:baseInstance:";

    /// The selector of [`patch_texels`]'s indexed patch draw.
    const DRAW_INDEXED_PATCHES: &str = "drawIndexedPatches:patchStart:patchCount:\
        patchIndexBuffer:patchIndexBufferOffset:controlPointIndexBuffer:\
        controlPointIndexBufferOffset:instanceCount:baseInstance:";

    /// The kind of [`crate::answer::DIES`] a draw of [`patch_texels`]'s pipeline that draws a
    /// patch dies of: the note `TessellationTriangleOrder` on a device whose tessellator draws it
    /// and is not Apple's (`tessellationShader`, on every device but MoltenVK on an Apple GPU),
    /// which the runtime announces ahead of the next; elsewhere `TessellationFactorEdgeCases`,
    /// which it announces at every draw of a tessellation pipeline that draws a patch, whose
    /// factors it does not read, though these factors are 1
    fn patch_draw_death() -> rt::Degradation {
        let d = dev();
        let desc = &d.raw.properties().description;
        let apple = desc.driver_id == Some(metal2vulkan::vk::DriverId::MOLTENVK)
            && desc.vendor_id == 0x106b;
        let tessellator = desc
            .features
            .contains(&metal2vulkan::device::VkFeature::TessellationShader);
        if tessellator && !apple {
            rt::Degradation::TessellationTriangleOrder
        } else {
            rt::Degradation::TessellationFactorEdgeCases
        }
    }

    /// A patch draw of no instances or no patches, plain or indexed, draws nothing and the pass
    /// completes, as Metal's does, whatever the private setters left (a restart, a line width of
    /// 2 and triangles filled as lines). One of a patch and an instance dies at its selector
    /// naming [`patch_draw_death`]'s kind
    #[test]
    fn empty_patch_draws_draw_nothing() {
        let setters = vec![
            RenderCmd::SetPrimitiveRestartEnabled {
                enabled: true,
                index: 0xffff,
            },
            RenderCmd::SetLineWidth { line_width: 2.0 },
            RenderCmd::SetTriangleFillMode { fill_mode: 1 },
        ];
        let empty = [(false, 1, 0), (false, 0, 1), (true, 1, 0), (true, 0, 1)];
        assert_eq!(patch_texels(setters, &empty), [0, 0, 0, 0xff].repeat(16));
        let kind = patch_draw_death();
        for (draw, at) in [
            ((false, 1, 1), DRAW_PATCHES),
            ((true, 1, 1), DRAW_INDEXED_PATCHES),
        ] {
            let site = crate::answer::dies_of(kind, || patch_texels(Vec::new(), &[draw]));
            assert_eq!(site.file, at, "{draw:?}");
        }
    }

    /// Indexed patches drawn after `setPrimitiveRestartEnabled:index:`, enabled or not, and
    /// without it, each die at the draw naming [`patch_draw_death`]'s kind
    #[test]
    fn indexed_patches_draw_after_a_restart_set_as_without_it() {
        let kind = patch_draw_death();
        let site = crate::answer::dies_of(kind, || patch_texels(Vec::new(), &[(true, 1, 1)]));
        assert_eq!(site.file, DRAW_INDEXED_PATCHES);
        for index in [0, 1, 2] {
            for enabled in [false, true] {
                let set = vec![RenderCmd::SetPrimitiveRestartEnabled { enabled, index }];
                let site = crate::answer::dies_of(kind, || patch_texels(set, &[(true, 1, 1)]));
                assert_eq!(site.file, DRAW_INDEXED_PATCHES, "{enabled} {index}");
            }
        }
    }

    /// A render pass executes an indirect command buffer's draws: the triangle of a command's
    /// pipeline state covers the target in its fragment's colour, the codes a plain draw of the
    /// state stores; an execution past the buffer's size dies at its selector, naming the
    /// runtime's refusal.
    ///
    /// The fragment's green, 0.5, is a tie of `BGRA8Unorm`'s codes (127.5): every Mac stores 0x80
    /// (measured: M3 Max, AMD 5300M, Intel 630, in `BGRA8Unorm` and `RGBA8Unorm`), and so does a
    /// device the runtime holds to store Metal's codes (`VkProperty::MetalColorWrites`, MoltenVK
    /// on Apple's GPUs). On any other the device's own conversion stores either neighbour (0x7f on
    /// an RTX 5080), which the runtime records for the state's write as
    /// `device-color-conversion` (`PrecisionNote::DeviceColorConversion`)
    #[test]
    fn a_pass_executes_an_icbs_draws() {
        use metal2vulkan::metal::PixelFormat;
        use metal2vulkan::pipeline::{
            ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource,
        };
        use reims_vgpu_core::platform::icb::{
            IcbRenderCmd, IndirectCommandBuffer as _, IndirectRenderCommand as _,
        };
        use reims_vgpu_core::platform::values::IcbDesc;

        const VERT: &[u8] = include_bytes!("../../../../tests/shaders/vert.metallib");
        const FRAG: &[u8] = include_bytes!("../../../../tests/shaders/frag.metallib");
        let p = platform();
        let d = dev();
        let function = |bytes, name| {
            d.raw
                .new_library_with_data(bytes)
                .unwrap()
                .new_function_with_name(name)
                .unwrap()
        };
        let mut desc = RenderPipelineDescriptor::new(StageSource::Function(
            function(VERT, "tri_vert").function_descriptor(),
        ));
        desc.fragment = Some(StageSource::Function(
            function(FRAG, "tri_frag").function_descriptor(),
        ));
        desc.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm));
        desc.support_indirect_command_buffers = true;
        let pipeline = crate::gpu::resource::render_pipeline(
            d.raw
                .new_render_pipeline_state(&desc, &rt::PipelineOptions::default())
                .unwrap(),
        );
        let icb = d
            .new_indirect_command_buffer(
                &IcbDesc {
                    command_types: 1,
                    ..IcbDesc::default()
                },
                2,
                0,
            )
            .unwrap()
            .unwrap();
        let c = icb.render_command(1).unwrap().unwrap();
        for cmd in [
            IcbRenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&pipeline),
            },
            IcbRenderCmd::ClearBarrier,
            IcbRenderCmd::DrawPrimitives {
                primitive_type: 3,
                vertex_start: 0,
                vertex_count: 3,
                instance_count: 1,
                base_instance: 0,
            },
        ] {
            c.encode(cmd).unwrap();
        }
        let (dst, pass) = pass_of(&p, &d, (4, 4));
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        for cmd in [
            RenderCmd::UseResources {
                resources: &[Some(ResourceRef::Icb(&icb))],
                usage: 1,
            },
            RenderCmd::ExecuteCommandsInBuffer {
                indirect_command_buffer: &icb,
                execution_range: NsRange::new(0, 2),
            },
        ] {
            e.encode(OnFault::Die, cmd).unwrap();
        }
        let site = catch_death(|| {
            e.encode(
                OnFault::Return,
                RenderCmd::ExecuteCommandsInBuffer {
                    indirect_command_buffer: &icb,
                    execution_range: NsRange::new(1, 2),
                },
            )
        })
        .unwrap_err();
        assert_eq!(site.file, "executeCommandsInBuffer:withRange:");
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
        let out = buffer::<VulkanPlatform>(&d, 64, 0, &[]);
        let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
        for cmd in [
            BlitCmd::CopyTextureToBuffer {
                source_texture: &dst,
                source_slice: 0,
                source_level: 0,
                source_origin: MtlOrigin::new(0, 0, 0),
                source_size: MtlSize::new(4, 4, 1),
                destination_buffer: &out,
                destination_offset: 0,
                destination_bytes_per_row: 16,
                destination_bytes_per_image: 64,
            },
            BlitCmd::EndEncoding,
        ] {
            blit.encode(OnFault::Die, cmd).unwrap();
        }
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert_eq!(cb.error(), None);
        // SAFETY: the Shared buffer's 64 bytes, read after the blit completed.
        let texels = unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), 64) };

        // The same state drawn plainly, for the codes the device stores for it.
        let (plain_dst, plain_pass) = pass_of(&p, &d, (4, 4));
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb
            .render_encoder(OnFault::Die, &plain_pass)
            .unwrap()
            .unwrap();
        for cmd in [
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&pipeline),
            },
            RenderCmd::DrawPrimitives {
                primitive_type: 3,
                vertex_start: 0,
                vertex_count: 3,
            },
            RenderCmd::EndEncoding,
        ] {
            e.encode(OnFault::Die, cmd).unwrap();
        }
        let plain_out = buffer::<VulkanPlatform>(&d, 64, 0, &[]);
        let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
        for cmd in [
            BlitCmd::CopyTextureToBuffer {
                source_texture: &plain_dst,
                source_slice: 0,
                source_level: 0,
                source_origin: MtlOrigin::new(0, 0, 0),
                source_size: MtlSize::new(4, 4, 1),
                destination_buffer: &plain_out,
                destination_offset: 0,
                destination_bytes_per_row: 16,
                destination_bytes_per_image: 64,
            },
            BlitCmd::EndEncoding,
        ] {
            blit.encode(OnFault::Die, cmd).unwrap();
        }
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert_eq!(cb.error(), None);
        // SAFETY: the Shared buffer's 64 bytes, read after the blit completed.
        let plain = unsafe { std::slice::from_raw_parts(plain_out.contents().as_ptr::<u8>(), 64) };
        eprintln!(
            "the state's green: {:#04x} executed, {:#04x} drawn plainly",
            texels[1], plain[1]
        );
        assert_eq!(texels, plain);
        let greens: &[u8] = if ties_as_metals(&d) {
            &[0x80]
        } else {
            &[0x7f, 0x80]
        };
        for texel in texels.chunks(4) {
            assert!(
                texel[0] == 0x40 && greens.contains(&texel[1]) && texel[2..] == [0xff, 0xff],
                "{texel:02x?}"
            );
        }
    }

    /// An execution whose commands run pipeline states of their own writes no texture bound on
    /// the encoder ([`Reach::at_slots`]): with a texture on a surface made read-only for the GPU
    /// bound at fragment texture 0, and made resident for reading, it goes on; with the texture
    /// made resident for writing, it dies at the execution
    #[test]
    fn an_execution_of_states_of_its_own_writes_no_bound_texture() {
        use metal2vulkan::metal::PixelFormat;
        use metal2vulkan::pipeline::{
            ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource,
        };
        use reims_vgpu_core::host::PhysicalMemoryRange;
        use reims_vgpu_core::platform::icb::{
            IcbRenderCmd, IndirectCommandBuffer as _, IndirectRenderCommand as _,
        };
        use reims_vgpu_core::platform::os::surface::SurfaceSpec;
        use reims_vgpu_core::platform::os::vm::HostVm as _;
        use reims_vgpu_core::platform::values::IcbDesc;

        const VERT: &[u8] = include_bytes!("../../../../tests/shaders/vert.metallib");
        const FRAG: &[u8] = include_bytes!("../../../../tests/shaders/frag.metallib");
        let p = platform();
        let d = dev();
        let function = |bytes, name| {
            d.raw
                .new_library_with_data(bytes)
                .unwrap()
                .new_function_with_name(name)
                .unwrap()
        };
        let mut desc = RenderPipelineDescriptor::new(StageSource::Function(
            function(VERT, "tri_vert").function_descriptor(),
        ));
        desc.fragment = Some(StageSource::Function(
            function(FRAG, "tri_frag").function_descriptor(),
        ));
        desc.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm));
        desc.support_indirect_command_buffers = true;
        let pipeline = crate::gpu::resource::render_pipeline(
            d.raw
                .new_render_pipeline_state(&desc, &rt::PipelineOptions::default())
                .unwrap(),
        );
        let icb = d
            .new_indirect_command_buffer(
                &IcbDesc {
                    command_types: 1,
                    ..IcbDesc::default()
                },
                1,
                0,
            )
            .unwrap()
            .unwrap();
        let c = icb.render_command(0).unwrap().unwrap();
        for cmd in [
            IcbRenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&pipeline),
            },
            IcbRenderCmd::DrawPrimitives {
                primitive_type: 3,
                vertex_start: 0,
                vertex_count: 3,
                instance_count: 1,
                base_instance: 0,
            },
        ] {
            c.encode(cmd).unwrap();
        }
        let vm = crate::os::OsVm::new();
        let page = vm.page_size();
        let guest = crate::os::allocate_aliasable(page).unwrap();
        let spec = SurfaceSpec {
            pixel_format: u32::from_be_bytes(*b"BGRA"),
            alloc_size: page,
            width: 16,
            height: 16,
            bytes_per_element: 4,
            bytes_per_row: 64,
            element_width: 1,
            element_height: 1,
            client_direction: Some(2),
            client_address_ranges: Some(vec![PhysicalMemoryRange {
                physical_address: reims_vgpu_core::num::ToU64::to_u64(guest.addr()),
                physical_length: page,
            }]),
            ..SurfaceSpec::default()
        };
        let s = p.surfaces().create(&spec).unwrap();
        let read = p.texture_descriptor(&TextureDesc {
            pixel_format: BGRA8,
            width: 16,
            height: 16,
            usage: READ,
            ..TextureDesc::default()
        });
        let t = d
            .new_texture_on_surface(&read, Some(&s), 0)
            .unwrap()
            .unwrap();
        assert!(t.gpu_read_only());
        let execute = |usage: u64| {
            let (_dst, pass) = pass_of(&p, &d, (4, 4));
            let q = d.new_command_queue().unwrap();
            let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
            let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
            let (listed, used) = (
                [Some(ResourceRef::Icb(&icb))],
                [Some(ResourceRef::Texture(&t))],
            );
            for cmd in [
                RenderCmd::SetFragmentTexture {
                    texture: Some(&t),
                    index: 0,
                },
                RenderCmd::UseResources {
                    resources: &listed,
                    usage: 1,
                },
                RenderCmd::UseResources {
                    resources: &used,
                    usage,
                },
                RenderCmd::ExecuteCommandsInBuffer {
                    indirect_command_buffer: &icb,
                    execution_range: NsRange::new(0, 1),
                },
                RenderCmd::EndEncoding,
            ] {
                e.encode(OnFault::Die, cmd).unwrap();
            }
            cb.commit(OnFault::Die).unwrap();
            cb.wait_until_completed(OnFault::Die).unwrap();
            cb.status().0
        };
        assert_eq!(execute(1), 4);
        let site = catch_death(|| execute(3)).unwrap_err();
        assert_eq!(site.file, "executeCommandsInBuffer:withRange:");
        drop((t, s));
        assert!(vm.deallocate(guest, page).is_success());
    }

    /// A pass of a mutable rate map not yet reset rasterizes nothing, a scissor rectangle set in
    /// it too: the colour and depth attachments keep their clears and the fragment function's
    /// write does not land, as on Metal; a tile dispatch in it dies. A reset after such a
    /// pass ended and before its command buffer's commit draws the pass by the map as reset, as
    /// Metal's GPU reads it; so does one after a pass of the map as reset ended, to the same map.
    /// A reset to a map of no layers before that pass has run dies: its draw would have died
    /// there
    #[test]
    fn a_pass_of_a_mutable_rate_map_not_yet_reset_rasterizes_nothing() {
        use metal2vulkan::metal::PixelFormat;
        use metal2vulkan::pipeline::{
            ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource,
        };
        use reims_vgpu_core::platform::rate_map::{
            RateMap as _, RateMapDescriptor as _, RateMapLayerDescriptor as _,
        };
        use reims_vgpu_core::platform::values::{
            DepthAttachmentSet, DepthStencilDesc, PassSet, RateMapDesc,
        };

        const VERT: &[u8] = include_bytes!("../../../../tests/shaders/vert.metallib");
        const FRAG: &[u8] = include_bytes!("../../../../tests/shaders/mutdec_fwrite.metallib");
        /// `MTLPixelFormatDepth32Float`, `MTLCompareFunctionAlways`.
        const DEPTH32F: u64 = 252;
        const ALWAYS: u64 = 7;
        const SIDE: u64 = 32;
        const TEXELS: usize = (SIDE * SIDE) as usize;
        let p = platform();
        let d = dev();
        let function = |bytes, name| {
            d.raw
                .new_library_with_data(bytes)
                .unwrap()
                .new_function_with_name(name)
                .unwrap()
        };
        let mut desc = RenderPipelineDescriptor::new(StageSource::Function(
            function(VERT, "tri_vert").function_descriptor(),
        ));
        desc.fragment = Some(StageSource::Function(
            function(FRAG, "mutdec_fwrite").function_descriptor(),
        ));
        desc.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm));
        desc.depth_format = Some(PixelFormat::Depth32Float);
        let pipeline = crate::gpu::resource::render_pipeline(
            d.raw
                .new_render_pipeline_state(&desc, &rt::PipelineOptions::default())
                .unwrap(),
        );
        let writes_depth = d
            .new_depth_stencil_state(&DepthStencilDesc {
                depth_compare_function: Some(ALWAYS),
                depth_write_enabled: Some(true),
                ..DepthStencilDesc::default()
            })
            .unwrap()
            .unwrap();
        let map_descriptor = |mutability| {
            let m = p.rate_map_descriptor(&RateMapDesc {
                screen_size: MtlSize::new(SIDE, SIDE, 1),
                mutability: Some(mutability),
                ..RateMapDesc::default()
            });
            let layer = p
                .new_rate_map_layer(OnFault::Die, MtlSize::new(1, 1, 0))
                .unwrap()
                .unwrap();
            for at in [
                layer.horizontal_sample_storage(),
                layer.vertical_sample_storage(),
            ] {
                // SAFETY: the layer's storage holds one sample on each axis.
                unsafe { *at.as_ptr::<f32>() = 1.0 };
            }
            m.set_layer(OnFault::Die, &layer, 0).unwrap();
            m
        };
        let map = d.new_rate_map(&map_descriptor(1)).unwrap().unwrap();
        let (dst, pass) = pass_of(&p, &d, (SIDE, SIDE));
        let depth = texture(&p, &d, DEPTH32F, TARGET, (SIDE, SIDE), None);
        pass.set_depth_attachment(&DepthAttachmentSet {
            texture: Some(Some(&depth)),
            load_action: Some(CLEAR),
            store_action: Some(STORE),
            clear_depth: Some(0.5),
            ..DepthAttachmentSet::default()
        });
        pass.set_pass(&PassSet {
            rasterization_rate_map: Some(Some(&map)),
            ..Default::default()
        });
        let q = d.new_command_queue().unwrap();
        let read = |cb: &VulkanCommandBuffer, texture: &VulkanTexture, out: &VulkanBuffer| {
            let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
            for cmd in [
                BlitCmd::CopyTextureToBuffer {
                    source_texture: texture,
                    source_slice: 0,
                    source_level: 0,
                    source_origin: MtlOrigin::new(0, 0, 0),
                    source_size: MtlSize::new(SIDE, SIDE, 1),
                    destination_buffer: out,
                    destination_offset: 0,
                    destination_bytes_per_row: SIDE * 4,
                    destination_bytes_per_image: SIDE * SIDE * 4,
                },
                BlitCmd::EndEncoding,
            ] {
                blit.encode(OnFault::Die, cmd).unwrap();
            }
        };
        let words = |b: &VulkanBuffer| {
            // SAFETY: the Shared buffer's words, read after its command buffer completed.
            unsafe { std::slice::from_raw_parts(b.contents().as_ptr::<u32>(), TEXELS).to_vec() }
        };
        let run = |between: &dyn Fn()| {
            let flag = buffer::<VulkanPlatform>(&d, 4, 0, &[0; 4]);
            let colors = buffer::<VulkanPlatform>(&d, SIDE * SIDE * 4, 0, &[]);
            let depths = buffer::<VulkanPlatform>(&d, SIDE * SIDE * 4, 0, &[]);
            let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
            let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
            for cmd in [
                RenderCmd::SetRenderPipelineState {
                    pipeline_state: Some(&pipeline),
                },
                RenderCmd::SetDepthStencilState {
                    depth_stencil_state: Some(&writes_depth),
                },
                RenderCmd::SetFragmentBuffer {
                    buffer: Some(&flag),
                    offset: 0,
                    index: 0,
                },
                RenderCmd::SetScissorRect {
                    rect: MtlScissorRect {
                        x: 0,
                        y: 0,
                        width: SIDE,
                        height: SIDE,
                    },
                },
                RenderCmd::DrawPrimitives {
                    primitive_type: 3,
                    vertex_start: 0,
                    vertex_count: 3,
                },
                RenderCmd::EndEncoding,
            ] {
                e.encode(OnFault::Die, cmd).unwrap();
            }
            read(&cb, &dst, &colors);
            read(&cb, &depth, &depths);
            between();
            cb.commit(OnFault::Die).unwrap();
            cb.wait_until_completed(OnFault::Die).unwrap();
            assert_eq!(cb.error(), None);
            (words(&colors), words(&depths), words(&flag)[0])
        };
        let clear = u32::from_le_bytes([0, 0, 0, 0xff]);
        let green = u32::from_le_bytes([0, 0xff, 0, 0xff]);
        assert_eq!(
            run(&|| {}),
            (vec![clear; TEXELS], vec![0.5f32.to_bits(); TEXELS], 0)
        );
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        // A tile dispatch in the pass, which the runtime's would run over every tile, dies.
        let site = catch_death(|| {
            e.encode(
                OnFault::Die,
                RenderCmd::DispatchThreadsPerTile {
                    threads_per_tile: MtlSize::new(1, 1, 1),
                },
            )
        })
        .unwrap_err();
        assert_eq!(site.file, "dispatchThreadsPerTile:");
        for cmd in [
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&pipeline),
            },
            RenderCmd::DrawPrimitives {
                primitive_type: 3,
                vertex_start: 0,
                vertex_count: 3,
            },
            RenderCmd::EndEncoding,
        ] {
            e.encode(OnFault::Die, cmd).unwrap();
        }
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        let drawn = (vec![green; TEXELS], vec![0.0f32.to_bits(); TEXELS], 1);
        assert_eq!(run(&|| map.reset(&map_descriptor(2))), drawn);
        assert_eq!(run(&|| map.reset(&map_descriptor(2))), drawn);
        let none = p.rate_map_descriptor(&RateMapDesc {
            screen_size: MtlSize::new(SIDE, SIDE, 1),
            ..RateMapDesc::default()
        });
        let site = catch_death(|| run(&|| map.reset(&none))).unwrap_err();
        assert_eq!(site.file, "resetUsingDescriptor:");
    }

    /// A rate map descriptor of a `side` by `side` screen, of `mutability`, with one layer of
    /// one sample of 1 on each axis.
    fn rate_map_descriptor(
        p: &VulkanPlatform,
        side: u64,
        mutability: u64,
    ) -> crate::gpu::rate_map::VulkanRateMapDescriptor {
        use reims_vgpu_core::platform::rate_map::{
            RateMapDescriptor as _, RateMapLayerDescriptor as _,
        };
        use reims_vgpu_core::platform::values::RateMapDesc;

        let m = p.rate_map_descriptor(&RateMapDesc {
            screen_size: MtlSize::new(side, side, 1),
            mutability: Some(mutability),
            ..RateMapDesc::default()
        });
        let layer = p
            .new_rate_map_layer(OnFault::Die, MtlSize::new(1, 1, 0))
            .unwrap()
            .unwrap();
        for at in [
            layer.horizontal_sample_storage(),
            layer.vertical_sample_storage(),
        ] {
            // SAFETY: the layer's storage holds one sample on each axis.
            unsafe { *at.as_ptr::<f32>() = 1.0 };
        }
        m.set_layer(OnFault::Die, &layer, 0).unwrap();
        m
    }

    /// A reset of a rate map to one the class a committed pass took does not hold over (from the
    /// identity to no layers) dies while that pass may still run, after core dropped its handles
    /// of the command buffer and the encoder, as it does right after the commit; once that
    /// command buffer completed, the reset is taken. A pass whose creation answers no encoder
    /// leaves the map free to reset
    #[test]
    fn a_reset_waits_for_a_committed_pass_core_let_go_of() {
        use std::sync::mpsc;

        use reims_vgpu_core::platform::event::{EventRef, SharedEvent as _};
        use reims_vgpu_core::platform::rate_map::RateMap as _;
        use reims_vgpu_core::platform::values::PassSet;

        use reims_vgpu_core::platform::values::RateMapDesc;

        let p = platform();
        let d = dev();
        let map = d
            .new_rate_map(&rate_map_descriptor(&p, 32, 2))
            .unwrap()
            .unwrap();
        let none = p.rate_map_descriptor(&RateMapDesc {
            screen_size: MtlSize::new(32, 32, 1),
            ..RateMapDesc::default()
        });
        let (_dst, pass) = pass_of(&p, &d, (32, 32));
        pass.set_pass(&PassSet {
            rasterization_rate_map: Some(Some(&map)),
            ..Default::default()
        });
        let q = d.new_command_queue().unwrap();
        let event = d.new_shared_event().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        cb.encode_wait(Some(EventRef::Shared(&event)), 1).unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
        let (tx, rx) = mpsc::channel();
        cb.add_completed_handler(OnFault::Die, Box::new(move |_| tx.send(()).unwrap()))
            .unwrap();
        cb.commit(OnFault::Die).unwrap();
        drop((e, cb));
        let site = catch_death(|| map.reset(&none)).unwrap_err();
        assert_eq!(site.file, "resetUsingDescriptor:");
        event.set_signaled_value(1).unwrap();
        rx.recv().unwrap();
        map.reset(&none);
        let empty = p.render_pass_descriptor();
        empty.set_pass(&PassSet {
            rasterization_rate_map: Some(Some(&map)),
            ..Default::default()
        });
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        assert!(cb.render_encoder(OnFault::Die, &empty).unwrap().is_none());
        map.reset(&rate_map_descriptor(&p, 32, 2));
    }

    /// A rate map whose rates vary is made and relayed (`rasterization-rate-map`), at an
    /// immutable map's creation and at a mutable map's reset, and a pass over either that only
    /// clears and stores runs, the mutable one's ending with a draw of no vertices
    /// ([`crate::gpu::rate_map::no_draw`]); a draw and a tile dispatch in either die at their
    /// selectors, naming the kind.
    /// A map whose tiles span as many screen pixels as physical ones announces nothing, at its
    /// creation or a reset, and a draw in a pass over it is taken
    #[test]
    fn draws_over_a_map_of_varying_rates_die() {
        use reims_vgpu_core::platform::rate_map::{
            RateMap as _, RateMapDescriptor as _, RateMapLayerDescriptor as _,
        };
        use reims_vgpu_core::platform::values::{PassSet, RateMapDesc};
        use reims_vgpu_core::raise::death::DeathKind;

        use crate::gpu::rate_map::{VulkanRateMap, VulkanRateMapDescriptor};

        const SIDE: u64 = 256;
        /// `MTLMutabilityMutable`, `MTLMutabilityImmutable`.
        const MUTABLE: u64 = 1;
        const IMMUTABLE: u64 = 2;
        const VARYING: [f32; 2] = [0.25, 1.0];
        const IDENTITY: [f32; 2] = [1.0, 1.0];

        /// A descriptor of a `SIDE` by `SIDE` screen of `mutability`, with one layer of two
        /// samples on each axis, `rates`.
        fn of_rates(
            p: &VulkanPlatform,
            mutability: u64,
            rates: [f32; 2],
        ) -> VulkanRateMapDescriptor {
            let desc = p.rate_map_descriptor(&RateMapDesc {
                screen_size: MtlSize::new(SIDE, SIDE, 1),
                mutability: Some(mutability),
                ..RateMapDesc::default()
            });
            let layer = p
                .new_rate_map_layer(OnFault::Die, MtlSize::new(2, 2, 0))
                .unwrap()
                .unwrap();
            for at in [
                layer.horizontal_sample_storage(),
                layer.vertical_sample_storage(),
            ] {
                // SAFETY: the layer's storage holds two samples on each axis.
                unsafe { std::ptr::copy_nonoverlapping(rates.as_ptr(), at.as_ptr::<f32>(), 2) };
            }
            desc.set_layer(OnFault::Die, &layer, 0).unwrap();
            desc
        }

        /// How many times the runtime announced the kind in this process.
        fn announced() -> usize {
            (crate::answer::RELAYED.lock().iter())
                .filter(|&&k| k == rt::Degradation::RasterizationRateMap)
                .count()
        }

        let p = platform();
        let d = dev();
        let state = fixture_state::<VulkanPlatform>(&d, "vert:tri_vert", "frag:tri_frag");
        let q = d.new_command_queue().unwrap();
        let draw = || RenderCmd::DrawPrimitives {
            primitive_type: 3,
            vertex_start: 0,
            vertex_count: 3,
        };
        // A pass over `map` running `cmds` after the state, committed and completed.
        let run = |map: &VulkanRateMap, cmds: Vec<RenderCmd<'_, VulkanPlatform>>| {
            let (_dst, pass) = pass_of(&p, &d, (SIDE, SIDE));
            pass.set_pass(&PassSet {
                rasterization_rate_map: Some(Some(map)),
                ..Default::default()
            });
            let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
            let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
            let set = RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&state),
            };
            for cmd in std::iter::once(set).chain(cmds) {
                e.encode(OnFault::Die, cmd).unwrap();
            }
            e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
            cb.commit(OnFault::Die).unwrap();
            cb.wait_until_completed(OnFault::Die).unwrap();
            assert!(cb.error().is_none(), "{:?}", cb.error());
        };

        let before = announced();
        let identity = d
            .new_rate_map(&of_rates(&p, IMMUTABLE, IDENTITY))
            .unwrap()
            .unwrap();
        let reset = d
            .new_rate_map(&of_rates(&p, MUTABLE, IDENTITY))
            .unwrap()
            .unwrap();
        reset.reset(&of_rates(&p, IMMUTABLE, IDENTITY));
        assert_eq!(announced(), before, "an identity map announced");
        run(&identity, vec![draw()]);
        run(&reset, vec![draw()]);
        let made = d
            .new_rate_map(&of_rates(&p, IMMUTABLE, VARYING))
            .unwrap()
            .unwrap();
        assert_eq!(announced(), before + 1);
        reset.reset(&of_rates(&p, IMMUTABLE, VARYING));
        assert_eq!(announced(), before + 2);
        run(&made, Vec::new());
        run(&reset, Vec::new());
        for map in [&made, &reset] {
            let (_dst, pass) = pass_of(&p, &d, (SIDE, SIDE));
            pass.set_pass(&PassSet {
                rasterization_rate_map: Some(Some(map)),
                ..Default::default()
            });
            let tile = RenderCmd::DispatchThreadsPerTile {
                threads_per_tile: MtlSize::new(1, 1, 1),
            };
            for cmd in [draw(), tile] {
                let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
                let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
                e.encode(
                    OnFault::Die,
                    RenderCmd::SetRenderPipelineState {
                        pipeline_state: Some(&state),
                    },
                )
                .unwrap();
                *crate::answer::LAST_LINE.lock() = None;
                let sel = cmd.selector();
                let site = catch_death(|| e.encode(OnFault::Die, cmd)).unwrap_err();
                let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
                assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{line}");
                assert!(line.contains("(rasterization-rate-map)"), "{line}");
            }
        }
    }

    /// A run of commands ([`RenderEncoder::encoding`]) holds the runtime's encoder and what the
    /// encoder holds of its draws for the thread that sends them: its commands take both again
    /// on that thread, and run as they would alone; another thread takes neither until the run
    /// ends, which lets both go
    #[test]
    fn a_run_of_commands_holds_the_encoder_for_its_thread() {
        let p = platform();
        let d = dev();
        let (_dst, pass) = pass_of(&p, &d, (4, 4));
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        e.encoding(|| {
            assert!(e.encoder.raw.is_owned_by_current_thread());
            assert!(e.drawing.is_owned_by_current_thread());
            std::thread::scope(|s| {
                s.spawn(|| {
                    assert!(e.encoder.raw.try_lock().is_none());
                    assert!(e.drawing.try_lock().is_none());
                });
            });
            e.encode(OnFault::Die, RenderCmd::SetCullMode { cull_mode: 1 })
                .unwrap();
            e.encode(
                OnFault::Die,
                RenderCmd::SetFrontFacingWinding {
                    front_facing_winding: 1,
                },
            )
            .unwrap();
        });
        assert!(!e.encoder.raw.is_locked());
        assert!(!e.drawing.is_locked());
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
        assert!(e.encoder.raw.lock().borrow().is_none());
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert_eq!(cb.error(), None);
    }

    /// The platform's no-draw pipeline is made only for a pass over a mutable map that no draw of
    /// the guest's took the place of, at the pass's end, once per device: after passes that draw
    /// 3 vertices of a triangle list and of a strip, once, the next plain state's
    /// `uniqueIdentifier` is one past the last; a pass whose only draw is of 2 vertices keeps the
    /// platform's draw, whose pipeline takes the number between; a pass that draws nothing then
    /// reuses it. Every pass completes with no error, none failed as a GPU hang
    #[test]
    fn no_draw_moves_later_plain_identifiers_only_for_a_pass_that_drew_nothing() {
        use reims_vgpu_core::platform::rate_map::RateMap as _;
        use reims_vgpu_core::platform::values::PassSet;

        /// `MTLPrimitiveTypeTriangle` and `TriangleStrip`.
        const TRIANGLE: u64 = 3;
        const STRIP: u64 = 4;
        const SIDE: u64 = 16;
        let p = platform();
        let d = dev();
        let state = fixture_state::<VulkanPlatform>(&d, "vert:tri_vert", "frag:tri_frag");
        let map = d
            .new_rate_map(&rate_map_descriptor(&p, SIDE, 1))
            .unwrap()
            .unwrap();
        map.reset(&rate_map_descriptor(&p, SIDE, 2));
        let next = || {
            fixture_state::<VulkanPlatform>(&d, "vert:tri_vert", "frag:tri_frag")
                .unique_identifier()
        };
        let pass = |draw: Option<(u64, u64)>| {
            let (_dst, pass) = pass_of(&p, &d, (SIDE, SIDE));
            pass.set_pass(&PassSet {
                rasterization_rate_map: Some(Some(&map)),
                ..Default::default()
            });
            let q = d.new_command_queue().unwrap();
            let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
            let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
            let set = RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&state),
            };
            e.encode(OnFault::Die, set).unwrap();
            if let Some((primitive_type, vertex_count)) = draw {
                let cmd = RenderCmd::DrawPrimitives {
                    primitive_type,
                    vertex_start: 0,
                    vertex_count,
                };
                e.encode(OnFault::Die, cmd).unwrap();
            }
            e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
            cb.commit(OnFault::Die).unwrap();
            cb.wait_until_completed(OnFault::Die).unwrap();
            assert!(cb.error().is_none(), "{draw:?}: {:?}", cb.error());
        };
        let first = next();
        pass(Some((TRIANGLE, 3)));
        let after_triangles = next();
        assert_eq!(after_triangles, first + 1, "a triangle list drawn");
        pass(Some((STRIP, 3)));
        let after_strip = next();
        assert_eq!(after_strip, after_triangles + 1, "a strip drawn");
        pass(Some((TRIANGLE, 2)));
        let after_two = next();
        assert_eq!(
            after_two,
            after_strip + 2,
            "2 vertices: the platform's draw"
        );
        pass(None);
        assert_eq!(
            next(),
            after_two + 1,
            "nothing drawn: the pipeline made before"
        );
    }

    /// A nil entry of useHeaps:count: or of useHeaps:count:stages: is skipped and the pass runs,
    /// as on Metal
    #[test]
    fn a_nil_heap_is_skipped_where_metal_goes_on() {
        let (e, _dst) = open_encoder();
        e.encode(OnFault::Die, RenderCmd::UseHeaps { heaps: &[None] })
            .unwrap();
        e.encode(
            OnFault::Die,
            RenderCmd::UseHeapsStages {
                heaps: &[None],
                stages: 3,
            },
        )
        .unwrap();
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
    }

    /// A nil entry of useResources:count:usage: or of useResources:count:usage:stages: is skipped
    /// and the pass runs, as on Metal
    #[test]
    fn a_nil_resource_is_skipped_where_metal_goes_on() {
        let d = dev();
        let (e, _dst) = open_encoder();
        let b = buffer::<VulkanPlatform>(&d, 16, 0, &[]);
        e.encode(
            OnFault::Die,
            RenderCmd::UseResources {
                resources: &[None, Some(ResourceRef::Buffer(&b)), None],
                usage: 1,
            },
        )
        .unwrap();
        e.encode(
            OnFault::Die,
            RenderCmd::UseResources {
                resources: &[None],
                usage: 1,
            },
        )
        .unwrap();
        e.encode(
            OnFault::Die,
            RenderCmd::UseResourcesStages {
                resources: &[None, Some(ResourceRef::Buffer(&b))],
                usage: 1,
                stages: 1,
            },
        )
        .unwrap();
        e.encode(
            OnFault::Die,
            RenderCmd::UseResourcesStages {
                resources: &[None],
                usage: 1,
                stages: 1,
            },
        )
        .unwrap();
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
    }

    /// A command after `endEncoding` dies at its selector
    #[test]
    fn a_command_after_end_encoding_dies() {
        let p = platform();
        let d = dev();
        let (_dst, pass) = pass_of(&p, &d, (4, 4));
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
        let site = catch_death(|| {
            e.encode(
                OnFault::Die,
                RenderCmd::SetViewport {
                    viewport: MtlViewport {
                        origin_x: 0.0,
                        origin_y: 0.0,
                        width: 4.0,
                        height: 4.0,
                        znear: 0.0,
                        zfar: 1.0,
                    },
                },
            )
        })
        .unwrap_err();
        assert_eq!(site.file, "setViewport:");
    }

    mod many_passes;

    mod mesh;

    #[cfg(target_os = "macos")]
    pub(crate) mod metal_tests;
}
