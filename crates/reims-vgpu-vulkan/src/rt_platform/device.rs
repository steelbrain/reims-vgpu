//! The rt platform's device ([`RtDevice`]): one metal2vulkan-runtime device for the process, made
//! of the physical device the library's choice of device picks ([`open`]), with its answers
//! (`super::answers`), its command queues (`super::queue`), its buffers (`super::buffer`), heaps
//! (`super::heap`) and the blocks it knows ([`KnownBlocks`]), its textures (`super::texture`),
//! events and fences, and the reservation of resource indices as rt's index grants. Every other
//! creator is refused while its piece is unbuilt (`super::unbuilt`), as Metal's creator answers
//! where it fails.
//!
//! **One device.** The process makes one rt device, the first time a device is asked for, and
//! every ask answers it (one `VkDevice` per VM, the host process), until the release of the last
//! graphics device lets go of it (`RtPlatform::let_go_of_device`); the next ask makes a new one.
//! A failure to make it is counted once as rt's refusal and again at each later ask, which
//! answers nil.
//!
//! **The choice.** The candidates are the physical devices that meet rt's floor, which rt lists in
//! the loader's order and picks none of (`Instance::physical_devices`, read from code). The pick
//! among them is the library's (`crate::device_choice::chosen_untimed`): the device the host's
//! user's selection variables name, else the one GPU, else the first by kind: discrete GPUs, then
//! integrated, virtual and CPU ones, each kind in the loader's order. It times no GPU, so a weak
//! discrete GPU beside a strong integrated one is picked wrongly: an RX 550 beside a Radeon 780M,
//! which composites a 3840x2400 frame in 24.4 ms to the RX 550's 40.4 ms (**measured**,
//! 2026-11-08, Mesa 26.2.6, by reims-vgpu's `composite_cost`). There the host's user's selection
//! variable is the answer, as the code reads it (`crate::device_choice::SELECTION_VARIABLES`):
//! `MESA_VK_DEVICE_SELECT` (`MESA_VK_DEVICE_SELECT=1002:15bf` names that 780M by its vendor and
//! device IDs), `DRI_PRIME` or `VK_LOADER_DEVICE_SELECT`. The library's rule that a candidate
//! lists the device extensions a display's window takes holds here: the window takes
//! `VK_KHR_swapchain` alone, which rt's floor takes, so it passes over no device that meets the
//! floor.
//!
//! **Its window system.** The instance enables the OS's surface extensions and present fences'
//! where the loader lists them, and the device the extension of present fences where it lists
//! it and its instance extensions are on (`crate::os`'s `rt_instance_extensions` and
//! `rt_present_fences`); `VK_KHR_surface` and `VK_KHR_swapchain` are rt's floor's. A display's
//! window presents through a swapchain on the device (`super::swapchain`), which keeps the
//! instance, and the device's DRM nodes tell its window whether the compositor composites on
//! this GPU ([`RtDevice::composited_here`]).
//!
//! **Its registry ID** folds its `deviceUUID` (`super::answers::registry_id`).
//!
//! **Its over-copies.** Where the device's linear image cannot hold a texture over a buffer in
//! place, rt keeps the texels in an image of its own and copies the bytes between the two,
//! counting each copy and its bytes (rt's `Metrics`). The process's counts line carries them as
//! amounts ([`fold_metrics`], which `crate::answer::write_counts` runs first):
//! [`OVER_COPIES_CODE`] and [`OVER_COPY_BYTES_CODE`]; of those over an imported block,
//! [`OVER_COPIES_IMPORTED_CODE`] and [`OVER_COPY_BYTES_IMPORTED_CODE`]; and by the cause rt gives
//! each copy, `rt-over-copies-<cause>` and `rt-over-copy-bytes-<cause>` ([`over_copy_counts`]).
//! rt counts each copy under one cause, so the causes sum to the over-copies; a copy over an
//! imported block is counted under its import's cause too, so the imported ones are an axis of
//! their own and no cause.
//!
//! **Its over-copies' kinds.** rt keeps the first 256 kinds of texture each device over-copied,
//! in the order it first copied them: the cause, and the texture's pixel format, size and usage
//! (`Counters::over_copy_kinds`). The counts' folds read the kinds a device added since the last
//! ([`fold_metrics`], and [`follow`]'s of the device before the next), and each kind the process
//! had not logged goes to the log at info once ([`note_over_copy_kinds`]), by its fields
//! ([`over_copy_kind_line`]): a kind two devices copied is logged once, at the first fold that
//! read it. A copy of a kind rt did not keep, the 256 being full, is counted as an amount,
//! [`OVER_COPIES_UNKEPT_KIND_CODE`] (rt's `Metrics::over_copies_unkept_kind`): where it is not
//! 0, some kinds were not logged.
//!
//! **Its command buffers held for room.** rt keeps on its `VkQueue` only what the queue has room
//! for, and holds a command buffer scheduled behind a group it could not join until there is room
//! (rt's `Metrics::held_for_room`). The counts line carries how many as an amount too,
//! [`HELD_FOR_ROOM_CODE`], folded with the over-copies ([`folded_counts`]).
//!
//! **Its cache writes.** rt hands the lowerings, refusals and render contents' records it keeps in
//! its cache directory to a writer off the guest's threads, and drops one rather than wait while
//! those waiting hold 8 MiB (rt's `Metrics::cache_writes_dropped`). The counts line carries the
//! dropped writes, [`CACHE_WRITES_DROPPED_CODE`], and the bytes written to the directory, its
//! pipeline cache blob's among them, [`CACHE_BYTES_WRITTEN_CODE`] (`Metrics::cache_bytes_written`),
//! as amounts: rt counts the writes it kept by their bytes alone.
//!
//! **Its translations.** Beside the cache's bytes the counts line carries, as amounts, the
//! modules rt's translator lowered, [`TRANSLATIONS_CODE`] (`Metrics::translations`: of a compute
//! content, its partial-threadgroups variant, the plain module of a kernel that observes its
//! threadgroup partition, which its state keeps beside the variant, and an emulated-SIMD-group
//! variant; of a render content, each function the cache directory did not hold), and the
//! lowered kernels pipeline states took from the cache directory in their stead,
//! [`TRANSLATIONS_LOADED_CODE`] (`Metrics::translations_loaded`), the render contents' records
//! pipeline states read from the directory, [`RENDER_RECORDS_READ_CODE`]
//! (`Metrics::render_records_read`, each record's stages among the translations loaded), and those
//! that named a lowering the directory no longer held, [`RENDER_RECORDS_STALE_CODE`]
//! (`Metrics::render_records_stale`), and the most bytes of the device's pipeline cache blob,
//! [`PIPELINE_CACHE_MOST_CODE`] (below), so that a boot's reader can tell a boot on a cold
//! directory, which lowers what a warm one loads, and set the bytes it wrote beside its blob's
//! size. Beside the translations it carries the two sides of the pipelines' makes, as amounts: the
//! nanoseconds rt's calls into the translator took, [`TRANSLATION_NS_CODE`]
//! (`Metrics::translation_nanos`), and the `VkPipeline`s rt made, [`PIPELINES_CREATED_CODE`]
//! (`Metrics::pipelines_created`), with the nanoseconds the driver's calls that made them took,
//! [`PIPELINE_CREATION_NS_CODE`] (`Metrics::pipeline_creation_nanos`). Calls on several threads at
//! once each count their own time, so where they overlap the nanoseconds pass the time that went
//! by. The counts line carries only the codes counted above zero, so an absent code reads as 0.
//!
//! **Its render stages past the kernels' bound.** rt compiles a kernel whose stage weighs more
//! than the kernels' stage bound only in the compile helper, and a render stage that heavy in the
//! process, up to the render stages' own bound, past which it refuses the state (rt's README,
//! "Kernels past the stage bound"). The counts line carries, as an amount, the render pipelines rt
//! translated with a stage past the kernels' bound, the refused among them,
//! [`RENDER_MAKES_PAST_KERNEL_BOUND_CODE`] (`Metrics::render_makes_past_kernel_bound`), and the
//! heaviest such stage's weight, [`RENDER_WEIGHT_PAST_KERNEL_BOUND_MOST_CODE`]
//! (`Metrics::render_weight_past_kernel_bound_max`). The weight is a maximum since the device was
//! made, not a total, so the counts line carries the highest over the process's devices, as an
//! amount risen to it ([`note_highest_metrics`]), as it does the gauges' below.
//!
//! **Its compute dispatches' first-use waits.** rt makes a compute kernel's `VkPipeline` for each
//! dispatch variant (threadgroup size, threadgroup memory lengths, tiles) at its first use where
//! no make ahead made it, and the dispatch's encoding waits for that make, or for the one another
//! thread is running (rt's `Metrics::first_use_variant_waits`), on every compute state. The
//! counts line carries the waits, [`FIRST_USE_WAITS_CODE`], and their nanoseconds,
//! [`FIRST_USE_WAIT_NS_CODE`], as amounts, and the longest, [`FIRST_USE_WAIT_MAX_NS_CODE`], as an
//! amount risen to it over the process's devices ([`note_highest_metrics`]); reims-vgpu's slow
//! answers count those that met a guest answer of a frame or more, with the compute dispatches'
//! waits for an unfinished main pipeline below (`super::overlap`).
//!
//! **Its states' finishes.** rt's creators answer before the driver's create of a state's main
//! pipeline, which the state's finish then makes on the device's compile thread, after the
//! create's reply and before its store (`finish_pipeline`, `super::pipeline`, "Its main
//! pipeline"), and, for the display's present pipelines, on the thread that made them, before
//! the display keeps them (core's `DisplayPipelineCache`); a dispatch or draw that needs the main
//! pipeline before a finish made it waits for the finish in flight, or makes it itself (rt's
//! `Metrics::first_use_unfinished_waits`, a share of its first-use waits apart from the
//! variants'), so those waits are the guest's states' alone. The counts line carries the main
//! pipelines the finishes made, [`FINISHED_PIPELINES_CODE`], and their nanoseconds,
//! [`FINISHED_PIPELINE_NS_CODE`], as amounts; the waits on an unfinished main pipeline,
//! [`UNFINISHED_WAITS_CODE`], and their nanoseconds, [`UNFINISHED_WAIT_NS_CODE`], as amounts; and
//! the longest, [`UNFINISHED_WAIT_MAX_NS_CODE`], as an amount risen to it over the process's
//! devices ([`note_highest_metrics`]). rt's finishes that failed (`Metrics::failed_finishes`) are
//! not folded: the device counts each as the create's late failure, reims-vgpu's
//! `pipeline-finish-failed`, at the platform's seam, where every platform's finish answers, a
//! raise or death in the platform's refusal scope among them, which rt never sees, and where the
//! failure's guest-visible effect is decided, so the counts line carries each failure once.
//!
//! **Its exported blocks whose address went out.** At a submission that writes through a device
//! address, rt owes the textures over an exported block their copy in only once a public call
//! handed out an address in that block: a buffer's `gpu_address` in it, or a texture buffer's
//! storage element over its bytes (rt's `Metrics::exported_blocks_addressed`, each block counted
//! at the first). The block's own handle states none (`BlockInfo::gpu_address` is `None` for an
//! exported block), so a block's make counts nothing, and a video cache's buffer in a block of its
//! own (`super::export`) counts its block only once its address is read. The counts line carries
//! them as an amount, [`EXPORTED_BLOCKS_ADDRESSED_CODE`], so a boot's line says whether any
//! exported block's address went out, and with it whether such a submission could copy a texture
//! over one in.
//!
//! **Its pipeline cache past its bound.** rt's reaper replaces the device's `VkPipelineCache` with
//! a fresh, empty one at a look that finds it past its bound of 128 MiB, and the pipelines the old
//! one held compile again where made again (rt's `Metrics::pipeline_caches_replaced`); and rt skips
//! a write of the cache's data over the device's blob where the data is past that bound, as at the
//! device's teardown where the cache grew past it after the reaper's last look, so the blob keeps
//! what it held (`Metrics::pipeline_cache_writes_skipped`). The counts line carries both as
//! amounts, [`PIPELINE_CACHES_REPLACED_CODE`] and [`PIPELINE_CACHE_WRITES_SKIPPED_CODE`], so a
//! boot's line says whether its pipelines met a fresh cache, and whether the blob the next boot
//! seeds from missed a write.
//!
//! **Its pipeline cache's looks skipped.** rt's reaper reads the cache's size, which
//! [`PIPELINE_CACHE_MOST_CODE`]'s gauge holds from the seed on, and the data it writes over the
//! blob only while no creation runs: a look where a creation began after creation was quiet is
//! skipped, and the reaper looks again once creation is quiet again (rt's
//! `Metrics::pipeline_cache_looks_skipped`). The counts line carries the looks skipped as an
//! amount, [`PIPELINE_CACHE_LOOKS_SKIPPED_CODE`], so that a boot's line whose cache size stays at
//! its seed's says whether the reaper's looks met creations. rt counts neither the looks it made
//! nor the blob's writes in a metric of its own; the blob's bytes are among
//! [`CACHE_BYTES_WRITTEN_CODE`]'s.
//!
//! **Its creations before the seed.** rt's reaper reads the device's blob into its
//! `VkPipelineCache` off the guest's threads, and a `VkPipeline` made before that is done goes
//! through the empty cache: it compiles without the blob's data and is not in this run's blob,
//! so a later run compiles it again (rt's `Metrics::creations_before_seed`). The counts line
//! carries them as an amount, [`CREATIONS_BEFORE_SEED_CODE`], so a boot's line says whether
//! pipelines were made before the blob was in.
//!
//! **Its emulated SIMD groups and its dropped makes.** rt counts the dispatches, direct and
//! indirect, that ran a kernel's emulated-SIMD-group module, a threadgroup not whole groups of the
//! width the kernel's own module runs, as 80 threads, or 32 on a device of 64-lane subgroups
//! (`Metrics::emulated_simdgroup_dispatches`); and the compiled contents, compute or render, a
//! state's make dropped because an equal one landed in the device's map of contents first, the
//! state taking the map's (`Metrics::kernel_makes_dropped`). The counts line carries both as
//! amounts, [`EMULATED_SIMDGROUP_DISPATCHES_CODE`] and [`KERNEL_MAKES_DROPPED_CODE`], so a boot's
//! line says how often its kernels took the emulated SIMD-group form, and whether two makes of one
//! content compiled it at once.
//!
//! **Its resolve copies.** A pass that resolves into a texture in place over a buffer whose linear
//! image takes no attachment usage resolves into the texture's resolve image, a one-sample image rt
//! makes at the first such pass and frees with the texture, and as the pass ends rt copies the
//! pass's render area from it into the buffer's bytes at the texture's pitch, a copy Metal does
//! not make (`Metrics::resolve_copies`, one a pass, and their bytes, `resolve_copy_bytes`, which
//! count in `bytes_copied` too). The counts line carries both as amounts, [`RESOLVE_COPIES_CODE`]
//! and [`RESOLVE_COPY_BYTES_CODE`], so a boot's line says whether its resolves into textures over
//! buffers went in place or through a resolve image, and the images go to it at their highest
//! (below). rt also counts every resolve of a pass's multisample attachment, one per attachment a
//! pass resolves as its encoder ends (`Metrics::resolves`), an amount on the counts line,
//! [`RESOLVES_CODE`]: on a device that resolves in place, where the copies and the images stay 0,
//! it is what shows that a boot's resolves ran.
//!
//! **Its partial-threadgroups variant's grids.** A kernel that observes its threadgroup
//! partition runs its whole grids (`dispatchThreads` of whole threadgroups,
//! `dispatchThreadgroups`) and its indirect dispatches on its plain module at their own
//! threadgroup size, and its ragged grids on its partial-threadgroups variant, so rt's state
//! lowers both modules for it, where Metal compiles one function. rt counts the dispatches of
//! whole threadgroups, direct and indirect, that ran the variant
//! (`Metrics::partial_variant_whole_grids`), which none does, and the counts line carries them as
//! an amount, [`PARTIAL_VARIANT_WHOLE_GRIDS_CODE`], so a boot's line shows that its whole grids
//! ran the plain module. A ragged grid, `dispatchThreads` of a grid that is not a whole number of
//! threadgroups, runs only on the variant, its regions flagged so that each threadgroup runs only
//! the grid's threads; one the variant cannot take is rt's refusal (`Gap::PartialThreadgroups`),
//! as is one whose threadgroup memory, the variant's predicated walk's 4-byte vote word included,
//! passes the device's `maxComputeSharedMemorySize`, decided per dispatch from the encoder's
//! lengths. rt counts the ragged grids that ran the variant apart
//! (`Metrics::partial_variant_ragged_grids`; a refused one counts in neither), and the counts line
//! carries them as an amount too, [`PARTIAL_VARIANT_RAGGED_GRIDS_CODE`], so a boot's line says how
//! many of its ragged grids ran.
//!
//! **Its views written through an extended usage.** A texture made for shader writes whose own
//! format the device does not store (an 8-bit sRGB format) is made, where the device's image
//! format query takes it, with `VK_IMAGE_CREATE_EXTENDED_USAGE_BIT` and the storage usage for its
//! views of a format it lists that the device does store (the sRGB format's unorm sibling), so a
//! kernel writes the texture through such a view, as Metal does. Where the query does not take it,
//! a dispatch that writes the view is rt's refusal, `Gap::StorageElement` of case `Unextended`.
//! rt counts the dispatches whose kernel writes, at one of its texture slots, a view through the
//! storage usage its texture's image took for that view's format alone
//! (`Metrics::extended_view_dispatches`; a refused dispatch, a read-only slot, a write through the
//! texture's own storage element and a draw count none), and the counts line carries them as an
//! amount, [`EXTENDED_VIEW_DISPATCHES_CODE`], so a boot's line says whether its kernels wrote
//! textures through such views.
//!
//! **Its storage textures' levels.** A function that writes or reads a `write` or `read_write`
//! texture at a level other than the constant 0, or asks it its level count, reads what it needs
//! of the texture's levels from words rt pushes at each dispatch or draw: the texture's level
//! count, and for a write or read the first element of the texture's run of level elements, 16
//! single-level storage views of its levels that set 1's storage-texture array holds past its
//! textures. A texture of more than one level takes its run at its first such bind and holds it
//! until it is released and its work retired; a dispatch or draw that finds every one of the
//! 4,096 runs held is rt's refusal (`Limit::LevelElements`), which records nothing. rt counts the
//! dispatches and draws whose push carried those words (`Metrics::level_field_commands`; a refused
//! one counts none), and the counts line carries them as an amount, [`LEVEL_FIELD_COMMANDS_CODE`],
//! so a boot's line says whether its functions took that path. The runs held reach the counts line
//! at their highest (`level_elements`, below).
//!
//! **Its heap textures' moves.** A texture placed in a heap whose bytes another resource of that
//! heap may have written since the texture's own last write (Vulkan's memory aliasing) has its
//! image moved out of `VK_IMAGE_LAYOUT_UNDEFINED`, whole, before the next command that writes it:
//! a dispatch's slot or declared write, a pass's attachment, a draw's write or a blit's
//! destination. A heap buffer's use that may write it marks every heap texture over its bytes so,
//! and a read before the texture's next write takes no move. rt counts those moves
//! (`Metrics::heap_moves`), and the counts line carries them as an amount, [`HEAP_MOVES_CODE`], so
//! a boot's line says whether its heap textures took them.
//!
//! **Its draws' reads of their own attachments.** A draw that reads an attachment of its own pass,
//! a fragment function's colour read or a texture argument over the attachment, reads it in place
//! through the attachment feedback loop, with the image in the feedback layout
//! (`Metrics::attachment_loop_reads`, a count, no bytes); on a device without the loop it reads
//! the image's shadow instead, which rt copies from the image first, copies Metal does not make
//! (`Metrics::attachment_read_copies`, and their bytes, `attachment_read_copy_bytes`, which count
//! in `bytes_copied` too). The counts line carries all three as amounts,
//! [`ATTACHMENT_LOOP_READS_CODE`], [`ATTACHMENT_READ_COPIES_CODE`] and
//! [`ATTACHMENT_READ_COPY_BYTES_CODE`], so a host's line says which route its draws took.
//!
//! **Its draws of stencil output by bits.** On a device without `VK_EXT_shader_stencil_export`, a
//! draw through a pipeline state whose fragment function writes `[[stencil]]` is drawn once, then
//! once for each stencil bit its depth-stencil state can replace, and once more for the depth
//! where the state writes it under a compare other than `Always` and `Never`: draws Metal does not
//! make, which store Metal's values but where fragments of one draw meet at a sample (rt's static
//! note `PrecisionNote::StencilOutputOverlap`). Where those draws would not give Metal's answer,
//! rt refuses the draw (`Gap::Draw` of `StencilOutput`). rt counts the draws beyond each draw's
//! own (`Metrics::stencil_bit_draws`, 0 on a device with the extension), and the counts line
//! carries them as an amount, [`STENCIL_BIT_DRAWS_CODE`], so a host's line says whether its draws
//! took that route.
//!
//! **Its Direct3D 11 planes' hand-overs.** On Windows, each submission whose command buffers keep
//! a plane of a Direct3D 11 surface (`super::d3d11_surface`), or whose uploads copy into or out of
//! one, acquires those planes from `VK_QUEUE_FAMILY_EXTERNAL` in a command buffer of rt's own
//! ahead of everything, after a global barrier from and to every stage, and releases them to that
//! family in another after everything, so Direct3D 11 owns the planes' memory between rt's
//! submissions. rt counts those submissions, a group that goes member by member once
//! (`Metrics::d3d11_plane_submissions`), and the counts line carries them as an amount,
//! [`D3D11_PLANE_SUBMISSIONS_CODE`], so a boot's line says how many of its submissions handed
//! planes over. Off Windows no Direct3D 11 surface is made, and the amount stays 0.
//!
//! **Its compute dispatches.** rt counts no dispatch, so the rt platform's compute encoders count
//! those they encode, by form, one relaxed add each (`super::compute`): the direct ones
//! (`dispatchThreads`, `dispatchThreadgroups`) and the indirect ones
//! (`dispatchThreadgroupsWithIndirectBuffer`). [`fold_metrics`] folds them onto the counts line as
//! amounts, `rt-dispatches-direct` and `rt-dispatches-indirect`
//! (`super::compute::DISPATCHES_DIRECT_CODE`, `super::compute::DISPATCHES_INDIRECT_CODE`), so a
//! boot's line says what share of its dispatches, direct and indirect, ran whole threadgroups on
//! a partial-threadgroups variant.
//!
//! **Its memory at its highest.** Four of rt's gauges split the device memory a device holds past
//! its places and blocks: the preprocess regions executions of indirect command buffers hold
//! (`Gauges::preprocess_used_bytes`) and those kept idle for the next (`preprocess_idle_bytes`),
//! the texture blocks held whole (`texture_block_held_bytes`), and the Private blocks taken and not
//! given back (`private_take_bytes`). A fifth is the size of the data in the device's
//! `VkPipelineCache` (`pipeline_cache_bytes`), which rt measures at its seed and when its reaper
//! last looked, once creation was quiet, and which falls when the reaper replaces a cache past 128
//! MiB with a fresh one. Two more are the resolve images a device holds (`resolve_images`) and the
//! bytes of their memory (`resolve_image_bytes`). Two are the compiled kernels alive (`kernels`),
//! each the translation of one compute pipeline content with its shader modules and their
//! `VkPipeline`s, which lives while a state or the device's map of compiled contents holds it, and
//! that map's entries, compute and render together, at most 1,024 (`kernel_entries`). The last
//! three are those kernels' `VkPipeline`s, each of one workgroup size, threadgroup memory lengths,
//! imageblock size and tiles, of which a module keeps the 8 it made first and the 56 dispatched
//! last past them, beside those that work not yet retired held at its last make
//! (`kernel_variants`), the variants of sampler states bound with LOD clamps of their own, at
//! most 4,000 (`lod_clamp_variants`), and the runs of level elements textures hold, at most 4,096
//! (`level_elements`, above). A gauge rises and falls, so the counts line carries the
//! highest each reached at a look, `rt-<gauge>-most` ([`highest_gauges`]), as an amount risen to it
//! ([`note_gauges`]): the in-run highest, to within the looks. The gauges' thread looks every 5 s
//! from the process's first device's open on, whether or not its lines are in the log
//! (`super::gauge_log`); so do the counts' folds ([`fold_metrics`]: a guest device's reset, after
//! its objects dropped, and the process's exit), and the let-go of a device once every handle but
//! the platform's is gone, before rt's quiesce and teardown of it (`RtPlatform::let_go_of_device`),
//! since the teardown sets every gauge to zero and the counts line of an exit after a let-go, as at
//! a host's quit, comes after it. A value rt held between two looks is not seen. The trace's
//! `rtcounts` carries the compiled kernels and the map's entries as they are at each of its events
//! (`super::command`), at most one each 50 ms while command buffers commit or complete, so a traced
//! run shows them over time.

use std::sync::{Arc, OnceLock};

use ash::vk;
use metal2vulkan_runtime as rt;

use crate::core::platform::caps::{DeviceCaps, IdentityCaps, SAMPLE_COUNTS};
use crate::core::platform::device::{GpuDevice, NoCopyDeallocator};
use crate::core::platform::host_range::HostRange;
use crate::core::platform::stage::StageFills;
use crate::core::platform::types::{
    ArgumentLayout, NsErrorText, PixelFormatInfo, SizeAndAlign, TextureLevelInfo, TextureLevelQuery,
};
use crate::core::platform::values::{
    ArgumentDesc, BufferDesc, DepthStencilDesc, FullComputePipelineDesc,
    FullMeshRenderPipelineDesc, FullRenderPipelineDesc, HeapDesc, HeapMade, IcbDesc,
    RenderPipelineDesc, SamplerDesc,
};
use crate::core::platform::{Handle, Platform};
use crate::core::raise::{OnFault, Raise, RaiseResult};
use crate::gpu::caps::METAL_LINEAR_ALIGNMENT;
use crate::gpu::device::{HEAP_DESCRIPTOR_OPTIONS, descriptor_options};
use crate::gpu::identity;

use super::answers::{self, Facts};
use super::argument;
use super::buffer::{self, KnownBlocks};
use super::command::DeviceLoss;
use super::heap;
use super::library::{self, RtLibrary};
use super::pipeline::{self, Numbering};
use super::queue::RtQueue;
use super::render_pipeline;
use super::stage::StageKey;
use super::texture::OwnIndices;
use super::unbuilt::{
    RtRefusal, Unbuilt, refuse, refuse_create, refused_as, refused_error, rt_refused,
    rt_refused_create, rt_refused_texture,
};
use super::{RtObj, RtPlatform};

/// The log target of the device's lines.
const TARGET: &str = "reims_vgpu_vulkan::rt_platform";

/// The code of rt's copies between a texture over a buffer and its image of its own (rt's
/// `Metrics::over_copies`), an amount on the counts line ([`fold_metrics`]).
pub const OVER_COPIES_CODE: &str = "rt-over-copies";

/// The code of those copies' bytes (`Metrics::over_copy_bytes`).
pub const OVER_COPY_BYTES_CODE: &str = "rt-over-copy-bytes";

/// The code of those copies over a buffer in an imported block (`Metrics::over_copies_imported`).
pub const OVER_COPIES_IMPORTED_CODE: &str = "rt-over-copies-imported";

/// The code of their bytes (`Metrics::over_copy_bytes_imported`).
pub const OVER_COPY_BYTES_IMPORTED_CODE: &str = "rt-over-copy-bytes-imported";

/// The code of the over-copies of a kind rt kept none of, its 256 kinds being full
/// (`Metrics::over_copies_unkept_kind`), an amount on the counts line ([`fold_metrics`]): the
/// copies whose kind [`note_over_copy_kinds`] cannot log.
pub const OVER_COPIES_UNKEPT_KIND_CODE: &str = "rt-over-copies-unkept-kind";

/// The code of the command buffers rt held for room on its `VkQueue` (`Metrics::held_for_room`),
/// an amount on the counts line ([`fold_metrics`]).
pub const HELD_FOR_ROOM_CODE: &str = "rt-held-for-room";

/// The code of the writes rt dropped rather than keep in its cache directory
/// (`Metrics::cache_writes_dropped`), an amount on the counts line ([`fold_metrics`]).
pub const CACHE_WRITES_DROPPED_CODE: &str = "rt-cache-writes-dropped";

/// The code of the bytes rt wrote to its cache directory (`Metrics::cache_bytes_written`), an
/// amount on the counts line ([`fold_metrics`]).
pub const CACHE_BYTES_WRITTEN_CODE: &str = "rt-cache-bytes-written";

/// The code of the modules rt's translator lowered (`Metrics::translations`, one for each module a
/// pipeline state lowers: the module docs, "Its translations"), an amount on the counts line
/// ([`fold_metrics`]).
pub const TRANSLATIONS_CODE: &str = "rt-translations";

/// The code of the lowered kernels pipeline states took from rt's cache directory in place of a
/// lowering (`Metrics::translations_loaded`), an amount on the counts line ([`fold_metrics`]).
pub const TRANSLATIONS_LOADED_CODE: &str = "rt-translations-loaded";

/// The code of the nanoseconds rt's calls into the translator took, those refused among them
/// (`Metrics::translation_nanos`), an amount on the counts line ([`fold_metrics`]).
pub const TRANSLATION_NS_CODE: &str = "rt-translation-ns";

/// The code of the `VkPipeline`s rt made (`Metrics::pipelines_created`), an amount on the counts
/// line ([`fold_metrics`]).
pub const PIPELINES_CREATED_CODE: &str = "rt-pipelines-created";

/// The code of the nanoseconds the driver's `vkCreate*Pipelines` calls took, those that failed
/// among them (`Metrics::pipeline_creation_nanos`), an amount on the counts line
/// ([`fold_metrics`]).
pub const PIPELINE_CREATION_NS_CODE: &str = "rt-pipeline-creation-ns";

/// The code of the render contents' records pipeline states read from rt's cache directory, each
/// naming the lowerings of the content's stages, which count among the translations loaded
/// (`Metrics::render_records_read`), an amount on the counts line ([`fold_metrics`]).
pub const RENDER_RECORDS_READ_CODE: &str = "rt-render-records-read";

/// The code of the render contents' records that named a lowering rt's cache directory no longer
/// held, so the content was lowered again (`Metrics::render_records_stale`), an amount on the
/// counts line ([`fold_metrics`]).
pub const RENDER_RECORDS_STALE_CODE: &str = "rt-render-records-stale";

/// The code of the render pipelines rt translated with a stage heavier than the kernels' stage
/// bound, those it refused past the render stages' own bound among them
/// (`Metrics::render_makes_past_kernel_bound`), an amount on the counts line ([`fold_metrics`]).
pub const RENDER_MAKES_PAST_KERNEL_BOUND_CODE: &str = "rt-render-makes-past-kernel-bound";

/// The code of the heaviest stage's weight among those render pipelines, the highest over the
/// process's devices (`Metrics::render_weight_past_kernel_bound_max`, [`note_highest_metrics`]).
pub const RENDER_WEIGHT_PAST_KERNEL_BOUND_MOST_CODE: &str =
    "rt-render-weight-past-kernel-bound-most";

/// The code of the dispatches whose encoding waited while rt made the `VkPipeline` of their
/// compute kernel's variant, their first-use waits (`Metrics::first_use_variant_waits`), an
/// amount on the counts line ([`fold_metrics`]).
pub const FIRST_USE_WAITS_CODE: &str = "rt-first-use-variant-waits";

/// The code of those waits' nanoseconds in all (`Metrics::first_use_variant_wait_nanos`).
pub const FIRST_USE_WAIT_NS_CODE: &str = "rt-first-use-variant-wait-ns";

/// The code of the longest of those waits, in nanoseconds
/// (`Metrics::first_use_variant_wait_max_nanos`), an amount risen to it ([`note_highest_metrics`]).
pub const FIRST_USE_WAIT_MAX_NS_CODE: &str = "rt-first-use-variant-wait-max-ns";

/// The code of the states' main pipelines that their finishes made, after the state's creation
/// answered (`Metrics::finished_pipelines`), an amount on the counts line ([`fold_metrics`]).
pub const FINISHED_PIPELINES_CODE: &str = "rt-finished-pipelines";

/// The code of those finishes' nanoseconds in all (`Metrics::finished_pipeline_nanos`).
pub const FINISHED_PIPELINE_NS_CODE: &str = "rt-finished-pipeline-ns";

/// The code of the dispatches and draws whose encoding waited for a state's main pipeline that no
/// finish had made yet, for the finish in flight or for their own make of it
/// (`Metrics::first_use_unfinished_waits`), an amount on the counts line ([`fold_metrics`]).
pub const UNFINISHED_WAITS_CODE: &str = "rt-first-use-unfinished-waits";

/// The code of those waits' nanoseconds in all (`Metrics::first_use_unfinished_wait_nanos`).
pub const UNFINISHED_WAIT_NS_CODE: &str = "rt-first-use-unfinished-wait-ns";

/// The code of the longest of those waits, in nanoseconds
/// (`Metrics::first_use_unfinished_wait_max_nanos`), an amount risen to it
/// ([`note_highest_metrics`]).
pub const UNFINISHED_WAIT_MAX_NS_CODE: &str = "rt-first-use-unfinished-wait-max-ns";

/// The code of the exported blocks a public call handed out a device address in, each counted at
/// the first, which a block's make never is (`Metrics::exported_blocks_addressed`), an amount on
/// the counts line ([`fold_metrics`]).
pub const EXPORTED_BLOCKS_ADDRESSED_CODE: &str = "rt-exported-blocks-addressed";

/// The code of the replacements of the device's `VkPipelineCache` by a fresh, empty one past its
/// bound of 128 MiB (`Metrics::pipeline_caches_replaced`), an amount on the counts line
/// ([`fold_metrics`]).
pub const PIPELINE_CACHES_REPLACED_CODE: &str = "rt-pipeline-caches-replaced";

/// The code of the writes of the pipeline cache's data over the device's blob that rt skipped
/// because the data was past the bound of 128 MiB (`Metrics::pipeline_cache_writes_skipped`),
/// an amount on the counts line ([`fold_metrics`]).
pub const PIPELINE_CACHE_WRITES_SKIPPED_CODE: &str = "rt-pipeline-cache-writes-skipped";

/// The code of the reaper's looks at the pipeline cache that rt skipped because a creation began
/// after creation was quiet (`Metrics::pipeline_cache_looks_skipped`), an amount on the counts
/// line ([`fold_metrics`]).
pub const PIPELINE_CACHE_LOOKS_SKIPPED_CODE: &str = "rt-pipeline-cache-looks-skipped";

/// The code of the `VkPipeline` creations that went through the device's empty pipeline cache
/// while rt's reaper read the device's blob into it (`Metrics::creations_before_seed`), an amount
/// on the counts line ([`fold_metrics`]).
pub const CREATIONS_BEFORE_SEED_CODE: &str = "rt-creations-before-seed";

/// The code of the dispatches that ran a kernel's emulated-SIMD-group module
/// (`Metrics::emulated_simdgroup_dispatches`), an amount on the counts line ([`fold_metrics`]).
pub const EMULATED_SIMDGROUP_DISPATCHES_CODE: &str = "rt-emulated-simdgroup-dispatches";

/// The code of the compiled contents a state's make dropped because an equal one landed in the
/// device's map of contents first (`Metrics::kernel_makes_dropped`), an amount on the counts line
/// ([`fold_metrics`]).
pub const KERNEL_MAKES_DROPPED_CODE: &str = "rt-kernel-makes-dropped";

/// The code of the resolves of passes' multisample attachments, one per attachment a pass
/// resolves, counted as its encoder ends (`Metrics::resolves`), an amount on the counts line
/// ([`fold_metrics`]).
pub const RESOLVES_CODE: &str = "rt-resolves";

/// The code of the copies of what a pass resolved into a texture's resolve image into the bytes
/// of the buffer the texture is in place over (`Metrics::resolve_copies`), an amount on the counts
/// line ([`fold_metrics`]).
pub const RESOLVE_COPIES_CODE: &str = "rt-resolve-copies";

/// The code of the bytes of those copies (`Metrics::resolve_copy_bytes`), an amount on the counts
/// line ([`fold_metrics`]).
pub const RESOLVE_COPY_BYTES_CODE: &str = "rt-resolve-copy-bytes";

/// The code of the dispatches of whole threadgroups, direct and indirect, that ran a kernel's
/// partial-threadgroups variant (`Metrics::partial_variant_whole_grids`), an amount on the counts
/// line ([`fold_metrics`]).
pub const PARTIAL_VARIANT_WHOLE_GRIDS_CODE: &str = "rt-partial-variant-whole-grids";

/// The code of the dispatches of a grid that ends in partial threadgroups, `dispatchThreads` of a
/// grid that is not a whole number of threadgroups, that ran a kernel's partial-threadgroups
/// variant (`Metrics::partial_variant_ragged_grids`), an amount on the counts line
/// ([`fold_metrics`]).
pub const PARTIAL_VARIANT_RAGGED_GRIDS_CODE: &str = "rt-partial-variant-ragged-grids";

/// The code of the dispatches whose kernel writes, at one of its texture slots, a view through the
/// storage usage its texture's image took for that view's format alone
/// (`Metrics::extended_view_dispatches`), an amount on the counts line ([`fold_metrics`]).
pub const EXTENDED_VIEW_DISPATCHES_CODE: &str = "rt-extended-view-dispatches";

/// The code of the submissions that acquired planes of a Direct3D 11 surface from
/// `VK_QUEUE_FAMILY_EXTERNAL` at their start and released them to it at their end
/// (`Metrics::d3d11_plane_submissions`), an amount on the counts line ([`fold_metrics`]).
pub const D3D11_PLANE_SUBMISSIONS_CODE: &str = "rt-d3d11-plane-submissions";

/// The code of the dispatches and draws whose push carried a storage texture's level count or the
/// first of its level elements (`Metrics::level_field_commands`), an amount on the counts line
/// ([`fold_metrics`]).
pub const LEVEL_FIELD_COMMANDS_CODE: &str = "rt-level-field-commands";

/// The code of the heap textures' images moved out of `VK_IMAGE_LAYOUT_UNDEFINED` before a command
/// that writes them, after another resource of their heap may have written their bytes
/// (`Metrics::heap_moves`), an amount on the counts line ([`fold_metrics`]).
pub const HEAP_MOVES_CODE: &str = "rt-heap-moves";

/// The code of the draws that read an attachment of their own pass in place, through the
/// attachment feedback loop, with the image in the feedback layout
/// (`Metrics::attachment_loop_reads`), an amount on the counts line ([`fold_metrics`]).
pub const ATTACHMENT_LOOP_READS_CODE: &str = "rt-attachment-loop-reads";

/// The code of the copies into an image's shadow that a draw reading an attachment of its own
/// pass takes first on a device without the attachment feedback loop
/// (`Metrics::attachment_read_copies`), an amount on the counts line ([`fold_metrics`]).
pub const ATTACHMENT_READ_COPIES_CODE: &str = "rt-attachment-read-copies";

/// The code of the bytes of those copies (`Metrics::attachment_read_copy_bytes`), an amount on
/// the counts line ([`fold_metrics`]).
pub const ATTACHMENT_READ_COPY_BYTES_CODE: &str = "rt-attachment-read-copy-bytes";

/// The code of the draws beyond a draw's own that a pipeline state whose fragment function writes
/// `[[stencil]]` takes on a device without `VK_EXT_shader_stencil_export`, one for each stencil
/// bit and one for the depth (`Metrics::stencil_bit_draws`), an amount on the counts line
/// ([`fold_metrics`]).
pub const STENCIL_BIT_DRAWS_CODE: &str = "rt-stencil-bit-draws";

/// The code of the most bytes of preprocess regions that executions of indirect command buffers
/// held at a look (`Gauges::preprocess_used_bytes`, [`note_gauges`]).
pub const PREPROCESS_USED_MOST_CODE: &str = "rt-preprocess-used-bytes-most";

/// The code of the most bytes of preprocess regions kept idle for the next executions at a look
/// (`Gauges::preprocess_idle_bytes`, [`note_gauges`]).
pub const PREPROCESS_IDLE_MOST_CODE: &str = "rt-preprocess-idle-bytes-most";

/// The code of the most bytes of texture blocks held, each whole, at a look
/// (`Gauges::texture_block_held_bytes`, [`note_gauges`]).
pub const TEXTURE_BLOCK_HELD_MOST_CODE: &str = "rt-texture-block-held-bytes-most";

/// The code of the most bytes of Private blocks taken and not given back at a look
/// (`Gauges::private_take_bytes`, [`note_gauges`]).
pub const PRIVATE_TAKE_MOST_CODE: &str = "rt-private-take-bytes-most";

/// The code of the most bytes of data in the device's `VkPipelineCache` at a look
/// (`Gauges::pipeline_cache_bytes`, [`note_gauges`]): rt's size of it at its seed and when its
/// reaper last looked, read only at the looks, so a look the reaper skipped
/// ([`PIPELINE_CACHE_LOOKS_SKIPPED_CODE`]) leaves it at the size before.
pub const PIPELINE_CACHE_MOST_CODE: &str = "rt-pipeline-cache-bytes-most";

/// The code of the most resolve images a device held at a look (`Gauges::resolve_images`,
/// [`note_gauges`]).
pub const RESOLVE_IMAGES_MOST_CODE: &str = "rt-resolve-images-most";

/// The code of the most bytes of the resolve images' memory at a look
/// (`Gauges::resolve_image_bytes`, [`note_gauges`]).
pub const RESOLVE_IMAGE_BYTES_MOST_CODE: &str = "rt-resolve-image-bytes-most";

/// The code of the most compiled kernels alive at a look (`Gauges::kernels`, [`note_gauges`]):
/// those a state holds and those only the device's map of compiled contents holds.
pub const KERNELS_MOST_CODE: &str = "rt-kernels-most";

/// The code of the most entries of the device's map of compiled contents, compute and render, at
/// a look (`Gauges::kernel_entries`, [`note_gauges`]).
pub const KERNEL_ENTRIES_MOST_CODE: &str = "rt-kernel-entries-most";

/// The code of the most `VkPipeline`s of compiled kernels' modules at a look
/// (`Gauges::kernel_variants`, [`note_gauges`]): those each module keeps for later dispatches and
/// those past them that work not yet retired held.
pub const KERNEL_VARIANTS_MOST_CODE: &str = "rt-kernel-variants-most";

/// The code of the most variants of sampler states, each a state bound with LOD clamps of its
/// own, at a look (`Gauges::lod_clamp_variants`, [`note_gauges`]).
pub const LOD_CLAMP_VARIANTS_MOST_CODE: &str = "rt-lod-clamp-variants-most";

/// The code of the most runs of level elements textures held, each a texture's single-level
/// storage views, at a look (`Gauges::level_elements`, [`note_gauges`]).
pub const LEVEL_ELEMENTS_MOST_CODE: &str = "rt-level-elements-most";

/// The gauges the counts line carries at their highest ([`highest_gauges`]).
const HIGHEST_GAUGES: usize = 12;

/// The highest of each of [`highest_gauges`]'s gauges a look saw, in its order, over the
/// process's devices.
static GAUGES_HIGHEST: crate::core::sync::Mutex<[u64; HIGHEST_GAUGES]> =
    crate::core::sync::Mutex::new([0; HIGHEST_GAUGES]);

/// The over-copy metrics the counts line and the trace's `rtcounts` carry ([`over_copy_counts`]):
/// the over-copies and their bytes, those over an imported block, and the 12 causes' pairs.
pub(super) const OVER_COPY_METRICS: usize = 4 + 2 * 12;

/// The metrics the counts line carries ([`folded_counts`]): the over-copy metrics, then the
/// command buffers held for room, the cache writes dropped, the cache bytes written, the
/// translations, the translations loaded, the translator's nanoseconds, the `VkPipeline`s made
/// and the driver's nanoseconds making them, the render records read and stale, the render
/// pipelines made with a stage past the kernels' bound, the compute dispatches' first-use waits
/// and their nanoseconds, the exported blocks whose address went out, the pipeline caches
/// replaced, the cache's writes skipped past its bound, the reaper's looks at it skipped and the
/// creations made before its seed, the dispatches that ran an emulated-SIMD-group module, the
/// compiled contents a state's make dropped, the resolves of multisample attachments, the copies
/// out of resolve images and their bytes, the whole grids and the ragged grids that ran a
/// partial-threadgroups variant, the dispatches that wrote a view through its image's extended
/// storage usage, the submissions that handed Direct3D 11 planes over, the dispatches and draws
/// that pushed a storage texture's level fields, the heap textures moved out of the undefined
/// layout, the draws that read an attachment of their own pass through the feedback loop, the
/// copies into attachments' shadows on a device without it and their bytes, the draws of stencil
/// output by bits, the main pipelines the states' finishes made and their nanoseconds, the
/// first-use waits for an unfinished main pipeline and their nanoseconds, and the over-copies of
/// a kind rt kept none of.
const FOLDED_METRICS: usize = OVER_COPY_METRICS + 38;

/// The metrics the counts line carries at their highest ([`highest_metrics`]): rt's maxima since
/// a device was made.
const HIGHEST_METRICS: usize = 3;

/// The highest of each of [`highest_metrics`]'s metrics a fold saw, in its order, over the
/// process's devices.
static METRICS_HIGHEST: crate::core::sync::Mutex<[u64; HIGHEST_METRICS]> =
    crate::core::sync::Mutex::new([0; HIGHEST_METRICS]);

/// The counters of the device the process made last ([`open`]), which [`fold_metrics`] reads,
/// with what it folded so far of each of their metrics, in [`folded_counts`]'s order, and how
/// many of its over-copies' kinds it read ([`note_over_copy_kinds`]).
type Folding = Option<(rt::Counters, [u64; FOLDED_METRICS], usize)>;

/// The process's [`Folding`], once a device is made.
static COUNTERS: crate::core::sync::Mutex<Folding> = crate::core::sync::Mutex::new(None);

/// The over-copies' kinds the process logged ([`note_over_copy_kinds`]), over its devices, in the
/// order logged: at most rt's 256 of each device.
static OVER_COPY_KINDS_LOGGED: crate::core::sync::Mutex<Vec<rt::OverCopyKind>> =
    crate::core::sync::Mutex::new(Vec::new());

/// Counts by code (`crate::answer::count_kind_by`) what rt's over-copy metrics of the process's
/// device, its command buffers held for room, its cache writes, its translations, its pipelines'
/// makes and their time, its render makes past the kernels' bound, its compute dispatches'
/// first-use waits, its exported blocks addressed, its pipeline caches replaced and cache writes
/// skipped past their bound, its reaper's looks at the cache skipped, its creations before the
/// seed, its emulated-SIMD-group dispatches, its dropped makes, its resolves, its resolve copies,
/// its whole and ragged grids on a partial-threadgroups variant, its dispatches that wrote views
/// through an extended usage, its submissions that handed Direct3D 11 planes over, its dispatches
/// and draws that pushed level fields, its heap textures' moves out of the undefined layout, its
/// draws' reads of their own attachments through the feedback loop and its copies into the
/// attachments' shadows on a device without it, its draws of stencil output by bits, its states'
/// finishes and their waits and its over-copies of a kind rt kept none of rose by since the last
/// fold ([`folded_counts`]), so the counts line carries their totals, then its maxima
/// ([`note_highest_metrics`]) and its gauges ([`note_gauges`]) at their highest, and logs its
/// over-copies' kinds the process had not logged ([`note_over_copy_kinds`]) (the module's docs).
/// Nothing of rt's before the device is made. First it counts the dispatches the rt platform's compute encoders encoded since the last fold
/// (`super::compute::fold_dispatches`).
pub fn fold_metrics() {
    super::compute::fold_dispatches();
    let (metrics, gauges, kinds) = {
        let mut latest = COUNTERS.lock();
        let Some((counters, folded, read)) = latest.as_mut() else {
            return;
        };
        let metrics = counters.metrics();
        fold(folded, folded_counts(&metrics));
        (metrics, counters.gauges(), unread_kinds(counters, read))
    };
    note_highest_metrics(&metrics);
    note_gauges(&gauges);
    note_over_copy_kinds(kinds);
}

/// Looks at `m`, the metrics of the process's device now: counts each of [`highest_metrics`]'s by
/// its rise over the highest a fold saw before, so that the counts line carries the highest (the
/// module's docs).
fn note_highest_metrics(m: &rt::Metrics) {
    raise_to(&mut METRICS_HIGHEST.lock(), highest_metrics(m));
}

/// The metrics of `m` the counts line carries at their highest, each with its code: the heaviest
/// stage's weight of the render makes past the kernels' bound
/// ([`RENDER_WEIGHT_PAST_KERNEL_BOUND_MOST_CODE`]), the longest first-use wait of a compute
/// dispatch ([`FIRST_USE_WAIT_MAX_NS_CODE`]), and the longest wait for an unfinished main
/// pipeline ([`UNFINISHED_WAIT_MAX_NS_CODE`]).
fn highest_metrics(m: &rt::Metrics) -> [(&'static str, u64); HIGHEST_METRICS] {
    let weight = m.render_weight_past_kernel_bound_max;
    let wait = m.first_use_variant_wait_max_nanos;
    let unfinished = m.first_use_unfinished_wait_max_nanos;
    [
        (RENDER_WEIGHT_PAST_KERNEL_BOUND_MOST_CODE, weight),
        (FIRST_USE_WAIT_MAX_NS_CODE, wait),
        (UNFINISHED_WAIT_MAX_NS_CODE, unfinished),
    ]
}

/// Looks at `g`, the gauges of the process's device now: counts each of [`highest_gauges`]'s by
/// its rise over the highest a look saw before, so that the counts line carries the highest (the
/// module's docs). The looks: the gauges' thread's, every 5 s from the process's first device's
/// open on, whether or not its lines are in the log (`super::gauge_log`); each fold
/// ([`fold_metrics`]) and the open of a device after another ([`follow`]); and a let-go's, before
/// rt's teardown of the device sets its gauges to zero (`RtPlatform::let_go_of_device`).
pub(super) fn note_gauges(g: &rt::Gauges) {
    raise_to(&mut GAUGES_HIGHEST.lock(), highest_gauges(g));
}

/// The gauges of `g` the counts line carries at their highest, each with its code: the
/// preprocess regions used ([`PREPROCESS_USED_MOST_CODE`]) and idle
/// ([`PREPROCESS_IDLE_MOST_CODE`]), the texture blocks held ([`TEXTURE_BLOCK_HELD_MOST_CODE`]),
/// the Private blocks taken ([`PRIVATE_TAKE_MOST_CODE`]), the pipeline cache's data
/// ([`PIPELINE_CACHE_MOST_CODE`]), the resolve images ([`RESOLVE_IMAGES_MOST_CODE`]) and their
/// memory ([`RESOLVE_IMAGE_BYTES_MOST_CODE`]), the compiled kernels alive ([`KERNELS_MOST_CODE`])
/// and the entries of the map of compiled contents ([`KERNEL_ENTRIES_MOST_CODE`]), the kernels'
/// `VkPipeline`s ([`KERNEL_VARIANTS_MOST_CODE`]), the sampler states' LOD clamp variants
/// ([`LOD_CLAMP_VARIANTS_MOST_CODE`]), and the runs of level elements textures hold
/// ([`LEVEL_ELEMENTS_MOST_CODE`]).
fn highest_gauges(g: &rt::Gauges) -> [(&'static str, u64); HIGHEST_GAUGES] {
    [
        (PREPROCESS_USED_MOST_CODE, g.preprocess_used_bytes),
        (PREPROCESS_IDLE_MOST_CODE, g.preprocess_idle_bytes),
        (TEXTURE_BLOCK_HELD_MOST_CODE, g.texture_block_held_bytes),
        (PRIVATE_TAKE_MOST_CODE, g.private_take_bytes),
        (PIPELINE_CACHE_MOST_CODE, g.pipeline_cache_bytes),
        (RESOLVE_IMAGES_MOST_CODE, g.resolve_images),
        (RESOLVE_IMAGE_BYTES_MOST_CODE, g.resolve_image_bytes),
        (KERNELS_MOST_CODE, g.kernels),
        (KERNEL_ENTRIES_MOST_CODE, g.kernel_entries),
        (KERNEL_VARIANTS_MOST_CODE, g.kernel_variants),
        (LOD_CLAMP_VARIANTS_MOST_CODE, g.lod_clamp_variants),
        (LEVEL_ELEMENTS_MOST_CODE, g.level_elements),
    ]
}

/// Counts the rise of each gauge or maximum of `now` over its highest in `highest` under its
/// code, where it is higher, and keeps it as the highest.
fn raise_to<const N: usize>(highest: &mut [u64; N], now: [(&'static str, u64); N]) {
    for ((code, now), most) in now.into_iter().zip(highest.iter_mut()) {
        if now > *most {
            crate::answer::count_kind_by(code, now - *most);
            *most = now;
        }
    }
}

/// Makes `counters`, those of a device just made, the ones [`fold_metrics`] reads. What the
/// device before it rose by since the last fold is folded first, its maxima and gauges looked at
/// ([`note_highest_metrics`], [`note_gauges`]) and its over-copies' kinds read
/// ([`note_over_copy_kinds`]), so that the counts line keeps its totals after a let-go
/// (`RtPlatform::let_go_of_device`) and adds the new device's.
fn follow(counters: rt::Counters) {
    let mut latest = COUNTERS.lock();
    let mut kinds = Vec::new();
    if let Some((before, folded, read)) = latest.as_mut() {
        let metrics = before.metrics();
        fold(folded, folded_counts(&metrics));
        note_highest_metrics(&metrics);
        note_gauges(&before.gauges());
        kinds = unread_kinds(before, read);
    }
    *latest = Some((counters, [0; FOLDED_METRICS], 0));
    drop(latest);
    note_over_copy_kinds(kinds);
}

/// The over-copies' kinds `counters` names past the first `read` (`rt::Counters::over_copy_kinds`),
/// which a fold has not read, in the order rt first copied them; `read` then counts them too.
fn unread_kinds(counters: &rt::Counters, read: &mut usize) -> Vec<rt::OverCopyKind> {
    let kinds = counters.over_copy_kinds(*read);
    *read += kinds.len();
    kinds
}

/// Logs at info, once a process, each of `kinds`, a device's over-copies' kinds a fold had not
/// read, that the process had not logged ([`first_seen`], [`over_copy_kind_line`]): a diagnostic,
/// which names no guest object, only what rt keeps of the kind.
fn note_over_copy_kinds(kinds: Vec<rt::OverCopyKind>) {
    if kinds.is_empty() {
        return;
    }
    let new = first_seen(&mut OVER_COPY_KINDS_LOGGED.lock(), kinds);
    for kind in new {
        let line = over_copy_kind_line(
            kind.cause,
            kind.pixel_format,
            [kind.width, kind.height],
            kind.usage,
        );
        log::info!(target: TARGET, "{line}");
    }
}

/// Of `kinds`, in their order, those not in `logged` and not before them in `kinds`, each then
/// added to `logged`.
fn first_seen<K: PartialEq + Copy>(logged: &mut Vec<K>, kinds: Vec<K>) -> Vec<K> {
    let mut new = Vec::new();
    for kind in kinds {
        if !logged.contains(&kind) {
            logged.push(kind);
            new.push(kind);
        }
    }
    new
}

/// The line of an over-copy's kind ([`note_over_copy_kinds`]), by its fields: the cause rt gives
/// it (`rt::OverCause`), then the texture's pixel format by its Metal name and value, its width
/// by its height in texels, and its usage's bits as its descriptor stated them.
fn over_copy_kind_line(
    cause: rt::OverCause,
    pixel_format: rt::PixelFormat,
    [width, height]: [u32; 2],
    usage: rt::TextureUsage,
) -> String {
    format!(
        "rt's first over-copy of a kind: cause {cause:?}, pixel format {} ({}), {width} by \
         {height}, usage {:#x}",
        pixel_format.name(),
        pixel_format.as_raw(),
        usage.as_raw()
    )
}

/// The metrics of `m` the counts line carries, each with its code: the over-copy metrics
/// ([`over_copy_counts`]), then the command buffers held for room ([`HELD_FOR_ROOM_CODE`]), the
/// cache writes dropped ([`CACHE_WRITES_DROPPED_CODE`]), the cache bytes written
/// ([`CACHE_BYTES_WRITTEN_CODE`]), the translations ([`TRANSLATIONS_CODE`]), the translations
/// loaded ([`TRANSLATIONS_LOADED_CODE`]), the translator's nanoseconds ([`TRANSLATION_NS_CODE`]),
/// the `VkPipeline`s made ([`PIPELINES_CREATED_CODE`]) and the driver's nanoseconds making them
/// ([`PIPELINE_CREATION_NS_CODE`]), the render records read ([`RENDER_RECORDS_READ_CODE`])
/// and stale ([`RENDER_RECORDS_STALE_CODE`]), the render pipelines made with a stage past the
/// kernels' bound ([`RENDER_MAKES_PAST_KERNEL_BOUND_CODE`]), the compute dispatches' first-use
/// waits ([`FIRST_USE_WAITS_CODE`]) and their nanoseconds ([`FIRST_USE_WAIT_NS_CODE`]), the
/// exported blocks whose address went out ([`EXPORTED_BLOCKS_ADDRESSED_CODE`]), the pipeline
/// caches replaced ([`PIPELINE_CACHES_REPLACED_CODE`]) and the cache's writes skipped
/// ([`PIPELINE_CACHE_WRITES_SKIPPED_CODE`]) past its bound, the reaper's looks at the cache
/// skipped ([`PIPELINE_CACHE_LOOKS_SKIPPED_CODE`]), the creations made before the cache's seed
/// ([`CREATIONS_BEFORE_SEED_CODE`]), the dispatches that ran an emulated-SIMD-group module
/// ([`EMULATED_SIMDGROUP_DISPATCHES_CODE`]), the compiled contents a state's make dropped
/// ([`KERNEL_MAKES_DROPPED_CODE`]), the resolves of multisample attachments ([`RESOLVES_CODE`]),
/// the copies out of resolve images ([`RESOLVE_COPIES_CODE`]) and their bytes
/// ([`RESOLVE_COPY_BYTES_CODE`]), the whole grids ([`PARTIAL_VARIANT_WHOLE_GRIDS_CODE`]) and the
/// ragged grids ([`PARTIAL_VARIANT_RAGGED_GRIDS_CODE`]) that ran a partial-threadgroups variant,
/// the dispatches that wrote a view through its image's extended storage usage
/// ([`EXTENDED_VIEW_DISPATCHES_CODE`]), the submissions that handed Direct3D 11 planes over
/// ([`D3D11_PLANE_SUBMISSIONS_CODE`]), the dispatches and draws that pushed a storage texture's
/// level fields ([`LEVEL_FIELD_COMMANDS_CODE`]), the heap textures moved out of the undefined
/// layout ([`HEAP_MOVES_CODE`]), the draws that read an attachment of their own pass through the
/// feedback loop ([`ATTACHMENT_LOOP_READS_CODE`]), the copies into attachments' shadows on a
/// device without it ([`ATTACHMENT_READ_COPIES_CODE`]) and their bytes
/// ([`ATTACHMENT_READ_COPY_BYTES_CODE`]), the draws of stencil output by bits
/// ([`STENCIL_BIT_DRAWS_CODE`]), the main pipelines the states' finishes made
/// ([`FINISHED_PIPELINES_CODE`]) and their nanoseconds ([`FINISHED_PIPELINE_NS_CODE`]), the
/// first-use waits for an unfinished main pipeline ([`UNFINISHED_WAITS_CODE`]) and their
/// nanoseconds ([`UNFINISHED_WAIT_NS_CODE`]), and the over-copies of a kind rt kept none of
/// ([`OVER_COPIES_UNKEPT_KIND_CODE`]).
fn folded_counts(m: &rt::Metrics) -> [(&'static str, u64); FOLDED_METRICS] {
    let over = over_copy_counts(m);
    let more = [
        (HELD_FOR_ROOM_CODE, m.held_for_room),
        (CACHE_WRITES_DROPPED_CODE, m.cache_writes_dropped),
        (CACHE_BYTES_WRITTEN_CODE, m.cache_bytes_written),
        (TRANSLATIONS_CODE, m.translations),
        (TRANSLATIONS_LOADED_CODE, m.translations_loaded),
        (TRANSLATION_NS_CODE, m.translation_nanos),
        (PIPELINES_CREATED_CODE, m.pipelines_created),
        (PIPELINE_CREATION_NS_CODE, m.pipeline_creation_nanos),
        (RENDER_RECORDS_READ_CODE, m.render_records_read),
        (RENDER_RECORDS_STALE_CODE, m.render_records_stale),
        (
            RENDER_MAKES_PAST_KERNEL_BOUND_CODE,
            m.render_makes_past_kernel_bound,
        ),
        (FIRST_USE_WAITS_CODE, m.first_use_variant_waits),
        (FIRST_USE_WAIT_NS_CODE, m.first_use_variant_wait_nanos),
        (EXPORTED_BLOCKS_ADDRESSED_CODE, m.exported_blocks_addressed),
        (PIPELINE_CACHES_REPLACED_CODE, m.pipeline_caches_replaced),
        (
            PIPELINE_CACHE_WRITES_SKIPPED_CODE,
            m.pipeline_cache_writes_skipped,
        ),
        (
            PIPELINE_CACHE_LOOKS_SKIPPED_CODE,
            m.pipeline_cache_looks_skipped,
        ),
        (CREATIONS_BEFORE_SEED_CODE, m.creations_before_seed),
        (
            EMULATED_SIMDGROUP_DISPATCHES_CODE,
            m.emulated_simdgroup_dispatches,
        ),
        (KERNEL_MAKES_DROPPED_CODE, m.kernel_makes_dropped),
        (RESOLVES_CODE, m.resolves),
        (RESOLVE_COPIES_CODE, m.resolve_copies),
        (RESOLVE_COPY_BYTES_CODE, m.resolve_copy_bytes),
        (
            PARTIAL_VARIANT_WHOLE_GRIDS_CODE,
            m.partial_variant_whole_grids,
        ),
        (
            PARTIAL_VARIANT_RAGGED_GRIDS_CODE,
            m.partial_variant_ragged_grids,
        ),
        (EXTENDED_VIEW_DISPATCHES_CODE, m.extended_view_dispatches),
        (D3D11_PLANE_SUBMISSIONS_CODE, m.d3d11_plane_submissions),
        (LEVEL_FIELD_COMMANDS_CODE, m.level_field_commands),
        (HEAP_MOVES_CODE, m.heap_moves),
        (ATTACHMENT_LOOP_READS_CODE, m.attachment_loop_reads),
        (ATTACHMENT_READ_COPIES_CODE, m.attachment_read_copies),
        (
            ATTACHMENT_READ_COPY_BYTES_CODE,
            m.attachment_read_copy_bytes,
        ),
        (STENCIL_BIT_DRAWS_CODE, m.stencil_bit_draws),
        (FINISHED_PIPELINES_CODE, m.finished_pipelines),
        (FINISHED_PIPELINE_NS_CODE, m.finished_pipeline_nanos),
        (UNFINISHED_WAITS_CODE, m.first_use_unfinished_waits),
        (UNFINISHED_WAIT_NS_CODE, m.first_use_unfinished_wait_nanos),
        (OVER_COPIES_UNKEPT_KIND_CODE, m.over_copies_unkept_kind),
    ];
    std::array::from_fn(|i| match i.checked_sub(OVER_COPY_METRICS) {
        None => over[i],
        Some(at) => more[at],
    })
}

/// The over-copy metrics of `m`, each with its code (the module's docs): the over-copies and
/// their bytes, those over an imported block, then each cause's copies and their bytes, the code
/// named for rt's field.
pub(super) fn over_copy_counts(m: &rt::Metrics) -> [(&'static str, u64); OVER_COPY_METRICS] {
    macro_rules! by_cause {
        ($($cause:literal $copies:ident $bytes:ident),* $(,)?) => {
            [
                (OVER_COPIES_CODE, m.over_copies),
                (OVER_COPY_BYTES_CODE, m.over_copy_bytes),
                (OVER_COPIES_IMPORTED_CODE, m.over_copies_imported),
                (OVER_COPY_BYTES_IMPORTED_CODE, m.over_copy_bytes_imported),
                $(
                    (concat!("rt-over-copies-", $cause), m.$copies),
                    (concat!("rt-over-copy-bytes-", $cause), m.$bytes),
                )*
            ]
        };
    }
    by_cause!(
        "host-memory" over_copies_host_memory over_copy_bytes_host_memory,
        "dma-buf" over_copies_dma_buf over_copy_bytes_dma_buf,
        "dma-bufs" over_copies_dma_bufs over_copy_bytes_dma_bufs,
        "dedicated" over_copies_dedicated over_copy_bytes_dedicated,
        "surface" over_copies_surface over_copy_bytes_surface,
        "spilled" over_copies_spilled over_copy_bytes_spilled,
        "linear" over_copies_linear over_copy_bytes_linear,
        "linear-features" over_copies_linear_features over_copy_bytes_linear_features,
        "pitch" over_copies_pitch over_copy_bytes_pitch,
        "render-reach" over_copies_render_reach over_copy_bytes_render_reach,
        "alignment" over_copies_alignment over_copy_bytes_alignment,
        "image-memory" over_copies_image_memory over_copy_bytes_image_memory,
    )
}

/// rt's metrics of the process's device now (`super::diag`); `None` before it is made.
pub(super) fn metrics() -> Option<rt::Metrics> {
    (COUNTERS.lock().as_ref()).map(|(counters, ..)| counters.metrics())
}

/// rt's gauges of the process's device now (`super::diag`); `None` before it is made.
pub(super) fn gauges() -> Option<rt::Gauges> {
    (COUNTERS.lock().as_ref()).map(|(counters, ..)| counters.gauges())
}

/// rt's last ended makes of compute kernels' pipelines for a dispatch's variant, of the process's
/// device, oldest first (`rt::Counters::variant_makes`, `super::overlap`); `None` before it is
/// made.
pub(super) fn variant_makes() -> Option<Vec<rt::VariantMake>> {
    (COUNTERS.lock().as_ref()).map(|(counters, ..)| counters.variant_makes(0))
}

/// Counts the rise of each metric of `now` ([`folded_counts`]) over `folded` under its code, and
/// keeps `now` as folded.
fn fold(folded: &mut [u64; FOLDED_METRICS], now: [(&'static str, u64); FOLDED_METRICS]) {
    for ((code, now), was) in now.into_iter().zip(folded.iter_mut()) {
        crate::answer::count_kind_by(code, now.saturating_sub(*was));
        *was = now;
    }
}

/// Metal's `maxCommandBufferCount` of a queue made by `newCommandQueue`.
pub const METAL_MAX_COMMAND_BUFFER_COUNT: u32 = 64;

/// The rt device's cache directory in the user's cache directory, beside the library's own
/// (`crate::os::cache_directory`), which it shares nothing with: rt keeps its lowered functions
/// and its pipeline caches there, within [`SHADER_CACHE_CAPACITY`].
#[cfg(not(any(test, feature = "testkit")))]
const CACHE: &str = "reims-vgpu-rt";

/// A test build's rt cache directory, the tests' own: a test of what a device compiles runs on an
/// empty one as on a full one, and the tests write nothing into the one the library's hosts share.
#[cfg(any(test, feature = "testkit"))]
const CACHE: &str = "reims-vgpu-rt-tests";

/// The most bytes [`CACHE`] holds, 25 GiB. A macOS 26 guest's built-in apps, each launched once,
/// need more than 1 GiB of translations and driver records on an RTX 5080 laptop. An entry dropped
/// past the capacity is compiled again at the next boot that draws with it.
#[cfg(not(any(test, feature = "testkit")))]
pub(crate) const SHADER_CACHE_CAPACITY: u64 = 25 << 30;

/// A test build's capacity, 1 GiB: its directory holds what the tests compile and nothing a
/// guest's boot reuses, so it is kept small on disk.
#[cfg(any(test, feature = "testkit"))]
pub(crate) const SHADER_CACHE_CAPACITY: u64 = 1 << 30;

/// `id<MTLDevice>` on metal2vulkan-runtime: rt's device and what the platform read of it.
pub struct RtDevice {
    /// rt's device.
    rt: rt::Device,
    /// What its answers depend on.
    facts: Facts,
    /// Its capabilities, read as it was made.
    caps: Arc<DeviceCaps>,
    /// The texture indices its textures the guest does not place take (`super::texture`).
    indices: OwnIndices,
    /// The sampler indices its sampler states the guest does not place take (`super::state`), and
    /// its pipeline states' samplers (`super::pipeline`).
    sampler_indices: OwnIndices,
    /// The sampler indices its pipeline states' samplers hold.
    numbering: Numbering,
    /// The platform's shaders as its library, read at the first ask (`super::library`).
    shaders: OnceLock<RtObj<RtLibrary>>,
    /// The blocks it knows by address, which pinned buffers and heaps inside them are ranges of.
    known: KnownBlocks,
    /// The instance it was made on, whose surfaces its windows' swapchains present to.
    instance: rt::Instance,
    /// Its DRM nodes, where it lists `VK_EXT_physical_device_drm`.
    drm: Option<crate::device_choice::DrmNodes>,
    /// Whether it enabled present fences' extension (the module's docs).
    present_fences: bool,
    /// Where its GPU lies on the host.
    host_gpu: HostGpu,
    /// Its loss as its command buffers see it (`super::command`'s docs, "What a completed handler
    /// learns" and "A lost device's line").
    loss: Arc<DeviceLoss>,
    /// The fills of the stages rt refuses for a compile not in its store (`super::stage`), which
    /// its encoders' dispatches wait for.
    stage_fills: Arc<StageFills<StageKey>>,
}

/// Where a device's GPU lies on the host, as Vulkan answers it, which the host's video opens its
/// APIs at: CUDA by the PCI address, VA-API on the render node, and on Windows on the DXGI adapter
/// of the LUID.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HostGpu {
    /// Its PCI domain, bus, device and function (`VK_EXT_pci_bus_info`); `None` where the device
    /// does not list the extension.
    pub pci: Option<(u32, u32, u32, u32)>,
    /// Its render node's major and minor numbers (`VK_EXT_physical_device_drm`); `None` where the
    /// device does not list the extension or has no render node.
    pub render: Option<(u32, u32)>,
    /// Its DXGI adapter's LUID (`VkPhysicalDeviceIDProperties::deviceLUID`); `None` where the
    /// device says it has no valid one, as off Windows.
    pub luid: Option<[u8; vk::LUID_SIZE]>,
}

impl RtDevice {
    /// rt's device.
    pub fn rt(&self) -> &rt::Device {
        &self.rt
    }

    /// What the platform read of it.
    pub fn facts(&self) -> &Facts {
        &self.facts
    }

    /// The texture indices of the platform's own.
    pub fn indices(&self) -> &OwnIndices {
        &self.indices
    }

    /// The sampler indices of the platform's own.
    pub fn sampler_indices(&self) -> &OwnIndices {
        &self.sampler_indices
    }

    /// The sampler indices its pipeline states' samplers hold.
    pub fn numbering(&self) -> &Numbering {
        &self.numbering
    }

    /// Its loss as its command buffers see it (`super::command`'s docs, "What a completed handler
    /// learns" and "A lost device's line"), which each of them keeps.
    pub fn loss(&self) -> &Arc<DeviceLoss> {
        &self.loss
    }

    /// The fills of the stages rt refuses for a compile not in its store (`super::stage`).
    pub fn stage_fills(&self) -> &Arc<StageFills<StageKey>> {
        &self.stage_fills
    }

    /// Its history of released ranges (`super::fault`), which the tables' objects record their
    /// releases in as they go.
    pub fn releases(&self) -> &super::fault::Releases {
        self.loss.watch().releases()
    }

    /// Walks `tables`, a decoder's, at its loss too, placing the loss's fault among their live
    /// objects (`super::fault`).
    pub fn watch_faults(&self, tables: Arc<dyn super::fault::LiveObjects>) {
        self.loss.watch().watch(tables);
    }

    /// Whether it is lost, which nothing makes again: rt's `Device::loss` names a cause. rt sets
    /// it at its own Vulkan call that meets the loss, before any of its command buffers ends with
    /// the device lost, and a window's swapchain reports a loss its own call meets first
    /// (`super::swapchain`'s docs, "A lost device"). A window's swapchain makes no Vulkan call on
    /// it from then on but its end's destroys.
    pub fn is_lost(&self) -> bool {
        self.rt.loss().is_some()
    }

    /// The blocks it knows by address (`super::buffer`'s docs, "A pinned address"): the graphics
    /// lane keeps a clone, the same set, to know and forget its blocks in.
    pub fn known_blocks(&self) -> &KnownBlocks {
        &self.known
    }

    /// The instance it was made on.
    pub fn instance(&self) -> &rt::Instance {
        &self.instance
    }

    /// Whether DRM node `node` (major, minor) is one of this GPU's: `None` where the device lists
    /// no DRM nodes.
    pub fn composited_here(&self, node: (u32, u32)) -> Option<bool> {
        self.drm.map(|drm| drm.names(node))
    }

    /// Whether each present can signal a fence of its own (`swapchainMaintenance1`).
    pub fn has_present_fences(&self) -> bool {
        self.present_fences
    }

    /// Where its GPU lies on the host.
    pub fn host_gpu(&self) -> HostGpu {
        self.host_gpu
    }
}

/// Why the platform has no device.
#[derive(Debug)]
pub enum Unopened {
    /// rt refused a call making it.
    Rt(rt::Error),
    /// No physical device meets rt's floor: why each listed one does not, or that none is listed.
    NoDevice(String),
}

impl Unopened {
    /// Counts the refusal at `sel` ([`super::unbuilt::rt_refused_create`]); its kind.
    pub fn refuse(&self, sel: &'static str) -> RtRefusal {
        match self {
            Unopened::Rt(error) => rt_refused_create(sel, error),
            Unopened::NoDevice(why) => {
                crate::core::platform::device::note_relayed_gap();
                refused_as(sel, RtRefusal::NoDevice, why);
                RtRefusal::NoDevice
            }
        }
    }
}

impl From<rt::Error> for Unopened {
    fn from(error: rt::Error) -> Unopened {
        Unopened::Rt(error)
    }
}

/// Makes the process's rt device (the module's docs), on a host of pages of `page_size` bytes,
/// and logs it, then the answers rt chose its forms on there ([`RtForms`]).
pub fn open(page_size: u64) -> Result<RtObj<RtDevice>, Unopened> {
    let instance = crate::os::rt_instance(&crate::os::rt_instance_extensions())?;
    let listed = instance.physical_devices()?;
    let (_, raw) = instance.raw();
    let mut below = Vec::new();
    let mut candidates = Vec::new();
    for physical in listed {
        // SAFETY: the instance listed the device and is alive.
        let properties = unsafe { raw.get_physical_device_properties(physical.raw()) };
        let name = (properties.device_name_as_c_str())
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        match physical.check() {
            Ok(floor) => candidates.push((physical, floor)),
            Err(why) => below.push(format!("{name}: {why}")),
        }
    }
    for line in &below {
        log::info!(target: TARGET, "not a candidate for the rt platform's device: {line}");
    }
    let seen: Vec<crate::device_choice::Seen> = (candidates.iter())
        .map(|(physical, _)| {
            // SAFETY: the instance listed the device and is alive.
            let names = unsafe { raw.enumerate_device_extension_properties(physical.raw()) }
                .unwrap_or_default();
            let lists = |name: &std::ffi::CStr| {
                (names.iter()).any(|p| p.extension_name_as_c_str() == Ok(name))
            };
            let drm = lists(ash::ext::physical_device_drm::NAME);
            let pci = lists(ash::ext::pci_bus_info::NAME);
            // The one device extension a display's window takes here, which rt's floor takes
            // too (the module's docs, "Its window system").
            let window_lacks = [ash::khr::swapchain::NAME]
                .into_iter()
                .filter(|&name| !lists(name))
                .collect();
            // SAFETY: as above; rt's instance is at Vulkan 1.3, where `properties2` is core.
            unsafe { crate::os::seen(raw, physical.raw(), true, drm, pci, window_lacks) }
        })
        .collect();
    let Some(at) = crate::device_choice::chosen(&seen) else {
        return Err(Unopened::NoDevice(if below.is_empty() {
            "the loader lists no physical device".to_owned()
        } else {
            below.join("; ")
        }));
    };
    let drm = seen.get(at).and_then(|s| s.drm);
    let luid = candidates.get(at).and_then(|(physical, _)| {
        let mut id = vk::PhysicalDeviceIDProperties::default();
        let mut asked = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
        // SAFETY: the instance listed the device and is alive; rt's instance is at Vulkan 1.3,
        // where the structure is core.
        unsafe { raw.get_physical_device_properties2(physical.raw(), &mut asked) };
        (id.device_luid_valid == vk::TRUE).then_some(id.device_luid)
    });
    let host_gpu = seen.get(at).map_or_else(HostGpu::default, |seen| HostGpu {
        pci: seen.pci,
        render: seen.drm.and_then(|nodes| nodes.render),
        luid,
    });
    let (Some((physical, floor)), Some(uuid)) =
        (candidates.get(at), seen.get(at).and_then(|s| s.uuid))
    else {
        return Err(Unopened::NoDevice(
            "the chosen device has no device UUID".to_owned(),
        ));
    };
    let (mut options, fences) = shared_options(&instance, physical);
    // rt's pool allocations, noted as rt maps them and forgotten as it unmaps them, among which a
    // lost device's fault is placed (`super::fault`).
    let allocations = Arc::new(super::fault::Allocations::default());
    options.block_mappings = Some(Arc::clone(&allocations) as Arc<dyn rt::BlockMappings>);
    options.pipeline_cache_dir = crate::os::user_cache_directory().map(|dir| dir.join(CACHE));
    options.pipeline_cache_cap = Some(SHADER_CACHE_CAPACITY);
    let device = rt::Device::new(physical, options)?;
    // Where it binds no several dma-bufs behind one buffer, a view more than one holds is mapped
    // from its files from the first (`buffer`'s docs, "Pages no pin takes").
    #[cfg(target_os = "linux")]
    crate::os::learn_dma_buf_granule(device.dma_buf_granule());
    let present_fences =
        fences.is_some_and(|name| (device.extensions().iter()).any(|n| n.as_c_str() == name));
    // The process's latest device (the module's docs), the one before it folded first.
    follow(device.counters());
    // Its gauges in the log, where the log asks for them (`super::gauge_log`).
    super::gauge_log::start();
    let facts = facts(&device, floor, &uuid, page_size);
    let driver = {
        let mut driver = vk::PhysicalDeviceDriverProperties::default();
        let mut asked = vk::PhysicalDeviceProperties2::default().push_next(&mut driver);
        // SAFETY: the instance listed the device and is alive; rt's instance is at Vulkan 1.3,
        // where the structure is core.
        unsafe { raw.get_physical_device_properties2(physical.raw(), &mut asked) };
        let text = |name: Result<&std::ffi::CStr, _>| {
            name.map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default()
        };
        format!(
            "{} {}",
            text(driver.driver_name_as_c_str()),
            text(driver.driver_info_as_c_str())
        )
    };
    log::info!(
        target: TARGET,
        "the rt platform's device: {} ({:04x}:{:04x}), driver {driver}, registry ID {:#x}; \
         instance extensions {:?}; device extensions {:?}; present fences: {present_fences}",
        facts.name,
        facts.vendor_id,
        facts.device_id,
        facts.registry_id,
        instance.extensions(),
        device.extensions()
    );
    log::info!(
        target: TARGET,
        "{}",
        RtForms::of(floor, &device, facts.unified_memory)
    );
    let caps = answers::caps(&facts);
    let gpu = format!(
        "{} ({:04x}:{:04x})",
        facts.name, facts.vendor_id, facts.device_id
    );
    let known = KnownBlocks::default();
    let watch = super::fault::FaultWatch::new(allocations, known.clone());
    let loss = Arc::new(DeviceLoss::watching(gpu, watch));
    let order = || Arc::clone(loss.flights());
    let stage_fills = Arc::new(super::stage::fills(device.clone(), uuid));
    Ok(RtObj::new(RtDevice {
        rt: device,
        facts,
        caps,
        indices: OwnIndices::new(rt::IndexClass::Texture, order()),
        sampler_indices: OwnIndices::new(rt::IndexClass::Sampler, order()),
        numbering: Numbering::default(),
        shaders: OnceLock::new(),
        known,
        instance,
        drm,
        present_fences,
        host_gpu,
        loss,
        stage_fills,
    }))
}

/// The answers on which rt chose between its forms on a device, which [`open`] logs after the
/// device's line, so that a log tells which form a refusal met: `rt capabilities: <field>
/// <value>, …; profile: max_compute_shared_memory_size <bytes>, push_window <bytes>, subgroups
/// <its Debug>; unified_memory <bool>`. A diagnostic: the guest sees nothing of it.
struct RtForms<'a> {
    /// rt's floor answers (`PhysicalDevice::check`), each by its field's name.
    floor: [(&'static str, bool); 10],
    /// The translator's profile's `limits().max_compute_shared_memory_size`.
    max_compute_shared_memory_size: u32,
    /// The profile's `push_window()`.
    push_window: u32,
    /// The profile's `subgroups()`.
    subgroups: &'a dyn std::fmt::Debug,
    /// rt's `has_unified_memory()`.
    unified_memory: bool,
}

impl<'a> RtForms<'a> {
    /// The answers of `device`, of rt's floor answers `floor`, whose `has_unified_memory()` is
    /// `unified_memory`: read from what rt keeps, with no call into the driver.
    fn of(floor: &rt::floor::Capabilities, device: &'a rt::Device, unified_memory: bool) -> Self {
        // Each name is its field's, so the line says the name rt's code reads.
        macro_rules! named {
            ($($field:ident),*) => { [$((stringify!($field), floor.$field)),*] };
        }
        let profile = device.translator().profile();
        RtForms {
            floor: named!(
                float16,
                storage_read_without_format,
                buffer_int64_atomics,
                shared_int64_atomics,
                float64,
                maximal_reconvergence,
                subgroup_uniform_control_flow,
                texel_offset_operand,
                attachment_reads,
                feedback_loop
            ),
            max_compute_shared_memory_size: profile.limits().max_compute_shared_memory_size,
            push_window: profile.push_window(),
            subgroups: profile.subgroups(),
            unified_memory,
        }
    }
}

impl std::fmt::Display for RtForms<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("rt capabilities: ")?;
        for (at, (name, value)) in self.floor.iter().enumerate() {
            let comma = if at == 0 { "" } else { ", " };
            write!(f, "{comma}{name} {value}")?;
        }
        write!(
            f,
            "; profile: max_compute_shared_memory_size {}, push_window {}, subgroups {:?}; \
             unified_memory {}",
            self.max_compute_shared_memory_size,
            self.push_window,
            self.subgroups,
            self.unified_memory
        )
    }
}

/// The options a device on `physical`, one of `instance`'s, is opened with both in the VM's
/// process ([`open`]) and in a stage's helper (`super::stage::helper_main`), where rt takes the
/// helper's compile only from a device that enabled what the VM's did (its `StageFill::Enables`
/// otherwise): the extension of present fences, with the feature it takes, where the device lists
/// it and its instance extensions are on (the module's docs, "Its window system"); and the
/// helper's memory ceiling on this host (`Limits::of_host`), which rt keeps with a memory breach,
/// so a breach kept under a lower ceiling is a miss on this device and the helper is asked again.
/// With that extension's name, where it is asked for.
pub(crate) fn shared_options(
    instance: &rt::Instance,
    physical: &rt::PhysicalDevice,
) -> (rt::DeviceOptions, Option<&'static std::ffi::CStr>) {
    let mut options = rt::DeviceOptions::default();
    options.stage_memory_ceiling = crate::os::compile_helper::Limits::of_host().memory;
    let (_, raw) = instance.raw();
    // SAFETY: the instance listed the device and is alive.
    let names =
        unsafe { raw.enumerate_device_extension_properties(physical.raw()) }.unwrap_or_default();
    let lists =
        |name: &std::ffi::CStr| (names.iter()).any(|p| p.extension_name_as_c_str() == Ok(name));
    let fences = crate::os::rt_present_fences(instance.extensions(), lists);
    if let Some(name) = fences {
        let wanted = vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT::default()
            .swapchain_maintenance1(true);
        let needs = |f: &vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT<'static>| {
            f.swapchain_maintenance1 == vk::TRUE
        };
        (options.extensions).push(rt::DeviceExtension::with_features(name, wanted, needs));
    }
    (options, fences)
}

/// What the answers of `device`, made with rt's floor answers `floor`, of `deviceUUID` `uuid`, on
/// a host of `page_size`-byte pages, depend on ([`Facts`]).
fn facts(
    device: &rt::Device,
    floor: &rt::floor::Capabilities,
    uuid: &[u8; vk::UUID_SIZE],
    page_size: u64,
) -> Facts {
    let raw = device.raw();
    let (instance, physical) = (raw.instance, raw.physical_device);
    // SAFETY: the instance and physical device are rt's, read only, while `device` lives.
    let (properties, features) = unsafe {
        (
            instance.get_physical_device_properties(physical),
            instance.get_physical_device_features(physical),
        )
    };
    let float = vk::Format::R32G32B32A32_SFLOAT;
    // SAFETY: as above.
    let filtering = unsafe { instance.get_physical_device_format_properties(physical, float) }
        .optimal_tiling_features
        .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR);
    // SAFETY: as above.
    let msaa = unsafe {
        instance.get_physical_device_image_format_properties(
            physical,
            float,
            vk::ImageType::TYPE_2D,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
            vk::ImageCreateFlags::empty(),
        )
    }
    .is_ok_and(|p| p.sample_counts.contains(vk::SampleCountFlags::TYPE_4));
    let sampled = |count: u64| {
        u32::try_from(count).is_ok_and(|count| device.supports_texture_sample_count(count))
    };
    let barycentrics = (device.extensions().iter())
        .any(|name| name.as_c_str() == ash::khr::fragment_shader_barycentric::NAME);
    Facts {
        name: (properties.device_name_as_c_str())
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default(),
        registry_id: answers::registry_id(uuid),
        vendor_id: properties.vendor_id,
        device_id: properties.device_id,
        pipeline_cache_uuid: properties.pipeline_cache_uuid,
        limits: properties.limits,
        float16: floor.float16,
        unified_memory: device.has_unified_memory(),
        sample_counts: SAMPLE_COUNTS.map(sampled),
        bc: features.texture_compression_bc == vk::TRUE,
        barycentrics,
        float32_filtering: filtering,
        float32_msaa: msaa,
        // SAFETY: as above.
        device_local: unsafe { identity::device_local_bytes_of(instance, physical) },
        page_size,
        heap_buffer_page_private: {
            let sized = device.heap_buffer_size_and_align(page_size, STORAGE_PRIVATE);
            SizeAndAlign {
                size: sized.size,
                align: sized.align,
            }
        },
        linear_alignment_largest: (crate::core::platform::caps::SDK_PIXEL_FORMATS.iter())
            .filter_map(|&format| linear_alignment(device, format))
            .fold(METAL_LINEAR_ALIGNMENT, u64::max),
        linear_alignment_bgra8: device
            .minimum_linear_texture_alignment(rt::PixelFormat::BGRA8Unorm)
            .ok(),
        indirect_tokens: floor.indirect_commands.map(|limits| limits.tokens),
        function_pointers: device.supports_function_pointers(),
        max_threadgroup_memory_length: device.max_threadgroup_memory_length(),
    }
}

/// rt's `minimumLinearTextureAlignmentForPixelFormat:` of the raw pixel format `format`; none
/// where rt has no format of it or refuses it.
fn linear_alignment(device: &rt::Device, format: u64) -> Option<u64> {
    let pf = crate::gpu::runtime_pixel_format(format)?;
    device.minimum_linear_texture_alignment(pf).ok()
}

/// rt's `minimumTextureBufferAlignmentForPixelFormat:` of the raw pixel format `format`, the
/// offset alignment its `Buffer::new_texture` holds a texture buffer of it to; none where rt has
/// no format of it or makes no texture buffer of it.
fn texture_buffer_alignment(device: &rt::Device, format: u64) -> Option<u64> {
    let pf = crate::gpu::runtime_pixel_format(format)?;
    device.minimum_texture_buffer_alignment(pf).ok()
}

/// `MTLResourceStorageModePrivate`, as resource options.
const STORAGE_PRIVATE: u64 = 0x20;

/// Whether a GPU whose driver reports Vulkan's device type `device_type` is an integrated one: of
/// type `INTEGRATED_GPU`, and of no other (discrete, virtual, CPU, other).
fn integrated(device_type: vk::PhysicalDeviceType) -> bool {
    device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
}

/// The raise of a query rt refused at `sel`, counted already under `kind`: Metal's exception at
/// the send, its detail the kind's code and why.
fn refused_raise(sel: &'static str, kind: RtRefusal, why: &dyn std::fmt::Display) -> Raise {
    Raise::platform(sel).with_detail(format!("{}: {why}", kind.code()))
}

/// Grants `indices` of `class` as runs of consecutive indices, each one grant; an index granted
/// already stays granted, as a second reservation of Metal's of an ID reserved changes nothing.
/// One rt refuses otherwise (past the class's range) is counted at `sel`.
fn grant(device: &rt::Device, sel: &'static str, class: rt::IndexClass, indices: &[u64]) {
    let mut sorted: Vec<u64> = indices.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut runs: Vec<(u64, u64)> = Vec::new();
    for index in sorted {
        match runs.last_mut() {
            Some((first, count)) if *first + *count == index => *count += 1,
            _ => runs.push((index, 1)),
        }
    }
    for (first, count) in runs {
        let (Ok(first), Ok(count)) = (u32::try_from(first), u32::try_from(count)) else {
            let why = format!("index {first} is past 32 bits");
            refused_as(sel, RtRefusal::Limit, &why);
            continue;
        };
        let Err(error) = device.grant_indices(class, first, count) else {
            continue;
        };
        if !granted_already(&error) {
            rt_refused(sel, &error);
            continue;
        }
        for index in first..first.saturating_add(count) {
            match device.grant_indices(class, index, 1) {
                Err(error) if !granted_already(&error) => {
                    rt_refused(sel, &error);
                }
                _ => {}
            }
        }
    }
}

/// Whether rt refused a grant because an index of it is granted already.
fn granted_already(error: &rt::Error) -> bool {
    matches!(
        error,
        rt::Error::Refused(rt::Refusal::IndicesGranted { .. })
    )
}

/// The selector of the resource indices' reservation.
const RESERVE: &str = "reserveResourceIndicesForResourceType:indices:indexCount:";

impl GpuDevice<RtPlatform> for RtObj<RtDevice> {
    /// Nothing: rt takes a library of any platform's.
    fn allow_foreign_libraries(&self) {}

    /// The translator's layout of the descriptors, Metal's encoder's (`super::argument`); `None`,
    /// counted, where Metal makes no encoder.
    fn argument_layout(&self, arguments: &[ArgumentDesc]) -> RaiseResult<Option<ArgumentLayout>> {
        Ok(argument::descriptors_layout(arguments))
    }

    /// The translator's layout of the function's argument buffer at the index, the one its
    /// translation reads (`super::argument`); `None`, counted, where Metal makes no encoder.
    fn function_argument_layout(
        &self,
        function: &<RtPlatform as Platform>::Function,
        buffer_index: u64,
    ) -> RaiseResult<Option<ArgumentLayout>> {
        Ok(argument::function_layout(function, buffer_index))
    }

    /// rt's answer: the device's image's own memory requirements, rounded to its buffer-image
    /// granularity, which the guest lays its heaps out by, not Metal's. Where rt refuses (a
    /// texture it makes none of, or `Gap::HeapTexture`, an image that takes an allocation of its
    /// own), the refusal is counted under its kind and raised ([`refused_raise`]), which the
    /// decoder answers the guest as refused, and the guest then lays the texture out by a bound
    /// of its own: never 0 bytes at 0, which places the texture at its neighbours' offset. A
    /// value rt's descriptor has no value of ([`super::descriptor::rt_texture_descriptor`]) is
    /// counted as Metal's refusal (`rt-refused-metal`) and raised alike.
    fn heap_texture_size_and_align(
        &self,
        desc: &<RtPlatform as Platform>::TextureDescriptor,
    ) -> RaiseResult<SizeAndAlign> {
        const SEL: &str = "heapTextureSizeAndAlignWithDescriptor:";
        let d = match super::descriptor::rt_texture_descriptor(&desc.desc()) {
            Ok(d) => d,
            Err(why) => {
                refused_as(SEL, RtRefusal::Refused, &why);
                return Err(refused_raise(SEL, RtRefusal::Refused, &why));
            }
        };
        match self.rt.heap_texture_size_and_align(&d) {
            Ok(rt::SizeAndAlign { size, align }) => Ok(SizeAndAlign { size, align }),
            Err(error) => {
                let desc = desc.desc();
                let options = desc.resource_options.unwrap_or(0);
                let what = super::texture::TextureDetail {
                    desc: &desc,
                    options,
                };
                super::unbuilt::refusal_detail(SEL, &error, Some(&what));
                let kind = rt_refused_texture(SEL, &error);
                Err(refused_raise(SEL, kind, &error))
            }
        }
    }

    /// A buffer over rt's import of the pages that hold `bytes` ([`buffer::over_host_memory`]);
    /// nil where Metal or rt refuses it, counted, the deallocator then dropped unrun, as Metal
    /// runs none for a buffer it does not make.
    fn new_buffer_over_host_memory(
        &self,
        on: OnFault,
        bytes: HostRange<'_>,
        options: u64,
        deallocator: Option<NoCopyDeallocator>,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::Buffer>> {
        let mut recorded_call = crate::recorder::rec_call!(
            "newBufferWithBytesNoCopy:length:options:deallocator:",
            self,
            bytes,
            options,
            deallocator
        );
        // The bytes the create copies into the device's own copy of a `Managed` buffer.
        if recorded_call.is_some() {
            let two_copies = !self.facts().unified_memory;
            crate::recorder::no_copy_contents_with(
                &mut recorded_call,
                two_copies,
                options,
                bytes,
                None,
            );
        }
        let recorded_answer = crate::recorder::run(
            || -> RaiseResult<Option<<RtPlatform as Platform>::Buffer>> {
                let _ = on;
                let sel = "newBufferWithBytesNoCopy:length:options:deallocator:";
                let length = bytes.len();
                let made = buffer::over_host_memory(
                    self,
                    sel,
                    bytes,
                    length,
                    options,
                    None,
                    None,
                    deallocator,
                );
                Ok(made)
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// A buffer of zeros in blocks of its own ([`buffer::of_length`]); nil where Metal or rt
    /// refuses it, counted.
    fn new_buffer_of_length(
        &self,
        on: OnFault,
        length: u64,
        options: u64,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::Buffer>> {
        let recorded_call =
            crate::recorder::rec_call!("newBufferWithLength:options:", self, length, options);
        let recorded_answer = crate::recorder::run(
            || -> RaiseResult<Option<<RtPlatform as Platform>::Buffer>> {
                let _ = on;
                let sel = "newBufferWithLength:options:";
                Ok(buffer::of_length(self, sel, length, options))
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// [`new_buffer_of_length`](GpuDevice::new_buffer_of_length)'s, a Shared one not zeroed
    /// ([`buffer::scratch_of_length`]).
    fn new_scratch_buffer_of_length(
        &self,
        on: OnFault,
        length: u64,
        options: u64,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::Buffer>> {
        let recorded_call =
            crate::recorder::rec_call!("newBufferWithLength:options:", self, length, options);
        let recorded_answer = crate::recorder::run(
            || -> RaiseResult<Option<<RtPlatform as Platform>::Buffer>> {
                let _ = on;
                let sel = "newBufferWithLength:options:";
                Ok(buffer::scratch_of_length(self, sel, length, options))
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// [`new_buffer_of_length`](GpuDevice::new_buffer_of_length)'s, a Private one's memory made
    /// at its first use ([`buffer::deferred_of_length`]). A recording holds it as the
    /// `newBufferWithLength:options:` it stands for, a buffer of zeros, as the Vulkan platform's
    /// is, so the copies and binds that name it (an IOSurface device cache's in the per-shape
    /// form) name its stream id.
    fn new_deferred_buffer_of_length(
        &self,
        on: OnFault,
        length: u64,
        options: u64,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::Buffer>> {
        let recorded_call =
            crate::recorder::rec_call!("newBufferWithLength:options:", self, length, options);
        let recorded_answer = crate::recorder::run(
            || -> RaiseResult<Option<<RtPlatform as Platform>::Buffer>> {
                let _ = on;
                let sel = "newBufferWithLength:options:";
                Ok(buffer::deferred_of_length(self, sel, length, options))
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// A buffer of zeros of its own exported block, whose memory CUDA imports as an opaque fd
    /// (`super::export`); `None` where rt makes none, counted, and the caller makes the buffer as
    /// any other.
    #[cfg(unix)]
    fn new_exportable_buffer_of_length(
        &self,
        on: OnFault,
        length: u64,
        options: u64,
        aligned_byte: Option<u64>,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::Buffer>> {
        let recorded_call = crate::recorder::rec_call!(
            "m2v.newExportableBufferOfLength",
            self,
            length,
            options,
            aligned_byte
        );
        let recorded_answer = crate::recorder::run(
            || -> RaiseResult<Option<<RtPlatform as Platform>::Buffer>> {
                let _ = on;
                let sel = "newBufferWithLength:options: (exportable)";
                let handle = rt::ExportHandle::OpaqueFd;
                let made =
                    super::export::exportable(self, sel, handle, length, options, aligned_byte);
                Ok(made)
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// A buffer of zeros of its own exported block, whose memory VA-API imports as a dma-buf
    /// (`super::export`), its byte `aligned_byte`, where given, on a multiple of 256 bytes of the
    /// memory; `None` where rt makes none, counted, and the caller makes the buffer as any other.
    /// The bytes between a texture's rows are no matter: rt makes no texture over a buffer.
    #[cfg(unix)]
    fn new_dma_buf_buffer_of_length_padding_unobserved(
        &self,
        on: OnFault,
        length: u64,
        options: u64,
        aligned_byte: Option<u64>,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::Buffer>> {
        let recorded_call = crate::recorder::rec_call!(
            "m2v.newDmaBufBufferOfLengthPaddingUnobserved",
            self,
            length,
            options,
            aligned_byte
        );
        let recorded_answer = crate::recorder::run(
            || -> RaiseResult<Option<<RtPlatform as Platform>::Buffer>> {
                let _ = on;
                let sel = "newBufferWithLength:options: (dma-buf)";
                let handle = rt::ExportHandle::DmaBuf;
                let made =
                    super::export::exportable(self, sel, handle, length, options, aligned_byte);
                Ok(made)
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// The make for `handle` (`new_exportable_buffer_of_length`'s,
    /// `new_dma_buf_buffer_of_length_padding_unobserved`'s), its exported block made at the
    /// buffer's first use (`super::export`, "Deferred exports"). A recording holds it as the
    /// make for `handle` it stands for, a buffer of zeros.
    #[cfg(unix)]
    fn new_deferred_exportable_buffer_of_length(
        &self,
        on: OnFault,
        length: u64,
        options: u64,
        aligned_byte: Option<u64>,
        handle: crate::core::platform::resource::SurfaceHandle,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::Buffer>> {
        use crate::core::platform::resource::SurfaceHandle;
        let (recorded_name, sel, handle) = match handle {
            SurfaceHandle::OpaqueFd => (
                "m2v.newExportableBufferOfLength",
                "newBufferWithLength:options: (exportable)",
                rt::ExportHandle::OpaqueFd,
            ),
            SurfaceHandle::DmaBuf => (
                "m2v.newDmaBufBufferOfLengthPaddingUnobserved",
                "newBufferWithLength:options: (dma-buf)",
                rt::ExportHandle::DmaBuf,
            ),
        };
        let recorded_call =
            crate::recorder::rec_call!(recorded_name, self, length, options, aligned_byte);
        let recorded_answer = crate::recorder::run(
            || -> RaiseResult<Option<<RtPlatform as Platform>::Buffer>> {
                let _ = on;
                let made =
                    super::export::deferred(self, sel, handle, length, options, aligned_byte);
                Ok(made)
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// rt's exported surface of `desc`'s planes (`super::surface`); `None` where rt makes none,
    /// counted, and the caller keeps the planes in images of its own.
    #[cfg(unix)]
    fn new_exported_surface(
        &self,
        on: OnFault,
        desc: &crate::core::platform::resource::SurfaceDesc<'_>,
    ) -> RaiseResult<Option<crate::core::platform::resource::ExportedSurface<RtPlatform>>> {
        let _ = on;
        Ok(super::surface::exported_surface(self, desc))
    }

    /// rt's surface over the Direct3D 11 textures `planes` name (`super::d3d11_surface`); `None`
    /// where rt makes none, counted, and the caller keeps the planes in images of its own.
    #[cfg(windows)]
    fn import_d3d11_surface(
        &self,
        on: OnFault,
        planes: &[crate::core::platform::resource::D3d11PlaneDesc<'_>],
    ) -> RaiseResult<Option<crate::core::platform::resource::ImportedSurface<RtPlatform>>> {
        let _ = on;
        Ok(super::d3d11_surface::imported_surface(self, planes))
    }

    /// rt's queue, of Metal's default `maxCommandBufferCount`; nil where rt refuses it.
    fn new_command_queue(&self) -> Option<RtObj<RtQueue>> {
        let recorded_call = crate::recorder::rec_call!("newCommandQueue", self);
        let recorded_answer = crate::recorder::run(|| -> Option<RtObj<RtQueue>> {
            match self.rt.new_command_queue(METAL_MAX_COMMAND_BUFFER_COUNT) {
                Ok(queue) => Some(RtObj::new(RtQueue::new(queue, self.clone_ref()))),
                Err(error) => {
                    rt_refused_create("newCommandQueue", &error);
                    None
                }
            }
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// [`new_command_queue`](GpuDevice::new_command_queue)'s: rt submits every command queue's
    /// work to its one GPU queue.
    fn new_command_queue_unshared(&self) -> Option<RtObj<RtQueue>> {
        let recorded_call = crate::recorder::rec_call!("m2v.newCommandQueueUnshared", self);
        let recorded_answer =
            crate::recorder::run(|| -> Option<RtObj<RtQueue>> { self.new_command_queue() });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// No: rt copies no guest RAM by its host addresses.
    fn copies_guest_ram(&self) -> bool {
        false
    }

    /// Suspends its stage fills (`super::stage`): each running ends cancelled and its helper is
    /// killed, and none starts until [`Self::reset_ends`].
    fn reset_begins(&self) {
        self.stage_fills.suspend();
    }

    /// Resumes its stage fills.
    fn reset_ends(&self) {
        self.stage_fills.resume();
    }

    /// rt's `minimumLinearTextureAlignmentForPixelFormat:` of BGRA8, the device's linear row
    /// rounding; 0 where rt refuses it.
    fn linear_row_alignment(&self) -> u64 {
        (self.rt)
            .minimum_linear_texture_alignment(rt::PixelFormat::BGRA8Unorm)
            .unwrap_or(0)
    }

    /// rt's state of the function alone (`super::pipeline`); nil for no function.
    fn new_compute_pipeline_state_for_function(
        &self,
        compute_function: Option<&<RtPlatform as Platform>::Function>,
    ) -> Result<Option<<RtPlatform as Platform>::ComputePipeline>, NsErrorText> {
        let recorded_call = crate::recorder::rec_call!(
            "newComputePipelineStateWithFunction:error:",
            self,
            compute_function
        );
        let recorded_answer = crate::recorder::run(
            || -> Result<Option<<RtPlatform as Platform>::ComputePipeline>, NsErrorText> {
                let Some(function) = compute_function else {
                    return Ok(None);
                };
                pipeline::of_function(self, function).map(Some)
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// The platform's shaders, read once (`super::library`).
    fn new_shader_library(&self) -> Result<Option<<RtPlatform as Platform>::Library>, NsErrorText> {
        let recorded_call = crate::recorder::rec_call!("newLibraryWithSource:options:error:", self);
        let recorded_answer = crate::recorder::run(
            || -> Result<Option<<RtPlatform as Platform>::Library>, NsErrorText> {
                library::shader_library(&self.shaders).map(Some)
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// rt's event; nil where rt refuses it.
    fn new_event(&self, on: OnFault) -> RaiseResult<Option<RtObj<rt::Event>>> {
        let recorded_call = crate::recorder::rec_call!("newEvent", self);
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<Option<RtObj<rt::Event>>> {
            let _ = on;
            Ok(match self.rt.new_event() {
                Ok(event) => Some(RtObj::new(event)),
                Err(error) => {
                    rt_refused_create("newEvent", &error);
                    None
                }
            })
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// rt's fence.
    fn new_fence(&self) -> RaiseResult<Option<RtObj<rt::Fence>>> {
        let recorded_call = crate::recorder::rec_call!("newFence", self);
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<Option<RtObj<rt::Fence>>> {
            Ok(Some(RtObj::new(self.rt.new_fence())))
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// The translator's library of a metallib (`super::library`).
    fn new_library(
        &self,
        data: &[u8],
    ) -> RaiseResult<Result<Option<<RtPlatform as Platform>::Library>, NsErrorText>> {
        let recorded_call = crate::recorder::rec_call!("newLibraryWithData:error:", self, data);
        let recorded_answer = crate::recorder::run(
            || -> RaiseResult<Result<Option<<RtPlatform as Platform>::Library>, NsErrorText>> {
                Ok(library::new_library(data).map(Some))
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// Unbuilt ([`Unbuilt::DynamicLibrary`]): the `NSError`.
    fn new_dynamic_library(
        &self,
        library: &<RtPlatform as Platform>::Library,
    ) -> RaiseResult<Result<Option<<RtPlatform as Platform>::DynamicLibrary>, NsErrorText>> {
        let recorded_call = crate::recorder::rec_call!("newDynamicLibrary:error:", self, library);
        let recorded_answer =
            crate::recorder::run(|| -> RaiseResult<Result<Option<<RtPlatform as Platform>::DynamicLibrary>, NsErrorText>> {
                let _ = library;
                Ok(Err(refused_error(
                    Unbuilt::DynamicLibrary,
                    "newDynamicLibrary:error:",
                )))
            });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// Unbuilt: nil ([`Unbuilt::RateMap`]).
    fn new_rate_map(
        &self,
        descriptor: &<RtPlatform as Platform>::RateMapDescriptor,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::RateMap>> {
        let recorded_call =
            crate::recorder::rec_call!("newRasterizationRateMapWithDescriptor:", self, descriptor);
        let recorded_answer = crate::recorder::run(
            || -> RaiseResult<Option<<RtPlatform as Platform>::RateMap>> {
                let _ = descriptor;
                refuse_create(Unbuilt::RateMap, "newRasterizationRateMapWithDescriptor:");
                Ok(None)
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// rt's shared event ([`super::event`]); nil where rt refuses it, counted.
    fn new_shared_event(&self) -> Option<<RtPlatform as Platform>::SharedEvent> {
        let recorded_call = crate::recorder::rec_call!("newSharedEvent", self);
        let recorded_answer =
            crate::recorder::run(|| -> Option<<RtPlatform as Platform>::SharedEvent> {
                match self.rt.new_shared_event() {
                    Ok(event) => Some(RtObj::new(event)),
                    Err(error) => {
                        rt_refused_create("newSharedEvent", &error);
                        None
                    }
                }
            });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// Unbuilt: nil ([`Unbuilt::SharedTexture`]).
    fn new_shared_texture(
        &self,
        descriptor: &<RtPlatform as Platform>::TextureDescriptor,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::Texture>> {
        let recorded_call =
            crate::recorder::rec_call!("newSharedTextureWithDescriptor:", self, descriptor);
        let recorded_answer = crate::recorder::run(
            || -> RaiseResult<Option<<RtPlatform as Platform>::Texture>> {
                let _ = descriptor;
                refuse_create(Unbuilt::SharedTexture, "newSharedTextureWithDescriptor:");
                Ok(None)
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// Unbuilt: nil ([`Unbuilt::SharedTexture`]).
    fn open_shared_texture(
        &self,
        handle: Option<&<RtPlatform as Platform>::SharedTextureHandle>,
        resource_index: u64,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::Texture>> {
        let recorded_call = crate::recorder::rec_call!(
            "newSharedTextureWithHandle:withResourceIndex:",
            self,
            handle,
            resource_index
        );
        let recorded_answer = crate::recorder::run(
            || -> RaiseResult<Option<<RtPlatform as Platform>::Texture>> {
                if let Some(handle) = handle {
                    handle.unreachable()
                }
                let _ = resource_index;
                refuse_create(
                    Unbuilt::SharedTexture,
                    "newSharedTextureWithHandle:withResourceIndex:",
                );
                Ok(None)
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// rt's texture of the descriptor ([`super::texture::make`]); a texture buffer rt's over a
    /// buffer of the platform's own that the texture holds
    /// ([`super::texture::make_texture_buffer`]). Nil where rt refuses it, counted, never a
    /// raise.
    fn new_texture(
        &self,
        on: OnFault,
        descriptor: &<RtPlatform as Platform>::TextureDescriptor,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::Texture>> {
        let recorded_call =
            crate::recorder::rec_call!("newTextureWithDescriptor:", self, descriptor);
        let recorded_answer = crate::recorder::run(
            || -> RaiseResult<Option<<RtPlatform as Platform>::Texture>> {
                let _ = on;
                let sel = "newTextureWithDescriptor:";
                let desc = descriptor.desc();
                if desc.texture_type == super::texture::TYPE_TEXTURE_BUFFER {
                    return Ok(super::texture::make_texture_buffer(self, sel, descriptor));
                }
                Ok(super::texture::make(self, sel, &desc))
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// Textures' (type 0) and sampler states' (type 1) indices as rt's grants of their class
    /// ([`grant`]), each also kept from the platform's own ([`OwnIndices::reserve`]);
    /// indirect command buffers' (type 5) are unbuilt ([`Unbuilt::Icb`]: rt gives a buffer no
    /// index), and those of another type ([`Unbuilt::ResourceType`]): each reserves nothing.
    /// Never a raise.
    fn reserve_resource_indices(&self, resource_type: u64, indices: &[u64]) -> RaiseResult<()> {
        let recorded_call = crate::recorder::rec_call!(
            "reserveResourceIndicesForResourceType:indices:indexCount:",
            self,
            resource_type,
            indices
        );
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<()> {
            let class = match resource_type {
                0 => {
                    self.indices.reserve(indices);
                    rt::IndexClass::Texture
                }
                1 => {
                    self.sampler_indices.reserve(indices);
                    rt::IndexClass::Sampler
                }
                5 => {
                    refuse(Unbuilt::Icb, RESERVE);
                    return Ok(());
                }
                _ => {
                    refuse(Unbuilt::ResourceType, RESERVE);
                    return Ok(());
                }
            };
            grant(&self.rt, RESERVE, class, indices);
            Ok(())
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// rt's description ([`super::format::description`]): Metal's on an Apple GPU, on every
    /// device.
    fn pixel_format_info(&self, format: u64) -> PixelFormatInfo {
        super::format::description(&self.rt, format)
    }

    /// rt's layout of the level ([`super::format::level_layout`]).
    fn texture_level_info(&self, q: &TextureLevelQuery) -> TextureLevelInfo {
        super::format::level_layout(&self.rt, q)
    }

    /// Read as the device was made ([`answers::caps`]).
    fn caps(&self) -> Arc<DeviceCaps> {
        Arc::clone(&self.caps)
    }

    /// [`answers::identity`], with rt's linear alignment of each format.
    fn identity_caps(&self) -> IdentityCaps {
        identity_of(self)
    }

    /// Yes, as on the Vulkan platform: the graphics lane offers its pipelines' private state, and
    /// rt's pipelines, once built, take it.
    fn takes_private_render_state(&self) -> bool {
        true
    }

    /// The device-local heaps' budgets, as the Vulkan platform answers
    /// (`crate::gpu::identity::device_local_budget`).
    fn own_memory_budget(&self) -> u64 {
        let raw = self.rt.raw();
        // SAFETY: the instance and physical device are rt's, read only, while the device lives.
        unsafe { identity::device_local_budget_of(raw.instance, raw.physical_device) }
    }

    /// Every heap's size, as the Vulkan platform answers.
    fn memory_heaps_total(&self) -> u64 {
        let raw = self.rt.raw();
        // SAFETY: as above.
        let memory =
            unsafe { (raw.instance).get_physical_device_memory_properties(raw.physical_device) };
        identity::memory_heaps_total_of(&memory)
    }

    /// Whether the driver reports rt's physical device as of type `INTEGRATED_GPU`
    /// (`integrated`).
    fn is_integrated(&self) -> bool {
        let raw = self.rt.raw();
        // SAFETY: as above.
        let properties =
            unsafe { (raw.instance).get_physical_device_properties(raw.physical_device) };
        integrated(properties.device_type)
    }

    /// Yes, as on the Vulkan platform: rt's translator links an OpenCL-mode kernel as Metal does.
    fn makes_opencl_mode_pipelines(&self) -> bool {
        true
    }

    /// [`buffer::of_desc`]'s: nil where Metal or rt refuses it, counted, the deallocator then
    /// dropped unrun.
    fn new_buffer(
        &self,
        desc: BufferDesc<'_>,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::Buffer>> {
        let mut recorded_call = crate::recorder::rec_call!("newBufferWithDescriptor:", self, desc);
        // The bytes a copy starts with; a no-copy buffer's arrive as writes, but those the create
        // copies into the device's own copy of a `Managed` one.
        if recorded_call.is_some()
            && let Some(bytes) = desc.contents
        {
            if desc.no_copy {
                let options = descriptor_options(buffer::BUFFER_DESCRIPTOR_OPTIONS, &desc.options);
                let two_copies = !self.facts().unified_memory;
                crate::recorder::no_copy_contents_with(
                    &mut recorded_call,
                    two_copies,
                    options,
                    bytes,
                    desc.zero_from,
                );
            } else {
                crate::recorder::contents(&mut recorded_call, bytes);
            }
        }
        let recorded_answer = crate::recorder::run(
            || -> RaiseResult<Option<<RtPlatform as Platform>::Buffer>> {
                Ok(buffer::of_desc(self, desc))
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// [`heap::of_desc`]'s, with the descriptor's options as Metal's descriptor holds them: nil
    /// where Metal or rt refuses it, or a sparse heap ([`Unbuilt::Heap`]), counted.
    fn new_heap(&self, desc: &HeapDesc) -> RaiseResult<HeapMade<RtPlatform>> {
        let recorded_call = crate::recorder::rec_call!("newHeapWithDescriptor:", self, desc);
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<HeapMade<RtPlatform>> {
            let held = descriptor_options(HEAP_DESCRIPTOR_OPTIONS, &desc.options);
            Ok(HeapMade {
                heap: heap::of_desc(self, desc, held),
                descriptor_resource_options: held,
            })
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// [`heap::over_host_memory`]'s, with the descriptor's options: nil where Metal or rt refuses
    /// it, counted. The recording names the bytes and the descriptor, not the deallocator.
    fn new_heap_over_host_memory(
        &self,
        bytes: HostRange<'_>,
        desc: &HeapDesc,
        deallocator: Option<NoCopyDeallocator>,
    ) -> RaiseResult<HeapMade<RtPlatform>> {
        let recorded_call =
            crate::recorder::rec_call!("m2v.newBlockWithLength:storageMode:", self, bytes, desc);
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<HeapMade<RtPlatform>> {
            let held = descriptor_options(HEAP_DESCRIPTOR_OPTIONS, &desc.options);
            Ok(HeapMade {
                heap: heap::over_host_memory(self, bytes, desc, held, deallocator),
                descriptor_resource_options: held,
            })
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// rt's sampler state ([`super::state`]); nil, counted, where rt refuses it or a value is one
    /// rt has none of.
    fn new_sampler_state(
        &self,
        desc: &SamplerDesc,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::SamplerState>> {
        let recorded_call =
            crate::recorder::rec_call!("newSamplerStateWithDescriptor:", self, desc);
        let recorded_answer = crate::recorder::run(
            || -> RaiseResult<Option<<RtPlatform as Platform>::SamplerState>> {
                Ok(super::state::new_sampler_state(self, desc))
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// rt's depth-stencil state ([`super::state`]); nil, counted, where a value is one rt has
    /// none of.
    fn new_depth_stencil_state(
        &self,
        desc: &DepthStencilDesc,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::DepthStencilState>> {
        let recorded_call =
            crate::recorder::rec_call!("newDepthStencilStateWithDescriptor:", self, desc);
        let recorded_answer = crate::recorder::run(
            || -> RaiseResult<Option<<RtPlatform as Platform>::DepthStencilState>> {
                Ok(super::state::new_depth_stencil_state(self, desc))
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// rt's buffer (`super::icb`); nil where rt refuses it, counted.
    fn new_indirect_command_buffer(
        &self,
        desc: &IcbDesc,
        max_count: u64,
        options: u64,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::Icb>> {
        let recorded_call = crate::recorder::rec_call!(
            "newIndirectCommandBufferWithDescriptor:maxCommandCount:options:",
            self,
            desc,
            max_count,
            options
        );
        let recorded_answer =
            crate::recorder::run(|| -> RaiseResult<Option<<RtPlatform as Platform>::Icb>> {
                let _ = options;
                Ok(super::icb::make(self, desc, max_count))
            });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// The display's state (`super::render_pipeline`).
    fn new_render_pipeline_state(
        &self,
        desc: &RenderPipelineDesc<'_, RtPlatform>,
    ) -> Result<Option<<RtPlatform as Platform>::RenderPipeline>, NsErrorText> {
        let recorded_call =
            crate::recorder::rec_call!("newRenderPipelineStateWithDescriptor:error:", self, desc);
        let recorded_answer = crate::recorder::run(
            || -> Result<Option<<RtPlatform as Platform>::RenderPipeline>, NsErrorText> {
                render_pipeline::of_value(self, desc).map(Some)
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// The state of the whole descriptor (`super::render_pipeline`).
    fn new_render_pipeline_state_full(
        &self,
        desc: &FullRenderPipelineDesc<'_, RtPlatform>,
    ) -> Result<Option<<RtPlatform as Platform>::RenderPipeline>, NsErrorText> {
        let recorded_call =
            crate::recorder::rec_call!("newRenderPipelineStateWithDescriptor:error:", self, desc);
        let recorded_answer = crate::recorder::run(
            || -> Result<Option<<RtPlatform as Platform>::RenderPipeline>, NsErrorText> {
                render_pipeline::of_descriptor(self, desc).map(Some)
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// Unbuilt ([`Unbuilt::MeshPipeline`]): the `NSError`.
    fn new_mesh_render_pipeline_state_full(
        &self,
        desc: &FullMeshRenderPipelineDesc<'_, RtPlatform>,
    ) -> Result<Option<<RtPlatform as Platform>::RenderPipeline>, NsErrorText> {
        let recorded_call = crate::recorder::rec_call!(
            "newRenderPipelineStateWithMeshDescriptor:options:reflection:error:",
            self,
            desc
        );
        let recorded_answer = crate::recorder::run(
            || -> Result<Option<<RtPlatform as Platform>::RenderPipeline>, NsErrorText> {
                render_pipeline::of_mesh_descriptor(desc).map(Some)
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// rt's state of the whole descriptor (`super::pipeline`).
    fn new_compute_pipeline_state_full(
        &self,
        desc: &FullComputePipelineDesc<'_, RtPlatform>,
    ) -> Result<Option<<RtPlatform as Platform>::ComputePipeline>, NsErrorText> {
        let recorded_call = crate::recorder::rec_call!(
            "newComputePipelineStateWithDescriptor:options:reflection:error:",
            self,
            desc
        );
        let recorded_answer = crate::recorder::run(
            || -> Result<Option<<RtPlatform as Platform>::ComputePipeline>, NsErrorText> {
                pipeline::of_descriptor(self, desc).map(Some)
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }
}

/// The identity of `device` ([`answers::identity`]): rt's linear and texture-buffer alignments
/// of each SDK format rt has a format of, none where it refuses one.
fn identity_of(device: &RtObj<RtDevice>) -> IdentityCaps {
    answers::identity(
        &device.facts,
        &device.caps,
        |format| linear_alignment(&device.rt, format),
        |format| texture_buffer_alignment(&device.rt, format),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::platform::device::relaying_gaps;
    use crate::core::platform::values::{OptionSet, TextureDesc};

    /// The options a device is opened with carry the helper's memory ceiling on this host, the
    /// one a memory breach is kept under.
    #[test]
    fn the_shared_options_carry_the_helpers_ceiling_on_vulkan() {
        let instance =
            crate::os::rt_instance(&crate::os::rt_instance_extensions()).expect("rt's instance");
        let listed = instance.physical_devices().expect("rt's devices");
        let physical = listed.first().expect("a device on the GPU the test holds");
        let (options, _) = shared_options(&instance, physical);
        let ceiling = crate::os::compile_helper::Limits::of_host().memory;
        assert_ne!(ceiling, 0);
        assert_eq!(options.stage_memory_ceiling, ceiling);
    }

    /// The line of the answers rt chose its forms on gives each floor answer by its name, in
    /// order, then the profile's, then whether memory is unified, on one line.
    #[test]
    fn the_forms_line_names_each_answer() {
        let forms = RtForms {
            floor: [
                ("float16", true),
                ("storage_read_without_format", false),
                ("buffer_int64_atomics", true),
                ("shared_int64_atomics", false),
                ("float64", true),
                ("maximal_reconvergence", false),
                ("subgroup_uniform_control_flow", true),
                ("texel_offset_operand", false),
                ("attachment_reads", true),
                ("feedback_loop", false),
            ],
            max_compute_shared_memory_size: 65536,
            push_window: 256,
            subgroups: &[32_u32, 64],
            unified_memory: false,
        };
        assert_eq!(
            forms.to_string(),
            "rt capabilities: float16 true, storage_read_without_format false, \
             buffer_int64_atomics true, shared_int64_atomics false, float64 true, \
             maximal_reconvergence false, subgroup_uniform_control_flow true, \
             texel_offset_operand false, attachment_reads true, feedback_loop false; profile: \
             max_compute_shared_memory_size 65536, push_window 256, subgroups [32, 64]; \
             unified_memory false"
        );
    }

    /// Only Vulkan's `INTEGRATED_GPU` is an integrated GPU: a discrete, virtual, CPU or other
    /// device is not
    #[test]
    fn only_the_integrated_device_type_is_integrated() {
        use vk::PhysicalDeviceType as T;
        assert!(integrated(T::INTEGRATED_GPU));
        for other in [T::OTHER, T::DISCRETE_GPU, T::VIRTUAL_GPU, T::CPU] {
            assert!(!integrated(other), "device type {}", other.as_raw());
        }
    }

    /// Each fold counts what each over-copy metric rose by since the last, under its code, so the
    /// counts line carries the metrics' totals: the over-copies and their bytes, those over an
    /// imported block, and each cause's, by the cause's name; and after them the command buffers
    /// held for room, the cache writes dropped, the cache bytes written, the translations, the
    /// translations loaded, the translator's nanoseconds, the `VkPipeline`s made and the driver's
    /// nanoseconds making them, the render records read and stale, the render pipelines made with
    /// a stage past the kernels' bound, the compute dispatches' first-use waits and their
    /// nanoseconds, the exported blocks whose address went out, the pipeline caches replaced, the
    /// cache's writes skipped past its bound, the reaper's looks at it skipped and the creations
    /// made before its seed, the dispatches that ran an emulated-SIMD-group module, the compiled
    /// contents a state's make dropped, the resolves of multisample attachments, the copies out of
    /// resolve images and their bytes, the whole grids and the ragged grids that ran a
    /// partial-threadgroups variant, the dispatches that wrote a view through its image's extended
    /// storage usage, the submissions that handed Direct3D 11 planes over, the dispatches and draws
    /// that pushed a storage texture's level fields, the heap textures moved out of the undefined
    /// layout, the draws that read an attachment of their own pass through the feedback loop, the
    /// copies into attachments' shadows on a device without it and their bytes, the draws of
    /// stencil output by bits, the main pipelines the states' finishes made and their
    /// nanoseconds, the first-use waits for an unfinished main pipeline and their nanoseconds, and
    /// the over-copies of a kind rt kept none of.
    #[test]
    fn over_copies_fold_into_the_counts_by_their_rise() {
        let codes = [
            OVER_COPIES_CODE,
            OVER_COPY_BYTES_CODE,
            OVER_COPIES_IMPORTED_CODE,
            OVER_COPY_BYTES_IMPORTED_CODE,
            "rt-over-copies-host-memory",
            "rt-over-copy-bytes-host-memory",
            "rt-over-copies-pitch",
            "rt-over-copy-bytes-pitch",
            "rt-over-copies-image-memory",
            "rt-over-copy-bytes-image-memory",
            HELD_FOR_ROOM_CODE,
            CACHE_WRITES_DROPPED_CODE,
            CACHE_BYTES_WRITTEN_CODE,
            TRANSLATIONS_CODE,
            TRANSLATIONS_LOADED_CODE,
            TRANSLATION_NS_CODE,
            PIPELINES_CREATED_CODE,
            PIPELINE_CREATION_NS_CODE,
            RENDER_RECORDS_READ_CODE,
            RENDER_RECORDS_STALE_CODE,
            RENDER_MAKES_PAST_KERNEL_BOUND_CODE,
            FIRST_USE_WAITS_CODE,
            FIRST_USE_WAIT_NS_CODE,
            EXPORTED_BLOCKS_ADDRESSED_CODE,
            PIPELINE_CACHES_REPLACED_CODE,
            PIPELINE_CACHE_WRITES_SKIPPED_CODE,
            PIPELINE_CACHE_LOOKS_SKIPPED_CODE,
            CREATIONS_BEFORE_SEED_CODE,
            EMULATED_SIMDGROUP_DISPATCHES_CODE,
            KERNEL_MAKES_DROPPED_CODE,
            RESOLVES_CODE,
            RESOLVE_COPIES_CODE,
            RESOLVE_COPY_BYTES_CODE,
            PARTIAL_VARIANT_WHOLE_GRIDS_CODE,
            PARTIAL_VARIANT_RAGGED_GRIDS_CODE,
            EXTENDED_VIEW_DISPATCHES_CODE,
            D3D11_PLANE_SUBMISSIONS_CODE,
            LEVEL_FIELD_COMMANDS_CODE,
            HEAP_MOVES_CODE,
            ATTACHMENT_LOOP_READS_CODE,
            ATTACHMENT_READ_COPIES_CODE,
            ATTACHMENT_READ_COPY_BYTES_CODE,
            STENCIL_BIT_DRAWS_CODE,
            FINISHED_PIPELINES_CODE,
            FINISHED_PIPELINE_NS_CODE,
            UNFINISHED_WAITS_CODE,
            UNFINISHED_WAIT_NS_CODE,
            OVER_COPIES_UNKEPT_KIND_CODE,
        ];
        let counts = || {
            let all = crate::answer::degradation_counts();
            codes.map(|code| (all.iter()).find(|(c, _)| c == code).map_or(0, |&(_, n)| n))
        };
        let before = counts();
        let mut m = rt::Metrics::default();
        m.over_copies = 2;
        m.over_copy_bytes = 8192;
        m.over_copies_imported = 1;
        m.over_copy_bytes_imported = 4096;
        m.over_copies_host_memory = 1;
        m.over_copy_bytes_host_memory = 4096;
        m.over_copies_pitch = 1;
        m.over_copy_bytes_pitch = 4096;
        m.held_for_room = 2;
        m.cache_writes_dropped = 1;
        m.cache_bytes_written = 65_536;
        m.translations = 7;
        m.translation_nanos = 3_000_000;
        m.pipelines_created = 4;
        m.pipeline_creation_nanos = 20_000_000;
        m.render_records_read = 2;
        m.render_makes_past_kernel_bound = 1;
        m.first_use_variant_waits = 2;
        m.first_use_variant_wait_nanos = 5_000;
        m.exported_blocks_addressed = 1;
        m.pipeline_caches_replaced = 1;
        m.pipeline_cache_looks_skipped = 2;
        m.creations_before_seed = 2;
        m.emulated_simdgroup_dispatches = 4;
        m.resolves = 3;
        m.resolve_copies = 1;
        m.resolve_copy_bytes = 3_686_400;
        m.partial_variant_whole_grids = 6;
        m.partial_variant_ragged_grids = 2;
        m.extended_view_dispatches = 1;
        m.d3d11_plane_submissions = 3;
        m.level_field_commands = 5;
        m.heap_moves = 2;
        m.attachment_loop_reads = 7;
        m.attachment_read_copies = 1;
        m.attachment_read_copy_bytes = 1_048_576;
        m.stencil_bit_draws = 8;
        m.finished_pipelines = 2;
        m.finished_pipeline_nanos = 12_000_000;
        m.first_use_unfinished_waits = 1;
        m.first_use_unfinished_wait_nanos = 400_000;
        m.over_copies_unkept_kind = 1;
        let mut folded = [0; FOLDED_METRICS];
        fold(&mut folded, folded_counts(&m));
        m.over_copies = 3;
        m.over_copy_bytes = 12288;
        m.over_copies_image_memory = 1;
        m.over_copy_bytes_image_memory = 4096;
        m.held_for_room = 5;
        m.cache_writes_dropped = 3;
        m.cache_bytes_written = 98_304;
        m.translations = 9;
        m.translations_loaded = 4;
        m.translation_nanos = 3_000_000 + 1_250_000;
        m.pipelines_created = 6;
        m.pipeline_creation_nanos = 20_000_000 + 8_500_000;
        m.render_records_read = 3;
        m.render_records_stale = 1;
        m.render_makes_past_kernel_bound = 2;
        m.first_use_variant_waits = 3;
        m.first_use_variant_wait_nanos = 9_000;
        m.exported_blocks_addressed = 2;
        m.pipeline_caches_replaced = 3;
        m.pipeline_cache_writes_skipped = 1;
        m.pipeline_cache_looks_skipped = 7;
        m.creations_before_seed = 5;
        m.emulated_simdgroup_dispatches = 11;
        m.kernel_makes_dropped = 2;
        m.resolves = 10;
        m.resolve_copies = 4;
        m.resolve_copy_bytes = 4 * 3_686_400;
        m.partial_variant_whole_grids = 13;
        m.partial_variant_ragged_grids = 9;
        m.extended_view_dispatches = 4;
        m.d3d11_plane_submissions = 12;
        m.level_field_commands = 8;
        m.heap_moves = 6;
        m.attachment_loop_reads = 7 + 12;
        m.attachment_read_copies = 3;
        m.attachment_read_copy_bytes = 3 * 1_048_576;
        m.stencil_bit_draws = 8 + 16;
        m.finished_pipelines = 5;
        m.finished_pipeline_nanos = 12_000_000 + 6_750_000;
        m.first_use_unfinished_waits = 3;
        m.first_use_unfinished_wait_nanos = 400_000 + 900_000;
        m.over_copies_unkept_kind = 4;
        fold(&mut folded, folded_counts(&m));
        assert_eq!(folded, folded_counts(&m).map(|(_, n)| n));
        let after = counts();
        let rose: Vec<u64> = (after.iter().zip(before)).map(|(a, b)| a - b).collect();
        assert_eq!(
            rose,
            [
                3, 12288, 1, 4096, 1, 4096, 1, 4096, 1, 4096, 5, 3, 98_304, 9, 4, 4_250_000, 6,
                28_500_000, 3, 1, 2, 3, 9_000, 2, 3, 1, 7, 5, 11, 2, 10, 4, 14_745_600, 13, 9, 4,
                12, 8, 6, 19, 3, 3_145_728, 24, 5, 18_750_000, 3, 1_300_000, 4
            ]
        );
        let named = over_copy_counts(&m).map(|(code, _)| code);
        let unique: std::collections::BTreeSet<&str> = named.iter().copied().collect();
        assert_eq!(unique.len(), OVER_COPY_METRICS, "{named:?}");
        assert!(
            named.iter().all(|code| code.starts_with("rt-over-cop")),
            "{named:?}"
        );
    }

    /// The longest first-use wait goes to the counts at its longest over the process's devices
    /// ([`note_highest_metrics`]), as the heaviest render stage does: counted by its rise over the
    /// longest counted, so a device whose longest is shorter leaves the count as it was.
    #[test]
    fn the_longest_first_use_wait_counts_its_rise_over_the_longest_counted() {
        let count = || {
            (crate::answer::degradation_counts().iter())
                .find(|(c, _)| c == FIRST_USE_WAIT_MAX_NS_CODE)
                .map_or(0, |&(_, n)| n)
        };
        let before = count();
        let mut highest = [0; HIGHEST_METRICS];
        let mut m = rt::Metrics::default();
        m.first_use_variant_wait_max_nanos = 7_000;
        raise_to(&mut highest, highest_metrics(&m));
        assert_eq!(count() - before, 7_000);
        m.first_use_variant_wait_max_nanos = 2_000;
        raise_to(&mut highest, highest_metrics(&m));
        assert_eq!(highest, [0, 7_000, 0]);
        assert_eq!(count() - before, 7_000);
    }

    /// The longest wait for an unfinished main pipeline goes to the counts at its longest over
    /// the process's devices ([`note_highest_metrics`]), apart from the variants' longest: counted
    /// by its rise over the longest counted, so a device whose longest is shorter leaves the count
    /// as it was.
    #[test]
    fn the_longest_unfinished_wait_counts_its_rise_over_the_longest_counted() {
        let count = || {
            (crate::answer::degradation_counts().iter())
                .find(|(c, _)| c == UNFINISHED_WAIT_MAX_NS_CODE)
                .map_or(0, |&(_, n)| n)
        };
        let before = count();
        let mut highest = [0; HIGHEST_METRICS];
        let mut m = rt::Metrics::default();
        m.first_use_unfinished_wait_max_nanos = 3_500_000;
        raise_to(&mut highest, highest_metrics(&m));
        assert_eq!(count() - before, 3_500_000);
        let mut second = rt::Metrics::default();
        second.first_use_unfinished_wait_max_nanos = 1_000_000;
        raise_to(&mut highest, highest_metrics(&second));
        second.first_use_unfinished_wait_max_nanos = 5_000_000;
        raise_to(&mut highest, highest_metrics(&second));
        assert_eq!(highest, [0, 0, 5_000_000]);
        assert_eq!(count() - before, 5_000_000);
    }

    /// Twelve of rt's gauges go to the counts at their highest ([`note_gauges`]): each counted by
    /// its rise over the highest a look saw, so that one that falls leaves its count as it was and
    /// one that rises past it raises it, the pipeline cache's data falling as its reaper replaces
    /// a full cache with a fresh one, the resolve images as their textures drop, the compiled
    /// kernels as their states drop while the map of compiled contents stays full, their
    /// `VkPipeline`s as work retires past the 64 a module keeps, the LOD clamp variants as their
    /// sampler states drop, and the runs of level elements as their textures drop.
    #[test]
    fn twelve_gauges_count_at_their_highest() {
        let codes = [
            PREPROCESS_USED_MOST_CODE,
            PREPROCESS_IDLE_MOST_CODE,
            TEXTURE_BLOCK_HELD_MOST_CODE,
            PRIVATE_TAKE_MOST_CODE,
            PIPELINE_CACHE_MOST_CODE,
            RESOLVE_IMAGES_MOST_CODE,
            RESOLVE_IMAGE_BYTES_MOST_CODE,
            KERNELS_MOST_CODE,
            KERNEL_ENTRIES_MOST_CODE,
            KERNEL_VARIANTS_MOST_CODE,
            LOD_CLAMP_VARIANTS_MOST_CODE,
            LEVEL_ELEMENTS_MOST_CODE,
        ];
        let counts = || {
            let all = crate::answer::degradation_counts();
            codes.map(|code| (all.iter()).find(|(c, _)| c == code).map_or(0, |&(_, n)| n))
        };
        let before = counts();
        let mut g = rt::Gauges::default();
        g.preprocess_used_bytes = 4 << 20;
        g.preprocess_idle_bytes = 64 << 20;
        g.texture_block_held_bytes = 48 << 20;
        g.private_take_bytes = 16 << 20;
        g.pipeline_cache_bytes = 30 << 20;
        g.resolve_images = 2;
        g.resolve_image_bytes = 8 << 20;
        g.kernels = 1_100;
        g.kernel_entries = 1_024;
        g.kernel_variants = 90;
        g.lod_clamp_variants = 12;
        g.level_elements = 6;
        let mut highest = [0; HIGHEST_GAUGES];
        raise_to(&mut highest, highest_gauges(&g));
        g.preprocess_used_bytes = 12 << 20;
        g.preprocess_idle_bytes = 0;
        g.texture_block_held_bytes = 32 << 20;
        g.pipeline_cache_bytes = 1 << 20;
        g.resolve_images = 1;
        g.resolve_image_bytes = 4 << 20;
        g.kernels = 1_030;
        g.kernel_variants = 64;
        g.lod_clamp_variants = 3;
        g.level_elements = 2;
        raise_to(&mut highest, highest_gauges(&g));
        g.pipeline_cache_bytes = 31 << 20;
        raise_to(&mut highest, highest_gauges(&g));
        g.resolve_images = 3;
        g.kernels = 1_150;
        g.lod_clamp_variants = 40;
        g.level_elements = 9;
        raise_to(&mut highest, highest_gauges(&g));
        assert_eq!(
            highest,
            [
                12 << 20,
                64 << 20,
                48 << 20,
                16 << 20,
                31 << 20,
                3,
                8 << 20,
                1_150,
                1_024,
                90,
                40,
                9
            ]
        );
        let after = counts();
        let rose: Vec<u64> = (after.iter().zip(before)).map(|(a, b)| a - b).collect();
        assert_eq!(rose, highest);
    }

    /// Each over-copy's kind is logged once a process ([`first_seen`]): a kind a fold read before,
    /// of this device or another, is not logged again, nor one twice in one read; the rest go in
    /// rt's order.
    #[test]
    fn each_over_copy_kind_is_logged_once_a_process() {
        let mut logged = Vec::new();
        assert_eq!(first_seen(&mut logged, vec![3, 1, 3]), [3, 1]);
        assert_eq!(first_seen(&mut logged, vec![]), [0; 0]);
        assert_eq!(first_seen(&mut logged, vec![1, 4, 2, 4, 3]), [4, 2]);
        assert_eq!(logged, [3, 1, 4, 2]);
    }

    /// An over-copy's kind's line names each of its fields ([`over_copy_kind_line`]): the cause by
    /// its name in rt, the pixel format by its Metal name and value, the size in texels and the
    /// usage's bits.
    #[test]
    fn an_over_copy_kind_s_line_names_its_fields() {
        let usage = rt::TextureUsage::SHADER_READ | rt::TextureUsage::RENDER_TARGET;
        let line = over_copy_kind_line(
            rt::OverCause::Pitch,
            rt::PixelFormat::BGRA8Unorm,
            [1920, 1080],
            usage,
        );
        assert_eq!(
            line,
            "rt's first over-copy of a kind: cause Pitch, pixel format MTLPixelFormatBGRA8Unorm \
             (80), 1920 by 1080, usage 0x5"
        );
    }

    /// The heaviest render stage past the kernels' bound goes to the counts at its highest
    /// ([`note_highest_metrics`]), as the gauges do: rt's maximum since a device was made, so a
    /// second device's smaller maximum leaves the count as it was and a larger one raises it to
    /// itself, never by the sum.
    #[test]
    fn the_heaviest_render_stage_past_the_bound_counts_at_its_highest() {
        let count = || {
            let all = crate::answer::degradation_counts();
            (all.iter())
                .find(|(c, _)| *c == RENDER_WEIGHT_PAST_KERNEL_BOUND_MOST_CODE)
                .map_or(0, |&(_, n)| n)
        };
        let before = count();
        let mut highest = [0; HIGHEST_METRICS];
        let mut first = rt::Metrics::default();
        first.render_weight_past_kernel_bound_max = 52_000;
        raise_to(&mut highest, highest_metrics(&first));
        raise_to(&mut highest, highest_metrics(&first));
        let mut second = rt::Metrics::default();
        second.render_weight_past_kernel_bound_max = 41_000;
        raise_to(&mut highest, highest_metrics(&second));
        second.render_weight_past_kernel_bound_max = 61_500;
        raise_to(&mut highest, highest_metrics(&second));
        assert_eq!(highest, [61_500, 0, 0]);
        assert_eq!(count() - before, 61_500);
    }

    /// The counts read the device the process made last ([`follow`]), on the GPU the test holds:
    /// two devices made in turn, the first let go of before the second is made
    /// (`RtPlatform::let_go_of_device`), each one's submissions read as its own as it is made and
    /// after its work, the first's no more once the second is made; and the over-copies folded
    /// are both devices' ([`fold_metrics`]).
    #[test]
    fn the_counts_read_each_device_made_in_turn_on_vulkan() {
        use crate::core::platform::command::{CommandBuffer as _, CommandQueue as _};
        use crate::core::platform::device::GpuDevice as _;

        let platform = RtPlatform::new();
        let read = || metrics().map(|m| m.submissions);
        let over_copies = || {
            let all = crate::answer::degradation_counts();
            (all.iter())
                .find(|(code, _)| *code == OVER_COPIES_CODE)
                .map_or(0, |&(_, n)| n)
        };
        let counted = over_copies();
        let mut made: Vec<rt::Counters> = Vec::new();
        // Each command buffer fills a buffer, as rt submits nothing for one with no work, and
        // each commit is waited for before the next, so each is a submission of its own: the
        // first device's count differs from the second's at each read.
        for (turn, commits) in [(1, 8), (2, 1)] {
            let device = (platform.system_default_device()).expect("an rt device on the GPU");
            let counters = device.rt().counters();
            let own = || Some(counters.metrics().submissions);
            let others = || (made.iter()).map(|c| Some(c.metrics().submissions));
            assert_eq!(read(), own(), "device {turn} made: its count");
            let before = counters.metrics().submissions;
            let queue = device.new_command_queue().expect("a command queue");
            let filled = device.new_buffer_of_length(OnFault::Return, 4096, 0);
            let filled = filled.expect("never a raise").expect("a Shared buffer");
            for _ in 0..commits {
                let buffer = (queue.command_buffer(OnFault::Return))
                    .expect("never a raise")
                    .expect("a command buffer");
                let mut blit = buffer.rt().blit_command_encoder().expect("a blit encoder");
                blit.fill_buffer(filled.rt(), 0..4096, 0x5a)
                    .expect("a fill");
                blit.end_encoding();
                assert_eq!(buffer.commit(OnFault::Return), Ok(()));
                assert_eq!(buffer.wait_until_completed(OnFault::Return), Ok(()));
            }
            assert_eq!(read(), own(), "device {turn}'s work: its count");
            let rose = counters.metrics().submissions - before;
            assert!(
                rose >= commits,
                "device {turn}: {rose} submissions for {commits} commits"
            );
            assert!(
                others().all(|other| other != own()),
                "device {turn}: the devices' counts tell them apart: {:?} against {:?}",
                own(),
                others().collect::<Vec<_>>()
            );
            drop((filled, queue, device));
            assert_eq!(
                platform.let_go_of_device(std::time::Duration::from_secs(5)),
                Ok(())
            );
            made.push(counters);
        }
        fold_metrics();
        let copied: u64 = (made.iter()).map(|c| c.metrics().over_copies).sum();
        assert_eq!(over_copies() - counted, copied, "both devices' over-copies");
    }

    /// The rt platform's device on the GPU the test holds: made once and answered at every ask,
    /// its registry ID opening it again; its answers bounded by its GPU, its texture-buffer
    /// alignments and `maxThreadgroupMemoryLength` rt's; its queues, events and fences made; the
    /// guest's texture indices granted, twice without a refusal; a heap texture's size rt's, and a
    /// shape rt or Metal refuses a counted raise, never 0 bytes at 0; every creator of an unbuilt
    /// piece nil and counted.
    #[test]
    fn the_device_and_its_answers_on_vulkan() {
        use crate::core::platform::device::GpuDevice as _;
        let platform = RtPlatform::new();
        let device = platform
            .system_default_device()
            .expect("an rt device on the GPU the test holds");
        let again = platform.system_default_device().expect("the same device");
        assert_eq!(again.id(), device.id());
        let caps = device.caps();
        assert_eq!(
            platform.open_device(caps.registry_id).map(|d| d.id()),
            Some(device.id())
        );
        assert!(platform.open_device(caps.registry_id ^ 1).is_none());
        let limits = device.facts().limits;
        assert!(caps.limits.max_texture_width_2d <= u64::from(limits.max_image_dimension2_d));
        // maxThreadgroupMemoryLength is rt's: the device's maxComputeSharedMemorySize less rt's
        // 4-byte vote word, rounded down to 16, at most Metal's 32 KiB (rt's
        // `Device::max_threadgroup_memory_length` docs); 32,752 on a device of 32,768 bytes.
        let told = caps.limits.max_threadgroup_memory_length;
        let shared = u64::from(limits.max_compute_shared_memory_size);
        eprintln!("maxThreadgroupMemoryLength {told} of maxComputeSharedMemorySize {shared}");
        assert_eq!(told, u64::from(device.rt().max_threadgroup_memory_length()));
        assert_eq!(told, (shared.saturating_sub(4) / 16 * 16).min(32768));
        assert!(caps.sample_counts[0] == Ok(true));
        let page = SizeAndAlign {
            size: device.facts().page_size,
            align: 256,
        };
        let on = OnFault::Die;
        assert_eq!(caps.heap_buffer_page_private(on), Ok(page));
        let identity = device.identity_caps();
        assert_eq!(
            identity.formats.len(),
            crate::core::platform::caps::SDK_PIXEL_FORMATS.len()
        );
        // The linear texture alignments are the largest format's, and BGRA8's is the device's
        // linear row rounding.
        let largest = (identity.formats.iter())
            .filter_map(|f| f.linear_alignment)
            .fold(16, u64::max);
        assert_eq!(caps.limits.linear_texture_alignment_bytes, largest);
        assert_eq!(caps.limits.device_linear_texture_alignment_bytes, largest);
        assert_eq!(caps.limits.iosurface_texture_alignment_bytes, largest);
        assert_eq!(
            caps.min_linear_alignment_bgra8(),
            Ok(device.linear_row_alignment())
        );
        // Each texture-buffer format answers rt's texture-buffer alignment of it, where rt
        // answers one, and none answers 0; a format Metal answers none of answers none.
        let texel_formats = crate::core::platform::caps::TEXTURE_BUFFER_FORMATS;
        for answers in &identity.formats {
            let rt_answer = texture_buffer_alignment(device.rt(), answers.format);
            let answered = answers.texture_buffer_alignment;
            assert_ne!(answered, Some(0), "format {}", answers.format);
            match texel_formats.contains(&answers.format) {
                true if rt_answer.is_some() => assert_eq!(answered, rt_answer),
                true => {}
                false => assert_eq!(answered, None, "format {}", answers.format),
            }
        }
        let texel_answers: Vec<(u64, u64)> = (identity.formats.iter())
            .filter_map(|f| Some((f.format, f.texture_buffer_alignment?)))
            .collect();
        eprintln!("texture-buffer alignments (format, bytes): {texel_answers:?}");
        assert!(device.own_memory_budget() > 0);
        assert!(device.memory_heaps_total() >= device.own_memory_budget());
        assert!(device.linear_row_alignment() > 0);
        assert!(device.new_command_queue().is_some());
        assert!(device.new_command_queue_unshared().is_some());
        assert!(matches!(device.new_event(OnFault::Return), Ok(Some(_))));
        assert!(matches!(device.new_fence(), Ok(Some(_))));
        let guests: Vec<u64> = (1..=16320).collect();
        let before = crate::answer::degradation_counts();
        assert_eq!(device.reserve_resource_indices(0, &guests), Ok(()));
        assert_eq!(device.reserve_resource_indices(0, &guests[..64]), Ok(()));
        assert_eq!(crate::answer::degradation_counts(), before);
        let sparse = HeapDesc {
            size: 0x4000,
            heap_type: 2,
            options: vec![OptionSet::StorageMode(2)],
            pinned_gpu_address: None,
        };
        let (made, relayed) = relaying_gaps(|| device.new_heap(&sparse));
        assert!(made.is_ok_and(|made| made.heap.is_none()) && relayed);
        assert!(device.new_library(&[]).is_ok_and(|made| made.is_err()));
        let texture = platform.texture_descriptor(&TextureDesc {
            pixel_format: 80,
            width: 64,
            height: 64,
            resource_options: Some(0x20),
            ..TextureDesc::default()
        });
        let sized = device.heap_texture_size_and_align(&texture);
        let Ok(SizeAndAlign { size, align }) = sized else {
            panic!("no size: {sized:?}");
        };
        assert!(
            size >= 64 * 64 * 4 && align.is_power_of_two(),
            "{size} at {align}"
        );
        let counted = |code: &str| {
            (crate::answer::degradation_counts().into_iter())
                .find(|(c, _)| c == code)
                .map_or(0, |(_, n)| n)
        };
        let metal = RtRefusal::Refused.code();
        let before = counted(metal);
        let refused = [
            TextureDesc {
                texture_type: 99,
                ..TextureDesc::default()
            },
            TextureDesc {
                width: 0,
                ..TextureDesc::default()
            },
        ];
        for shape in refused {
            let descriptor = platform.texture_descriptor(&shape);
            let asked = device.heap_texture_size_and_align(&descriptor);
            assert!(asked.is_err(), "{shape:?}: {asked:?}");
        }
        assert_eq!(counted(metal), before + 2);

        let bgra8 = device.pixel_format_info(80);
        assert_eq!(bgra8.name.as_deref(), Some("MTLPixelFormatBGRA8Unorm"));
        let q = TextureLevelQuery {
            pixel_format: 80,
            width: 64,
            height: 32,
            depth: 1,
            array_length: 1,
            ..TextureLevelQuery::default()
        };
        let level = device.texture_level_info(&q);
        assert_eq!((level.bytes_per_row, level.aligned_height), (256, 32));
    }
}
