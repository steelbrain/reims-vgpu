//! The stage-3 Metal-call decoder: it reads the guest bundle's serialised Metal
//! calls off a command queue's ring and drives the library's `Platform`
//! (docs/graphics-kext/design.md, "Stage 3 wire protocol"). It is generic over
//! the platform, so it drives Metal on macOS and the Vulkan platform on Linux,
//! the same trait the Apple lane's deserializer drives.
//!
//! The decoder keeps an `obj_id` → handle table: a create ([`GpuDevice`] and the
//! resource creators) inserts the new handle, a destroy drops it (handles are
//! refcounted, so the drop releases the object). Everything the decoder drives is
//! the library's `Platform` directly — this lane keeps no resource manager of the
//! Apple lane's; the handle table is the whole of an object's host lifetime.
//!
//! What this covers:
//! - the descriptor-free **state objects** (events, fences, shared events,
//!   command queues) and the descriptor creates (samplers, depth/stencil,
//!   libraries) and **memory resources** (buffers, heaps, textures), each on the
//!   `decode` message dispatch;
//! - the **command buffer** op stream ([`cmdbuf`]): a
//!   [`RGFX_CMD_COMMAND_BUFFER`] opens a `Platform` command buffer on its queue,
//!   walks the encoder ops (the render group in [`render`], the compute and blit
//!   groups stubbed in [`compute`] and [`blit`]), commits, and registers the
//!   completion path;
//! - the **object-resolution API** (the `require_*`/`get_*` methods below) the
//!   encoder-op decoders call to turn a wire `obj_id` into a host handle.
//!
//! The decoder runs under a live raise handler, as the Apple lane's command
//! processing does: a platform raise propagates out of [`GraphicsDecoder::decode`]
//! for the caller to catch. An op-stream reference the host cannot honour faults
//! the exec the same way ([`bad_reference`], [`op_fault`]), but for one of an object
//! whose posted create the host refused, whose uses are dropped and whose command
//! buffers go on ([`refused`]).
//!
//! [`super::GraphicsControl`] drives it from the device's threads: ring 0's messages on their
//! context's lane (the presents on the presents' lane, a block's pages and downloads on the
//! block's), each
//! queue's command buffers on the queue's service thread (or a doorbell's),
//! and the compiles' ends on the compile pool. Each message is carried out through a
//! [`GraphicsDecoder`] of its own, a handle on the state the decoder keeps for every context
//! ([`Shared`]) with the message's sender and the sinks its records go to; no lock is held across a
//! message by any of them, so one context's command buffer never holds up another's, nor ring 0.
//! The object-resolution API below ([`GraphicsDecoder::require_buffer`] and the rest) is the whole
//! contract the encoder-op decoders call; some of its resolvers are for ops the first op slice
//! does not emit yet, and so are not reached, but they are kept whole (project rule "Unread is not
//! dead").
//!
//! **The locks** (each held only for what it guards, never across a command buffer's walk, a
//! platform call that waits, a wait of another kind, or a post, but where said):
//! - `GraphicsControl`'s reset gate, outside every lock here: held shared by each message carried
//!   out off ring 0's threads, from before its staleness check to its end, and alone by the
//!   device's reset ([`GraphicsDecoder::reset`]), which so finds no message in flight (ring 0's
//!   own have returned by then: the device tells the host of a reset once its lanes' have).
//! - `contexts`: which contexts have state ([`tables::Context`]), to find, make or drop one.
//! - each context's table ([`tables`]): read at a message's first naming of an object, and
//!   written to take or close a view and at each make and removal. A message sees the table as of
//!   its view ([`tables::View`]).
//! - each context's drain: a command buffer joins it as it is begun.
//! - `surfaces`: the IOSurfaces' names (the textures over each one's plane 0, and their copies'
//!   writers), which presents read.
//! - `libraries`: the library chunks, makes and cache, on ring 0 and the compile pool. Held across
//!   a library's store in its context's table, so a destroy never comes between.
//! - `id_chunks`: the texture resource ids' chunks, on ring 0.
//! - `surface_views`, then `surface_caches` ([`surface_cache::Caches`]), the blocks' views of
//!   their surfaces' pages and their device caches. A command buffer that uses a cache holds the
//!   caches' lock for its end's signal, never across its walk or its commit, and counts among the
//!   caches' users until its commit returns; a download with nothing to copy is answered under
//!   it. A view's mapping and import are made with
//!   neither held ([`surface_pages::SurfaceViews::surface_pages`]).
//! - `unsettled` ([`worker::Unsettled`]), the blocks whose `SURFACE_PAGES` the surface
//!   worker has yet to carry out, taken alone; a message that names one waits for it holding no
//!   lock here, nor the reset gate alone (the worker takes the gate shared, recursively).
//! - the commit order ([`sequencer`]), and the contexts closed ([`ClosedContexts`]).
//! - `surface_order` ([`surface_order::SurfaceOrder`]), the queues' surface timelines: a leaf,
//!   taken under no other lock and holding none, by a command buffer that holds `SURFACE_ORDER`
//!   (or is waited for) while its waits and its signal are encoded, and at a queue's make and end.
//!
//! Where two are held at once, they are taken in this order: `libraries`, `contexts`, a context's
//! table, then the compiles' pending set (`super::compile`, asked under the table's lock as a
//! compiled pipeline is stored); and `surface_views` before `surface_caches`.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashMap};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, Weak};

use reims_graphics::region::Region;
use reims_graphics::wire::{
    RGFX_CMD_BUFFER_AT_OFFSET, RGFX_CMD_BUFFER_CREATE, RGFX_CMD_BUFFER_DID_MODIFY_RANGE,
    RGFX_CMD_BUFFER_MANAGED_AT_OFFSET, RGFX_CMD_BUFFER_OVER_HOST_MEMORY,
    RGFX_CMD_BUFFER_OVER_PAGES, RGFX_CMD_BUFFER_OVER_SURFACE_PAGES, RGFX_CMD_COMMAND_BUFFER,
    RGFX_CMD_COMPUTE_PIPELINE_CREATE, RGFX_CMD_DEPTH_STENCIL_CREATE, RGFX_CMD_EVENT_CREATE,
    RGFX_CMD_FENCE_CREATE, RGFX_CMD_HEAP_CREATE, RGFX_CMD_HEAP_OVER_HOST_MEMORY,
    RGFX_CMD_HEAP_OVER_PAGES, RGFX_CMD_ICB_CREATE, RGFX_CMD_ICB_SLOTS, RGFX_CMD_ICB_WRITE,
    RGFX_CMD_LIBRARY_CREATE, RGFX_CMD_LIBRARY_FROM_REGION, RGFX_CMD_LINEAR_TEXTURE_FROM_BUFFER,
    RGFX_CMD_MANAGED_BUFFER_CREATE, RGFX_CMD_MANAGED_TEXTURE_CREATE,
    RGFX_CMD_MESH_RENDER_PIPELINE_CREATE, RGFX_CMD_OBJECT_DESTROY, RGFX_CMD_QUEUE_CREATE,
    RGFX_CMD_RENDER_PIPELINE_CREATE, RGFX_CMD_RESOURCE_ID_CHUNK, RGFX_CMD_RESOURCE_ID_SMALL_CHUNK,
    RGFX_CMD_SAMPLER_CREATE, RGFX_CMD_SHARED_EVENT_CREATE, RGFX_CMD_SHARED_EVENT_SET_VALUE,
    RGFX_CMD_TEXTURE_CREATE, RGFX_CMD_TEXTURE_FROM_BUFFER, RGFX_CMD_TEXTURE_FROM_IOSURFACE,
    RGFX_CMD_TEXTURE_REPLACE_REGION, RGFX_CMD_TEXTURE_VIEW, RGFX_CMD_VIDEO_SESSION_CREATE,
    RGFX_COMPUTE_PIPELINE_SUPPORT_ICB, RGFX_COMPUTE_PIPELINE_TG_MULTIPLE_OF_EXEC_WIDTH,
    RGFX_CONTEXT_ALLOCATIONS_OFFSET, RGFX_CONTEXT_OBJ_ID_SHIFT, RGFX_CURSOR_SPAN,
    RGFX_EVT_PIPELINE_REFLECTION, RGFX_EVT_SAMPLER_STATE, RGFX_EVT_SIZE_AND_ALIGN,
    RGFX_LINKED_BINARY_FUNCTIONS, RGFX_LINKED_FUNCTIONS, RGFX_LINKED_GROUP,
    RGFX_LINKED_PRIVATE_FUNCTIONS, RGFX_MESH_PIPELINE_MESH_TG_MULTIPLE_OF_EXEC_WIDTH,
    RGFX_MESH_PIPELINE_OBJECT_TG_MULTIPLE_OF_EXEC_WIDTH, RGFX_PIPELINE_KIND_COMPUTE,
    RGFX_PIPELINE_KIND_MESH, RGFX_PIPELINE_KIND_RENDER, RGFX_PIPELINE_OK,
    RGFX_PIPELINE_REFLECT_SUPPORT_ICB, RGFX_PIPELINE_REFLECT_THREADGROUP_MATCHES_TILE,
    RGFX_PIPELINE_REFUSED, RGFX_RENDER_PIPELINE_ALPHA_TO_COVERAGE,
    RGFX_RENDER_PIPELINE_ALPHA_TO_ONE, RGFX_RENDER_PIPELINE_RASTERIZATION_ENABLED,
    RGFX_RENDER_PIPELINE_SUPPORT_ICB, RGFX_RESOURCE_ID_CHUNK, RGFX_RESOURCE_ID_LIMIT,
    RGFX_RESOURCE_ID_SMALL_CHUNK, RGFX_RESOURCE_ID_SMALL_CHUNKS, RGFX_STATUS_INVALID_ARGUMENT,
    RGFX_STATUS_OK, RGFX_STATUS_REFUSED, RGFX_SURFACE_FACTS_OFFSET, RGFX_SURFACE_NO_WRITERS,
    RGFX_SURFACE_PAGES_ATTACH, RGFX_SURFACE_PAGES_CACHED, RGFX_SURFACE_PAGES_DEFERRED,
    RGFX_SURFACE_PAGES_PRIVATE, RGFX_YCBCR_MATRIX_DCI_P3, RGFX_YCBCR_MATRIX_ITU_R_601_4,
    RGFX_YCBCR_MATRIX_ITU_R_709_2, RGFX_YCBCR_MATRIX_ITU_R_2020, RGFX_YCBCR_MATRIX_NONE,
    RGFX_YCBCR_MATRIX_OTHER, RGFX_YCBCR_MATRIX_P3_D65, RGFX_YCBCR_MATRIX_SMPTE_240M_1995,
    RgfxBufferAtOffset, RgfxBufferCreate, RgfxBufferOverHostMemory, RgfxBufferOverPages,
    RgfxCbComplete, RgfxCbScheduled, RgfxCbSubmitted, RgfxColorAttachmentDesc,
    RgfxComputePipelineCreate, RgfxDepthStencilCreate, RgfxEventSignalled, RgfxFunctionConstant,
    RgfxHeapCreate, RgfxHeapOverHostMemory, RgfxHeapTextureSizeAndAlign,
    RgfxLinearTextureFromBuffer, RgfxLinkedFunction, RgfxMeshPipelineReflection,
    RgfxMeshRenderPipelineCreate, RgfxMsgHeader, RgfxObjectCreate, RgfxObjectDestroy, RgfxPageRun,
    RgfxPipelineReflection, RgfxQueueCreate, RgfxRenderPipelineCreate, RgfxResourceIdChunk,
    RgfxSamplerCreate, RgfxSamplerState, RgfxSharedEventCreate, RgfxSharedEventSetValue,
    RgfxSizeAndAlign, RgfxStencilDescriptor, RgfxSurfaceFacts, RgfxSurfacePages, RgfxSurfaceWriter,
    RgfxSurfaceWriters, RgfxTextureCreate, RgfxTextureDescriptor, RgfxTextureFromBuffer,
    RgfxTextureFromIOSurface, RgfxVertexAttribute, RgfxVertexBufferLayout, RgfxVideoSessionCreate,
};
use reims_vgpu_core::host::HostPtr;
use reims_vgpu_core::memory::map::MemoryMap;
use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
use reims_vgpu_core::platform::device::GpuDevice;
use reims_vgpu_core::platform::event::{EventRef, SharedEvent};
use reims_vgpu_core::platform::os::clock::{Clock as _, MonoTime};
use reims_vgpu_core::platform::resource::{
    Buffer, ComputePipeline, FunctionConstant, Heap, Library, RenderPipeline, SamplerState,
};
use reims_vgpu_core::platform::types::{MtlSize, Swizzle};
use reims_vgpu_core::platform::values::{
    AttributeDesc, BufferDesc, BufferLayoutDesc, ColorAttachmentDesc, DepthStencilDesc,
    FullComputePipelineDesc, FullMeshRenderPipelineDesc, FullRenderPipelineDesc, HeapDesc,
    LinkedFunctionsDesc, OptionSet, SamplerDesc, StageInputDesc, StencilDesc, SurfaceFacts,
    SurfaceMatrix, SurfacePlane, TextureDesc, VertexDesc,
};
use reims_vgpu_core::platform::{Handle, HostRange, Platform};
use reims_vgpu_core::raise::{OnFault, Raise, RaiseResult, refusals};
use reims_vgpu_core::sync::Mutex;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// Metal's page on Apple silicon, to which a Shared heap's memory is rounded up: the guest's
/// block holds it whole ([`RGFX_CMD_HEAP_OVER_HOST_MEMORY`]).
const HEAP_PAGE: u64 = 0x4000;

mod argument;
mod blit;
#[cfg(test)]
mod close_cost;
mod cmdbuf;
mod commit_order;
#[cfg(test)]
mod composite_cost;
mod compute;
#[cfg(test)]
mod draw_cost;
#[cfg(test)]
mod fuzz;
#[cfg(test)]
mod handler_cost;
mod icb;
#[cfg(test)]
mod icb_slots_cost;
mod lane;
mod library;
mod managed;
mod refused;
mod render;
mod sequencer;
mod split;
mod surface_cache;
mod surface_order;
mod surface_pages;
mod tables;
mod video;
mod view;
mod waits;
mod work;
pub(crate) mod worker;

pub use video::{GraphicsVideo, GraphicsVideoSession, VideoDecode, VideoPlane};

pub(crate) use cmdbuf::Begun;
use lane::{Lane, Turn};
#[cfg(test)]
pub(crate) use library::tests::{metallib as test_metallib, sha as test_sha};
#[cfg(test)]
pub(crate) use library::verified_key as test_key;
use library::{Libraries, LibraryRef};
pub(crate) use library::{LibraryJob, LibraryStart, MadeLibrary, complete, make_library};
use managed::{STORAGE_MODE_MANAGED, storage_mode};
use sequencer::Sequencer;
pub(crate) use sequencer::Ticket;
use surface_cache::Caches;
pub(crate) use surface_cache::{FillHook, FillWait};
use tables::{Context, Entry, GuestBytes, QueueCounts, View};

/// A sink the decoder calls with each command buffer's completion record once its
/// work finishes (the handler the stream's close registers). The ring loop sets
/// one that writes the record to the queue's completion ring; until it is set, a
/// completion is built and dropped.
pub(crate) type CompletionSink = Arc<dyn Fn(RgfxCbComplete) + Send + Sync>;

/// A sink the decoder calls with each shared event a command buffer signalled
/// (`encodeSignalEvent:value:`), once the buffer's work finishes and before its completion record
/// goes to the [`CompletionSink`]. The ring loop sets one that writes it to the queue's completion
/// ring; until it is set, it is built and dropped.
pub(crate) type SignalSink = Arc<dyn Fn(RgfxEventSignalled) + Send + Sync>;

/// A sink the decoder calls with each command buffer's `CB_SCHEDULED` record once the buffer is
/// scheduled (its platform command buffer's scheduled handler), ahead of its completion record.
/// The ring loop sets one that writes it to the queue's completion ring; until it is set, a
/// command buffer gets no scheduled handler.
pub(crate) type ScheduledSink = Arc<dyn Fn(RgfxCbScheduled) + Send + Sync>;

/// A sink the decoder calls with the `CB_SUBMITTED` record of each command buffer that asks for it
/// (`RGFX_OP_SUBMITTED`) once the buffer is submitted, ahead of its scheduled and completion
/// records. The ring loop sets one that writes it to the queue's completion ring; until it is set,
/// no buffer's ask is answered.
pub(crate) type SubmittedSink = Arc<dyn Fn(RgfxCbSubmitted) + Send + Sync>;

/// A sink the decoder calls with a pipeline create's reflection reply once the
/// pipeline is made (or refused). The ring loop sets one that writes the reply to
/// the queue's completion ring, keyed to the create by its `seqno`; until it is
/// set, the reply is built and dropped.
pub(crate) type ReflectionSink = Arc<dyn Fn(PipelineReply) + Send + Sync>;

/// A `PIPELINE_REFLECTION` reply: its record, and, after it, a made mesh render pipeline's
/// limits ([`RgfxMeshPipelineReflection`], at [`RGFX_PIPELINE_REFLECTION_MESH_OFFSET`]), which
/// no other reply carries.
///
/// [`RGFX_PIPELINE_REFLECTION_MESH_OFFSET`]: reims_graphics::wire::RGFX_PIPELINE_REFLECTION_MESH_OFFSET
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PipelineReply {
    /// The record.
    pub(crate) reflection: RgfxPipelineReflection,
    /// A made mesh render pipeline's limits.
    pub(crate) mesh: Option<RgfxMeshPipelineReflection>,
}

impl PipelineReply {
    /// A reply of `reflection` alone.
    pub(crate) fn of(reflection: RgfxPipelineReflection) -> Self {
        PipelineReply {
            reflection,
            mesh: None,
        }
    }

    /// The reply's bytes: the record, its header's size counting the mesh limits after it where
    /// it carries them.
    pub(crate) fn bytes(&self) -> Vec<u8> {
        let mut reflection = self.reflection;
        let mut out = Vec::with_capacity(
            size_of::<RgfxPipelineReflection>() + size_of::<RgfxMeshPipelineReflection>(),
        );
        if let Some(mesh) = &self.mesh {
            reflection.header.size = (size_of::<RgfxPipelineReflection>()
                + size_of::<RgfxMeshPipelineReflection>())
                as u32;
            out.extend_from_slice(reflection.as_bytes());
            out.extend_from_slice(mesh.as_bytes());
        } else {
            out.extend_from_slice(reflection.as_bytes());
        }
        out
    }
}

/// The sinks a message's records go to: those of the ring it came from. A sink not set builds and
/// drops its records, and a command buffer gets no scheduled handler or `CB_SUBMITTED` without
/// one.
#[derive(Clone, Default)]
pub(crate) struct Sinks {
    pub(crate) completion: Option<CompletionSink>,
    pub(crate) signal: Option<SignalSink>,
    pub(crate) scheduled: Option<ScheduledSink>,
    pub(crate) submitted: Option<SubmittedSink>,
    pub(crate) reflection: Option<ReflectionSink>,
}

/// An answer the host gives once a context's command buffers have finished (a `CONTEXT_CLOSE`'s,
/// a `BLOCK_RELEASE`'s).
pub(crate) type DrainAnswer = Box<dyn FnOnce() + Send>;

/// The command buffers one context committed since its last drain began: each holds a clone
/// until its completion handler has run, and the answers waiting on them are given when the last
/// clone goes ([`GraphicsDecoder::drain_context`]). A drain holds the one begun after it, so a
/// later drain ends only after every earlier one: its answers wait for every buffer committed
/// before it, those of the drains before it too.
#[derive(Default)]
pub(crate) struct Drain {
    /// Given when the drain drops.
    answers: Mutex<Vec<DrainAnswer>>,
    /// The drain begun after this one, which ends no sooner.
    later: Mutex<Option<Arc<Drain>>>,
}

impl Drop for Drain {
    fn drop(&mut self) {
        for answer in self.answers.get_mut().drain(..) {
            answer();
        }
        // The later drains this one held, each ended here where this was its last hold, in turn
        // rather than by recursion: a buffer that runs long may hold up many.
        let mut later = self.later.get_mut().take();
        while let Some(next) = later {
            let Some(mut next) = Arc::into_inner(next) else {
                break;
            };
            for answer in next.answers.get_mut().drain(..) {
                answer();
            }
            later = next.later.get_mut().take();
        }
    }
}

/// The contexts closed since the device's reset, and the resets so far. What comes to the
/// decoder for a closed context makes it no state (a command buffer taken off its ring before
/// the close and handed over after it, a release the kext posted behind the close), but a create
/// of its own, as of a process given its number again, which opens it again ([`Self::reopen`]).
/// The resets tell an answer given on a doorbell's thread before a reset from one after it
/// ([`cmdbuf::Answered`]): the rebooted guest's processes take the same context numbers again,
/// from 1.
///
/// No context is failed for a failure of its work: a command buffer whose completion the
/// platform refused on its own thread ([`CompletedInfo::refused`]), where the paravirt device ends
/// the process, fails alone, as a buffer Metal could not run does, and its process's later work
/// is carried out (docs/graphics-kext/design.md, "Guest input the host refuses").
///
/// [`CompletedInfo::refused`]: reims_vgpu_core::platform::command::CompletedInfo::refused
#[derive(Default)]
pub(crate) struct ClosedContexts {
    /// The contexts closed since the device's reset.
    closed: Mutex<Runs>,
    /// The device's resets so far ([`Self::forget_all`]), changed under `closed`'s lock.
    resets: AtomicU64,
}

impl ClosedContexts {
    /// The device's resets so far.
    pub(crate) fn resets(&self) -> u64 {
        self.resets.load(Ordering::Acquire)
    }

    /// `context` closed (its `CONTEXT_CLOSE`): kept closed until the device's reset.
    fn close(&self, context: u32) {
        self.closed.lock().insert(context);
    }

    /// Whether `context` closed since the device's reset, and was not opened again since.
    fn closed(&self, context: u32) -> bool {
        self.closed.lock().contains(context)
    }

    /// `context`, closed before, is opened again: a create of its own came (a process given its
    /// number again).
    fn reopen(&self, context: u32) {
        self.closed.lock().remove(context);
    }

    /// Forgets every close at the device's reset, and counts the reset.
    fn forget_all(&self) {
        let mut closed = self.closed.lock();
        closed.clear();
        self.resets.fetch_add(1, Ordering::AcqRel);
    }
}

/// A set of context numbers, kept as the runs of consecutive numbers it holds, each by its first
/// and its last: the kext numbers contexts in the order of their opens, and the closed ones then
/// make about one run for each context still open between them.
#[derive(Debug, Default)]
struct Runs(BTreeMap<u32, u32>);

impl Runs {
    /// Whether `n` is held.
    fn contains(&self, n: u32) -> bool {
        (self.0.range(..=n).next_back()).is_some_and(|(_, &last)| n <= last)
    }

    /// Holds `n`, joining it to the runs it is next to.
    fn insert(&mut self, n: u32) {
        if self.contains(n) {
            return;
        }
        let before = (self.0.range(..n).next_back())
            .filter(|&(_, &last)| last.checked_add(1) == Some(n))
            .map(|(&first, _)| first);
        let after = n.checked_add(1).and_then(|next| self.0.remove(&next));
        let first = before.unwrap_or(n);
        self.0.insert(first, after.unwrap_or(n));
    }

    /// Holds `n` no longer, splitting its run.
    fn remove(&mut self, n: u32) {
        let Some((&first, &last)) = self.0.range(..=n).next_back() else {
            return;
        };
        if n > last {
            return;
        }
        self.0.remove(&first);
        if first < n {
            self.0.insert(first, n - 1);
        }
        if n < last {
            self.0.insert(n + 1, last);
        }
    }

    /// Holds nothing.
    fn clear(&mut self) {
        self.0.clear();
    }
}

/// A `SAMPLER_CREATE`'s answer (`SAMPLER_STATE`): the sampler made, handed to `keep` with the
/// message's id, and its `gpuResourceID`, the id the platform numbers it by, which the guest's
/// sampler answers and its argument buffers hold; [`RGFX_STATUS_REFUSED`] with 0 where the
/// platform makes none, and [`RGFX_STATUS_INVALID_ARGUMENT`] for a short create. `None` for a
/// message too short to carry its seqno, which no one can be answered by.
fn sampler_answer<P: Platform>(
    device: &P::Device,
    message: &[u8],
    keep: impl FnOnce(u64, P::SamplerState),
) -> RaiseResult<Option<RgfxSamplerState>> {
    let Some(header) = read::<RgfxMsgHeader>(message) else {
        return Ok(None);
    };
    let answer = |status, gpu_resource_id| RgfxSamplerState {
        header: RgfxMsgHeader {
            r#type: RGFX_EVT_SAMPLER_STATE,
            size: size_of::<RgfxSamplerState>() as u32,
            seqno: header.seqno,
        },
        status,
        reserved_14: 0,
        gpu_resource_id,
    };
    Ok(Some(match new_sampler::<P>(device, message)? {
        Ok((obj_id, sampler)) => {
            let id = sampler.gpu_resource_id();
            keep(obj_id, sampler);
            answer(RGFX_STATUS_OK, id)
        }
        Err(DecodeError::Malformed) => answer(RGFX_STATUS_INVALID_ARGUMENT, 0),
        Err(_) => answer(RGFX_STATUS_REFUSED, 0),
    }))
}

/// The sampler a `SAMPLER_CREATE` describes, made on `device`, with the id it is stored under.
fn new_sampler<P: Platform>(
    device: &P::Device,
    message: &[u8],
) -> RaiseResult<Result<(u64, P::SamplerState), DecodeError>> {
    let Some(m) = read::<RgfxSamplerCreate>(message) else {
        return Ok(Err(DecodeError::Malformed));
    };
    // Our wire sends the whole descriptor, so every field is set;
    // the LOD clamps travel as `f32` bits, and `flags` packs the
    // booleans. `reduction_mode` is not on the wire yet (the Metal
    // default stands).
    let desc = SamplerDesc {
        min_filter: Some(m.min_filter.into()),
        mag_filter: Some(m.mag_filter.into()),
        mip_filter: Some(m.mip_filter.into()),
        s_address_mode: Some(m.s_address_mode.into()),
        t_address_mode: Some(m.t_address_mode.into()),
        r_address_mode: Some(m.r_address_mode.into()),
        border_color: Some(m.border_color.into()),
        compare_function: Some(m.compare_function.into()),
        max_anisotropy: Some(m.max_anisotropy.into()),
        lod_min_clamp: Some(f32::from_bits(m.lod_min_clamp_bits)),
        lod_max_clamp: Some(f32::from_bits(m.lod_max_clamp_bits)),
        normalized_coordinates: Some((m.flags & 1) != 0),
        lod_average: Some((m.flags & 2) != 0),
        support_argument_buffers: Some((m.flags & 4) != 0),
        reduction_mode: None,
        resource_index: None,
    };
    Ok(match device.new_sampler_state(&desc)? {
        Some(sampler) => Ok((m.obj_id, sampler)),
        None => Err(DecodeError::Refused),
    })
}

/// A host handle for one guest object, by kind. Later increments read these to
/// bind resources into command buffers; this slice only inserts them on a create
/// and drops them on a destroy.
enum ObjHandle<P: Platform> {
    Event(P::Event),
    Fence(P::Fence),
    SharedEvent(P::SharedEvent),
    CommandQueue(P::CommandQueue),
    Library(P::Library),
    Sampler(P::SamplerState),
    DepthStencil(P::DepthStencilState),
    Buffer(P::Buffer),
    Texture(P::Texture),
    Heap(P::Heap),
    RenderPipeline(P::RenderPipeline),
    ComputePipeline(P::ComputePipeline),
    /// An indirect command buffer ([`RGFX_CMD_ICB_CREATE`]), which `ICB_WRITE`s encode into and
    /// command buffers execute, copy, optimize and reset.
    Icb(P::Icb),
    /// A video session ([`RGFX_CMD_VIDEO_SESSION_CREATE`]), which its command buffers' video
    /// encoders name, let go of on the video worker ([`video::SessionHold`]).
    VideoSession(video::SessionHold<P>),
    /// What an id stands for whose posted create the host refused ([`refused`]): what names it
    /// is dropped, not the command buffer that names it.
    Refused(refused::Refused),
}

/// Why a message was not carried out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// The message was shorter than its command, or unreadable.
    Malformed,
    /// The device has no command of that type.
    UnknownCommand,
    /// The platform made no object.
    Refused,
    /// The message named an object the host does not hold, or of the wrong kind.
    BadReference,
    /// A texture create's resource id lies in no chunk its context holds
    /// (`RESOURCE_ID_CHUNK`, `RESOURCE_ID_SMALL_CHUNK`): 0, at or past
    /// [`RGFX_RESOURCE_ID_LIMIT`], or another context's. The platform is not called, so no id the host's Metal ends the
    /// process for reaches it.
    BadResourceId,
    /// The command buffer's context was closed before the buffer came to the decoder (taken off
    /// its queue's ring before the close, handed over after it): nothing of it is carried out,
    /// and nothing is owed, its queue having ended at the close.
    ContextClosed,
}

/// How many `SURFACE_PAGES` pass between the logs of what the surface views did, at most
/// ([`views_log_due`]).
const VIEWS_LOGGED_EVERY: u64 = 256;
/// The longest the logs of what the surface views did wait for [`VIEWS_LOGGED_EVERY`]
/// `SURFACE_PAGES`: the first after it logs them too. Where dormant blocks keep most device
/// caches, a desktop makes a few new ones a second, and 256 `SURFACE_PAGES` took minutes, so the
/// lines bounded the caches' rates and never gave them. The kext logs its dormant blocks on the
/// same interval.
const VIEWS_LOGGED_AFTER: std::time::Duration = std::time::Duration::from_secs(10);

/// Whether a `SURFACE_PAGES`, the `counted`th the views' counts hold, logs what the surface
/// views did at `now`, the last log having been at `last`: every [`VIEWS_LOGGED_EVERY`]th, and
/// the first [`VIEWS_LOGGED_AFTER`] or more after the last log.
fn views_log_due(counted: u64, now: MonoTime, last: MonoTime) -> bool {
    counted.is_multiple_of(VIEWS_LOGGED_EVERY)
        || now.saturating_duration_since(last) >= VIEWS_LOGGED_AFTER
}

/// What a present of an IOSurface shows ([`GraphicsDecoder::surface_present`]).
pub(crate) struct SurfacePresent<P: Platform> {
    /// The latest texture a guest process made over the surface's plane 0 with
    /// `TEXTURE_FROM_IOSURFACE` and has not destroyed.
    pub(crate) texture: P::Texture,
    /// The region block it lies in (its buffer's): its offset and size.
    pub(crate) span: Option<(u64, u64)>,
    /// The command buffers committed that write the texture's copy, which the present waits for:
    /// the slots of its [`RgfxSurfaceWriters`] that name a queue the decoder holds (a slot naming
    /// another, or none, has nothing to wait for).
    pub(crate) writers: Vec<RgfxSurfaceWriter>,
    /// The context that made the texture.
    pub(crate) owner: u32,
}

/// The context an object id lies in.
fn context_of(obj_id: u64) -> u32 {
    (obj_id >> RGFX_CONTEXT_OBJ_ID_SHIFT) as u32
}

/// What the decoder keeps for every context, behind locks of its own (the module doc's list).
pub(crate) struct Shared<P: Platform> {
    /// The platform whose device the decoder drives, held for the descriptor builders
    /// ([`Platform::texture_descriptor`]) some per-device creates need.
    platform: Arc<P>,
    /// Each context's state, by id, made at its first create and dropped at its close.
    contexts: Mutex<HashMap<u32, Arc<Context<P>>>>,
    /// The last drains of the contexts closed (`CONTEXT_CLOSE`), by context, while their command
    /// buffers may still run: a later drain of such a context, a release the kext posted behind
    /// its close, ends after it ([`GraphicsDecoder::drain_context`]). Taken under `contexts`'
    /// lock.
    closing: Mutex<HashMap<u32, Weak<Drain>>>,
    /// The shared blocks a holder let go of (`BLOCK_UNSHARE`), by offset, until their release:
    /// each one's drain ends once every command buffer such a holder began before its unshare has
    /// finished, and gives the release's answer ([`GraphicsDecoder::block_unshared`]).
    unshared: Mutex<HashMap<u64, Arc<Drain>>>,
    /// The IOSurfaces' names.
    surfaces: Mutex<SurfaceNames>,
    /// The host's hardware video, which makes the video sessions; unset where it has none.
    video: OnceLock<Arc<dyn GraphicsVideo<P>>>,
    /// The libraries' chunks, makes and cache (`library`).
    libraries: Mutex<Libraries<P>>,
    /// The context holding each small chunk of the texture resource ids
    /// ([`RGFX_RESOURCE_ID_SMALL_CHUNK`] ids), by small chunk, as `RESOURCE_ID_SMALL_CHUNK` gave
    /// it, or `RESOURCE_ID_CHUNK` the four of its chunk, until its `CONTEXT_CLOSE`: every
    /// process's textures share the one device's ids, so a texture create's id must lie in a
    /// chunk of its own context ([`GraphicsDecoder::holds_resource_id`]).
    id_chunks: Mutex<Box<[Option<u32>]>>,
    /// The watch over command buffers that wait on events, which logs the ones that wait long
    /// (diagnostics only: [`waits`]).
    waits: waits::WaitWatch<P>,
    /// The contexts closed since the device's reset, and the resets ([`ClosedContexts`]).
    closed: ClosedContexts,
    /// Whether a `Managed` resource is one copy, over its block
    /// ([`GraphicsDecoder::set_unified_memory`]).
    unified_memory: AtomicBool,
    /// The views of the IOSurfaces whose blocks name their own guest pages ([`surface_pages`]).
    surface_views: Mutex<surface_pages::SurfaceViews<P>>,
    /// When what the surface views did was last logged, or the decoder made, in the platform
    /// clock's nanoseconds ([`views_log_due`]).
    views_logged_at: AtomicU64,
    /// The blocks whose `SURFACE_PAGES` is not yet carried out ([`worker::Unsettled`]).
    unsettled: Arc<worker::Unsettled>,
    /// The threads the views are made and let go of on, a block's jobs in their order
    /// ([`worker::Worker`]), and a block's device cache at its release ([`Caches::release`]).
    surface_worker: Arc<worker::Worker>,
    /// The thread a closed context's objects and views are let go of on, its close answered
    /// after them ([`GraphicsDecoder::let_go_later`]).
    close_worker: worker::Worker,
    /// The threads video sessions are let go of on, each session's holds in their order
    /// ([`video::SessionHold`]).
    video_worker: Arc<worker::Worker>,
    /// The device caches of the blocks the kext marked cached ([`surface_cache`]).
    surface_caches: Arc<Caches<P>>,
    /// The commit order `CB_SUBMITTED` promises ([`sequencer`]).
    sequencer: Arc<Sequencer>,
    /// The contexts with a command buffer whose parts are out, which hold the contexts' other
    /// commits and range messages until its last commit ([`split`]).
    split_gate: Arc<split::SplitGate>,
    /// The queues' surface timelines, which order command buffers of different queues over the
    /// IOSurfaces they share ([`surface_order`]).
    surface_order: Mutex<surface_order::SurfaceOrder<P>>,
    /// The device's resets, which end every wait for another queue's buffers
    /// ([`commit_order`]).
    commit_cancel: commit_order::Cancel,
    /// The decoder's own queue, which signals a gone context's plain events past every value
    /// ([`GraphicsDecoder::release_events`]): made at the first such signal, unshared, so no
    /// command buffer of the guest's, nor one waiting on another's GPU queue, holds it up; `None`
    /// where the platform made none.
    release_queue: OnceLock<Option<P::CommandQueue>>,
    /// The budget of the queues' lanes' extra threads ([`lane`]).
    lanes: Arc<lane::Budget>,
    /// The listener the command buffers' notifications of the values they signal run on, made at
    /// the first ([`Shared::listener`]).
    listener: OnceLock<P::SharedEventListener>,
    /// Called after each op of every command buffer's walk (the tests' look inside a walk).
    #[cfg(test)]
    walk_hook: Mutex<Option<WalkHook>>,
    /// The tests' own [`split::SPLIT_AFTER`], where one is set ([`GraphicsDecoder::split_after`]).
    #[cfg(test)]
    split_after: Mutex<Option<std::time::Duration>>,
    /// The device's BAR 2 region: the host memory the over-host-memory buffers
    /// and heaps are placed over. Held so it stays mapped while the decoder's
    /// objects live; declared last so the contexts drop before it (a buffer
    /// also keeps its own clone alive in its no-copy deallocator).
    region: Arc<Region>,
}

impl<P: Platform> Shared<P> {
    /// The device's one listener, which the notifications of the values command buffers signal
    /// their shared events to run on, one after another (`cmdbuf.rs`, `Signals`), made at the
    /// first: Metal's runs them on its dispatch queue, the Vulkan platform's on the thread of the
    /// process's one runtime listener.
    pub(crate) fn listener(&self) -> &P::SharedEventListener {
        self.listener
            .get_or_init(|| self.platform.new_shared_event_listener())
    }
}

/// The decoder's state, shared by the threads that carry out messages, each on a decoder of its
/// own ([`Self::sender`]).
#[derive(Clone)]
pub(crate) struct Decoders<P: Platform>(Arc<Shared<P>>);

impl<P: Platform> Decoders<P> {
    /// A decoder for a message from `context` (`None`: what is no context's), its records going
    /// to `sinks`.
    pub(crate) fn sender(&self, context: Option<u32>, sinks: Sinks) -> GraphicsDecoder<P> {
        GraphicsDecoder::on(Arc::clone(&self.0), context, sinks)
    }

    /// A reset of the device is under way: every wait for another queue's buffers ends, and none
    /// waits until [`Self::reset_done`], so none holds up the reset, which waits for the work begun
    /// before it ([`commit_order`]).
    pub(crate) fn resetting(&self) {
        self.0.commit_cancel.begin();
    }

    /// The reset [`Self::resetting`] told of is done.
    pub(crate) fn reset_done(&self) {
        self.0.commit_cancel.end();
    }
}

/// The IOSurfaces' names: the textures over each one's plane 0, and where each copy keeps its
/// writers.
#[derive(Default)]
struct SurfaceNames {
    /// The textures over each IOSurface's plane 0, by the surface's global id:
    /// every `TEXTURE_FROM_IOSURFACE` naming the surface whose texture the table
    /// holds, in the order made, so a destroy of one leaves the others (another
    /// process's, or an older one's) to name the surface. A present names its
    /// surface by that id and shows the latest.
    by_surface: HashMap<u32, Vec<u64>>,
    /// Where in the region each of those textures' copy keeps its
    /// [`RgfxSurfaceWriters`], by the texture's id, where its message gave one.
    writers: HashMap<u64, u64>,
}

/// The stage-3 decoder: a handle on the device's [`Shared`] state through which one message is
/// carried out, with the message's sender and the sinks its records go to. It holds what the
/// message named (its views of its contexts' tables, [`tables::View`]) until it drops, or until
/// a new message begins on it ([`Self::decode`]) or it ends ([`Self::end_message`]), and a command
/// buffer's place in the commit order ([`sequencer`]).
pub struct GraphicsDecoder<P: Platform> {
    shared: Arc<Shared<P>>,
    /// The platform, as [`Shared`]'s.
    platform: Arc<P>,
    /// The region, as [`Shared`]'s.
    region: Arc<Region>,
    /// The device caches, as [`Shared`]'s.
    surface_caches: Arc<Caches<P>>,
    /// Where the message's records go.
    sinks: Sinks,
    /// The context (the process) of the message being carried out, whose objects alone it may
    /// name ([`Self::named`]); `None` names any (the decoder's own tests).
    context: Option<u32>,
    /// The message's views of the tables it named objects in, one a context.
    views: RefCell<Vec<View<P>>>,
    /// The commit order's mark at the command buffer's begin ([`Sequencer::mark`]).
    mark: Cell<Option<u64>>,
    /// The range messages of its context that came before the command buffer's begin, which it is
    /// committed after, where any was still to be carried out then ([`Sequencer::ranges_asked`]);
    /// `None` until its begin.
    ranges: Cell<Option<Option<u64>>>,
    /// The command buffer's ticket in the commit order, where it was answered at its take; given
    /// back with the handle, after its commit or its stand-in's.
    ticket: RefCell<Option<Ticket>>,
    /// The command buffer's turn in its queue's lane ([`lane`]), from its begin; it passes with
    /// the handle, once the buffer's commit, stand-in or failure is done.
    turn: RefCell<Option<Turn>>,
    /// The command buffer's `CB_SUBMITTED` was posted on a doorbell's thread before its begin
    /// ([`cmdbuf::Answered`]).
    posted_at_doorbell: Cell<bool>,
    /// The drain of the message's context, taken at its command buffer's begin
    /// ([`Self::command_buffer_drain`]).
    drain: RefCell<Option<Arc<Drain>>>,
    /// The waits its command buffer makes for buffers of other queues, taken at its begin, for
    /// the work the host starts off it ([`Self::wait_for_order_on_cpu`]).
    order_waits: RefCell<Vec<surface_order::OrderWait<P>>>,
    /// The queues of its process its command buffer is committed after, each with the seqno its
    /// `COMMIT_ORDER` named, taken at its begin ([`commit_order`]).
    commit_after: RefCell<Vec<(Arc<surface_order::Marks>, u64)>>,
    /// Its command buffer's queue's marks and its seqno, from its begin: the queue's begun mark
    /// rises to the seqno once it takes its place among the tickets, and its commit mark as the
    /// message ends, before its turn passes ([`commit_order`]).
    committed: RefCell<Option<(Arc<surface_order::Marks>, u64)>>,
    /// The entries the message found last ([`Recent`]).
    recent: Recent<P>,
}

impl<P: Platform> Drop for GraphicsDecoder<P> {
    /// As [`GraphicsDecoder::end_message`] does: its command buffer's queue's commit mark rises
    /// before its turn passes, which a lane's job takes out of the handle first and lets go of
    /// after ([`commit_order`]).
    fn drop(&mut self) {
        self.note_committed();
    }
}

/// How many entries [`Recent`] keeps.
const RECENT: usize = 8;

/// The entries a message found last, each by its id, at a slot its id picks: a lookup of an id
/// the message found before finds its entry here, with no look into its view's map. A draw's
/// binds name the same few objects again and again (each of `draws.10k`'s 10,000 draws binds its
/// upload chunk, 256 draws a chunk), and the view's lookup was most of what naming one cost.
/// What it keeps is what the view answered, which it answers again until the message makes or
/// removes that id itself ([`GraphicsDecoder::saw`]) or ends ([`GraphicsDecoder::end_message`]),
/// which empty it; the context it may name is the message's, fixed until it ends.
struct Recent<P: Platform>([RecentSlot<P>; RECENT]);

/// A slot of [`Recent`]: an id and the entry found for it.
type RecentSlot<P> = Cell<Option<(u64, NonNull<Entry<P>>)>>;

// SAFETY: as a view's ([`View`]), which lends the entries it points to: used on one thread at a
// time (it is not `Sync`); moved to another, it moves its pointers, to entries that are
// `Send + Sync`.
unsafe impl<P: Platform> Send for Recent<P> where Entry<P>: Send + Sync {}

impl<P: Platform> Recent<P> {
    fn new() -> Recent<P> {
        Recent(std::array::from_fn(|_| Cell::new(None)))
    }

    /// `obj_id`'s slot.
    fn slot(&self, obj_id: u64) -> &RecentSlot<P> {
        &self.0[obj_id as usize % RECENT]
    }

    /// The entry the message found last for `obj_id`, where it is kept.
    fn get(&self, obj_id: u64) -> Option<NonNull<Entry<P>>> {
        self.slot(obj_id)
            .get()
            .and_then(|(id, at)| (id == obj_id).then_some(at))
    }

    /// Keeps `at`, the entry the message found for `obj_id`.
    fn put(&self, obj_id: u64, at: NonNull<Entry<P>>) {
        self.slot(obj_id).set(Some((obj_id, at)));
    }

    /// Keeps nothing.
    fn clear(&self) {
        for slot in &self.0 {
            slot.set(None);
        }
    }
}

/// What the tests call after each op of a command buffer's walk, with its index, but for the
/// render ops, which run inside their encoder's runs
/// ([`reims_vgpu_core::platform::render::RenderEncoder::encoding`]) and are not counted.
#[cfg(test)]
pub(crate) type WalkHook = Arc<dyn Fn(usize) + Send + Sync>;

impl<P: Platform> GraphicsDecoder<P> {
    /// A new decoder for `platform` over its `region`, with no objects. The first one made turns
    /// on the contexts' counts where the log names their target ([`work`]).
    pub fn new(platform: Arc<P>, region: Arc<Region>) -> Self {
        work::start(&platform);
        let made_at = platform.clock().now_mono();
        let shared = Arc::new(Shared {
            waits: waits::WaitWatch::new(Arc::clone(&platform)),
            platform,
            contexts: Mutex::new(HashMap::new()),
            closing: Mutex::default(),
            unshared: Mutex::default(),
            surfaces: Mutex::default(),
            video: OnceLock::new(),
            listener: OnceLock::new(),
            libraries: Mutex::new(Libraries::new()),
            id_chunks: Mutex::new(
                vec![None; RGFX_RESOURCE_ID_SMALL_CHUNKS as usize].into_boxed_slice(),
            ),
            closed: ClosedContexts::default(),
            unified_memory: AtomicBool::new(false),
            surface_views: Mutex::new(surface_pages::SurfaceViews::new(None, None)),
            views_logged_at: AtomicU64::new(made_at.as_nanos()),
            unsettled: Arc::default(),
            surface_worker: Arc::new(worker::Worker::new("reims-vgpu.graphics-surfaces")),
            close_worker: worker::Worker::new("reims-vgpu.graphics-closes"),
            video_worker: Arc::new(worker::Worker::new("reims-vgpu.graphics-video")),
            surface_caches: Arc::new(Caches::new(true)),
            sequencer: Arc::default(),
            split_gate: Arc::default(),
            surface_order: Mutex::new(surface_order::SurfaceOrder::new()),
            commit_cancel: commit_order::Cancel::default(),
            release_queue: OnceLock::new(),
            lanes: lane::Budget::of_host(),
            #[cfg(test)]
            walk_hook: Mutex::new(None),
            #[cfg(test)]
            split_after: Mutex::new(None),
            region,
        });
        GraphicsDecoder::on(shared, None, Sinks::default())
    }

    /// A handle on `shared` for a message of `context`'s, its records going to `sinks`.
    fn on(shared: Arc<Shared<P>>, context: Option<u32>, sinks: Sinks) -> Self {
        GraphicsDecoder {
            platform: Arc::clone(&shared.platform),
            region: Arc::clone(&shared.region),
            surface_caches: Arc::clone(&shared.surface_caches),
            shared,
            sinks,
            context,
            views: RefCell::new(Vec::new()),
            mark: Cell::new(None),
            ranges: Cell::new(None),
            ticket: RefCell::new(None),
            turn: RefCell::new(None),
            posted_at_doorbell: Cell::new(false),
            drain: RefCell::new(None),
            order_waits: RefCell::new(Vec::new()),
            commit_after: RefCell::new(Vec::new()),
            committed: RefCell::new(None),
            recent: Recent::new(),
        }
    }

    /// A handle on this decoder's state for a message from `context`, its records going to
    /// `sinks`.
    #[cfg(test)]
    pub(crate) fn sender(&self, context: Option<u32>, sinks: Sinks) -> Self {
        GraphicsDecoder::on(Arc::clone(&self.shared), context, sinks)
    }

    /// This decoder's state, shared ([`Decoders`]).
    pub(crate) fn decoders(&self) -> Decoders<P> {
        Decoders(Arc::clone(&self.shared))
    }

    /// Ends the message carried out on this handle: what it named is let go of, and its place in
    /// the commit order given back.
    pub(crate) fn end_message(&mut self) {
        self.recent.clear();
        self.views.get_mut().clear();
        self.mark.set(None);
        self.ranges.set(None);
        self.ticket.get_mut().take();
        // Its queue's commit mark rises before its turn passes, so a queue's buffers raise it in
        // its ring's order ([`commit_order`]).
        self.note_committed();
        self.commit_after.get_mut().clear();
        self.turn.get_mut().take();
        self.posted_at_doorbell.set(false);
        self.drain.get_mut().take();
        self.order_waits.get_mut().clear();
    }

    /// The message's command buffer took its place among the commit order's tickets, or got as
    /// far as it will: its queue's begun mark rises to its seqno ([`commit_order`]).
    fn note_begun(&self) {
        if let Some((marks, seqno)) = self.committed.borrow().as_ref() {
            marks.commit.begin(*seqno);
        }
    }

    /// The message's command buffer was committed, stood in for or failed: its queue's commit
    /// mark rises to its seqno ([`commit_order`]). Called as the message ends, in its turn.
    fn note_committed(&self) {
        if let Some((marks, seqno)) = self.committed.borrow_mut().take() {
            marks.commit.begin(seqno);
            marks.commit.raise(seqno);
        }
    }

    /// The guest's RAM, through which an IOSurface's own pages are mapped for its GPU view
    /// ([`surface_pages`]): with it, and a host page of the guest's size, the device carries out
    /// `SURFACE_PAGES` ([`Self::surface_pages_available`]).
    /// `lease` is the platform's hold on the guest's RAM, which each view keeps.
    pub(crate) fn set_guest_memory(
        &mut self,
        memory: Arc<MemoryMap<P>>,
        lease: Arc<dyn std::any::Any + Send + Sync>,
    ) {
        let teardown = Some(Arc::clone(&self.shared.surface_worker));
        *self.shared.surface_views.lock() =
            surface_pages::SurfaceViews::new(Some((memory, lease)), teardown);
    }

    /// Sets the bytes deferred blocks' device caches may hold at once ([`surface_cache`], the
    /// device's `DEFERRED_BUDGET`, read once at its start); without it the decoder defers none.
    pub(crate) fn set_deferred_budget(&self, budget: u64) {
        self.surface_caches.set_deferred_budget(budget);
    }

    /// Whether a cached block on `device` is named by its runs, with no view: the device copies
    /// the guest's RAM by them and keeps caches (`surface_pages`, "By runs").
    fn copies_by_runs(&self, device: &P::Device) -> bool {
        device.copies_guest_ram() && self.surface_caches.lock().on()
    }

    /// Whether `SURFACE_PAGES` makes views ([`RGFX_FEATURE_SURFACE_PAGES`]).
    ///
    /// [`RGFX_FEATURE_SURFACE_PAGES`]: reims_graphics::wire::RGFX_FEATURE_SURFACE_PAGES
    pub(crate) fn surface_pages_available(&self) -> bool {
        self.shared.surface_views.lock().available(&self.platform)
    }

    /// `SURFACE_PAGES` carried out here, at once ([`Self::begin_surface_pages`],
    /// [`Self::carry_out_surface_pages`]): its answer (for the tests).
    #[cfg(test)]
    pub(crate) fn surface_pages(
        &self,
        device: &P::Device,
        pages: &RgfxSurfacePages,
        runs: &[RgfxPageRun],
        flags: u32,
    ) -> u32 {
        let settle = self.begin_surface_pages(pages.block_offset);
        self.carry_out_surface_pages(device, pages, runs, flags, &settle)
            .unwrap_or(RGFX_STATUS_REFUSED)
    }

    /// Begins `SURFACE_PAGES` of the block at `block`, in ring order: until the returned
    /// [`worker::Settle`] drops, every message that names the block waits for it.
    pub(crate) fn begin_surface_pages(&self, block: u64) -> worker::Settle {
        let resets = self.shared.surface_views.lock().resets();
        let begun = self.platform.clock().now_mono();
        self.shared.unsettled.begin(block, resets, begun)
    }

    /// Hands the `SURFACE_PAGES` of the block at `block` to the surface worker, behind the
    /// block's jobs handed to it before ([`worker::Worker`]), begun here
    /// ([`Self::begin_surface_pages`]): `job` runs there, with its settle, which it lets go of once
    /// it has answered, and waits for nothing.
    pub(crate) fn hand_off_surface_pages(
        &self,
        block: u64,
        job: impl FnOnce(worker::Settle) + Send + 'static,
    ) {
        let settle = self.begin_surface_pages(block);
        let begun = settle.begun;
        self.shared
            .surface_worker
            .submit_waitless(block, Box::new(move || job(settle)));
        log::debug!(
            target: "reims_vgpu::graphics::surface_pages",
            "SURFACE_PAGES of the block at {block:#x} handed to the worker in {} us",
            self.platform
                .clock()
                .now_mono()
                .saturating_duration_since(begun)
                .as_micros()
        );
    }

    /// Carries out the `SURFACE_PAGES` `settle` began: the view of `runs` for the block `pages`
    /// names, which its `BUFFER_OVER_SURFACE_PAGES` buffers lie over, with a buffer over it on
    /// `device` that holds its import: [`RGFX_STATUS_OK`], else [`RGFX_STATUS_REFUSED`] with
    /// none, logged as a guest error; `None`, with nothing made, where the device was reset since
    /// its begin. `flags` are the message's `RGFX_SURFACE_PAGES_*`: a block marked
    /// [`RGFX_SURFACE_PAGES_PRIVATE`] is a pages block, its view made at its first
    /// `BUFFER_OVER_PAGES` and its buffers' alone, with no device cache ([`surface_pages`]); else
    /// one marked [`RGFX_SURFACE_PAGES_CACHED`] may be given a device cache, and with it one
    /// marked [`RGFX_SURFACE_PAGES_DEFERRED`] or [`RGFX_SURFACE_PAGES_ATTACH`] is a deferred
    /// block or its pages ([`Self::deferred_surface_pages`]). A summary of what the views did is
    /// logged every [`VIEWS_LOGGED_EVERY`], and at the first [`VIEWS_LOGGED_AFTER`] after the
    /// last ([`views_log_due`]).
    pub(crate) fn carry_out_surface_pages(
        &self,
        device: &P::Device,
        pages: &RgfxSurfacePages,
        runs: &[RgfxPageRun],
        flags: u32,
        settle: &worker::Settle,
    ) -> Option<u32> {
        let clock = self.platform.clock();
        let start = clock.now_mono();
        let private = flags & RGFX_SURFACE_PAGES_PRIVATE != 0;
        let views = &self.shared.surface_views;
        let status = if flags & (RGFX_SURFACE_PAGES_DEFERRED | RGFX_SURFACE_PAGES_ATTACH) != 0 {
            self.deferred_surface_pages(device, pages, runs, flags, settle)?
        } else {
            self.view_surface_pages(device, pages, runs, flags, settle)?
        };
        log::debug!(
            target: "reims_vgpu::graphics::surface_pages",
            "SURFACE_PAGES of the block at {:#x} ({:#x} bytes, {} runs{}): {status} in {} us, {} \
             us in the queue",
            pages.block_offset,
            pages.length,
            runs.len(),
            if private {
                ", a pages block"
            } else if flags & RGFX_SURFACE_PAGES_DEFERRED != 0 {
                ", deferred"
            } else if flags & RGFX_SURFACE_PAGES_ATTACH != 0 {
                ", a deferred block's pages"
            } else if flags & RGFX_SURFACE_PAGES_CACHED != 0 {
                ", cached"
            } else {
                ""
            },
            clock.now_mono().saturating_duration_since(start).as_micros(),
            start.saturating_duration_since(settle.begun).as_micros()
        );
        let c = views.lock().counts();
        let d = self.surface_caches.lock().counts();
        // A deferred block, and a block named by its runs, make no view, so they are counted
        // here too.
        let counted = (c.made + c.reused + c.bad_runs + c.not_mapped + c.private + c.by_runs)
            .saturating_add(d.deferred + d.deferred_refused);
        let now = clock.now_mono();
        let logged_at = &self.shared.views_logged_at;
        let last = logged_at.load(Ordering::Relaxed);
        // One message logs for each time due, where two blocks' lanes find it due at once.
        if views_log_due(counted, now, MonoTime::from_nanos(last))
            && logged_at
                .compare_exchange(last, now.as_nanos(), Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
        {
            log::info!(
                target: "reims_vgpu::graphics",
                "IOSurface pages in place: {} views made, {} taken from the cache; {} pages \
                 blocks' views made; {} named by their runs; refused: {} for bad runs, {} not \
                 mapped",
                c.made, c.reused, c.private, c.by_runs, c.bad_runs, c.not_mapped
            );
            if d.made != 0 {
                log::info!(
                    target: "reims_vgpu::graphics",
                    "IOSurface device caches: {} made; {} fills of {} MiB ({} as made), {} \
                     downloads of {} MiB, {} asked of a cache nothing wrote; {} fills or \
                     downloads the host's devices' writes changed; {} written caches copied at a \
                     release with no client left; {} dropped uncopied, held by the kext alone \
                     ({} MiB), {} copied as taken again; {} copies out not made, the CPU's \
                     generation past the cache's; {} \
                     deferred, {} refused, {} named their pages ({} copied), {} released \
                     unnamed; {} early copies of {} MiB ({} on the read-back queue), {} \
                     downloads they answered, {} unread; {} copy by runs, {} copies by runs \
                     refused",
                    d.made, d.fills, d.filled_bytes >> 20, d.made_fills, d.downloads,
                    d.downloaded_bytes >> 20, d.clean, d.host_writes, d.unread, d.unheld,
                    d.unheld_bytes >> 20, d.retaken, d.behind,
                    d.deferred, d.deferred_refused, d.attached, d.attached_copied, d.unattached,
                    d.early, d.early_bytes >> 20, d.early_aside, d.early_read, d.early_unread,
                    d.by_runs, d.runs_refused
                );
            }
        }
        Some(status)
    }

    /// [`Self::carry_out_surface_pages`] of a block with pages, which it maps.
    fn view_surface_pages(
        &self,
        device: &P::Device,
        pages: &RgfxSurfacePages,
        runs: &[RgfxPageRun],
        flags: u32,
        settle: &worker::Settle,
    ) -> Option<u32> {
        let private = flags & RGFX_SURFACE_PAGES_PRIVATE != 0;
        let cached = flags & RGFX_SURFACE_PAGES_CACHED != 0;
        let by_runs = cached && self.copies_by_runs(device);
        let views = &self.shared.surface_views;
        let made = if private {
            Ok(surface_pages::SurfaceViews::private_pages(
                views,
                &self.platform,
                pages,
                runs,
                settle,
            ))
        } else {
            surface_pages::SurfaceViews::surface_pages(
                views,
                &self.platform,
                device,
                pages,
                runs,
                settle,
                by_runs,
            )
        };
        let status = match made {
            Ok(None) => return None,
            Ok(Some(Ok(()))) if private => {
                self.uncache_surface(device, pages.block_offset);
                RGFX_STATUS_OK
            }
            Ok(Some(Ok(()))) => {
                self.cache_surface(device, pages.block_offset, runs, cached);
                RGFX_STATUS_OK
            }
            Ok(Some(Err(why))) => {
                self.uncache_surface(device, pages.block_offset);
                log::warn!(
                    target: "reims_vgpu::graphics",
                    "the IOSurface block of {:#x} bytes at {:#x}: no view of its {:#x} bytes in \
                     {} runs ({why:?}); the surface takes a copy block",
                    pages.block_size, pages.block_offset, pages.length, runs.len()
                );
                RGFX_STATUS_REFUSED
            }
            Err(raise) => {
                self.uncache_surface(device, pages.block_offset);
                log::warn!(
                    target: "reims_vgpu::graphics",
                    "the IOSurface block of {:#x} bytes at {:#x}: the platform raised at the \
                     buffer over its view ({raise:?}); the surface takes a copy block",
                    pages.block_size, pages.block_offset
                );
                RGFX_STATUS_REFUSED
            }
        };
        Some(status)
    }

    /// [`Self::carry_out_surface_pages`] of a deferred block ([`RGFX_SURFACE_PAGES_DEFERRED`]) or
    /// of its pages ([`RGFX_SURFACE_PAGES_ATTACH`]), each marked [`RGFX_SURFACE_PAGES_CACHED`] too
    /// ([`surface_cache`]):
    /// - a deferred block, with no runs, gets a device cache of zeros and no view, any view it had
    ///   going: [`RGFX_STATUS_OK`], else [`RGFX_STATUS_REFUSED`] with no cache and no view, where
    ///   the device keeps no caches, the deferred caches' budget has no room or the platform made
    ///   none, and the kext then names the block's pages as any block's;
    /// - its pages, found by the block whatever context sends them (the maker's may have closed),
    ///   get a view, which the cache takes, keeping its bytes: [`RGFX_STATUS_OK`]. Where the host
    ///   makes no view of them, or no buffer over it, the cache takes them copied, through the
    ///   host's mapping of the guest's RAM ([`surface_cache`]'s module doc): [`RGFX_STATUS_OK`]
    ///   too. [`RGFX_STATUS_REFUSED`], with the cache as it was and its pages not named, only
    ///   where the runs are not the surface's bytes of guest RAM the host maps;
    ///   [`RGFX_STATUS_INVALID_ARGUMENT`] for a block with no deferred cache whose pages are not
    ///   named, with no view made.
    ///
    /// A message with both flags, or without `CACHED`, or a deferred block's with runs, is
    /// [`RGFX_STATUS_INVALID_ARGUMENT`], and changes nothing. `None`, with nothing made, where the
    /// device was reset since `settle` began it.
    fn deferred_surface_pages(
        &self,
        device: &P::Device,
        pages: &RgfxSurfacePages,
        runs: &[RgfxPageRun],
        flags: u32,
        settle: &worker::Settle,
    ) -> Option<u32> {
        let block = pages.block_offset;
        let deferred = flags & RGFX_SURFACE_PAGES_DEFERRED != 0;
        let attach = flags & RGFX_SURFACE_PAGES_ATTACH != 0;
        let cached = flags & RGFX_SURFACE_PAGES_CACHED != 0;
        let private = flags & RGFX_SURFACE_PAGES_PRIVATE != 0;
        if !cached || private || (deferred && attach) || (deferred && !runs.is_empty()) {
            log::warn!(
                target: "reims_vgpu::graphics",
                "the IOSurface block at {block:#x}: a SURFACE_PAGES of {} runs with flags \
                 {flags:#x}, not a deferred block's nor its pages': refused",
                runs.len()
            );
            return Some(RGFX_STATUS_INVALID_ARGUMENT);
        }
        let views = &self.shared.surface_views;
        if deferred {
            if !surface_pages::SurfaceViews::deferred(views, block, settle) {
                return None;
            }
            let made =
                self.surface_caches
                    .make_deferred(device, block, pages.length, pages.block_size);
            return Some(match made {
                Ok(true) => RGFX_STATUS_OK,
                Ok(false) => RGFX_STATUS_REFUSED,
                Err(raise) => {
                    log::warn!(
                        target: "reims_vgpu::graphics",
                        "the IOSurface block at {block:#x}: the platform raised at its \
                         deferred device cache ({raise:?}); its pages are named instead"
                    );
                    RGFX_STATUS_REFUSED
                }
            });
        }
        if self.surface_caches.lock().unattached(block).is_none() {
            log::warn!(
                target: "reims_vgpu::graphics",
                "the IOSurface block at {block:#x}: its pages named as a deferred block's, and it \
                 has no deferred device cache without pages: refused"
            );
            return Some(RGFX_STATUS_INVALID_ARGUMENT);
        }
        let made = surface_pages::SurfaceViews::surface_pages(
            views,
            &self.platform,
            device,
            pages,
            runs,
            settle,
            self.copies_by_runs(device),
        );
        // Runs that are not whole guest pages summing to the surface, or with no guest RAM to map,
        // are refused; pages the host made no view of, or no buffer over, are copied instead.
        let copyable = !matches!(
            made,
            Ok(Some(Err(surface_pages::PagesRefusal::NoMemoryMap
                | surface_pages::PagesRefusal::PageSize
                | surface_pages::PagesRefusal::Unaligned
                | surface_pages::PagesRefusal::Length)))
        );
        let why = match made {
            Ok(None) => return None,
            Ok(Some(Ok(()))) => {
                let (named, host_pages) = {
                    let views = views.lock();
                    let named = match (views.view(block), views.by_runs(block)) {
                        (Some(view), _) => Some(surface_cache::Named::View(view)),
                        (None, Some((runs, unmade))) => {
                            Some(surface_cache::Named::Runs(runs, unmade))
                        }
                        (None, None) => None,
                    };
                    (named, views.host_runs(runs))
                };
                let attached = match named {
                    Some(named) => self.surface_caches.attach(device, block, named, host_pages),
                    None => Ok(Some(false)),
                };
                match attached {
                    Ok(Some(true)) => return Some(RGFX_STATUS_OK),
                    Ok(Some(false)) => {
                        "no buffer over its view, or a view shorter than the cache".into()
                    }
                    Ok(None) => return Some(RGFX_STATUS_INVALID_ARGUMENT),
                    Err(raise) => {
                        format!("the platform raised at the buffer over its view ({raise:?})")
                    }
                }
            }
            Ok(Some(Err(why))) => format!("no view of its pages ({why:?})"),
            Err(raise) => format!("the platform raised at its view ({raise:?})"),
        };
        if copyable {
            let (copied, host_pages) = {
                let views = views.lock();
                (views.host_pages_in_order(runs), views.host_runs(runs))
            };
            let taken = copied.map(|c| self.surface_caches.attach_copied(block, &c, host_pages));
            match taken {
                Some(Some(true)) => {
                    log::warn!(
                        target: "reims_vgpu::graphics",
                        "the IOSurface block at {block:#x}: its deferred device cache took its {} \
                         runs copied, the host's CPU moving its bytes: {why}",
                        runs.len()
                    );
                    return Some(RGFX_STATUS_OK);
                }
                Some(None) => return Some(RGFX_STATUS_INVALID_ARGUMENT),
                Some(Some(false)) | None => {}
            }
        }
        log::warn!(
            target: "reims_vgpu::graphics",
            "the IOSurface block at {block:#x}: its deferred device cache took none of its {} \
             runs: {why}; the cache keeps its bytes, its pages not named",
            runs.len()
        );
        Some(RGFX_STATUS_REFUSED)
    }

    /// Gives the block at `block`, whose view of `runs` was just made, or which is named by its
    /// runs, a device cache where the kext marked it `cached` and the device keeps caches
    /// ([`surface_cache`]); else the block keeps its view alone, and its buffers lie over the
    /// pages. A block named by its runs that gets no cache takes its view now.
    fn cache_surface(&self, device: &P::Device, block: u64, runs: &[RgfxPageRun], cached: bool) {
        if !cached || !self.surface_caches.lock().on() {
            self.uncache_surface(device, block);
            return;
        }
        let (view, by_runs, pages) = {
            let views = self.shared.surface_views.lock();
            (
                views.view(block),
                views.by_runs(block),
                views.host_runs(runs),
            )
        };
        // The CPU's generation the kext wrote into the block's header as it cached the block, or
        // since, which the cache's first fill is of.
        let generation = surface_cache::cpu_generation(&self.shared.region, block).unwrap_or(0);
        let (made, length) = match (view, by_runs) {
            (Some(view), _) => (
                (self.surface_caches).make(device, block, &view, pages, generation),
                view.length(),
            ),
            (None, Some((runs, unmade))) => {
                let length = unmade.length();
                let made = self.surface_caches.make_by_runs(
                    device,
                    block,
                    runs,
                    Arc::clone(&unmade),
                    pages,
                    generation,
                );
                if !matches!(made, Ok(true)) {
                    self.view_by_runs(device, block, &unmade);
                }
                (made, length)
            }
            (None, None) => return,
        };
        match made {
            Ok(true) => {}
            Ok(false) => log::warn!(
                target: "reims_vgpu::graphics",
                "the IOSurface block at {block:#x}: the platform made no device cache of its \
                 {length:#x} bytes; its buffers lie over its pages"
            ),
            Err(raise) => log::warn!(
                target: "reims_vgpu::graphics",
                "the IOSurface block at {block:#x}: the platform raised at its device cache \
                 ({raise:?}); its buffers lie over its pages"
            ),
        }
    }

    /// The block at `block`, named by its runs and given no device cache, takes the view
    /// `unmade` makes, which its buffers then lie over; one not made leaves the block with
    /// none, and its buffers are refused.
    fn view_by_runs(&self, device: &P::Device, block: u64, unmade: &surface_pages::Unmade<P>) {
        match unmade.make(device) {
            Ok(Ok((view, import))) => {
                self.shared.surface_views.lock().viewed(block, view);
                // The view's buffers hold it; the import's buffer goes on the worker, where it
                // may be the last hold of the import.
                self.shared.surface_worker.drop_later(block, block, import);
            }
            Ok(Err(why)) => log::warn!(
                target: "reims_vgpu::graphics",
                "the IOSurface block at {block:#x}: no device cache, and no view of its pages \
                 ({why:?}); its buffers are refused"
            ),
            Err(raise) => log::warn!(
                target: "reims_vgpu::graphics",
                "the IOSurface block at {block:#x}: no device cache, and the platform raised at \
                 the buffer over its view ({raise:?}); its buffers are refused"
            ),
        }
    }

    /// The block at `block`, named again without a cache, loses any it had, as at a release: what
    /// a buffer wrote of it goes into the pages first ([`surface_cache`]).
    fn uncache_surface(&self, device: &P::Device, block: u64) {
        if !self.surface_caches.lock().has(block) {
            return;
        }
        if let Err(raise) = self.surface_caches.released(device, block, Box::new(|| {})) {
            log::warn!(
                target: "reims_vgpu::graphics",
                "the IOSurface block at {block:#x}: the platform raised at the copy of its device \
                 cache into its pages ({raise:?})"
            );
        }
    }

    /// `SURFACE_DOWNLOAD` of the block at `block`: its device cache written into the surface's
    /// pages, after every command buffer that used it ([`surface_cache`]), with no wait here for
    /// one still walked; `done` once the pages hold it, at once where the block has no cache or
    /// nothing wrote it since its last copy, or `failed` where the platform raised at the copy.
    pub(crate) fn surface_download(
        &self,
        device: &P::Device,
        block: u64,
        done: Box<dyn FnOnce() + Send>,
        failed: Box<dyn FnOnce(Raise) + Send>,
    ) {
        self.settle_surface(block);
        let region = Arc::clone(&self.shared.region);
        (self.surface_caches).download(device, block, region, done, failed);
    }

    /// [`Self::surface_download`] where it waits for nothing and commits no copy now
    /// ([`surface_cache::Caches::try_download`]): the block's `SURFACE_PAGES` settled, and the
    /// caches' lock free. Whether it was carried out; where not, `done` and `failed` are dropped
    /// ungiven, for the download to be asked again on the block's lane.
    pub(crate) fn try_surface_download(
        &self,
        device: &P::Device,
        block: u64,
        done: Box<dyn FnOnce() + Send>,
        failed: Box<dyn FnOnce(Raise) + Send>,
    ) -> bool {
        if self.shared.unsettled.any(std::iter::once(block)) {
            return false;
        }
        let region = Arc::clone(&self.shared.region);
        (self.surface_caches).try_download(device, block, region, done, failed)
    }

    /// The block at `offset` was released, its `RGFX_BLOCK_RELEASE_*` `flags`: its surface view,
    /// if any, goes once the buffers over it have, and its device cache once the pages hold what
    /// a command buffer wrote of it, whether its surface had a client left or not
    /// ([`RGFX_BLOCK_RELEASE_NO_CLIENT`](reims_graphics::wire::RGFX_BLOCK_RELEASE_NO_CLIENT), which
    /// is only counted). Where nothing but the kext held the surface
    /// ([`RGFX_BLOCK_RELEASE_UNHELD`](reims_graphics::wire::RGFX_BLOCK_RELEASE_UNHELD)) the cache
    /// goes uncopied. There is no wait here for a buffer still walked; `done` then, or `failed`
    /// where the platform raised at the copy ([`surface_cache`]). The block's holds of them are
    /// let go of on the surface worker, where either may be the last of an import or a mapping.
    pub(crate) fn block_released(
        &self,
        device: &P::Device,
        offset: u64,
        flags: u32,
        done: Box<dyn FnOnce() + Send>,
        failed: Box<dyn FnOnce(Raise) + Send>,
    ) {
        self.settle_surface(offset);
        let gone = self.shared.surface_views.lock().released(offset);
        if let Some(gone) = gone {
            self.shared.surface_worker.drop_later(offset, offset, gone);
        }
        let worker = Arc::clone(&self.shared.surface_worker);
        let let_go = Box::new(move |cache| worker.drop_later(offset, offset, cache));
        let region = Arc::clone(&self.shared.region);
        (self.surface_caches).release(device, offset, flags, region, done, failed, let_go);
    }

    /// Fills the device cache of the block the texture a present of IOSurface `iosurface_id`
    /// shows lies in, where it holds an older CPU generation than `generation` ([`surface_cache`]):
    /// the wait for the fill, which the present makes; `None` where it need wait for none.
    pub(crate) fn fill_presented(
        &self,
        device: &P::Device,
        iosurface_id: u32,
        generation: u64,
    ) -> RaiseResult<Option<FillWait>> {
        let shown = (self.shared.surfaces.lock().by_surface.get(&iosurface_id))
            .and_then(|textures| textures.last().copied());
        let block = shown.and_then(|obj_id| {
            self.context_state(context_of(obj_id))?
                .with_live(obj_id, |e| e.span)
                .flatten()
                .map(|(at, _)| at)
        });
        match block {
            Some(block) => {
                self.settle_surface(block);
                self.surface_caches.fill_now(device, block, generation)
            }
            None => Ok(None),
        }
    }

    /// Has each fill of a device cache tell `hook` as it finishes: the host's presents that wait
    /// for one ([`Self::fill_presented`]).
    pub(crate) fn on_fill_done(&self, hook: FillHook) {
        self.surface_caches.on_fill_done(hook);
    }

    /// Sets the sink the decoder hands each command buffer's completion record to
    /// once its work finishes (the ring loop's, writing the queue's completion
    /// ring).
    #[cfg(test)]
    pub(crate) fn set_completion_sink(&mut self, sink: CompletionSink) {
        self.sinks.completion = Some(sink);
    }

    /// Sets the sink the decoder hands each shared event a command buffer signalled to, once the
    /// buffer's work finishes (the ring loop's, writing the queue's completion ring).
    #[cfg(test)]
    pub(crate) fn set_signal_sink(&mut self, sink: SignalSink) {
        self.sinks.signal = Some(sink);
    }

    /// Sets the sink the decoder hands each command buffer's `CB_SCHEDULED` to once the buffer is
    /// scheduled (the ring loop's, writing the queue's completion ring).
    #[cfg(test)]
    pub(crate) fn set_scheduled_sink(&mut self, sink: ScheduledSink) {
        self.sinks.scheduled = Some(sink);
    }

    /// Sets the sink the decoder hands the `CB_SUBMITTED` of each command buffer that asks for it
    /// to once the buffer is submitted (the ring loop's, writing the queue's completion ring).
    #[cfg(test)]
    pub(crate) fn set_submitted_sink(&mut self, sink: SubmittedSink) {
        self.sinks.submitted = Some(sink);
    }

    /// Sets the sink the decoder hands each pipeline create's reflection reply to
    /// (the ring loop's, writing the queue's completion ring).
    #[cfg(test)]
    pub(crate) fn set_reflection_sink(&mut self, sink: ReflectionSink) {
        self.sinks.reflection = Some(sink);
    }

    /// Sets the context of the messages carried out next, which begin anew: the process that sent
    /// them, whose objects alone they may name. On ring 0 it is the context of the object a
    /// message creates or destroys, which the kext checked is the sender's; on a queue's ring, the
    /// queue's.
    #[cfg(test)]
    pub(crate) fn set_context(&mut self, context: Option<u32>) {
        self.end_message();
        self.context = context;
    }

    /// Queue `queue` was made, its ring's published seqnos read through `published`: a buffer of
    /// another queue that names one of its buffers past them waits for nothing
    /// ([`surface_order`]).
    pub(crate) fn surface_order_queue_made(&self, queue: u64, published: surface_order::Published) {
        self.shared
            .surface_order
            .lock()
            .queue_made(queue, published);
    }

    /// Queue `queue_obj_id`, whose handle is `queue` where the host still holds it, is gone (its
    /// destroy, or its context's close): its end event goes where none of its buffers signalled it
    /// ([`Caches::queue_ended`]), and its surface timeline signals what other queues' buffers
    /// wait for of it, once the order's lock is let go of ([`surface_order::SurfaceOrder::ended`]).
    /// A refusal is logged, and the destroy goes on.
    fn queue_ended(&self, queue_obj_id: u64, queue: Option<&P::CommandQueue>) {
        self.surface_caches.queue_ended(queue_obj_id);
        let release = self.shared.surface_order.lock().ended(queue_obj_id);
        if let Some(release) = release
            && let Err(raise) = refusing(|| release.signal(queue, &self.shared.commit_cancel))
        {
            log::warn!(
                target: "reims_vgpu::graphics",
                "queue {queue_obj_id:#x}'s surface timeline was ended, and its last signal \
                 faulted: {raise}"
            );
        }
    }

    /// Whether `obj_id` is the current context's to name (any where none is set).
    fn owned(&self, obj_id: u64) -> bool {
        self.context
            .is_none_or(|context| obj_id >> RGFX_CONTEXT_OBJ_ID_SHIFT == u64::from(context))
    }

    /// The state of context `context`, where it has any.
    fn context_state(&self, context: u32) -> Option<Arc<Context<P>>> {
        self.shared.contexts.lock().get(&context).cloned()
    }

    /// The state of context `context`, for a create of its own, made where it has none. A create
    /// of a context closed before opens it again ([`ClosedContexts::reopen`]): only a process
    /// given its number again posts one after the close, which empties the context's
    /// compiles and waits for its no-copy buffers first. Nothing else makes a state.
    fn context_made(&self, context: u32) -> Arc<Context<P>> {
        let made = self.context_made_if(context, || true);
        made.unwrap_or_else(|| unreachable!("a state is made where `live` holds"))
    }

    /// [`Self::context_made`], for a create carried out off its context's ring order (a compile's
    /// pipeline, an indirect command buffer made on its worker): where the context has no state,
    /// `live` answers, under the states' lock, whether the create is still to be stored, and
    /// where it is not, no state is made and `None` is answered. A close takes the context's
    /// state out under that lock, after it cancelled what it overtook, so such a create, ending
    /// after the close, makes the closed context no state again.
    fn context_made_if(
        &self,
        context: u32,
        live: impl FnOnce() -> bool,
    ) -> Option<Arc<Context<P>>> {
        let mut contexts = self.shared.contexts.lock();
        if let Some(state) = contexts.get(&context) {
            return Some(Arc::clone(state));
        }
        if !live() {
            return None;
        }
        self.shared.closed.reopen(context);
        let state = Arc::new(Context::new(context));
        contexts.insert(context, Arc::clone(&state));
        Some(state)
    }

    /// Makes the context of `obj_id`, a create's object, a state where it has none, in its lane's
    /// ring order, for a create carried out later off it ([`Self::store_late`]): a create of a
    /// context closed before opens it again here, as one carried out at once does.
    pub(crate) fn open_for_create(&self, obj_id: u64) {
        drop(self.context_made(context_of(obj_id)));
    }

    /// Whether the current context was closed before this message came, and has made no state
    /// since: what of it needs a state is owed nothing ([`DecodeError::ContextClosed`]).
    fn context_closed(&self) -> bool {
        self.context.is_some_and(|context| {
            self.context_state(context).is_none() && self.shared.closed.closed(context)
        })
    }

    /// How many contexts the decoder holds a state of.
    #[cfg(test)]
    pub(crate) fn context_states(&self) -> usize {
        self.shared.contexts.lock().len()
    }

    /// The entry of `obj_id` the message sees (its view of the object's context's table, taken
    /// at its first naming of an object there where not before; [`tables`]), where it is the
    /// current context's to name: a process names only its own objects, and an id of another's
    /// resolves to nothing (a bad reference where it is named).
    fn entry(&self, obj_id: u64) -> Option<&Entry<P>> {
        if let Some(at) = self.recent.get(obj_id) {
            // SAFETY: an entry a view of `views` found below, which the view lends for its own
            // life; the view stays there until the message ends, which empties the cache first.
            return Some(unsafe { at.as_ref() });
        }
        if !self.owned(obj_id) {
            return None;
        }
        let context = context_of(obj_id);
        let found = {
            let views = self.views.borrow();
            views
                .iter()
                .find(|v| v.context().id == context)
                .map(|v| v.get(obj_id).map(std::ptr::from_ref))
        };
        let found = match found {
            Some(found) => found,
            None => {
                let view = View::take(self.context_state(context)?);
                let found = view.get(obj_id).map(std::ptr::from_ref);
                self.views.borrow_mut().push(view);
                found
            }
        };
        let found = found.and_then(|p| NonNull::new(p.cast_mut()))?;
        self.recent.put(obj_id, found);
        // SAFETY: the view lends the entry for its own life, and it stays in `views` until the
        // handle drops or a message begins anew on it, which takes `&mut self`: so for `&self`'s.
        Some(unsafe { found.as_ref() })
    }

    /// The object `obj_id` names, where it is the current context's: a process names only its own
    /// objects, and an id of another's resolves to nothing (a bad reference where it is named).
    fn named(&self, obj_id: u64) -> Option<&ObjHandle<P>> {
        self.entry(obj_id).map(|e| &e.handle)
    }

    /// The counts of the command queue `obj_id` names, where it is one the current context may
    /// name.
    pub(super) fn queue_counts(&self, obj_id: u64) -> Option<&QueueCounts> {
        self.entry(obj_id)
            .filter(|e| matches!(e.handle, ObjHandle::CommandQueue(_)))
            .and_then(|e| e.queue.as_ref())
    }

    /// Whether `len` bytes at region offset `offset` lie in one block the current context holds
    /// (`BLOCK_GRANT`, [`Region::held_by`]). With no context set (the decoder's own tests), in the
    /// processes' part of the region, where the kext places their blocks: past the control rings,
    /// the device-info blob and the scanout ([`RGFX_CONTEXT_ALLOCATIONS_OFFSET`]), before the
    /// cursor's span at the end.
    fn in_own_block(&self, offset: u64, len: u64) -> bool {
        match self.context {
            Some(context) => self.region.held_by(context, offset, len),
            None => {
                let end = self.region.size().saturating_sub(RGFX_CURSOR_SPAN);
                offset >= RGFX_CONTEXT_ALLOCATIONS_OFFSET
                    && offset.checked_add(len).is_some_and(|e| e <= end)
            }
        }
    }

    /// The `count` records of `stride` bytes from region offset `offset`, where they lie in one
    /// block the current context holds ([`Self::in_own_block`]), each with its first `taken`
    /// words swapped for 0 as it is copied ([`Region::take_records`]).
    fn take_own(&self, offset: u64, stride: u64, count: u64, taken: usize) -> Option<Vec<u8>> {
        match self.context {
            Some(context) => self
                .region
                .take_records_held(context, offset, stride, count, taken),
            None => {
                let len = stride.checked_mul(count)?;
                self.in_own_block(offset, len)
                    .then(|| self.region.take_records(offset, stride, count, taken))
                    .flatten()
            }
        }
    }

    /// The objects the tables hold (for the tests).
    #[cfg(any(test, feature = "testkit"))]
    pub fn object_count(&self) -> usize {
        let contexts: Vec<Arc<Context<P>>> =
            self.shared.contexts.lock().values().cloned().collect();
        contexts.iter().map(|c| c.len()).sum()
    }

    /// The object `obj_id` names as the message sees it (for the tests).
    #[cfg(test)]
    fn object(&self, obj_id: u64) -> Option<&ObjHandle<P>> {
        self.named(obj_id)
    }

    /// The ids of the objects the tables hold, by context, each in its table's order (for the
    /// tests).
    #[cfg(test)]
    pub(crate) fn object_ids(&self) -> Vec<u64> {
        let mut contexts: Vec<Arc<Context<P>>> =
            self.shared.contexts.lock().values().cloned().collect();
        contexts.sort_by_key(|c| c.id);
        contexts.iter().flat_map(|c| c.ids()).collect()
    }

    /// The ids that stand for a refused create ([`refused`]), by context (for the tests).
    #[cfg(test)]
    pub(crate) fn refused_ids(&self) -> Vec<u64> {
        let mut contexts: Vec<Arc<Context<P>>> =
            self.shared.contexts.lock().values().cloned().collect();
        contexts.sort_by_key(|c| c.id);
        let mut ids: Vec<u64> = contexts.iter().flat_map(|c| c.refused_ids()).collect();
        ids.sort_unstable();
        ids
    }

    /// The device caches, locked (for the tests).
    #[cfg(test)]
    pub(crate) fn caches(
        &self,
    ) -> reims_vgpu_core::sync::MutexGuard<'_, surface_cache::SurfaceCaches<P>> {
        self.surface_caches.lock()
    }

    /// Waits for the block at `block` to be as its `SURFACE_PAGES` left it, where one is being
    /// carried out ([`worker::Unsettled`]); true.
    pub(crate) fn surface_settled(&self, block: u64) -> bool {
        self.settle_surface(block);
        true
    }

    /// Waits for the block at `block` to be as its `SURFACE_PAGES` left it, where the surface
    /// worker is still carrying one out ([`worker::Unsettled`]), and logs the wait: the guest's
    /// kext names a block only once it is answered, so a wait is a message the guest posted before.
    fn settle_surface(&self, block: u64) {
        if !self.shared.unsettled.any(std::iter::once(block)) {
            return;
        }
        let clock = self.platform.clock();
        let start = clock.now_mono();
        self.shared.unsettled.wait(block);
        log::debug!(
            target: "reims_vgpu::graphics::surface_pages",
            "a message naming the block at {block:#x} waited {} us for its SURFACE_PAGES",
            clock.now_mono().saturating_duration_since(start).as_micros()
        );
    }

    /// Whether the `BUFFER_OVER_PAGES` `message` makes its pages block's view (the block's first),
    /// or follows its block's `SURFACE_PAGES` still being carried out on the surface worker: what
    /// is carried out there ([`surface_pages`]).
    pub(crate) fn buffer_over_pages_makes(&self, message: &[u8]) -> bool {
        let Some(m) = read::<RgfxBufferOverPages>(message) else {
            return false;
        };
        self.shared.unsettled.any(std::iter::once(m.block_offset))
            || self
                .shared
                .surface_views
                .lock()
                .private_unmade(m.block_offset)
    }

    /// Hands `job`, of the block at `block`, to the surface worker, behind the block's jobs handed
    /// to it before.
    pub(crate) fn on_surface_worker(&self, block: u64, job: worker::Job) {
        self.shared.surface_worker.submit(block, job);
    }

    /// The surface worker (for the tests).
    #[cfg(test)]
    pub(crate) fn surface_worker(&self) -> &worker::Worker {
        &self.shared.surface_worker
    }

    /// The video worker (for the tests).
    #[cfg(test)]
    pub(crate) fn video_worker(&self) -> &worker::Worker {
        &self.shared.video_worker
    }

    /// The surfaces' views, locked (for the tests).
    #[cfg(test)]
    pub(crate) fn views_of_surfaces(
        &self,
    ) -> reims_vgpu_core::sync::MutexGuard<'_, surface_pages::SurfaceViews<P>> {
        self.shared.surface_views.lock()
    }

    /// Has `hook` called after each op of every command buffer's walk with the op's index (for
    /// the tests).
    #[cfg(test)]
    pub(crate) fn set_walk_hook(&self, hook: Option<WalkHook>) {
        *self.shared.walk_hook.lock() = hook;
    }

    /// The hook the walks call (for the tests).
    #[cfg(test)]
    pub(crate) fn walk_hook(&self) -> Option<WalkHook> {
        self.shared.walk_hook.lock().clone()
    }

    /// Has the walks commit a part of a buffer once it took `after` ([`Self::split_after`]), or
    /// [`split::SPLIT_AFTER`] where `None` (for the tests and the benches).
    #[cfg(test)]
    pub(crate) fn set_split_after(&self, after: Option<std::time::Duration>) {
        *self.shared.split_after.lock() = after;
    }

    /// How long a part of a command buffer is walked, at least, before the walk commits it
    /// ([`split::SPLIT_AFTER`]; the tests may set their own).
    pub(crate) fn split_after(&self) -> std::time::Duration {
        #[cfg(test)]
        if let Some(after) = *self.shared.split_after.lock() {
            return after;
        }
        split::SPLIT_AFTER
    }

    /// The texture a present of IOSurface `iosurface_id` shows ([`Self::surface_present`]).
    #[cfg(test)]
    pub(crate) fn surface_texture(&self, iosurface_id: u32) -> Option<P::Texture> {
        self.surface_present(iosurface_id)
            .map(|present| present.texture)
    }

    /// The textures IOSurface `iosurface_id` is named by, oldest first: those over its plane 0,
    /// the newest of which a present of it shows ([`Self::surface_present`]).
    pub(crate) fn surface_textures(&self, iosurface_id: u32) -> Vec<u64> {
        (self.shared.surfaces.lock().by_surface.get(&iosurface_id))
            .cloned()
            .unwrap_or_default()
    }

    /// What a present of IOSurface `iosurface_id` shows ([`SurfacePresent`]). The caller holds its
    /// own reference to the texture: a destroy of it after this leaves it drawable.
    pub(crate) fn surface_present(&self, iosurface_id: u32) -> Option<SurfacePresent<P>> {
        let (obj_id, writers_at) = {
            let names = self.shared.surfaces.lock();
            let obj_id = *names.by_surface.get(&iosurface_id)?.last()?;
            (obj_id, names.writers.get(&obj_id).copied())
        };
        let (texture, span) = self
            .context_state(context_of(obj_id))?
            .with_live(obj_id, |e| match &e.handle {
                ObjHandle::Texture(texture) => Some((texture.clone_ref(), e.span)),
                _ => None,
            })
            .flatten()?;
        let is_queue = |queue: u64| {
            self.context_state(context_of(queue))
                .and_then(|c| {
                    c.with_live(queue, |e| matches!(e.handle, ObjHandle::CommandQueue(_)))
                })
                .unwrap_or(false)
        };
        let writers = writers_at
            .and_then(|at| self.region.surface_writers(at))
            .map(|record| {
                record
                    .writers
                    .into_iter()
                    .filter(|w| w.seqno != 0 && is_queue(w.queue_obj_id))
                    .collect()
            })
            .unwrap_or_default();
        Some(SurfacePresent {
            texture,
            span,
            writers,
            owner: context_of(obj_id),
        })
    }

    /// Forgets texture `obj_id` as a name of the surfaces it lies over.
    fn forget_surface_texture(&self, obj_id: u64) {
        let mut names = self.shared.surfaces.lock();
        names.by_surface.retain(|_, textures| {
            textures.retain(|t| *t != obj_id);
            !textures.is_empty()
        });
        names.writers.remove(&obj_id);
    }

    /// Forgets the surface names of context `context`'s textures: at its close, so a later object
    /// that takes a dropped id is never taken for the surface's texture.
    fn forget_context_surfaces(&self, context: u32) {
        let ours = |t: &u64| context_of(*t) == context;
        let mut names = self.shared.surfaces.lock();
        names.writers.retain(|t, _| !ours(t));
        names.by_surface.retain(|_, textures| {
            textures.retain(|t| !ours(t));
            !textures.is_empty()
        });
    }

    /// Drops every object of context `context` ([`Self::close_tables`]), letting go of them here
    /// ([`Closed::let_go`]): how many objects were dropped.
    pub(crate) fn drop_context(&self, device: &P::Device, context: u32) -> usize {
        let (dropped, closed) = self.close_tables(device, context);
        closed.let_go();
        dropped
    }

    /// Takes every object of context `context` out of the decoder's tables and names: those whose
    /// id lies in its range, `context << RGFX_CONTEXT_OBJ_ID_SHIFT` and up, below the next
    /// context's (its process closed the device or died), its chunks of the texture resource ids,
    /// its surfaces' names, its libraries and its pages blocks' views: its events set past every
    /// value first ([`Self::release_events`]), and its queues ended. How many objects it held,
    /// and what it took out, for the caller to let go of ([`Closed`]).
    pub(crate) fn close_tables(&self, device: &P::Device, context: u32) -> (usize, Closed<P>) {
        for holder in self.shared.id_chunks.lock().iter_mut() {
            if *holder == Some(context) {
                *holder = None;
            }
        }
        let mut dropped = 0;
        let mut objects = Vec::new();
        let queues = self.shared.surface_order.lock().queues_of(context);
        if let Some(state) = self.context_state(context) {
            // Its events first, while its table holds them: a view open on it, a walk's, keeps
            // what the removal retires out of its answer until the view closes.
            self.release_events(device, &state);
            let (removed, count) = state.remove_all();
            dropped = count;
            // Its queues end first, each signalling on its own host queue where it is let go of
            // here ([`surface_order`]).
            for &queue in &queues {
                let handle = (removed.iter().find(|r| r.id() == queue)).and_then(|r| {
                    match &r.entry().handle {
                        ObjHandle::CommandQueue(q) => Some(q),
                        _ => None,
                    }
                });
                self.queue_ended(queue, handle);
            }
            // A buffer answered on a doorbell's thread and never begun gives back its turn and
            // its ticket in the commit order here, not with its queue's entry where the close
            // worker lets go of the objects: other contexts' commits marked after the ticket wait
            // for nothing of the closed context's. Only entries no view holds were handed back,
            // so no begin can still take such an answer ([`tables`]).
            for r in &removed {
                if let Some(counts) = &r.entry().queue {
                    drop(counts.answered.lock().take());
                }
            }
            objects = removed;
        }
        self.forget_context_surfaces(context);
        // Let go of with no lock held: a view may hold the last of an import of the guest's
        // pages, which the platform may take long to let go of, and every context's surface
        // pages take the views' lock.
        let views = self.shared.surface_views.lock().context_closed(context);
        let libraries = self.shared.libraries.lock().ids_of(context);
        for id in libraries {
            if let Err(refusal) = refusals::refusing(|| self.forget_library(id)) {
                refused_release(id, &refusal);
            }
        }
        let closed = Closed {
            context,
            objects,
            views,
        };
        (dropped, closed)
    }

    /// A context is gone, at its close or the device's reset, and with it everything that could
    /// set or signal its events: no event of the guest's crosses processes, and its queues
    /// have ended. A command buffer of its that waits for one of their values not yet reached
    /// would wait for good, holding its queue, every object it names, the GPU addresses its
    /// buffers pin among them, and its context's drain, which the close's answer waits for.
    /// So each event the context's table `state` holds is set past every value, before the
    /// close takes it out, as a view a walk holds open keeps what is taken out from the close:
    /// a shared event on the CPU, and the plain events, which only a command buffer signals, by
    /// one command buffer on the decoder's own queue ([`Shared::release_queue`]), which no
    /// buffer of the guest's holds up. A refusal is logged, and the close goes on.
    fn release_events(&self, device: &P::Device, state: &tables::Context<P>) {
        let context = state.id;
        let mut plain = 0usize;
        state.for_each_live(|id, entry| match &entry.handle {
            ObjHandle::SharedEvent(event) => {
                if let Err(raise) = refusing(|| event.set_signaled_value(u64::MAX)) {
                    log::warn!(
                        target: "reims_vgpu::graphics",
                        "context {context}'s shared event {id:#x} was not set past every value at \
                         its close: {raise}"
                    );
                }
            }
            ObjHandle::Event(_) => plain += 1,
            _ => {}
        });
        if plain == 0 {
            return;
        }
        let signalled = refusing(|| {
            let queue =
                (self.shared.release_queue).get_or_init(|| device.new_command_queue_unshared());
            let Some(cb) = (match queue {
                Some(queue) => queue.command_buffer(OnFault::Return)?,
                None => None,
            }) else {
                return Ok(false);
            };
            let mut encoded = Ok(());
            state.for_each_live(|_, entry| {
                if let ObjHandle::Event(event) = &entry.handle
                    && encoded.is_ok()
                {
                    encoded = cb.encode_signal(Some(EventRef::Event(event)), u64::MAX);
                }
            });
            encoded?;
            cb.commit(OnFault::Return)?;
            Ok(true)
        });
        match signalled {
            Ok(true) => {}
            Ok(false) => log::warn!(
                target: "reims_vgpu::graphics",
                "context {context}'s {plain} events were not signalled past every value at its \
                 close: the platform made no command buffer"
            ),
            Err(raise) => log::warn!(
                target: "reims_vgpu::graphics",
                "context {context}'s {plain} events were not signalled past every value at its \
                 close: {raise}"
            ),
        }
    }

    /// Lets go of `closed` on the close worker, then of `drain`, its close's: the close's answer,
    /// which the drain gives once its command buffers have finished too, comes once its objects
    /// are let go of, as the kext gives the context's texture resource ids and GPU addresses out
    /// again at the answer.
    pub(crate) fn let_go_later(&self, closed: Closed<P>, drain: Arc<Drain>) {
        self.shared.close_worker.submit(
            u64::from(closed.context),
            Box::new(move || {
                closed.let_go();
                drop(drain);
            }),
        );
    }

    /// Waits for the closes handed to the close worker before this to have let go of their
    /// objects ([`Self::let_go_later`]).
    pub(crate) fn closes_let_go(&self) {
        self.shared.close_worker.flush();
    }

    /// The device's reset (`reims_graphics::Host::device_reset`): every context's objects dropped
    /// as at its close ([`Self::drop_context`]), the lowest context first, then what the decoder
    /// keeps beside them for the guest's processes: the surfaces' names, the libraries' chunks,
    /// makes and cache, the blocks' views of their surfaces' pages and their device caches
    /// (dropped uncopied: the pages are the rebooted guest's again), the queues' counts, and every
    /// context's close, chunks of the texture resource ids and drains (their answers are owed
    /// to no one). Each release is made in a refusal scope. The decoder then holds what one just
    /// made holds, so a texture's resource id and a sampler's are free for the guest's next use,
    /// as on a new device. The contexts dropped, in order, and how many objects went.
    ///
    /// It is made with no message of the guest's being carried out on the decoder, as the
    /// device's reset waits for those that began before it (`GraphicsControl::reset`), so no
    /// view of a table is open and every object goes here.
    pub(crate) fn reset(&self, device: &P::Device) -> (Vec<u32>, usize) {
        let shared = &self.shared;
        let mut contexts: Vec<u32> = shared.contexts.lock().keys().copied().collect();
        contexts.extend(shared.libraries.lock().contexts());
        contexts.extend(shared.id_chunks.lock().iter().flatten().copied());
        contexts.sort_unstable();
        contexts.dedup();
        let dropped = contexts.iter().map(|&c| self.drop_context(device, c)).sum();
        *shared.surfaces.lock() = SurfaceNames::default();
        // Their drains go with them, those of the contexts closed and of the unshared blocks too:
        // the answers waiting on those are owed to no one.
        shared.closing.lock().clear();
        shared.unshared.lock().clear();
        let states = std::mem::take(&mut *shared.contexts.lock());
        if let Err(refusal) = refusals::refusing(move || drop(states)) {
            log::warn!(
                target: "reims_vgpu::graphics",
                "the contexts' tables were let go of at the device's reset, and a release was \
                 refused: {refusal}"
            );
        }
        shared.closed.forget_all();
        let cache = shared.libraries.lock().take_cache();
        if let Err(refusal) = refusals::refusing(move || drop(cache)) {
            log::warn!(
                target: "reims_vgpu::graphics",
                "the library cache was let go of at the device's reset, and a release was \
                 refused: {refusal}"
            );
        }
        shared.surface_views.lock().reset();
        self.surface_caches.reset();
        shared.surface_order.lock().reset();
        (contexts, dropped)
    }

    /// Whether texture resource id `id` lies in a chunk that the context of object
    /// `obj_id` holds: never 0, and always below [`RGFX_RESOURCE_ID_LIMIT`].
    fn holds_resource_id(&self, obj_id: u64, id: u64) -> bool {
        if id == 0 || id >= RGFX_RESOURCE_ID_LIMIT {
            return false;
        }
        let context = obj_id >> RGFX_CONTEXT_OBJ_ID_SHIFT;
        self.shared
            .id_chunks
            .lock()
            .get(((id - 1) / RGFX_RESOURCE_ID_SMALL_CHUNK) as usize)
            .copied()
            .flatten()
            .is_some_and(|holder| u64::from(holder) == context)
    }

    /// Gives context `context` chunk `chunk` of the texture resource ids, as a
    /// `RESOURCE_ID_CHUNK` does.
    #[cfg(test)]
    pub(crate) fn hold_resource_ids(&mut self, context: u32, chunk: u32) {
        let small = small_chunks(RGFX_CMD_RESOURCE_ID_CHUNK, chunk).unwrap();
        self.shared.id_chunks.lock()[small].fill(Some(context));
    }

    /// A `SAMPLER_CREATE`'s answer ([`sampler_answer`]), the sampler stored under the message's id
    /// before it is answered, so a command buffer of its process, which names it only after the
    /// answer, finds it. Its make holds no lock of the decoder's.
    pub(crate) fn sampler_state(
        &self,
        device: &P::Device,
        message: &[u8],
    ) -> RaiseResult<Option<RgfxSamplerState>> {
        sampler_answer::<P>(device, message, |obj_id, sampler| {
            self.store(obj_id, Entry::of(ObjHandle::Sampler(sampler)));
        })
    }

    /// The sampler of a `SAMPLER_CREATE`, stored under its id.
    fn make_sampler(
        &self,
        device: &P::Device,
        message: &[u8],
    ) -> RaiseResult<Result<(), DecodeError>> {
        match new_sampler::<P>(device, message)? {
            Ok((obj_id, sampler)) => self.insert(obj_id, ObjHandle::Sampler(sampler)),
            Err(e) => Ok(Err(e)),
        }
    }

    /// A `HEAP_TEXTURE_SIZE_AND_ALIGN`'s answer: `device`'s size and alignment for a texture of
    /// the message's shape (`heapTextureSizeAndAlignWithDescriptor:`), or, where the platform
    /// raises or refuses the shape, [`RGFX_STATUS_REFUSED`] with both 0, which the guest answers
    /// by a bound of its own. `None` for a message too short to carry its seqno, which no one can
    /// be answered by; a shorter body is answered [`RGFX_STATUS_INVALID_ARGUMENT`].
    pub(crate) fn heap_texture_size_and_align(
        &self,
        device: &P::Device,
        message: &[u8],
    ) -> Option<RgfxSizeAndAlign> {
        let header = read::<RgfxMsgHeader>(message)?;
        let answer = |status, size, align| RgfxSizeAndAlign {
            header: RgfxMsgHeader {
                r#type: RGFX_EVT_SIZE_AND_ALIGN,
                size: size_of::<RgfxSizeAndAlign>() as u32,
                seqno: header.seqno,
            },
            status,
            reserved_14: 0,
            size,
            align,
        };
        let Some(m) = read::<RgfxHeapTextureSizeAndAlign>(message) else {
            return Some(answer(RGFX_STATUS_INVALID_ARGUMENT, 0, 0));
        };
        // The shape alone: a texture's resource index is no part of its size.
        let desc = TextureDesc {
            resource_index: None,
            ..texture_desc(&m.desc, 0)
        };
        let asked = refusing(|| {
            let descriptor = self.platform.texture_descriptor(&desc);
            device.heap_texture_size_and_align(&descriptor)
        });
        Some(match asked {
            Ok(sa) => answer(RGFX_STATUS_OK, sa.size, sa.align),
            Err(raise) => {
                log::warn!(
                    target: "reims_vgpu::graphics",
                    "heap texture size of format {}, type {}, {}x{}x{} not answered: {raise}",
                    m.desc.pixel_format,
                    m.desc.texture_type,
                    m.desc.width,
                    m.desc.height,
                    m.desc.depth
                );
                answer(RGFX_STATUS_REFUSED, 0, 0)
            }
        })
    }

    /// [`Self::decode`] in the lane's refusal scope ([`refusing`]), as ring 0 carries each message
    /// out: where a create the bundle answered at the call without the host's answer is not
    /// carried out (the platform makes nothing or dies at it, or it names an object the host
    /// does not hold, a refused one among them), its id stands for a refused object from then on
    /// ([`Self::refuse`]), so what names it is dropped, not the command buffers that name it.
    pub(crate) fn decode_refusing(
        &mut self,
        device: &P::Device,
        message: &[u8],
    ) -> RaiseResult<Result<(), DecodeError>> {
        let outcome = refusing(|| self.decode(device, message));
        let carried = matches!(outcome, Ok(Ok(()) | Err(DecodeError::ContextClosed)));
        if !carried && let Some((kind, obj_id)) = refused::posted_create(message) {
            self.refuse(obj_id, kind);
        }
        outcome
    }

    /// `obj_id`, made by a posted create of message type `kind` the host did not carry out,
    /// stands for a refused object ([`refused`]) until its destroy, its next create, its
    /// context's close or the device's reset, in place of what it named: an id of a context
    /// other than the message's own is left as it is.
    pub(crate) fn refuse(&self, obj_id: u64, kind: u32) {
        if obj_id == 0 || !self.owned(obj_id) {
            return;
        }
        log::debug!(
            target: "reims_vgpu::graphics",
            "object {obj_id:#x} stands refused (create type {kind:#x}): what names it is dropped"
        );
        // As a create made again names only its new texture's surface.
        if kind == RGFX_CMD_TEXTURE_FROM_IOSURFACE {
            self.forget_surface_texture(obj_id);
        }
        self.store(
            obj_id,
            Entry::of(ObjHandle::Refused(refused::Refused::of(kind))),
        );
    }

    /// Carries out one wire message against `device`: a create inserts the new
    /// handle (keyed by the guest-picked `obj_id`), a destroy drops it. The
    /// inner `Result` says whether the message was carried out; a platform raise
    /// propagates through the outer `RaiseResult`.
    pub fn decode(
        &mut self,
        device: &P::Device,
        message: &[u8],
    ) -> RaiseResult<Result<(), DecodeError>> {
        // A new message: what the last named is let go of.
        self.end_message();
        let Some(header) = read::<RgfxMsgHeader>(message) else {
            return Ok(Err(DecodeError::Malformed));
        };
        match header.r#type {
            RGFX_CMD_OBJECT_DESTROY => {
                let Some(m) = read::<RgfxObjectDestroy>(message) else {
                    return Ok(Err(DecodeError::Malformed));
                };
                // A missing id is not an error: the guest may destroy an object
                // the host never made (a create that the platform refused). Each release is
                // refused alone where the platform dies at it, so the destroy is carried out
                // whole ([`drop_refused`]).
                if let Err(refusal) = refusals::refusing(|| self.forget_library(m.obj_id)) {
                    refused_release(m.obj_id, &refusal);
                }
                // A queue's surface timeline signals what other queues wait for of it. Read from
                // the table as it is, so the destroy takes no view of it.
                let queue = (self.context_state(context_of(m.obj_id))).and_then(|c| {
                    c.with_live(m.obj_id, |e| match &e.handle {
                        ObjHandle::CommandQueue(q) => Some(q.clone_ref()),
                        _ => None,
                    })
                    .flatten()
                });
                if let Some(queue) = &queue {
                    self.queue_ended(m.obj_id, Some(queue));
                }
                self.remove(m.obj_id);
                Ok(Ok(()))
            }
            RGFX_CMD_RESOURCE_ID_CHUNK | RGFX_CMD_RESOURCE_ID_SMALL_CHUNK => {
                let Some(m) = read::<RgfxResourceIdChunk>(message) else {
                    return Ok(Err(DecodeError::Malformed));
                };
                // The kext's grant: the context holds the chunk's small chunks until its
                // close. A chunk past the pool is not one.
                let Some(small) = small_chunks(header.r#type, m.chunk) else {
                    return Ok(Err(DecodeError::BadResourceId));
                };
                self.shared.id_chunks.lock()[small].fill(Some(m.context));
                Ok(Ok(()))
            }
            RGFX_CMD_EVENT_CREATE => {
                let Some(m) = read::<RgfxObjectCreate>(message) else {
                    return Ok(Err(DecodeError::Malformed));
                };
                match device.new_event(OnFault::Return)? {
                    Some(event) => self.insert(m.obj_id, ObjHandle::Event(event)),
                    None => Ok(Err(DecodeError::Refused)),
                }
            }
            RGFX_CMD_FENCE_CREATE => {
                let Some(m) = read::<RgfxObjectCreate>(message) else {
                    return Ok(Err(DecodeError::Malformed));
                };
                match device.new_fence()? {
                    Some(fence) => self.insert(m.obj_id, ObjHandle::Fence(fence)),
                    None => Ok(Err(DecodeError::Refused)),
                }
            }
            RGFX_CMD_SHARED_EVENT_CREATE => {
                let Some(m) = read::<RgfxSharedEventCreate>(message) else {
                    return Ok(Err(DecodeError::Malformed));
                };
                // A new shared event's value is 0: one made at another starts there.
                let Some(event) = device.new_shared_event() else {
                    return Ok(Err(DecodeError::Refused));
                };
                if m.initial_value != 0 {
                    event.set_signaled_value(m.initial_value)?;
                }
                self.insert(m.obj_id, ObjHandle::SharedEvent(event))
            }
            RGFX_CMD_SHARED_EVENT_SET_VALUE => {
                let Some(m) = read::<RgfxSharedEventSetValue>(message) else {
                    return Ok(Err(DecodeError::Malformed));
                };
                // `setSignaledValue:` from the guest's CPU; the platform's value only rises, as
                // Metal's does.
                let event = match self.named(m.obj_id) {
                    Some(ObjHandle::SharedEvent(event)) => event,
                    // One whose create the host refused: its waits on the host are dropped, and
                    // the guest keeps its value ([`refused`]).
                    Some(ObjHandle::Refused(r)) => {
                        r.used(m.obj_id);
                        return Ok(Ok(()));
                    }
                    _ => return Ok(Err(DecodeError::BadReference)),
                };
                event.set_signaled_value(m.value)?;
                Ok(Ok(()))
            }
            RGFX_CMD_QUEUE_CREATE => {
                use reims_graphics::wire::{
                    RGFX_QUEUE_CREATE_FLAGS_OFFSET, RGFX_QUEUE_UNSHARED, RgfxQueueCreateFlags,
                };
                let Some(m) = read::<RgfxQueueCreate>(message) else {
                    return Ok(Err(DecodeError::Malformed));
                };
                // The ring pair this message places (`submit_offset`, `complete_offset`, the data
                // sizes and the vector) is the device's; here the queue handle is made so later
                // command buffers can name it. A queue whose tail holds `RGFX_QUEUE_UNSHARED`
                // (the device took it only where the driver took its feature, and the kext asks it
                // for WindowServer's queues) is the platform's queue that shares its GPU queue
                // with no other command queue.
                let unshared =
                    read_at::<RgfxQueueCreateFlags>(message, RGFX_QUEUE_CREATE_FLAGS_OFFSET)
                        .is_some_and(|tail| tail.flags & RGFX_QUEUE_UNSHARED != 0);
                let made = if unshared {
                    device.new_command_queue_unshared()
                } else {
                    device.new_command_queue()
                };
                if unshared && made.is_some() {
                    log::info!(
                        target: "reims_vgpu::graphics",
                        "queue {:#x}: made unshared (RGFX_QUEUE_UNSHARED)", m.obj_id
                    );
                }
                match made {
                    Some(queue) => {
                        // Its surface timeline and its device caches' end event (and the caches'
                        // transfer queue, made with the first queue) are made here, on the queue's
                        // own thread before its first buffer, with no lock held, rather than at
                        // that buffer's finish under the locks every queue's buffers take
                        // ([`surface_order::SurfaceOrder::made`], [`Caches::queue_made`]). One not
                        // made here is made there, as before.
                        let timeline = device.new_shared_event();
                        if let Err(raise) = self.surface_caches.queue_made(device, m.obj_id) {
                            log::debug!(
                                target: "reims_vgpu::graphics",
                                "queue {:#x}: its end event was not made with it: {raise}",
                                m.obj_id
                            );
                        }
                        let mut entry = Entry::of(ObjHandle::CommandQueue(queue));
                        let (order, spare) =
                            self.shared.surface_order.lock().made(m.obj_id, timeline);
                        if let Some(spare) = spare {
                            drop_refused(m.obj_id, spare);
                        }
                        let lane = Lane::new(m.obj_id, Arc::clone(&self.shared.lanes));
                        entry.queue = Some(QueueCounts::on(lane, order));
                        self.store(m.obj_id, entry);
                        Ok(Ok(()))
                    }
                    None => Ok(Err(DecodeError::Refused)),
                }
            }
            RGFX_CMD_LIBRARY_CREATE | RGFX_CMD_LIBRARY_FROM_REGION => {
                self.decode_library(device, message)
            }
            RGFX_CMD_SAMPLER_CREATE => self.make_sampler(device, message),
            RGFX_CMD_ICB_CREATE => Ok(self.new_icb(device, message, false)?.map(|_| ())),
            RGFX_CMD_ICB_WRITE => self.icb_write(message),
            RGFX_CMD_ICB_SLOTS => self.icb_slots(message),
            RGFX_CMD_DEPTH_STENCIL_CREATE => {
                let Some(m) = read::<RgfxDepthStencilCreate>(message) else {
                    return Ok(Err(DecodeError::Malformed));
                };
                let desc = DepthStencilDesc {
                    depth_compare_function: Some(m.depth_compare.into()),
                    depth_write_enabled: Some(m.depth_write_enabled != 0),
                    front: Some(stencil(&m.front)),
                    back: Some(stencil(&m.back)),
                };
                match device.new_depth_stencil_state(&desc)? {
                    Some(state) => self.insert(m.obj_id, ObjHandle::DepthStencil(state)),
                    None => Ok(Err(DecodeError::Refused)),
                }
            }
            RGFX_CMD_BUFFER_CREATE => {
                let Some(m) = read::<RgfxBufferCreate>(message) else {
                    return Ok(Err(DecodeError::Malformed));
                };
                // A device-placed (Private/Shared-on-UMA) buffer. The guest picks
                // the GPU address and the host pins it; a 0 address lets the host
                // choose. `options` carries the whole MTLResourceOptions.
                let desc = BufferDesc {
                    length: m.length,
                    options: vec![OptionSet::ResourceOptions(m.options.into())],
                    pinned_gpu_address: (m.gpu_address != 0).then_some(m.gpu_address),
                    contents: None,
                    no_copy: false,
                    deallocator: None,
                };
                match device.new_buffer(desc)? {
                    Some(buffer) => self.insert(m.obj_id, ObjHandle::Buffer(buffer)),
                    None => Ok(Err(DecodeError::Refused)),
                }
            }
            RGFX_CMD_BUFFER_OVER_HOST_MEMORY => {
                let Some(m) = read::<RgfxBufferOverHostMemory>(message) else {
                    return Ok(Err(DecodeError::Malformed));
                };
                // A Shared buffer placed over the guest's own memory in the
                // device's region, `length` bytes at `region_offset`: the same
                // zero-copy import the scanout uses, so a blit reads and writes the
                // guest's pages in place. A span in no block of the sender's
                // (`in_own_block`) is a guest protocol error.
                if !self.in_own_block(m.region_offset, m.length) {
                    return Ok(Err(DecodeError::BadReference));
                }
                // A `Managed` one's block is its CPU copy ([`managed`]).
                if storage_mode(m.options.into()) == STORAGE_MODE_MANAGED {
                    return self.managed_buffer(device, &m);
                }
                // At the guest's address, which its pointers to the buffer hold, as a
                // BUFFER_CREATE's is.
                let pin = (m.gpu_address != 0).then_some(m.gpu_address);
                let made = self.buffer_over_block(
                    device,
                    m.region_offset,
                    m.length,
                    m.options.into(),
                    pin,
                )?;
                let Some(buffer) = made else {
                    return Ok(Err(DecodeError::Refused));
                };
                // Its contents are its block's bytes.
                let bytes = GuestBytes {
                    ram: None,
                    at: m.region_offset,
                    length: m.length,
                };
                let entry = Entry::of(ObjHandle::Buffer(buffer))
                    .over(Some((m.region_offset, m.length)))
                    .with_bytes(Some(bytes));
                self.store(m.obj_id, entry);
                Ok(Ok(()))
            }
            RGFX_CMD_BUFFER_OVER_SURFACE_PAGES => self.buffer_over_surface_pages(device, message),
            RGFX_CMD_BUFFER_MANAGED_AT_OFFSET => self.managed_buffer_at_offset(device, message),
            RGFX_CMD_HEAP_OVER_PAGES => self.heap_over_pages(device, message),
            RGFX_CMD_BUFFER_OVER_PAGES => self.buffer_over_pages(device, message),
            RGFX_CMD_BUFFER_DID_MODIFY_RANGE => {
                let drain = self.command_buffer_drain();
                self.did_modify_range(message, drain)
            }
            RGFX_CMD_MANAGED_TEXTURE_CREATE => self.managed_texture(device, message),
            RGFX_CMD_MANAGED_BUFFER_CREATE => self.managed_buffer_create(device, message),
            RGFX_CMD_TEXTURE_REPLACE_REGION => {
                let drain = self.command_buffer_drain();
                self.replace_region(message, drain)
            }
            RGFX_CMD_TEXTURE_VIEW => self.texture_view(message),
            RGFX_CMD_HEAP_CREATE => {
                let Some(m) = read::<RgfxHeapCreate>(message) else {
                    return Ok(Err(DecodeError::Malformed));
                };
                // A device-placed heap at a guest-picked GPU address (0 lets the
                // host choose). Heaps over host memory land with that region's decode.
                let desc = HeapDesc {
                    size: m.size_bytes,
                    heap_type: host_heap_type(m.heap_type),
                    options: vec![OptionSet::ResourceOptions(m.options.into())],
                    address_ranges: None,
                    pinned_gpu_address: (m.gpu_address != 0).then_some(m.gpu_address),
                };
                match device.new_heap(&desc)?.heap {
                    Some(heap) => self.insert(m.obj_id, ObjHandle::Heap(heap)),
                    None => {
                        heap_refused(
                            m.obj_id,
                            m.size_bytes,
                            m.heap_type,
                            m.options,
                            m.gpu_address,
                        );
                        Ok(Err(DecodeError::Refused))
                    }
                }
            }
            RGFX_CMD_HEAP_OVER_HOST_MEMORY => {
                let Some(m) = read::<RgfxHeapOverHostMemory>(message) else {
                    return Ok(Err(DecodeError::Malformed));
                };
                // A Shared heap over the guest's own memory in the device's region, in place:
                // its memory is the `size_bytes` at `region_offset`, rounded up to Metal's 16 KiB
                // page, so a resource the guest places in it lies in the guest's bytes at its
                // offset, which the CPU and the GPU share, as Metal's heap does. A span in no
                // block of the sender's is a guest protocol error. The region stays mapped while
                // the decoder holds it, and the heap (in the object table) drops before the
                // region (field order).
                let Some(length) = m.size_bytes.checked_next_multiple_of(HEAP_PAGE) else {
                    return Ok(Err(DecodeError::BadReference));
                };
                if !self.in_own_block(m.region_offset, length) {
                    return Ok(Err(DecodeError::BadReference));
                }
                // SAFETY: the block lies inside the region (checked above), which stays mapped
                // while the decoder, and so the heap, lives.
                let bytes = unsafe {
                    HostRange::new(
                        HostPtr::from_ptr(self.region.base().add(m.region_offset as usize)),
                        length,
                    )
                };
                let desc = HeapDesc {
                    size: m.size_bytes,
                    heap_type: host_heap_type(m.heap_type),
                    options: vec![OptionSet::ResourceOptions(m.options.into())],
                    address_ranges: None,
                    pinned_gpu_address: (m.gpu_address != 0).then_some(m.gpu_address),
                };
                match device.new_heap_over_host_memory(bytes, &desc)?.heap {
                    Some(heap) => self.insert_over(
                        m.obj_id,
                        ObjHandle::Heap(heap),
                        Some((m.region_offset, length)),
                    ),
                    None => {
                        heap_refused(
                            m.obj_id,
                            m.size_bytes,
                            m.heap_type,
                            m.options,
                            m.gpu_address,
                        );
                        Ok(Err(DecodeError::Refused))
                    }
                }
            }
            RGFX_CMD_TEXTURE_CREATE => {
                let Some(m) = read::<RgfxTextureCreate>(message) else {
                    return Ok(Err(DecodeError::Malformed));
                };
                if !self.holds_resource_id(m.obj_id, m.resource_id) {
                    return Ok(Err(DecodeError::BadResourceId));
                }
                // The guest picks the resource id, pinned as the texture's
                // resource index so its argument buffers name the same texture
                // the host made. `texture_descriptor` forwards the raw pixel
                // format, so a format Metal refuses aborts here; the guest bundle
                // only ever sends one it read from a real descriptor.
                let desc = texture_desc(&m.desc, m.resource_id);
                let descriptor = self.platform.texture_descriptor(&desc);
                let span = self.span_of(m.heap_obj_id);
                let made = if m.heap_obj_id != 0 {
                    // Heap-placed at `heap_offset` in the named heap; a heap the
                    // host does not hold is a guest protocol error.
                    let Some(ObjHandle::Heap(heap)) = self.named(m.heap_obj_id) else {
                        return Ok(Err(DecodeError::BadReference));
                    };
                    heap.new_texture(&descriptor, m.heap_offset)?
                } else {
                    // Device-placed at the guest-picked GPU address.
                    device.new_texture(OnFault::Return, &descriptor)?
                };
                match made {
                    Some(texture) => self.insert_over(m.obj_id, ObjHandle::Texture(texture), span),
                    None => Ok(Err(DecodeError::Refused)),
                }
            }
            RGFX_CMD_BUFFER_AT_OFFSET => {
                let Some(m) = read::<RgfxBufferAtOffset>(message) else {
                    return Ok(Err(DecodeError::Malformed));
                };
                // A buffer at an offset in an existing heap; a heap the host does
                // not hold is a guest protocol error.
                let made = {
                    let Some(ObjHandle::Heap(heap)) = self.named(m.heap_obj_id) else {
                        return Ok(Err(DecodeError::BadReference));
                    };
                    heap.new_buffer_of_length(m.length, m.options.into(), m.offset)?
                };
                let Some(buffer) = made else {
                    return Ok(Err(DecodeError::Refused));
                };
                // In a heap over the guest's memory, its contents are the heap's bytes at its
                // offset: the guest's RAM where the heap lies over a pages block's, else the
                // heap's block of the region. A heap of the GPU's own memory has neither.
                let (span, ram) = (self.span_of(m.heap_obj_id), self.ram_of(m.heap_obj_id));
                let bytes = match (ram, span) {
                    (Some((first, _)), _) => Some(GuestBytes {
                        ram: Some(first),
                        at: m.offset,
                        length: m.length,
                    }),
                    (None, Some((at, _))) => Some(GuestBytes {
                        ram: None,
                        at: at + m.offset,
                        length: m.length,
                    }),
                    (None, None) => None,
                };
                let entry = Entry::of(ObjHandle::Buffer(buffer))
                    .over(span)
                    .with_bytes(bytes);
                self.store(m.obj_id, entry);
                Ok(Ok(()))
            }
            RGFX_CMD_TEXTURE_FROM_BUFFER => {
                let Some(m) = read::<RgfxTextureFromBuffer>(message) else {
                    return Ok(Err(DecodeError::Malformed));
                };
                if !self.holds_resource_id(m.obj_id, m.resource_id) {
                    return Ok(Err(DecodeError::BadResourceId));
                }
                let desc = texture_desc(&m.desc, m.resource_id);
                let descriptor = self.platform.texture_descriptor(&desc);
                // The texture reads the guest's named buffer; a buffer the host
                // does not hold is a guest protocol error, not a refusal.
                let made = {
                    let Some(ObjHandle::Buffer(buffer)) = self.named(m.buffer_obj_id) else {
                        return Ok(Err(DecodeError::BadReference));
                    };
                    let made = buffer.new_texture(
                        OnFault::Return,
                        &descriptor,
                        m.offset,
                        m.bytes_per_row,
                    )?;
                    if made.is_none() {
                        log_refused_texture(&TextureOverBuffer {
                            obj_id: m.obj_id,
                            buffer_obj_id: m.buffer_obj_id,
                            buffer_length: buffer.length(),
                            desc: &m.desc,
                            offset: m.offset,
                            bytes_per_row: m.bytes_per_row,
                            bytes_per_image: None,
                            surface: None,
                        });
                    }
                    made
                };
                let span = self.span_of(m.buffer_obj_id);
                match made {
                    Some(texture) => self.insert_over(m.obj_id, ObjHandle::Texture(texture), span),
                    None => Ok(Err(DecodeError::Refused)),
                }
            }
            RGFX_CMD_LINEAR_TEXTURE_FROM_BUFFER => {
                let Some(m) = read::<RgfxLinearTextureFromBuffer>(message) else {
                    return Ok(Err(DecodeError::Malformed));
                };
                if !self.holds_resource_id(m.obj_id, m.resource_id) {
                    return Ok(Err(DecodeError::BadResourceId));
                }
                let desc = texture_desc(&m.desc, m.resource_id);
                let descriptor = self.platform.texture_descriptor(&desc);
                // As TEXTURE_FROM_BUFFER's, by the platform's linear form, which takes a 2D
                // array and the bytes between its slices.
                let made = {
                    let Some(ObjHandle::Buffer(buffer)) = self.named(m.buffer_obj_id) else {
                        return Ok(Err(DecodeError::BadReference));
                    };
                    let made = buffer.new_linear_texture(
                        &descriptor,
                        m.offset,
                        m.bytes_per_row,
                        m.bytes_per_image,
                    )?;
                    if made.is_none() {
                        log_refused_texture(&TextureOverBuffer {
                            obj_id: m.obj_id,
                            buffer_obj_id: m.buffer_obj_id,
                            buffer_length: buffer.length(),
                            desc: &m.desc,
                            offset: m.offset,
                            bytes_per_row: m.bytes_per_row,
                            bytes_per_image: Some(m.bytes_per_image),
                            surface: None,
                        });
                    }
                    made
                };
                let span = self.span_of(m.buffer_obj_id);
                match made {
                    Some(texture) => self.insert_over(m.obj_id, ObjHandle::Texture(texture), span),
                    None => Ok(Err(DecodeError::Refused)),
                }
            }
            RGFX_CMD_TEXTURE_FROM_IOSURFACE => {
                let Some(m) = read::<RgfxTextureFromIOSurface>(message) else {
                    return Ok(Err(DecodeError::Malformed));
                };
                let facts = match surface_facts(&m, message) {
                    Ok(facts) => facts,
                    Err(e) => return Ok(Err(e)),
                };
                let plane = match surface_plane(&m, facts.as_ref()) {
                    Ok(plane) => plane,
                    Err(e) => return Ok(Err(e)),
                };
                if !self.holds_resource_id(m.obj_id, m.resource_id) {
                    return Ok(Err(DecodeError::BadResourceId));
                }
                // Made by Metal's IOSurface creator over the buffer the guest keeps its copy
                // of the surface in, which stands for the surface: the plane lies in it as
                // TEXTURE_FROM_BUFFER places a texture, and plane 1 where the facts say. A
                // buffer the host does not hold is a guest protocol error.
                let mut desc = texture_desc(&m.desc, m.resource_id);
                if !matches!(self.named(m.buffer_obj_id), Some(ObjHandle::Buffer(_))) {
                    return Ok(Err(DecodeError::BadReference));
                }
                let span = self.span_of(m.buffer_obj_id);
                if let Some((at, _)) = span {
                    self.settle_surface(at);
                }
                // Over a device cache, a Private buffer, the texture is Private too, as Metal
                // makes a texture over a buffer in the buffer's storage mode; the guest's own
                // texture keeps the mode it asked (`surface_cache`).
                if span.is_some_and(|(at, _)| self.surface_caches.lock().has(at)) {
                    desc.resource_options = Some(
                        desc.resource_options.unwrap_or(0) & !surface_cache::STORAGE_MASK
                            | surface_cache::STORAGE_PRIVATE,
                    );
                }
                let descriptor = self.platform.texture_descriptor(&desc);
                // The copy's writers lie wholly inside the buffer, in the region.
                let writers = if m.writers_offset == RGFX_SURFACE_NO_WRITERS {
                    None
                } else {
                    let record = size_of::<RgfxSurfaceWriters>() as u64;
                    match span {
                        Some((at, length))
                            if m.writers_offset.is_multiple_of(8)
                                && m.writers_offset
                                    .checked_add(record)
                                    .is_some_and(|end| end <= length) =>
                        {
                            Some(at + m.writers_offset)
                        }
                        _ => return Ok(Err(DecodeError::Malformed)),
                    }
                };
                let made = {
                    let Some(ObjHandle::Buffer(buffer)) = self.named(m.buffer_obj_id) else {
                        return Ok(Err(DecodeError::BadReference));
                    };
                    let made = buffer.new_texture_over_surface_plane(
                        OnFault::Return,
                        &descriptor,
                        plane,
                    )?;
                    if made.is_none() {
                        log_refused_texture(&TextureOverBuffer {
                            obj_id: m.obj_id,
                            buffer_obj_id: m.buffer_obj_id,
                            buffer_length: buffer.length(),
                            desc: &m.desc,
                            offset: m.offset,
                            bytes_per_row: m.bytes_per_row,
                            bytes_per_image: None,
                            surface: Some((m.iosurface_id, m.plane, facts)),
                        });
                    }
                    made
                };
                match made {
                    Some(texture) => {
                        // An id made again names only its new texture's surface.
                        self.forget_surface_texture(m.obj_id);
                        let mut entry = Entry::of(ObjHandle::Texture(texture)).over(span);
                        let info = device.pixel_format_info(u64::from(m.desc.pixel_format));
                        entry.place = Some(video::PlanePlace {
                            buffer_obj_id: m.buffer_obj_id,
                            offset: m.offset,
                            bytes_per_row: m.bytes_per_row,
                            row_bytes: video::row_bytes(&info, m.desc.width, m.bytes_per_row),
                            width: u64::from(m.desc.width),
                            height: u64::from(m.desc.height),
                        });
                        self.store(m.obj_id, entry);
                        if m.plane == 0 {
                            let mut names = self.shared.surfaces.lock();
                            names
                                .by_surface
                                .entry(m.iosurface_id)
                                .or_default()
                                .push(m.obj_id);
                            if let Some(at) = writers {
                                names.writers.insert(m.obj_id, at);
                            }
                        }
                        Ok(Ok(()))
                    }
                    None => Ok(Err(DecodeError::Refused)),
                }
            }
            RGFX_CMD_VIDEO_SESSION_CREATE => {
                let Some(m) = read::<RgfxVideoSessionCreate>(message) else {
                    return Ok(Err(DecodeError::Malformed));
                };
                let Some(session) =
                    (self.shared.video.get()).and_then(|video| video.session(m.codec, m.direction))
                else {
                    log::warn!(
                        target: "reims_vgpu::graphics",
                        "video session {:#x} not made: the host has no codec {} in direction {}",
                        m.obj_id, m.codec, m.direction
                    );
                    return Ok(Err(DecodeError::Refused));
                };
                let hold = video::SessionHold::new(m.obj_id, session, &self.shared.video_worker);
                self.insert(m.obj_id, ObjHandle::VideoSession(hold))
            }
            RGFX_CMD_RENDER_PIPELINE_CREATE
            | RGFX_CMD_MESH_RENDER_PIPELINE_CREATE
            | RGFX_CMD_COMPUTE_PIPELINE_CREATE => self.decode_pipeline(device, message),
            RGFX_CMD_COMMAND_BUFFER => {
                let drain = self.command_buffer_drain();
                if drain.is_none() && self.context_closed() {
                    return Ok(Err(DecodeError::ContextClosed));
                }
                self.decode_command_buffer(device, message, drain)
            }
            _ => Ok(Err(DecodeError::UnknownCommand)),
        }
    }

    /// The current context's drain, which a command buffer of it holds until its completion
    /// handler has run. It is taken once a message, before the message's view of the context's
    /// table, so a drain begun after that view was taken waits for the buffer
    /// ([`Self::drain_context`]); a stand-in for the buffer holds the same. `None` with no
    /// context, or one with no state, whose buffer names none of its objects: it makes it none,
    /// as only a create does ([`Self::context_made`]). Taken under the states' lock, so a close
    /// either finds the drain taken or has taken the context's own.
    pub(crate) fn command_buffer_drain(&self) -> Option<Arc<Drain>> {
        if let Some(drain) = self.drain.borrow().as_ref() {
            return Some(Arc::clone(drain));
        }
        let context = self.context?;
        let drain = {
            let contexts = self.shared.contexts.lock();
            Arc::clone(&*contexts.get(&context)?.drain.lock())
        };
        *self.drain.borrow_mut() = Some(Arc::clone(&drain));
        Some(drain)
    }

    /// Carries out `op`, the platform's part of a range message of a `Managed` object (`what`
    /// `obj_id`), which reaches the platform's next submission, once each command buffer of the
    /// current context answered `CB_SUBMITTED` at its take before now is committed
    /// ([`sequencer`]): at once where none is out, else on the thread that commits the last of
    /// them, with no wait here. It holds `drain`, its context's, until it has run, so the
    /// context's close and its blocks' releases are answered after it; the context's command
    /// buffers begun after it are committed after it. A raise of the platform's at it is logged,
    /// as a message not carried out.
    fn after_submitted(
        &self,
        what: &'static str,
        obj_id: u64,
        drain: Option<Arc<Drain>>,
        op: impl FnOnce() -> RaiseResult<()> + Send + 'static,
    ) {
        let run = move || {
            if let Err(raise) = refusing(op) {
                log::warn!(
                    target: "reims_vgpu::graphics",
                    "{what} of object {obj_id:#x} was not carried out: the platform raised \
                     ({raise:?})"
                );
            }
            drop(drain);
        };
        match self.context {
            // While a command buffer of the context has parts out, after its last part
            // ([`split`]).
            Some(context) => {
                let gate = Arc::clone(&self.shared.split_gate);
                let run = Box::new(move || gate.range(context, Box::new(run)));
                self.shared.sequencer.after_context(context, run);
            }
            None => run(),
        }
    }

    /// Inserts a new object's handle under its guest-picked id.
    fn insert(&self, obj_id: u64, handle: ObjHandle<P>) -> RaiseResult<Result<(), DecodeError>> {
        self.insert_over(obj_id, handle, None)
    }

    /// Inserts a new object's handle under its guest-picked id, with the region span it lies over,
    /// where it lies over one.
    fn insert_over(
        &self,
        obj_id: u64,
        handle: ObjHandle<P>,
        span: Option<(u64, u64)>,
    ) -> RaiseResult<Result<(), DecodeError>> {
        self.store(obj_id, Entry::of(handle).over(span));
        Ok(Ok(()))
    }

    /// Stores `entry` under `obj_id`, in place of what the id named.
    fn store(&self, obj_id: u64, entry: Entry<P>) {
        self.forget_library(obj_id);
        self.put(obj_id, entry);
    }

    /// Stores `entry` under `obj_id` in its context's table, in place of what the id named, which
    /// is let go of where no view needs it; the message sees it from now on.
    fn put(&self, obj_id: u64, entry: Entry<P>) {
        let context = self.context_made(context_of(obj_id));
        let (at, replaced) = context.insert(obj_id, entry, self.view_at(&context));
        self.saw(&context, obj_id, Some(at));
        // As a store over a live id let go of what it named.
        drop(replaced.map(tables::Removed::into_entry));
    }

    /// [`Self::store`], where `live` answers, under the states' lock where the context has no
    /// state ([`Self::context_made_if`]) and under the table's lock, that the object is still to
    /// be stored: one `live` refuses is let go of, and makes its context no state.
    fn store_if(&self, obj_id: u64, entry: Entry<P>, live: impl Fn() -> bool) {
        let Some(context) = self.context_made_if(context_of(obj_id), &live) else {
            return;
        };
        if let Some((at, replaced)) =
            context.insert_if(obj_id, entry, self.view_at(&context), &live)
        {
            self.saw(&context, obj_id, Some(at));
            drop(replaced.map(tables::Removed::into_entry));
        }
    }

    /// [`Self::store`], for a create its context's lane handed to a worker
    /// ([`Self::open_for_create`] made the context's state then): where the context closed since,
    /// it has no state, the object is let go of, and the closed context is made no state again.
    fn store_late(&self, obj_id: u64, entry: Entry<P>) {
        let id_context = context_of(obj_id);
        let Some(context) =
            self.context_made_if(id_context, || !self.shared.closed.closed(id_context))
        else {
            return;
        };
        self.forget_library(obj_id);
        let (at, replaced) = context.insert(obj_id, entry, self.view_at(&context));
        self.saw(&context, obj_id, Some(at));
        drop(replaced.map(tables::Removed::into_entry));
    }

    /// The clock of the message's view of `context`'s table, where it took one: a make of the
    /// message's is named by that view ([`Self::saw`]), so it is kept for it ([`tables`]).
    fn view_at(&self, context: &Arc<Context<P>>) -> Option<u64> {
        let views = self.views.borrow();
        views
            .iter()
            .find(|v| Arc::ptr_eq(v.context(), context))
            .map(View::at)
    }

    /// The message's own change of `obj_id` in `context`'s table: made at `at`, or removed.
    fn saw(&self, context: &Arc<Context<P>>, obj_id: u64, at: Option<std::ptr::NonNull<Entry<P>>>) {
        self.recent.clear();
        for view in self.views.borrow().iter() {
            if Arc::ptr_eq(view.context(), context) {
                match at {
                    Some(at) => view.made(obj_id, at),
                    None => view.removed(obj_id),
                }
            }
        }
    }

    /// Removes what `obj_id` names (an `OBJECT_DESTROY`): its CPU copy and then its object are let
    /// go of, each in a refusal scope ([`drop_refused`]), at once, or where an open view names
    /// it, once the last such view closes ([`tables`]); a texture's surface names are forgotten. A
    /// buffer over a block's surface view is let go of on the surface worker, where it may be the
    /// last hold of the view's mapping ([`surface_pages`]), and a video session on the video worker
    /// ([`video::SessionHold`]).
    fn remove(&self, obj_id: u64) {
        let Some(context) = self.context_state(context_of(obj_id)) else {
            return;
        };
        self.saw(&context, obj_id, None);
        let Some((texture, removed)) = context.remove(obj_id) else {
            return;
        };
        if let Some(removed) = removed {
            let Entry {
                managed,
                handle,
                span,
                over_view,
                ..
            } = removed.into_entry();
            // Nothing else: a texture's drop gives back the resource id the guest picked, and a
            // buffer's over host memory the GPU address, which the guest's next create may take
            // at once.
            if over_view {
                // Behind its block's jobs, or its own where it has no block.
                let key = span.map_or(obj_id, |(offset, _)| offset);
                self.shared
                    .surface_worker
                    .drop_later(key, obj_id, (managed, handle));
            } else {
                if let Some(copy) = managed {
                    drop_refused(obj_id, copy);
                }
                drop_refused(obj_id, handle);
            }
        }
        if texture {
            self.forget_surface_texture(obj_id);
        }
    }

    /// The region span `obj_id`'s object lies over, where it is one the current context may name
    /// and lies over one.
    fn span_of(&self, obj_id: u64) -> Option<(u64, u64)> {
        self.entry(obj_id).and_then(|e| e.span)
    }

    /// The host bytes of the heap over a pages block's guest RAM `obj_id` names
    /// (`HEAP_OVER_PAGES`): its first byte and length.
    fn ram_of(&self, obj_id: u64) -> Option<(HostPtr, u64)> {
        self.entry(obj_id).and_then(|e| e.ram)
    }

    /// Where the contents of the buffer over the guest's memory `obj_id` names lie, where the
    /// current context may name it ([`GuestBytes`]).
    fn bytes_of(&self, obj_id: u64) -> Option<GuestBytes> {
        self.entry(obj_id).and_then(|e| e.bytes)
    }

    /// Whether an object the decoder holds lies in the `size` bytes at region offset `offset`:
    /// the block they make is not to be given out again while it does.
    pub(crate) fn span_in_use(&self, offset: u64, size: u64) -> bool {
        let contexts: Vec<Arc<Context<P>>> =
            self.shared.contexts.lock().values().cloned().collect();
        contexts.iter().any(|c| c.span_in(offset, size))
    }

    /// Begins a drain of `context`'s command buffers: `answer` is given once every one the decoder
    /// has begun for the context has finished its completion handler, those begun before an
    /// earlier drain among them, and not later ones. The drain is returned for the caller to drop
    /// with no lock held: the answer goes then where no buffer is in flight, else on the GPU
    /// thread of the last to finish. `closed` ends the context's drains and its state (its
    /// `CONTEXT_CLOSE`); else a new one begins for its later buffers, held by this one
    /// ([`Drain`]). A context with no state has no buffer to wait for, and is made none. A drain
    /// of a context closed before, whose close's drain has not ended (a release the kext posted
    /// behind the close), gives `answer` only once that one has ended too.
    pub(crate) fn drain_context(
        &self,
        context: u32,
        closed: bool,
        answer: DrainAnswer,
    ) -> Arc<Drain> {
        let (drain, closing) = {
            let mut contexts = self.shared.contexts.lock();
            if closed {
                let state = contexts.remove(&context);
                let drain: Arc<Drain> = state
                    .map(|state| std::mem::take(&mut *state.drain.lock()))
                    .unwrap_or_default();
                // Under the states' lock, so no state is made for it after this.
                self.shared.closed.close(context);
                let mut closing = self.shared.closing.lock();
                closing.retain(|_, d| d.strong_count() > 0);
                closing.insert(context, Arc::downgrade(&drain));
                (drain, None)
            } else {
                let drain = match contexts.get(&context) {
                    Some(state) => {
                        let mut current = state.drain.lock();
                        let next = Arc::new(Drain::default());
                        *current.later.lock() = Some(Arc::clone(&next));
                        std::mem::replace(&mut *current, next)
                    }
                    None => Arc::default(),
                };
                (drain, self.shared.closing.lock().get(&context).cloned())
            }
        };
        let answer = match closing {
            // At the drain's end: where the close's drain has not ended, its end gives it.
            Some(closing) => Box::new(move || match closing.upgrade() {
                Some(close) => close.answers.lock().push(answer),
                None => answer(),
            }),
            None => answer,
        };
        drain.answers.lock().push(answer);
        drain
    }

    /// `BLOCK_UNSHARE` by `context` of the shared block at `block`, which other contexts still
    /// hold, or its hold's end at its close or at the block's release where its unshare had not
    /// come: the block's release is answered only once every command buffer the decoder began for
    /// the context before this has finished as well ([`Self::after_unshares`]), since the kext
    /// gives the block's memory back at that answer. The drain is returned for the caller to drop
    /// with no lock held, as [`Self::drain_context`]'s is.
    pub(crate) fn block_unshared(&self, context: u32, block: u64) -> Arc<Drain> {
        let gate = Arc::clone(self.shared.unshared.lock().entry(block).or_default());
        self.drain_context(context, false, Box::new(move || drop(gate)))
    }

    /// `answer`, the answer to the release of the block at `block`, given once every command
    /// buffer its earlier holders began before their `BLOCK_UNSHARE` has finished too
    /// ([`Self::block_unshared`]): `answer` itself where none let the block go.
    pub(crate) fn after_unshares(&self, block: u64, answer: DrainAnswer) -> DrainAnswer {
        let Some(gate) = self.shared.unshared.lock().remove(&block) else {
            return answer;
        };
        gate.answers.lock().push(answer);
        Box::new(move || drop(gate))
    }

    /// `RENDER_PIPELINE_CREATE` and `COMPUTE_PIPELINE_CREATE`, carried out at once: the create is
    /// prepared ([`Self::prepare_pipeline`]), compiled ([`compile_pipeline`]) and stored
    /// ([`Self::finish_pipeline`]), and its reflection reply goes to the reflection sink. The
    /// device's host runs the same three steps with the compile on a thread of its own.
    fn decode_pipeline(
        &mut self,
        device: &P::Device,
        message: &[u8],
    ) -> RaiseResult<Result<(), DecodeError>> {
        let job = match self.prepare_pipeline(message) {
            Ok(job) => job,
            Err(e) => {
                if let (Some(sink), Some(refused)) =
                    (&self.sinks.reflection, refused_reply(message))
                {
                    sink(PipelineReply::of(refused));
                }
                return Ok(Err(e));
            }
        };
        let compiled = compile_refusing::<P>(device, job);
        let refused = compiled.is_refused();
        if let Some(sink) = self.sinks.reflection.clone() {
            sink(compiled.reply());
        }
        self.finish_pipeline(compiled);
        Ok(if refused {
            Err(DecodeError::Refused)
        } else {
            Ok(())
        })
    }

    /// The first step of a pipeline create, with the table at hand: its libraries resolved from
    /// the table, its entry points, constants and whole descriptor read from the message, so the
    /// compile needs nothing of the decoder's. A library reference that names no library is a
    /// guest protocol error, as the other creates treat one ([`DecodeError::BadReference`]); a
    /// span the message does not cover, a name that is not UTF-8, more than eight colour
    /// attachments, a vertex or stage-input buffer or attribute index past Metal's last (30), or a
    /// linked function in no list Metal has, is [`DecodeError::Malformed`]. Neither is answered. A
    /// render create naming library 0 and no name for its fragment function has none (a pipeline
    /// that does not rasterize).
    ///
    /// Every field the wire carries reaches the platform's creator: the render form's functions,
    /// vertex descriptor, colour attachments (format, blending, factors, operations, write mask),
    /// depth and stencil formats, raster sample count, alpha-to-coverage and alpha-to-one,
    /// rasterization, input primitive topology, maximum vertex amplification count, indirect
    /// command buffer support, the vertex and fragment stages' linked functions, and its label;
    /// the compute form's function, threadgroup-size multiple, maximum threads per threadgroup,
    /// maximum call stack depth, indirect command buffer support, stage-input descriptor and
    /// linked functions. Each function is specialised by the index-keyed constants the wire
    /// carries for it, and each linked function made with the options the guest made it with.
    pub(crate) fn prepare_pipeline(&self, message: &[u8]) -> Result<PipelineJob<P>, DecodeError> {
        let library = |obj_id: u64| self.library_ref(obj_id);
        let header = read::<RgfxMsgHeader>(message).ok_or(DecodeError::Malformed)?;
        if header.r#type == RGFX_CMD_COMPUTE_PIPELINE_CREATE {
            let m = read::<RgfxComputePipelineCreate>(message).ok_or(DecodeError::Malformed)?;
            let library = library(m.function_library_obj_id)?;
            let name =
                str_tail(message, m.name_offset, m.name_len).ok_or(DecodeError::Malformed)?;
            let constants = constants_tail(message, m.constants_offset, m.constant_count)
                .ok_or(DecodeError::Malformed)?;
            let linked = self.linked_tail(message, m.linked_offset, m.linked_count)?;
            let stage_input = stage_input(message, &m)?;
            return Ok(PipelineJob {
                obj_id: m.obj_id,
                seqno: m.header.seqno,
                shape: PipelineShape::Compute {
                    library,
                    name: name.to_owned(),
                    constants,
                    linked,
                    fields: ComputeFields {
                        threadgroup_size_is_multiple_of_execution_width: m.flags
                            & RGFX_COMPUTE_PIPELINE_TG_MULTIPLE_OF_EXEC_WIDTH
                            != 0,
                        max_total_threads_per_threadgroup: m
                            .max_total_threads_per_threadgroup
                            .into(),
                        max_call_stack_depth: m.max_call_stack_depth.into(),
                        support_indirect_command_buffers: m.flags
                            & RGFX_COMPUTE_PIPELINE_SUPPORT_ICB
                            != 0,
                        stage_input,
                    },
                },
            });
        }
        if header.r#type == RGFX_CMD_MESH_RENDER_PIPELINE_CREATE {
            return self.prepare_mesh_pipeline(message);
        }
        let m = read::<RgfxRenderPipelineCreate>(message).ok_or(DecodeError::Malformed)?;
        let vertex_library = library(m.vertex_library_obj_id)?;
        let vertex_name = str_tail(message, m.vertex_name_offset, m.vertex_name_len)
            .ok_or(DecodeError::Malformed)?;
        let vertex_constants =
            constants_tail(message, m.vertex_constants_offset, m.vertex_constant_count)
                .ok_or(DecodeError::Malformed)?;
        let fragment = if m.fragment_library_obj_id == 0 && m.fragment_name_len == 0 {
            None
        } else {
            let name = str_tail(message, m.fragment_name_offset, m.fragment_name_len)
                .ok_or(DecodeError::Malformed)?;
            let constants = constants_tail(
                message,
                m.fragment_constants_offset,
                m.fragment_constant_count,
            )
            .ok_or(DecodeError::Malformed)?;
            Some((
                library(m.fragment_library_obj_id)?,
                name.to_owned(),
                constants,
            ))
        };
        let label = if m.label_len == 0 {
            None
        } else {
            Some(
                str_tail(message, m.label_offset, m.label_len)
                    .ok_or(DecodeError::Malformed)?
                    .to_owned(),
            )
        };
        let vertex_linked =
            self.linked_tail(message, m.vertex_linked_offset, m.vertex_linked_count)?;
        let fragment_linked =
            self.linked_tail(message, m.fragment_linked_offset, m.fragment_linked_count)?;
        Ok(PipelineJob {
            obj_id: m.obj_id,
            seqno: m.header.seqno,
            shape: PipelineShape::Render {
                vertex_library,
                vertex_name: vertex_name.to_owned(),
                vertex_constants,
                fragment,
                label,
                vertex_linked,
                fragment_linked,
                fields: render_fields(message, &m)?,
            },
        })
    }

    /// [`Self::prepare_pipeline`] of a `MESH_RENDER_PIPELINE_CREATE`: its object, mesh and
    /// fragment functions, each stage's constants and linked functions, its label and the rest of
    /// its descriptor. A create naming library 0 and no name for its object function has none (a
    /// pipeline with no object stage), and likewise for its fragment function (one that does not
    /// rasterize); one with no mesh function is [`DecodeError::Malformed`], as the bundle ends
    /// the process at such a descriptor before posting it, as Metal does.
    fn prepare_mesh_pipeline(&self, message: &[u8]) -> Result<PipelineJob<P>, DecodeError> {
        let m = read::<RgfxMeshRenderPipelineCreate>(message).ok_or(DecodeError::Malformed)?;
        let stage = |library: u64, (name_offset, name_len): (u32, u32), (at, count): (u32, u32)| {
            if library == 0 && name_len == 0 {
                return Ok(None);
            }
            let name = str_tail(message, name_offset, name_len).ok_or(DecodeError::Malformed)?;
            let constants = constants_tail(message, at, count).ok_or(DecodeError::Malformed)?;
            Ok(Some((
                self.library_ref(library)?,
                name.to_owned(),
                constants,
            )))
        };
        let object = stage(
            m.object_library_obj_id,
            (m.object_name_offset, m.object_name_len),
            (m.object_constants_offset, m.object_constant_count),
        )?;
        let mesh = stage(
            m.mesh_library_obj_id,
            (m.mesh_name_offset, m.mesh_name_len),
            (m.mesh_constants_offset, m.mesh_constant_count),
        )?
        .ok_or(DecodeError::Malformed)?;
        let fragment = stage(
            m.fragment_library_obj_id,
            (m.fragment_name_offset, m.fragment_name_len),
            (m.fragment_constants_offset, m.fragment_constant_count),
        )?;
        let label = if m.label_len == 0 {
            None
        } else {
            Some(
                str_tail(message, m.label_offset, m.label_len)
                    .ok_or(DecodeError::Malformed)?
                    .to_owned(),
            )
        };
        let in_use = usize::try_from(m.color_attachment_count)
            .ok()
            .filter(|&n| n <= m.colors.len())
            .ok_or(DecodeError::Malformed)?;
        let colors = m.colors[..in_use]
            .iter()
            .zip(0u64..)
            .map(|(c, index)| (index, color_attachment(c)))
            .collect();
        let size = |[width, height, depth]: [u32; 3]| {
            MtlSize::new(width.into(), height.into(), depth.into())
        };
        Ok(PipelineJob {
            obj_id: m.obj_id,
            seqno: m.header.seqno,
            shape: PipelineShape::Mesh(Box::new(MeshShape {
                object,
                mesh,
                fragment,
                label,
                object_linked: self.linked_tail(
                    message,
                    m.object_linked_offset,
                    m.object_linked_count,
                )?,
                mesh_linked: self.linked_tail(
                    message,
                    m.mesh_linked_offset,
                    m.mesh_linked_count,
                )?,
                fragment_linked: self.linked_tail(
                    message,
                    m.fragment_linked_offset,
                    m.fragment_linked_count,
                )?,
                fields: MeshFields {
                    colors,
                    depth_pixel_format: m.depth_pixel_format.into(),
                    stencil_pixel_format: m.stencil_pixel_format.into(),
                    raster_sample_count: m.raster_sample_count.into(),
                    alpha_to_coverage: m.flags & RGFX_RENDER_PIPELINE_ALPHA_TO_COVERAGE != 0,
                    alpha_to_one: m.flags & RGFX_RENDER_PIPELINE_ALPHA_TO_ONE != 0,
                    rasterization_enabled: m.flags & RGFX_RENDER_PIPELINE_RASTERIZATION_ENABLED
                        != 0,
                    max_total_threads_per_object_threadgroup: m
                        .max_total_threads_per_object_threadgroup
                        .into(),
                    max_total_threads_per_mesh_threadgroup: m
                        .max_total_threads_per_mesh_threadgroup
                        .into(),
                    max_total_threadgroups_per_mesh_grid: m
                        .max_total_threadgroups_per_mesh_grid
                        .into(),
                    object_threadgroup_size_is_multiple_of_execution_width: m.flags
                        & RGFX_MESH_PIPELINE_OBJECT_TG_MULTIPLE_OF_EXEC_WIDTH
                        != 0,
                    mesh_threadgroup_size_is_multiple_of_execution_width: m.flags
                        & RGFX_MESH_PIPELINE_MESH_TG_MULTIPLE_OF_EXEC_WIDTH
                        != 0,
                    payload_memory_length: m.payload_memory_length.into(),
                    max_vertex_amplification_count: m.max_vertex_amplification_count.into(),
                    required_threads_per_object_threadgroup: size(
                        m.required_threads_per_object_threadgroup,
                    ),
                    required_threads_per_mesh_threadgroup: size(
                        m.required_threads_per_mesh_threadgroup,
                    ),
                    support_indirect_command_buffers: m.flags & RGFX_RENDER_PIPELINE_SUPPORT_ICB
                        != 0,
                },
            })),
        })
    }

    /// The `count` linked functions at byte `offset` of `message`: each one's library resolved
    /// from the table, its name, constants and options, and its list. A library that names none is
    /// [`DecodeError::BadReference`]; a span the message does not cover, a name that is not UTF-8,
    /// or a list Metal has none of is [`DecodeError::Malformed`].
    fn linked_tail(
        &self,
        message: &[u8],
        offset: u32,
        count: u32,
    ) -> Result<Vec<LinkedRef<P>>, DecodeError> {
        let records = records_tail::<RgfxLinkedFunction>(message, offset, count)
            .ok_or(DecodeError::Malformed)?;
        records
            .iter()
            .map(|r| {
                let list = match r.list {
                    RGFX_LINKED_FUNCTIONS => LinkedList::Functions,
                    RGFX_LINKED_BINARY_FUNCTIONS => LinkedList::Binary,
                    RGFX_LINKED_PRIVATE_FUNCTIONS => LinkedList::Private,
                    RGFX_LINKED_GROUP => LinkedList::Group(
                        str_tail(message, r.group_offset, r.group_len)
                            .ok_or(DecodeError::Malformed)?
                            .to_owned(),
                    ),
                    _ => return Err(DecodeError::Malformed),
                };
                Ok(LinkedRef {
                    library: self.library_ref(r.library_obj_id)?,
                    name: str_tail(message, r.name_offset, r.name_len)
                        .ok_or(DecodeError::Malformed)?
                        .to_owned(),
                    constants: constants_tail(message, r.constants_offset, r.constant_count)
                        .ok_or(DecodeError::Malformed)?,
                    options: r.options.into(),
                    specialized_name: if r.specialized_name_len == 0 {
                        None
                    } else {
                        Some(
                            str_tail(message, r.specialized_name_offset, r.specialized_name_len)
                                .ok_or(DecodeError::Malformed)?
                                .to_owned(),
                        )
                    },
                    list,
                })
            })
            .collect()
    }

    /// The last step of a pipeline create: stores the compiled pipeline under its id, or, for a
    /// refused one, what the id stands for ([`refused`]): the bundle posts a create equal to an
    /// accepted one unanswered ("Known pipeline creates"), and its draws are then dropped, not
    /// its command buffers.
    pub(crate) fn finish_pipeline(&self, compiled: CompiledPipeline<P>) {
        self.finish_pipeline_if(compiled, || true);
    }

    /// [`Self::finish_pipeline`], where `live` answers, under the lock of the table the pipeline
    /// goes in, that no destroy or close cancelled it: a destroy cancels the compile before it
    /// removes what the id names, so the pipeline is never stored after its destroy, and a close
    /// before it takes the context's state out, so a compile ending after its context's close
    /// makes the closed context no state again ([`Self::store_if`]).
    pub(crate) fn finish_pipeline_if(
        &self,
        compiled: CompiledPipeline<P>,
        live: impl Fn() -> bool,
    ) {
        let handle = compiled.handle.unwrap_or_else(|| {
            ObjHandle::Refused(refused::Refused::of(match compiled.reflection.kind {
                RGFX_PIPELINE_KIND_COMPUTE => RGFX_CMD_COMPUTE_PIPELINE_CREATE,
                RGFX_PIPELINE_KIND_MESH => RGFX_CMD_MESH_RENDER_PIPELINE_CREATE,
                _ => RGFX_CMD_RENDER_PIPELINE_CREATE,
            }))
        });
        self.store_if(compiled.obj_id, Entry::of(handle), live);
    }
}

/// A pipeline create the decoder has prepared ([`GraphicsDecoder::prepare_pipeline`]): what its
/// compile needs, owned, so it runs with no lock of the decoder's held.
pub(crate) struct PipelineJob<P: Platform> {
    /// The pipeline's guest-picked id.
    pub(crate) obj_id: u64,
    /// The create's seqno, which its reply carries.
    pub(crate) seqno: u64,
    /// The pipeline's functions and descriptor.
    shape: PipelineShape<P>,
}

impl<P: Platform> PipelineJob<P> {
    /// [`RGFX_PIPELINE_KIND_RENDER`], [`RGFX_PIPELINE_KIND_MESH`] or
    /// [`RGFX_PIPELINE_KIND_COMPUTE`].
    pub(crate) fn kind(&self) -> u32 {
        match self.shape {
            PipelineShape::Render { .. } => RGFX_PIPELINE_KIND_RENDER,
            PipelineShape::Mesh(_) => RGFX_PIPELINE_KIND_MESH,
            PipelineShape::Compute { .. } => RGFX_PIPELINE_KIND_COMPUTE,
        }
    }
}

/// A prepared pipeline's functions and descriptor.
enum PipelineShape<P: Platform> {
    Render {
        vertex_library: LibraryRef<P>,
        vertex_name: String,
        vertex_constants: Vec<FunctionConstant>,
        /// The fragment function's library, name and constants; `None` for none.
        fragment: Option<(LibraryRef<P>, String, Vec<FunctionConstant>)>,
        label: Option<String>,
        vertex_linked: Vec<LinkedRef<P>>,
        fragment_linked: Vec<LinkedRef<P>>,
        fields: RenderFields,
    },
    Mesh(Box<MeshShape<P>>),
    Compute {
        library: LibraryRef<P>,
        name: String,
        constants: Vec<FunctionConstant>,
        linked: Vec<LinkedRef<P>>,
        fields: ComputeFields,
    },
}

/// A stage's function as a create names it: its library, name and constants.
type StageRef<P> = (LibraryRef<P>, String, Vec<FunctionConstant>);

/// A prepared mesh render pipeline's functions and descriptor.
struct MeshShape<P: Platform> {
    /// The object function; `None` for none.
    object: Option<StageRef<P>>,
    mesh: StageRef<P>,
    /// The fragment function; `None` for none.
    fragment: Option<StageRef<P>>,
    label: Option<String>,
    object_linked: Vec<LinkedRef<P>>,
    mesh_linked: Vec<LinkedRef<P>>,
    fragment_linked: Vec<LinkedRef<P>>,
    fields: MeshFields,
}

/// A linked function a pipeline create names, prepared: its library, name, constants and
/// options, and the list of `MTLLinkedFunctions` it is in.
struct LinkedRef<P: Platform> {
    library: LibraryRef<P>,
    name: String,
    constants: Vec<FunctionConstant>,
    /// The `MTLFunctionOptions` the guest made it with.
    options: u64,
    /// The descriptor's `specializedName`, where the guest gave one.
    specialized_name: Option<String>,
    list: LinkedList,
}

/// The list of `MTLLinkedFunctions` a linked function is in.
#[derive(Debug, Clone, PartialEq, Eq)]
enum LinkedList {
    Functions,
    Binary,
    /// The call-site group of this name.
    Group(String),
    Private,
}

/// A render pipeline create's descriptor past its functions and label, owned: the fields of
/// [`FullRenderPipelineDesc`] the compile hands the platform with them.
#[derive(Debug, Clone, PartialEq, Eq)]
struct RenderFields {
    vertex_descriptor: Option<VertexDesc>,
    colors: Vec<(u64, ColorAttachmentDesc)>,
    depth_pixel_format: u64,
    stencil_pixel_format: u64,
    raster_sample_count: u64,
    alpha_to_coverage: bool,
    alpha_to_one: bool,
    rasterization_enabled: bool,
    input_primitive_topology: u64,
    max_vertex_amplification_count: u64,
    support_indirect_command_buffers: bool,
}

/// A compute pipeline create's descriptor past its function and linked functions: the fields of
/// [`FullComputePipelineDesc`] the compile hands the platform with them.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ComputeFields {
    threadgroup_size_is_multiple_of_execution_width: bool,
    max_total_threads_per_threadgroup: u64,
    max_call_stack_depth: u64,
    support_indirect_command_buffers: bool,
    stage_input: Option<StageInputDesc>,
}

/// A mesh render pipeline create's descriptor past its functions, label and linked functions:
/// the fields of [`FullMeshRenderPipelineDesc`] the compile hands the platform with them.
#[derive(Debug, Clone, PartialEq, Eq)]
struct MeshFields {
    colors: Vec<(u64, ColorAttachmentDesc)>,
    depth_pixel_format: u64,
    stencil_pixel_format: u64,
    raster_sample_count: u64,
    alpha_to_coverage: bool,
    alpha_to_one: bool,
    rasterization_enabled: bool,
    max_total_threads_per_object_threadgroup: u64,
    max_total_threads_per_mesh_threadgroup: u64,
    max_total_threadgroups_per_mesh_grid: u64,
    object_threadgroup_size_is_multiple_of_execution_width: bool,
    mesh_threadgroup_size_is_multiple_of_execution_width: bool,
    payload_memory_length: u64,
    max_vertex_amplification_count: u64,
    required_threads_per_object_threadgroup: MtlSize,
    required_threads_per_mesh_threadgroup: MtlSize,
    support_indirect_command_buffers: bool,
}

/// The highest vertex buffer and attribute index Metal's arrays hold: a larger one raises there.
const LAST_VERTEX_INDEX: u32 = 30;

/// A render create's descriptor fields ([`RenderFields`]): the colour attachments in use, the
/// vertex descriptor's runs from the tail (none where it carries neither layouts nor
/// attributes), and the rest from the head.
fn render_fields(
    message: &[u8],
    m: &RgfxRenderPipelineCreate,
) -> Result<RenderFields, DecodeError> {
    let in_use = usize::try_from(m.color_attachment_count)
        .ok()
        .filter(|&n| n <= m.colors.len())
        .ok_or(DecodeError::Malformed)?;
    let colors = m.colors[..in_use]
        .iter()
        .zip(0u64..)
        .map(|(c, index)| (index, color_attachment(c)))
        .collect();
    let (layouts, attributes) = layout_runs(
        message,
        (m.vertex_layouts_offset, m.vertex_layout_count),
        (m.vertex_attributes_offset, m.vertex_attribute_count),
    )?;
    let vertex_descriptor = (!layouts.is_empty() || !attributes.is_empty()).then_some(VertexDesc {
        layouts,
        attributes,
    });
    Ok(RenderFields {
        vertex_descriptor,
        colors,
        depth_pixel_format: m.depth_pixel_format.into(),
        stencil_pixel_format: m.stencil_pixel_format.into(),
        raster_sample_count: m.raster_sample_count.into(),
        alpha_to_coverage: m.flags & RGFX_RENDER_PIPELINE_ALPHA_TO_COVERAGE != 0,
        alpha_to_one: m.flags & RGFX_RENDER_PIPELINE_ALPHA_TO_ONE != 0,
        rasterization_enabled: m.flags & RGFX_RENDER_PIPELINE_RASTERIZATION_ENABLED != 0,
        input_primitive_topology: m.input_primitive_topology.into(),
        max_vertex_amplification_count: m.max_vertex_amplification_count.into(),
        support_indirect_command_buffers: m.flags & RGFX_RENDER_PIPELINE_SUPPORT_ICB != 0,
    })
}

/// The wire's stride for `MTLBufferLayoutStrideDynamic`, which the guest's 32 bits hold as their
/// highest value.
const WIRE_DYNAMIC_STRIDE: u32 = u32::MAX;

/// The layouts and attributes of a vertex or stage-input descriptor: the runs of
/// [`RgfxVertexBufferLayout`] and [`RgfxVertexAttribute`] at the `(offset, count)` spans of
/// `message`, each stride as Metal's (`MTLBufferLayoutStrideDynamic` its `NSUIntegerMax`). A run
/// the message does not hold, or a buffer or attribute index past Metal's last, is
/// [`DecodeError::Malformed`].
fn layout_runs(
    message: &[u8],
    (layouts_offset, layout_count): (u32, u32),
    (attributes_offset, attribute_count): (u32, u32),
) -> Result<(Vec<BufferLayoutDesc>, Vec<AttributeDesc>), DecodeError> {
    let layouts = records_tail::<RgfxVertexBufferLayout>(message, layouts_offset, layout_count)
        .ok_or(DecodeError::Malformed)?;
    let attributes =
        records_tail::<RgfxVertexAttribute>(message, attributes_offset, attribute_count)
            .ok_or(DecodeError::Malformed)?;
    let past_last = layouts.iter().any(|l| l.buffer_index > LAST_VERTEX_INDEX)
        || attributes
            .iter()
            .any(|a| a.attribute_index > LAST_VERTEX_INDEX || a.buffer_index > LAST_VERTEX_INDEX);
    if past_last {
        return Err(DecodeError::Malformed);
    }
    let layouts = layouts
        .iter()
        .map(|l| BufferLayoutDesc {
            buffer_index: l.buffer_index.into(),
            stride: if l.stride == WIRE_DYNAMIC_STRIDE {
                u64::MAX
            } else {
                l.stride.into()
            },
            step_function: l.step_function.into(),
            step_rate: l.step_rate.into(),
        })
        .collect();
    let attributes = attributes
        .iter()
        .map(|a| AttributeDesc {
            attribute_index: a.attribute_index.into(),
            format: a.format.into(),
            offset: a.offset.into(),
            buffer_index: a.buffer_index.into(),
        })
        .collect();
    Ok((layouts, attributes))
}

/// A compute create's stage-input descriptor: none where it carries neither layouts nor
/// attributes.
fn stage_input(
    message: &[u8],
    m: &RgfxComputePipelineCreate,
) -> Result<Option<StageInputDesc>, DecodeError> {
    let (layouts, attributes) = layout_runs(
        message,
        (m.stage_input_layouts_offset, m.stage_input_layout_count),
        (
            m.stage_input_attributes_offset,
            m.stage_input_attribute_count,
        ),
    )?;
    Ok(
        (!layouts.is_empty() || !attributes.is_empty()).then(|| StageInputDesc {
            layouts,
            attributes,
            index_type: m.stage_input_index_type.into(),
            index_buffer_index: m.stage_input_index_buffer_index.into(),
        }),
    )
}

/// A colour attachment of the wire as the platform's value: each field its own.
fn color_attachment(c: &RgfxColorAttachmentDesc) -> ColorAttachmentDesc {
    ColorAttachmentDesc {
        pixel_format: c.pixel_format.into(),
        blending_enabled: c.blending_enabled != 0,
        source_rgb_blend_factor: c.source_rgb_blend_factor.into(),
        destination_rgb_blend_factor: c.destination_rgb_blend_factor.into(),
        rgb_blend_operation: c.rgb_blend_operation.into(),
        source_alpha_blend_factor: c.source_alpha_blend_factor.into(),
        destination_alpha_blend_factor: c.destination_alpha_blend_factor.into(),
        alpha_blend_operation: c.alpha_blend_operation.into(),
        write_mask: c.write_mask.into(),
    }
}

/// A compiled pipeline create: the pipeline (none where it was refused) and the reflection reply
/// the create owes.
pub(crate) struct CompiledPipeline<P: Platform> {
    /// The pipeline's guest-picked id.
    obj_id: u64,
    /// The pipeline, or `None` where it was refused.
    handle: Option<ObjHandle<P>>,
    /// The reply: the pipeline's reflection, or a refusal.
    pub(crate) reflection: RgfxPipelineReflection,
    /// A made mesh render pipeline's limits, which its reply carries after the reflection.
    mesh: Option<RgfxMeshPipelineReflection>,
    /// Why it was refused, for the host's log; `None` for a made pipeline.
    refusal: Option<String>,
}

impl<P: Platform> CompiledPipeline<P> {
    /// A refused create of `kind` (a nil function, the platform refusing the descriptor, or a
    /// raise or a refusal in the compile), refused for `reason`.
    pub(crate) fn refused(obj_id: u64, seqno: u64, kind: u32, reason: impl Into<String>) -> Self {
        CompiledPipeline {
            obj_id,
            handle: None,
            reflection: refused_reflection(obj_id, seqno, kind),
            mesh: None,
            refusal: Some(reason.into()),
        }
    }

    /// A made pipeline and its reflection.
    fn made(obj_id: u64, handle: ObjHandle<P>, reflection: RgfxPipelineReflection) -> Self {
        CompiledPipeline {
            obj_id,
            handle: Some(handle),
            reflection,
            mesh: None,
            refusal: None,
        }
    }

    /// The reply the create owes: its reflection, with a made mesh pipeline's limits after it.
    pub(crate) fn reply(&self) -> PipelineReply {
        PipelineReply {
            reflection: self.reflection,
            mesh: self.mesh,
        }
    }

    /// The platform made no pipeline.
    pub(crate) fn is_refused(&self) -> bool {
        self.handle.is_none()
    }

    /// Why the platform made no pipeline; `None` for a made one.
    pub(crate) fn refusal(&self) -> Option<&str> {
        self.refusal.as_deref()
    }
}

/// [`compile_pipeline`] in the lane's refusal scope ([`refusing`]): a platform refusing the
/// compile refuses the create, and a refusal's reason names the pipeline's functions and the
/// constants each was specialised with, for the host's log.
pub(crate) fn compile_refusing<P: Platform>(
    device: &P::Device,
    job: PipelineJob<P>,
) -> CompiledPipeline<P> {
    let (obj_id, seqno, kind) = (job.obj_id, job.seqno, job.kind());
    let functions = job.functions();
    let mut compiled = match refusing(|| compile_pipeline::<P>(device, job)) {
        Ok(compiled) => compiled,
        Err(raise) => CompiledPipeline::refused(obj_id, seqno, kind, raise.to_string()),
    };
    if let Some(reason) = compiled.refusal.as_mut() {
        reason.push_str(&format!("; its functions: {functions}"));
    }
    compiled
}

/// A pipeline create's functions, for the host's log: each stage's function name and the
/// constants it is specialised with, then each linked function's.
pub(crate) struct PipelineFunctions(Vec<(&'static str, String, Vec<FunctionConstant>)>);

impl std::fmt::Display for PipelineFunctions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, (stage, name, constants)) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{stage} {name:?} [")?;
            for (j, c) in constants.iter().enumerate() {
                if j > 0 {
                    f.write_str(" ")?;
                }
                write!(f, "{}:{}=", c.index, c.data_type)?;
                for b in c.bytes().unwrap_or(&c.value) {
                    write!(f, "{b:02x}")?;
                }
            }
            f.write_str("]")?;
        }
        Ok(())
    }
}

impl<P: Platform> PipelineJob<P> {
    /// The create's functions and their constants ([`PipelineFunctions`]).
    pub(crate) fn functions(&self) -> PipelineFunctions {
        let linked = |stage: &'static str, refs: &[LinkedRef<P>]| {
            refs.iter()
                .map(move |r| {
                    let name = match &r.specialized_name {
                        Some(specialized) => format!("{} as {specialized:?}", r.name),
                        None => r.name.clone(),
                    };
                    (stage, name, r.constants.clone())
                })
                .collect::<Vec<_>>()
        };
        let mut out = Vec::new();
        match &self.shape {
            PipelineShape::Render {
                vertex_name,
                vertex_constants,
                fragment,
                vertex_linked,
                fragment_linked,
                ..
            } => {
                out.push(("vertex", vertex_name.clone(), vertex_constants.clone()));
                if let Some((_, name, constants)) = fragment {
                    out.push(("fragment", name.clone(), constants.clone()));
                }
                out.extend(linked("vertex-linked", vertex_linked));
                out.extend(linked("fragment-linked", fragment_linked));
            }
            PipelineShape::Mesh(shape) => {
                let MeshShape {
                    object,
                    mesh,
                    fragment,
                    object_linked,
                    mesh_linked,
                    fragment_linked,
                    ..
                } = &**shape;
                if let Some((_, name, constants)) = object {
                    out.push(("object", name.clone(), constants.clone()));
                }
                out.push(("mesh", mesh.1.clone(), mesh.2.clone()));
                if let Some((_, name, constants)) = fragment {
                    out.push(("fragment", name.clone(), constants.clone()));
                }
                out.extend(linked("object-linked", object_linked));
                out.extend(linked("mesh-linked", mesh_linked));
                out.extend(linked("fragment-linked", fragment_linked));
            }
            PipelineShape::Compute {
                name,
                constants,
                linked: refs,
                ..
            } => {
                out.push(("kernel", name.clone(), constants.clone()));
                out.extend(linked("linked", refs));
            }
        }
        PipelineFunctions(out)
    }
}

/// The middle step of a pipeline create, with no lock held: the functions specialised by their
/// constants and the pipeline made on `device`. A nil function (the library has no such entry
/// point, or refuses the specialisation) refuses the pipeline, as a nil state or an `NSError`
/// from the creator does; a platform raise propagates. The caller runs it in the lane's refusal
/// scope ([`refusing`]), where a platform refusing the compile refuses the create too.
pub(crate) fn compile_pipeline<P: Platform>(
    device: &P::Device,
    job: PipelineJob<P>,
) -> RaiseResult<CompiledPipeline<P>> {
    let kind = job.kind();
    let (obj_id, seqno) = (job.obj_id, job.seqno);
    let refused = |reason: String| CompiledPipeline::refused(obj_id, seqno, kind, reason);
    let no_library = |stage: &str| format!("its {stage} function's library was refused");
    let no_function =
        |name: &str| format!("the library made no function {name:?} of its constants");
    match job.shape {
        PipelineShape::Render {
            vertex_library,
            vertex_name,
            vertex_constants,
            fragment,
            label,
            vertex_linked,
            fragment_linked,
            fields,
        } => {
            // A library being made is waited for here; one the platform refused refuses the
            // pipeline.
            let Some(vertex_library) = vertex_library.resolve() else {
                return Ok(refused(no_library("vertex")));
            };
            let Some(vertex) = specialised::<P>(&vertex_library, &vertex_name, &vertex_constants)?
            else {
                return Ok(refused(no_function(&vertex_name)));
            };
            let fragment = match fragment {
                None => None,
                Some((library, name, constants)) => {
                    let Some(library) = library.resolve() else {
                        return Ok(refused(no_library("fragment")));
                    };
                    let Some(function) = specialised::<P>(&library, &name, &constants)? else {
                        return Ok(refused(no_function(&name)));
                    };
                    Some(function)
                }
            };
            let vertex_linked = match linked_functions::<P>(vertex_linked)? {
                Ok(linked) => linked,
                Err(reason) => return Ok(refused(reason)),
            };
            let fragment_linked = match linked_functions::<P>(fragment_linked)? {
                Ok(linked) => linked,
                Err(reason) => return Ok(refused(reason)),
            };
            let desc = FullRenderPipelineDesc::<P> {
                label: label.as_deref(),
                vertex: &vertex,
                fragment: fragment.as_ref(),
                vertex_descriptor: fields.vertex_descriptor,
                colors: fields.colors,
                depth_pixel_format: fields.depth_pixel_format,
                stencil_pixel_format: fields.stencil_pixel_format,
                raster_sample_count: fields.raster_sample_count,
                alpha_to_coverage: fields.alpha_to_coverage,
                alpha_to_one: fields.alpha_to_one,
                rasterization_enabled: fields.rasterization_enabled,
                input_primitive_topology: fields.input_primitive_topology,
                max_vertex_amplification_count: fields.max_vertex_amplification_count,
                support_indirect_command_buffers: fields.support_indirect_command_buffers,
                vertex_linked,
                fragment_linked,
            };
            // A nil state or an `NSError` refuses; a refusal Metal answers with none dies at the
            // send, inside the caller's refusal scope.
            Ok(match device.new_render_pipeline_state_full(&desc) {
                Ok(Some(pipeline)) => {
                    let reflection = render_reflection::<P>(&pipeline, seqno, obj_id);
                    CompiledPipeline::made(obj_id, ObjHandle::RenderPipeline(pipeline), reflection)
                }
                Ok(None) => refused("the platform answered nil".to_owned()),
                Err(e) => refused(format!("the platform answered an error: {}", e.description)),
            })
        }
        PipelineShape::Mesh(shape) => {
            let MeshShape {
                object,
                mesh,
                fragment,
                label,
                object_linked,
                mesh_linked,
                fragment_linked,
                fields,
            } = *shape;
            // Each stage's function: a library being made is waited for here, and one the
            // platform refused, or an entry point it makes no function of, refuses the pipeline.
            let function =
                |stage: &str,
                 (library, name, constants): StageRef<P>|
                 -> RaiseResult<Result<P::Function, String>> {
                    let Some(library) = library.resolve() else {
                        return Ok(Err(no_library(stage)));
                    };
                    Ok(specialised::<P>(&library, &name, &constants)?
                        .ok_or_else(|| no_function(&name)))
                };
            let object = match object.map(|f| function("object", f)).transpose()? {
                None => None,
                Some(Ok(made)) => Some(made),
                Some(Err(reason)) => return Ok(refused(reason)),
            };
            let mesh = match function("mesh", mesh)? {
                Ok(made) => made,
                Err(reason) => return Ok(refused(reason)),
            };
            let fragment = match fragment.map(|f| function("fragment", f)).transpose()? {
                None => None,
                Some(Ok(made)) => Some(made),
                Some(Err(reason)) => return Ok(refused(reason)),
            };
            let object_linked = match linked_functions::<P>(object_linked)? {
                Ok(linked) => linked,
                Err(reason) => return Ok(refused(reason)),
            };
            let mesh_linked = match linked_functions::<P>(mesh_linked)? {
                Ok(linked) => linked,
                Err(reason) => return Ok(refused(reason)),
            };
            let fragment_linked = match linked_functions::<P>(fragment_linked)? {
                Ok(linked) => linked,
                Err(reason) => return Ok(refused(reason)),
            };
            let desc = FullMeshRenderPipelineDesc::<P> {
                label: label.as_deref(),
                object: object.as_ref(),
                mesh: &mesh,
                fragment: fragment.as_ref(),
                colors: fields.colors,
                depth_pixel_format: fields.depth_pixel_format,
                stencil_pixel_format: fields.stencil_pixel_format,
                raster_sample_count: fields.raster_sample_count,
                alpha_to_coverage: fields.alpha_to_coverage,
                alpha_to_one: fields.alpha_to_one,
                rasterization_enabled: fields.rasterization_enabled,
                max_total_threads_per_object_threadgroup: fields
                    .max_total_threads_per_object_threadgroup,
                max_total_threads_per_mesh_threadgroup: fields
                    .max_total_threads_per_mesh_threadgroup,
                max_total_threadgroups_per_mesh_grid: fields.max_total_threadgroups_per_mesh_grid,
                object_threadgroup_size_is_multiple_of_execution_width: fields
                    .object_threadgroup_size_is_multiple_of_execution_width,
                mesh_threadgroup_size_is_multiple_of_execution_width: fields
                    .mesh_threadgroup_size_is_multiple_of_execution_width,
                payload_memory_length: fields.payload_memory_length,
                max_vertex_amplification_count: fields.max_vertex_amplification_count,
                required_threads_per_object_threadgroup: fields
                    .required_threads_per_object_threadgroup,
                required_threads_per_mesh_threadgroup: fields.required_threads_per_mesh_threadgroup,
                support_indirect_command_buffers: fields.support_indirect_command_buffers,
                object_linked,
                mesh_linked,
                fragment_linked,
            };
            // A nil state or an `NSError` refuses; a refusal Metal answers with none dies at the
            // send, inside the caller's refusal scope.
            Ok(match device.new_mesh_render_pipeline_state_full(&desc) {
                Ok(Some(pipeline)) => {
                    let mut reflection = render_reflection::<P>(&pipeline, seqno, obj_id);
                    reflection.kind = RGFX_PIPELINE_KIND_MESH;
                    let mesh = mesh_reflection::<P>(&pipeline);
                    let mut made = CompiledPipeline::made(
                        obj_id,
                        ObjHandle::RenderPipeline(pipeline),
                        reflection,
                    );
                    made.mesh = Some(mesh);
                    made
                }
                Ok(None) => refused("the platform answered nil".to_owned()),
                Err(e) => refused(format!("the platform answered an error: {}", e.description)),
            })
        }
        PipelineShape::Compute {
            library,
            name,
            constants,
            linked,
            fields,
        } => {
            let Some(library) = library.resolve() else {
                return Ok(refused(no_library("kernel")));
            };
            let Some(function) = specialised::<P>(&library, &name, &constants)? else {
                return Ok(refused(no_function(&name)));
            };
            let linked = match linked_functions::<P>(linked)? {
                Ok(linked) => linked,
                Err(reason) => return Ok(refused(reason)),
            };
            let desc = FullComputePipelineDesc::<P> {
                label: None,
                function: &function,
                threadgroup_size_is_multiple_of_execution_width: fields
                    .threadgroup_size_is_multiple_of_execution_width,
                max_total_threads_per_threadgroup: fields.max_total_threads_per_threadgroup,
                max_call_stack_depth: fields.max_call_stack_depth,
                support_indirect_command_buffers: fields.support_indirect_command_buffers,
                stage_input: fields.stage_input,
                linked,
            };
            Ok(match device.new_compute_pipeline_state_full(&desc) {
                Ok(Some(pipeline)) => {
                    let reflection = compute_reflection::<P>(&pipeline, seqno, obj_id);
                    CompiledPipeline::made(obj_id, ObjHandle::ComputePipeline(pipeline), reflection)
                }
                Ok(None) => refused("the platform answered nil".to_owned()),
                Err(e) => refused(format!("the platform answered an error: {}", e.description)),
            })
        }
    }
}

/// The small chunks of the texture resource ids, by index, that a grant of type `type`
/// (`RESOURCE_ID_CHUNK` or `RESOURCE_ID_SMALL_CHUNK`) of chunk `chunk` gives its context: chunk
/// `k` of [`RGFX_RESOURCE_ID_CHUNK`] ids is small chunks `4k` to `4k + 3`, small chunk `k` is
/// itself; `None` for a chunk past the pool.
fn small_chunks(r#type: u32, chunk: u32) -> Option<std::ops::Range<usize>> {
    let per = if r#type == RGFX_CMD_RESOURCE_ID_SMALL_CHUNK {
        1
    } else {
        (RGFX_RESOURCE_ID_CHUNK / RGFX_RESOURCE_ID_SMALL_CHUNK) as usize
    };
    let first = (chunk as usize).checked_mul(per)?;
    let end = first.checked_add(per)?;
    (end <= RGFX_RESOURCE_ID_SMALL_CHUNKS as usize).then_some(first..end)
}

/// The reply a pipeline create refused at its op owes: a refusal, keyed to the create by its
/// seqno, for the object its header names (`None` for a message too short to name one). The
/// create's process waits for it, so a create the host will not carry out is still answered.
pub(crate) fn refused_reply(message: &[u8]) -> Option<RgfxPipelineReflection> {
    let header = read::<RgfxMsgHeader>(message)?;
    let obj_id = u64::from_le_bytes(message.get(16..24)?.try_into().ok()?);
    let kind = match header.r#type {
        RGFX_CMD_RENDER_PIPELINE_CREATE => RGFX_PIPELINE_KIND_RENDER,
        RGFX_CMD_MESH_RENDER_PIPELINE_CREATE => RGFX_PIPELINE_KIND_MESH,
        _ => RGFX_PIPELINE_KIND_COMPUTE,
    };
    Some(refused_reflection(obj_id, header.seqno, kind))
}

/// A refused `PIPELINE_REFLECTION` for the create of `kind` at `seqno` of pipeline `obj_id`.
fn refused_reflection(obj_id: u64, seqno: u64, kind: u32) -> RgfxPipelineReflection {
    RgfxPipelineReflection {
        header: reflection_header(seqno),
        obj_id,
        status: RGFX_PIPELINE_REFUSED,
        kind,
        ..RgfxPipelineReflection::default()
    }
}

/// A `PIPELINE_REFLECTION` message header for a create at `seqno`.
fn reflection_header(seqno: u64) -> RgfxMsgHeader {
    RgfxMsgHeader {
        r#type: RGFX_EVT_PIPELINE_REFLECTION,
        size: size_of::<RgfxPipelineReflection>() as u32,
        seqno,
    }
}

/// A UTF-8 string at `[offset, offset + len)` of `message`, or `None` when the
/// span lies outside the message or is not UTF-8.
/// The `count` [`RgfxFunctionConstant`]s at byte `offset` of `message`, or `None`
/// where the message does not cover them.
fn constants_tail(message: &[u8], offset: u32, count: u32) -> Option<Vec<FunctionConstant>> {
    let size = size_of::<RgfxFunctionConstant>();
    let start = offset as usize;
    let end = start.checked_add((count as usize).checked_mul(size)?)?;
    let bytes = message.get(start..end)?;
    Some(
        bytes
            .chunks_exact(size)
            .filter_map(|b| RgfxFunctionConstant::read_from_bytes(b).ok())
            .map(|c| FunctionConstant {
                index: c.index,
                data_type: c.data_type,
                value: c.value,
            })
            .collect(),
    )
}

/// The function `name` of `library`, specialised by `constants` when there are any.
fn specialised<P: Platform>(
    library: &P::Library,
    name: &str,
    constants: &[FunctionConstant],
) -> RaiseResult<Option<P::Function>> {
    if constants.is_empty() {
        library.new_function(OnFault::Return, name)
    } else {
        library.new_function_with_constants(OnFault::Return, name, constants)
    }
}

/// The linked functions `refs` name, made: each from its library, specialised by its constants,
/// with its options where it was made with any, in its list. A library the platform refused, or
/// a function the library makes none of, is the refusal's reason.
fn linked_functions<P: Platform>(
    refs: Vec<LinkedRef<P>>,
) -> RaiseResult<Result<LinkedFunctionsDesc<P>, String>> {
    let mut out = LinkedFunctionsDesc::<P>::default();
    for r in refs {
        let Some(library) = r.library.resolve() else {
            return Ok(Err(format!(
                "linked function {:?}'s library was refused",
                r.name
            )));
        };
        let function = if r.options == 0 && r.specialized_name.is_none() {
            specialised::<P>(&library, &r.name, &r.constants)?
        } else {
            library.new_function_with_options(
                OnFault::Return,
                &r.name,
                &r.constants,
                r.options,
                r.specialized_name.as_deref(),
            )?
        };
        let Some(function) = function else {
            return Ok(Err(format!(
                "the library made no linked function {:?} of its constants and options",
                r.name
            )));
        };
        match r.list {
            LinkedList::Functions => out.functions.push(function),
            LinkedList::Binary => out.binary_functions.push(function),
            LinkedList::Private => out.private_functions.push(function),
            LinkedList::Group(name) => match out.groups.iter_mut().find(|(n, _)| *n == name) {
                Some((_, group)) => group.push(function),
                None => out.groups.push((name, vec![function])),
            },
        }
    }
    Ok(Ok(out))
}

/// `count` wire records of `T` at `offset` from the message start: `None` where the message does
/// not hold them all.
fn records_tail<T: FromBytes + KnownLayout + Immutable>(
    message: &[u8],
    offset: u32,
    count: u32,
) -> Option<Vec<T>> {
    let size = size_of::<T>();
    let start = offset as usize;
    let end = start.checked_add((count as usize).checked_mul(size)?)?;
    message
        .get(start..end)?
        .chunks_exact(size)
        .map(|b| T::read_from_bytes(b).ok())
        .collect()
}

fn str_tail(message: &[u8], offset: u32, len: u32) -> Option<&str> {
    let start = offset as usize;
    let end = start.checked_add(len as usize)?;
    std::str::from_utf8(message.get(start..end)?).ok()
}

/// The reflection reply for a made render pipeline: the state's own queries (the
/// platform exposes no per-argument binding reflection).
fn render_reflection<P: Platform>(
    pipeline: &P::RenderPipeline,
    seqno: u64,
    obj_id: u64,
) -> RgfxPipelineReflection {
    let mut flags = 0;
    if pipeline.support_indirect_command_buffers() {
        flags |= RGFX_PIPELINE_REFLECT_SUPPORT_ICB;
    }
    if pipeline.threadgroup_size_matches_tile_size() {
        flags |= RGFX_PIPELINE_REFLECT_THREADGROUP_MATCHES_TILE;
    }
    RgfxPipelineReflection {
        header: reflection_header(seqno),
        obj_id,
        gpu_resource_id: pipeline.gpu_resource_id(),
        unique_identifier: pipeline.guest_unique_identifier(),
        max_total_threads_per_threadgroup: pipeline.max_total_threads_per_threadgroup(),
        static_threadgroup_memory_length: 0,
        thread_execution_width: 0,
        imageblock_sample_length: pipeline.imageblock_sample_length() as u32,
        resource_index: pipeline.resource_index() as u32,
        status: RGFX_PIPELINE_OK,
        kind: RGFX_PIPELINE_KIND_RENDER,
        flags,
    }
}

/// A made mesh render pipeline's limits, which its reply carries after its reflection: the
/// state's own queries, each held in the wire's 32 bits (`maxTotalThreadgroupsPerMeshGrid`'s
/// `NSUIntegerMax` as their highest value).
fn mesh_reflection<P: Platform>(pipeline: &P::RenderPipeline) -> RgfxMeshPipelineReflection {
    let u32_of = |v: u64| u32::try_from(v).unwrap_or(u32::MAX);
    RgfxMeshPipelineReflection {
        max_total_threads_per_object_threadgroup: u32_of(
            pipeline.max_total_threads_per_object_threadgroup(),
        ),
        max_total_threads_per_mesh_threadgroup: u32_of(
            pipeline.max_total_threads_per_mesh_threadgroup(),
        ),
        object_thread_execution_width: u32_of(pipeline.object_thread_execution_width()),
        mesh_thread_execution_width: u32_of(pipeline.mesh_thread_execution_width()),
        max_total_threadgroups_per_mesh_grid: u32_of(
            pipeline.max_total_threadgroups_per_mesh_grid(),
        ),
        reserved_14: 0,
    }
}

/// The reflection reply for a made compute pipeline: the state's own queries.
fn compute_reflection<P: Platform>(
    pipeline: &P::ComputePipeline,
    seqno: u64,
    obj_id: u64,
) -> RgfxPipelineReflection {
    let mut flags = 0;
    if pipeline.support_indirect_command_buffers() {
        flags |= RGFX_PIPELINE_REFLECT_SUPPORT_ICB;
    }
    RgfxPipelineReflection {
        header: reflection_header(seqno),
        obj_id,
        gpu_resource_id: pipeline.gpu_resource_id(),
        unique_identifier: pipeline.guest_unique_identifier(),
        max_total_threads_per_threadgroup: pipeline.max_total_threads_per_threadgroup(),
        static_threadgroup_memory_length: pipeline.static_threadgroup_memory_length(),
        thread_execution_width: pipeline.thread_execution_width() as u32,
        imageblock_sample_length: 0,
        resource_index: pipeline.resource_index() as u32,
        status: RGFX_PIPELINE_OK,
        kind: RGFX_PIPELINE_KIND_COMPUTE,
        flags,
    }
}

/// An exec fault for an op that named an object the host does not hold, or of the
/// wrong kind: the object-resolution API's raise, which propagates out of the op
/// walk for the ring loop's handler to turn into a command-buffer error (as the
/// design's "invalid guest input ends the process at the same op"). Unlike a
/// control-ring create's [`DecodeError::BadReference`], an op-stream reference
/// faults the exec, since the stream is already running on the GPU's queue.
fn bad_reference() -> Raise {
    Raise::foreign("rgfx.decode.object")
        .with_detail("an op named an object the host does not hold, or of the wrong kind")
}

/// An exec fault for a structurally valid op the host cannot honour: an unknown
/// op tag, an unknown group, or an encoder op with no matching open encoder. Like
/// [`bad_reference`], it faults the exec rather than reporting a [`DecodeError`].
pub(crate) fn op_fault(detail: impl std::fmt::Display) -> Raise {
    Raise::foreign("rgfx.decode.op").with_detail(detail)
}

/// What a close took out of its context's tables, to let go of
/// ([`GraphicsDecoder::close_tables`]).
pub(crate) struct Closed<P: Platform> {
    context: u32,
    /// Its objects.
    objects: Vec<tables::Removed<P>>,
    /// Its pages blocks' views ([`surface_pages`]).
    views: Vec<surface_pages::Private<P>>,
}

impl<P: Platform> Closed<P> {
    /// Lets go of every object, then of their CPU copies, then of the views, each in a refusal
    /// scope of its own ([`drop_refused`]), so a release the platform dies at is refused, logged,
    /// and the rest go on. An object a message's view still needs goes when that message ends
    /// ([`tables`]).
    pub(crate) fn let_go(self) {
        let Closed {
            context,
            objects,
            views,
        } = self;
        let mut copies = Vec::new();
        for r in objects {
            let id = r.id();
            let Entry {
                managed, handle, ..
            } = r.into_entry();
            drop_refused(id, handle);
            if let Some(copy) = managed {
                copies.push((id, copy));
            }
        }
        for (id, copy) in copies {
            drop_refused(id, copy);
        }
        if !views.is_empty()
            && let Err(refusal) = refusals::refusing(move || drop(views))
        {
            log::warn!(
                target: "reims_vgpu::graphics",
                "context {context}'s pages blocks' views were let go of at its close, and a \
                 release was refused: {refusal}"
            );
        }
    }
}

/// Lets go of `value`, what the decoder held for object `id`, in a refusal scope
/// ([`refusals::refusing`]): a death in its release, which ends the process outside every scope,
/// is refused there and logged ([`refused_release`]), and the caller goes on. Where the platform
/// died partway, what it had not let go of yet stays held.
pub(crate) fn drop_refused<T>(id: u64, value: T) {
    if let Err(refusal) = refusals::refusing(move || drop(value)) {
        refused_release(id, &refusal);
    }
}

/// Logs the refused release of object `id` ([`drop_refused`]).
fn refused_release(id: u64, refusal: &refusals::Refusal) {
    log::warn!(
        target: "reims_vgpu::graphics",
        "object {id:#x} was let go of, and its release was refused: {refusal}"
    );
}

/// A copy's `failed` that fails the test: the platform raised at it.
#[cfg(test)]
pub(crate) fn no_raise() -> Box<dyn FnOnce(Raise) + Send> {
    Box::new(|raise| panic!("the platform raised at the copy: {raise:?}"))
}

/// `op`, one guest op of the lane's, inside the refusal scope ([`refusals::refusing`]): a death
/// on this thread inside it, a platform refusing a create or an encode among them, which ends
/// the process at its call on the paravirt device, refuses the op instead, as a raise the
/// caller's handler takes. So the op is refused to the guest at its op (a refused reflection, a
/// `COMPLETE` with its status, a failed command buffer) and the process goes on
/// (docs/graphics-kext/design.md, "Guest input the host refuses").
pub(crate) fn refusing<T>(op: impl FnOnce() -> RaiseResult<T>) -> RaiseResult<T> {
    refusals::refusing(op).unwrap_or_else(|refusal| {
        Err(Raise::foreign("rgfx.platform.refused").with_detail(format!("refused: {refusal}")))
    })
}

/// The object-resolution API: the contract the encoder-op decoders
/// ([`render`], [`compute`], [`blit`]) call to turn a wire `obj_id` into a host
/// handle. A `require_*` resolves a reference that must name an object of its kind
/// (nil or a miss is an exec fault, [`bad_reference`]); a `get_*` resolves a
/// nil-allowed reference (`0` is `None`, a non-zero miss is an exec fault). An id that
/// stands for an object whose posted create the host refused is neither: either answers
/// [`refused::reference`], which drops what named it ([`refused`]). Each
/// lends the table's own handle for the borrow of the decoder, which no op's encode
/// outlives: an encode retains what it keeps, and a decoder op of 10,000 a command
/// buffer takes no counted reference of its own ([`Handle::clone_ref`], an atomic
/// add and its release) to name an object.
macro_rules! resolvers {
    ($( ($req:ident, $get:ident, $variant:ident, $assoc:ident, $what:literal) ),* $(,)?) => {
        // The whole object-resolution API: some resolvers are for ops the first slice does not
        // emit yet, so they are not reached but are kept whole (project rule "Unread is not dead").
        #[allow(dead_code)]
        impl<P: Platform> GraphicsDecoder<P> {
            $(
                #[doc = concat!("The ", $what, " `id` names; an exec fault when `id` is 0 or names no ", $what, ".")]
                pub(crate) fn $req(&self, id: u64) -> RaiseResult<&P::$assoc> {
                    match self.named(id) {
                        Some(ObjHandle::$variant(h)) => Ok(h),
                        Some(ObjHandle::Refused(r)) => Err(r.reference(id)),
                        _ => Err(bad_reference()),
                    }
                }
                #[doc = concat!("The ", $what, " `id` names, `None` for 0; an exec fault when a non-zero `id` names no ", $what, ".")]
                pub(crate) fn $get(&self, id: u64) -> RaiseResult<Option<&P::$assoc>> {
                    if id == 0 {
                        return Ok(None);
                    }
                    match self.named(id) {
                        Some(ObjHandle::$variant(h)) => Ok(Some(h)),
                        Some(ObjHandle::Refused(r)) => Err(r.reference(id)),
                        _ => Err(bad_reference()),
                    }
                }
            )*
        }
    };
}

resolvers! {
    (require_buffer, get_buffer, Buffer, Buffer, "buffer"),
    (require_texture, get_texture, Texture, Texture, "texture"),
    (require_sampler, get_sampler, Sampler, SamplerState, "sampler"),
    (require_depth_stencil, get_depth_stencil, DepthStencil, DepthStencilState, "depth/stencil state"),
    (require_render_pipeline, get_render_pipeline, RenderPipeline, RenderPipeline, "render pipeline"),
    (require_compute_pipeline, get_compute_pipeline, ComputePipeline, ComputePipeline, "compute pipeline"),
    (require_fence, get_fence, Fence, Fence, "fence"),
    (require_event, get_event, Event, Event, "event"),
    (require_heap, get_heap, Heap, Heap, "heap"),
    (require_library, get_library, Library, Library, "library"),
    (require_indirect_command_buffer, get_indirect_command_buffer, Icb, Icb, "indirect command buffer"),
}

/// The line of a heap create the platform answered nil for (a size of 0, memory the device
/// cannot allocate, or an address its space cannot honour), which names what the guest asked
/// for, as the refusal's own line names only its message.
fn heap_refused(obj_id: u64, size: u64, heap_type: i32, options: u32, gpu_address: u64) {
    log::warn!(
        target: "reims_vgpu::graphics",
        "heap {obj_id:#x} not made: the platform answered nil for {size:#x} bytes of type \
         {heap_type}, options {options:#x}, at GPU address {gpu_address:#x}"
    );
}

/// The `MTLHeapType` the host makes a guest heap of `heap_type` with: an `Automatic` heap (0) is
/// a `Placement` heap (1), whose resources the guest places itself, at the offsets its own
/// allocator picks, as Metal's `Automatic` heap would (`guest/bundle/src/RgfxHeapRanges.hpp`);
/// any other type is its own. Metal makes both types of a heap at one hazard tracking mode, the
/// options', and answers `Default` as `Untracked` for each (measured: M3 Max, macOS 27), so the
/// heap tracks as the guest's would.
fn host_heap_type(heap_type: i32) -> i64 {
    match heap_type {
        HEAP_TYPE_AUTOMATIC => HEAP_TYPE_PLACEMENT,
        other => other.into(),
    }
}

/// `MTLHeapTypeAutomatic`, as `HEAP_CREATE` carries it.
const HEAP_TYPE_AUTOMATIC: i32 = 0;
/// `MTLHeapTypePlacement`.
const HEAP_TYPE_PLACEMENT: i64 = 1;

/// Reads a message struct from the start of `message`, or `None` if it is too
/// short.
fn read<T: FromBytes>(message: &[u8]) -> Option<T> {
    T::read_from_prefix(message).ok().map(|(t, _)| t)
}

/// The `T` that follows a message's own struct at byte `offset` of `message`, or `None` where the
/// message ends before it (a message that does not carry it).
fn read_at<T: FromBytes>(message: &[u8], offset: u32) -> Option<T> {
    read(message.get(offset as usize..)?)
}

/// The core texture descriptor values from a wire texture descriptor, with the
/// guest-picked resource id pinned as the texture's resource index (the SPI the
/// guest's argument buffers name it by), which the caller checked lies in its
/// context's chunks. A 0 swizzle leaves Metal's identity swizzle.
fn texture_desc(d: &RgfxTextureDescriptor, resource_id: u64) -> TextureDesc {
    TextureDesc {
        texture_type: d.texture_type.into(),
        pixel_format: d.pixel_format.into(),
        width: d.width.into(),
        height: d.height.into(),
        depth: d.depth.into(),
        mipmap_level_count: d.mipmap_level_count.into(),
        sample_count: d.sample_count.into(),
        array_length: d.array_length.into(),
        usage: d.usage.into(),
        resource_options: Some(d.options.into()),
        // Four `MTLTextureSwizzle` channels packed one byte each, red lowest.
        swizzle: (d.swizzle != 0).then(|| {
            let [red, green, blue, alpha] = d.swizzle.to_le_bytes();
            Swizzle {
                red,
                green,
                blue,
                alpha,
            }
        }),
        resource_index: Some((resource_id, true)),
        ..TextureDesc::default()
    }
}

/// A texture over a buffer as its create asked for it (`TEXTURE_FROM_BUFFER`,
/// `LINEAR_TEXTURE_FROM_BUFFER`, `TEXTURE_FROM_IOSURFACE`), for the line that logs its refusal
/// ([`log_refused_texture`]).
struct TextureOverBuffer<'a> {
    obj_id: u64,
    buffer_obj_id: u64,
    /// The buffer's length.
    buffer_length: u64,
    desc: &'a RgfxTextureDescriptor,
    offset: u64,
    bytes_per_row: u64,
    /// A linear texture's bytes between its slices.
    bytes_per_image: Option<u64>,
    /// An IOSurface texture's surface id and plane, and the surface's facts where the create
    /// carried them.
    surface: Option<(u32, u32, Option<RgfxSurfaceFacts>)>,
}

/// Logs a texture over a buffer the platform made none of: it names no reason (Metal answers nil
/// with none), so the line holds every argument the create gave it, from which its rules decide
/// the refusal.
fn log_refused_texture(t: &TextureOverBuffer<'_>) {
    let d = t.desc;
    let image = t
        .bytes_per_image
        .map_or_else(String::new, |bytes| format!(", {bytes} bytes an image"));
    let surface = t.surface.map_or_else(String::new, |(id, plane, facts)| {
        let facts = facts.map_or_else(String::new, |f| {
            format!(
                " (pixel format {}, {} planes, {} x {}, plane 1 at offset {}, {} bytes a row, \
                 YCbCr matrix {})",
                four_characters(f.surface_pixel_format),
                f.plane_count,
                f.width,
                f.height,
                f.chroma_offset,
                f.chroma_bytes_per_row,
                f.ycbcr_matrix
            )
        });
        format!(", IOSurface {id:#x} plane {plane}{facts}")
    });
    log::warn!(
        target: "reims_vgpu::graphics",
        "a texture (object {:#x}) over buffer {:#x} of {} bytes was refused: pixel format {}, \
         type {}, {} x {} x {}, {} levels, array length {}, {} samples, usage {:#x}, resource \
         options {:#x}, swizzle {:#010x}, at offset {}, {} bytes a row{image}{surface}",
        t.obj_id,
        t.buffer_obj_id,
        t.buffer_length,
        d.pixel_format,
        d.texture_type,
        d.width,
        d.height,
        d.depth,
        d.mipmap_level_count,
        d.array_length,
        d.sample_count,
        d.usage,
        d.options,
        d.swizzle,
        t.offset,
        t.bytes_per_row,
    );
}

/// The surface's facts after a `TEXTURE_FROM_IOSURFACE` message's struct `m`, by the message's
/// size ([`RGFX_SURFACE_FACTS_OFFSET`]): none for a message of the struct alone (or one whose size
/// says less, as before the facts), the facts for one of the struct and the facts or more; a size
/// between the two, or a message shorter than the facts, is malformed.
fn surface_facts(
    m: &RgfxTextureFromIOSurface,
    message: &[u8],
) -> Result<Option<RgfxSurfaceFacts>, DecodeError> {
    let start = RGFX_SURFACE_FACTS_OFFSET as usize;
    let end = start + size_of::<RgfxSurfaceFacts>();
    let size = m.header.size as usize;
    if size <= start {
        return Ok(None);
    }
    if size < end {
        return Err(DecodeError::Malformed);
    }
    message
        .get(start..end)
        .and_then(|bytes| RgfxSurfaceFacts::read_from_bytes(bytes).ok())
        .map(Some)
        .ok_or(DecodeError::Malformed)
}

/// Where a `TEXTURE_FROM_IOSURFACE` message's texture lies in its buffer, which stands for the
/// surface, and the surface's facts as the platform takes them, from the message `m` and its
/// `facts` ([`surface_facts`]): plane `m.plane` at `m.offset`, rows `m.bytes_per_row` apart, of a
/// surface with planes where the facts count any; without facts, of none, and the platform makes
/// the texture as over the buffer alone. A YCbCr matrix the wire names no value for is
/// malformed.
fn surface_plane(
    m: &RgfxTextureFromIOSurface,
    facts: Option<&RgfxSurfaceFacts>,
) -> Result<SurfacePlane, DecodeError> {
    let facts = match facts {
        Some(f) => {
            let ycbcr_matrix = match f.ycbcr_matrix {
                RGFX_YCBCR_MATRIX_NONE => None,
                RGFX_YCBCR_MATRIX_ITU_R_601_4 => Some(SurfaceMatrix::Itu601),
                RGFX_YCBCR_MATRIX_ITU_R_709_2 => Some(SurfaceMatrix::Itu709),
                RGFX_YCBCR_MATRIX_ITU_R_2020 => Some(SurfaceMatrix::Itu2020),
                RGFX_YCBCR_MATRIX_SMPTE_240M_1995 => Some(SurfaceMatrix::Smpte240M1995),
                RGFX_YCBCR_MATRIX_DCI_P3 => Some(SurfaceMatrix::DciP3),
                RGFX_YCBCR_MATRIX_P3_D65 => Some(SurfaceMatrix::P3D65),
                RGFX_YCBCR_MATRIX_OTHER => Some(SurfaceMatrix::Other),
                _ => return Err(DecodeError::Malformed),
            };
            Some(SurfaceFacts {
                pixel_format: f.surface_pixel_format,
                plane_count: u64::from(f.plane_count),
                width: u64::from(f.width),
                height: u64::from(f.height),
                chroma_offset: f.chroma_offset,
                chroma_bytes_per_row: f.chroma_bytes_per_row,
                ycbcr_matrix,
            })
        }
        None => None,
    };
    Ok(SurfacePlane {
        plane: u64::from(m.plane),
        offset: m.offset,
        bytes_per_row: m.bytes_per_row,
        planar: facts.is_some_and(|f| f.plane_count > 0),
        facts,
    })
}

/// An OSType as its four characters, quoted, where each is printable ASCII; else its value in hex.
fn four_characters(code: u32) -> String {
    let bytes = code.to_be_bytes();
    match bytes.iter().all(|b| b.is_ascii_graphic() || *b == b' ') {
        true => format!(
            "'{}'",
            bytes.iter().map(|&b| char::from(b)).collect::<String>()
        ),
        false => format!("{code:#010x}"),
    }
}

/// The core stencil descriptor for one face from its wire record.
fn stencil(s: &RgfxStencilDescriptor) -> StencilDesc {
    StencilDesc {
        stencil_compare_function: s.stencil_compare.into(),
        stencil_failure_operation: s.stencil_fail_op.into(),
        depth_failure_operation: s.depth_fail_op.into(),
        depth_stencil_pass_operation: s.pass_op.into(),
        read_mask: s.read_mask,
        write_mask: s.write_mask,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use reims_graphics::wire::{
        RGFX_CMD_HEAP_TEXTURE_SIZE_AND_ALIGN, RGFX_LIBRARY_BYTES_OFFSET, RGFX_RESOURCE_ID_CHUNKS,
        RgfxLibraryCreate, RgfxLibraryFromRegion,
    };
    use reims_vgpu_core::testkit::{Arg, FakePlatform, Step};
    use zerocopy::IntoBytes;

    use super::*;

    /// What the surface views did is logged at every 256th `SURFACE_PAGES`, and at the first one
    /// ten seconds or more after the last log, however few came between.
    #[test]
    fn the_views_counts_are_logged_by_count_and_by_time() {
        let last = MonoTime::from_nanos(5_000_000_000);
        let after = |d: std::time::Duration| last.saturating_add(d);
        let ms = std::time::Duration::from_millis;
        // The 256th logs at once, and so do its multiples.
        assert!(views_log_due(256, after(ms(1)), last));
        assert!(views_log_due(512, last, last));
        // Others wait for the interval.
        assert!(!views_log_due(255, after(ms(1)), last));
        assert!(!views_log_due(3, after(ms(9_999)), last));
        assert!(views_log_due(3, after(ms(10_000)), last));
        assert!(views_log_due(257, after(ms(60_000)), last));
        // A clock that reads earlier than the last log waits as if no time had passed.
        assert!(!views_log_due(3, MonoTime::from_nanos(1), last));
    }

    /// The closed contexts' runs hold each number inserted, joined with the runs it is next to,
    /// and nothing else
    #[test]
    fn runs_hold_what_was_inserted_joined() {
        let mut runs = Runs::default();
        for n in [3, 1, 2, 7, 5, 6, u32::MAX, 0] {
            runs.insert(n);
        }
        for n in [0, 1, 2, 3, 5, 6, 7, u32::MAX] {
            assert!(runs.contains(n), "{n}: {runs:?}");
        }
        for n in [4, 8, u32::MAX - 1] {
            assert!(!runs.contains(n), "{n}: {runs:?}");
        }
        assert_eq!(runs.0.len(), 3, "{runs:?}");
        runs.insert(4);
        runs.insert(4);
        assert_eq!(
            runs.0.iter().map(|(&a, &b)| (a, b)).collect::<Vec<_>>(),
            [(0, 7), (u32::MAX, u32::MAX)]
        );
        // A context opened again leaves its run split around it.
        runs.remove(4);
        runs.remove(0);
        runs.remove(u32::MAX);
        runs.remove(9);
        assert_eq!(
            runs.0.iter().map(|(&a, &b)| (a, b)).collect::<Vec<_>>(),
            [(1, 3), (5, 7)]
        );
        runs.clear();
        assert!(!runs.contains(1));
    }

    /// A library from region chunks: each chunk is read at its offset and acked, and
    /// the library is made, from the bytes unchanged, when the chunks reach its length;
    /// a span outside the processes' blocks is a bad reference and a chunk out of order
    /// is malformed.
    #[test]
    fn makes_a_library_from_region_chunks() {
        let p = Arc::new(FakePlatform::new());
        let device = p.device();
        // The processes' blocks start past the rings and scanout; one MiB of them, then the
        // cursor's span.
        let blocks = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
        let size = blocks + 0x10_0000 + RGFX_CURSOR_SPAN;
        let region = Arc::new(reims_graphics::region::Region::new(size as usize).unwrap());
        let (staging, ack) = (blocks + 0x4000, blocks + 0x100);
        let air: Vec<u8> = (0..0x2345u32).map(|i| (i * 7) as u8).collect();
        let mut dec = GraphicsDecoder::<FakePlatform>::new(Arc::clone(&p), Arc::clone(&region));

        let msg =
            |obj_id: u64, region_offset: u64, len: u64, total: u64, at: u64, ack_value: u64| {
                RgfxLibraryFromRegion {
                    header: RgfxMsgHeader {
                        r#type: RGFX_CMD_LIBRARY_FROM_REGION,
                        size: size_of::<RgfxLibraryFromRegion>() as u32,
                        seqno: 0,
                    },
                    obj_id,
                    region_offset,
                    byte_length: len,
                    total_length: total,
                    chunk_offset: at,
                    ack_offset: ack,
                    ack_value,
                    ..RgfxLibraryFromRegion::default()
                }
            };
        let total = air.len() as u64;
        // Two chunks through one staging block, each acked once copied out.
        let first = 0x1000usize;
        assert!(region.write(staging, &air[..first]));
        assert_eq!(
            dec.decode(
                &device,
                msg(20, staging, first as u64, total, 0, 1).as_bytes()
            )
            .unwrap(),
            Ok(())
        );
        assert_eq!(region.read(ack, 8).unwrap(), 1u64.to_le_bytes());
        assert_eq!(dec.object_count(), 0);
        assert!(region.write(staging, &air[first..]));
        let rest = total - first as u64;
        assert_eq!(
            dec.decode(
                &device,
                msg(20, staging, rest, total, first as u64, 2).as_bytes()
            )
            .unwrap(),
            Ok(())
        );
        assert_eq!(region.read(ack, 8).unwrap(), 2u64.to_le_bytes());
        assert_eq!(dec.object_count(), 1);
        let calls = p.call_log().calls_to("newLibraryWithData:error:");
        assert_eq!(calls.len(), 1);
        assert!(matches!(&calls[0].args[0], Arg::Bytes(b) if *b == air));

        // A span past the processes' blocks, one over the scanout, and a chunk with no first
        // chunk before it.
        for offset in [size - RGFX_CURSOR_SPAN - 0x1000, blocks - 0x1000] {
            assert_eq!(
                dec.decode(&device, msg(21, offset, 0x2000, 0x2000, 0, 3).as_bytes())
                    .unwrap(),
                Err(DecodeError::BadReference)
            );
        }
        assert_eq!(
            dec.decode(&device, msg(22, staging, 0x10, 0x20, 0x10, 4).as_bytes())
                .unwrap(),
            Err(DecodeError::Malformed)
        );
        assert_eq!(dec.object_count(), 1);
    }

    /// Host-memory buffers and heaps lie in the processes' blocks: a span over the rings and
    /// scanout, past the blocks' end or past the region's is refused, one inside is made; and a
    /// create names only its own process's objects once a context is set
    #[test]
    fn host_memory_and_references_stay_in_their_process() {
        let p = Arc::new(FakePlatform::new());
        let device = p.device();
        let blocks = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
        let size = blocks + 0x10_0000 + RGFX_CURSOR_SPAN;
        let region = Arc::new(reims_graphics::region::Region::new(size as usize).unwrap());
        let mut dec = GraphicsDecoder::<FakePlatform>::new(Arc::clone(&p), Arc::clone(&region));
        let buffer = |obj_id: u64, region_offset: u64, length: u64| RgfxBufferOverHostMemory {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_BUFFER_OVER_HOST_MEMORY,
                size: size_of::<RgfxBufferOverHostMemory>() as u32,
                seqno: 0,
            },
            obj_id,
            gpu_address: 0x1_0000_0000,
            region_offset,
            length,
            options: 0,
            reserved_34: 0,
        };
        let heap = |obj_id: u64, region_offset: u64, size_bytes: u64| RgfxHeapOverHostMemory {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_HEAP_OVER_HOST_MEMORY,
                size: size_of::<RgfxHeapOverHostMemory>() as u32,
                seqno: 0,
            },
            obj_id,
            gpu_address: 0x2_0000_0000,
            region_offset,
            size_bytes,
            ..RgfxHeapOverHostMemory::default()
        };
        let outside = [
            (0, 0x1000),
            (blocks - 0x1000, 0x2000),
            (size - RGFX_CURSOR_SPAN - 0x1000, 0x2000),
            (size - 0x1000, 0x1000),
            (u64::MAX - 0xfff, 0x1000),
        ];
        for (offset, len) in outside {
            assert_eq!(
                dec.decode(&device, buffer(1, offset, len).as_bytes())
                    .unwrap(),
                Err(DecodeError::BadReference),
                "a buffer at {offset:#x}"
            );
            assert_eq!(
                dec.decode(&device, heap(2, offset, len).as_bytes())
                    .unwrap(),
                Err(DecodeError::BadReference),
                "a heap at {offset:#x}"
            );
        }
        assert_eq!(dec.object_count(), 0);
        let one = 1u64 << RGFX_CONTEXT_OBJ_ID_SHIFT;
        let two = 2u64 << RGFX_CONTEXT_OBJ_ID_SHIFT;
        // With a context, a span must lie in one block the kext granted that context.
        region.grant(1, blocks, 0x2_0000);
        region.grant(2, blocks + 0x8_0000, 0x1_0000);
        dec.set_context(Some(1));
        for (offset, len, what) in [
            (blocks + 0x4_0000, 0x1000, "in no block"),
            (blocks + 0x8_0000, 0x1000, "in another process's block"),
            (blocks + 0x1_f000, 0x2000, "past its block's end"),
        ] {
            assert_eq!(
                dec.decode(&device, buffer(one | 9, offset, len).as_bytes())
                    .unwrap(),
                Err(DecodeError::BadReference),
                "a buffer {what}"
            );
        }
        assert_eq!(
            dec.decode(&device, heap(one | 2, blocks, 0x1_0000).as_bytes())
                .unwrap(),
            Ok(())
        );
        assert_eq!(
            dec.decode(
                &device,
                buffer(one | 3, blocks + 0x1_0000, 0x1000).as_bytes()
            )
            .unwrap(),
            Ok(())
        );
        // Process 2 names process 1's heap and buffer: refused, nothing made.
        dec.set_context(Some(2));
        let at_offset = RgfxBufferAtOffset {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_BUFFER_AT_OFFSET,
                size: size_of::<RgfxBufferAtOffset>() as u32,
                seqno: 0,
            },
            obj_id: two | 4,
            heap_obj_id: one | 2,
            length: 0x100,
            ..RgfxBufferAtOffset::default()
        };
        assert_eq!(
            dec.decode(&device, at_offset.as_bytes()).unwrap(),
            Err(DecodeError::BadReference)
        );
        assert!(dec.require_buffer(one | 3).is_err());
        dec.set_context(Some(1));
        assert!(dec.require_buffer(one | 3).is_ok());
        assert_eq!(dec.object_count(), 2);
    }

    /// A Shared heap over host memory is made over its block in place, the guest's size rounded
    /// up to Metal's 16 KiB page, at the guest's address; one whose rounded size runs past its
    /// block is a bad reference. A buffer over host memory lies over its bytes at the guest's
    /// address, as a BUFFER_CREATE's does.
    #[test]
    fn host_memory_heaps_and_buffers_lie_in_place_at_the_guest_address() {
        let p = Arc::new(FakePlatform::new());
        let device = p.device();
        let blocks = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
        let size = blocks + 0x10_0000 + RGFX_CURSOR_SPAN;
        let region = Arc::new(reims_graphics::region::Region::new(size as usize).unwrap());
        let mut dec = GraphicsDecoder::<FakePlatform>::new(Arc::clone(&p), Arc::clone(&region));
        region.grant(1, blocks, 0x2_0000);
        dec.set_context(Some(1));
        let one = 1u64 << RGFX_CONTEXT_OBJ_ID_SHIFT;
        let heap = |obj_id: u64, region_offset: u64, size_bytes: u64| RgfxHeapOverHostMemory {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_HEAP_OVER_HOST_MEMORY,
                size: size_of::<RgfxHeapOverHostMemory>() as u32,
                seqno: 0,
            },
            obj_id,
            gpu_address: 0x2_0000_8000,
            region_offset,
            size_bytes,
            ..RgfxHeapOverHostMemory::default()
        };
        let mark = p.call_log().mark();
        assert_eq!(
            dec.decode(&device, heap(one | 1, blocks, 0x1_0001).as_bytes())
                .unwrap(),
            Ok(())
        );
        let calls = p.call_log().since(mark);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].sel, "newHeapWithDescriptor:");
        assert_eq!(
            calls[0].args[..2],
            [
                Arg::Ptr(region.base() as usize + blocks as usize),
                Arg::U64(0x1_4000)
            ]
        );
        assert_eq!(
            dec.require_heap(one | 1).unwrap().gpu_address(),
            0x2_0000_8000
        );
        // 0x4001 bytes at 0x1c000 into the block take 0x8000, past its end.
        assert_eq!(
            dec.decode(&device, heap(one | 2, blocks + 0x1_c000, 0x4001).as_bytes())
                .unwrap(),
            Err(DecodeError::BadReference)
        );
        let buffer = RgfxBufferOverHostMemory {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_BUFFER_OVER_HOST_MEMORY,
                size: size_of::<RgfxBufferOverHostMemory>() as u32,
                seqno: 0,
            },
            obj_id: one | 3,
            gpu_address: 0x3_0000_0000,
            region_offset: blocks + 0x1_8000,
            length: 0x100,
            options: 0,
            reserved_34: 0,
        };
        assert_eq!(dec.decode(&device, buffer.as_bytes()).unwrap(), Ok(()));
        let made = dec.require_buffer(one | 3).unwrap();
        assert_eq!(made.gpu_address(), 0x3_0000_0000);
        assert_eq!(
            made.contents().addr(),
            region.base() as usize + (blocks + 0x1_8000) as usize
        );
    }

    /// An Automatic heap is made a Placement heap, the guest placing its resources itself; a
    /// Placement heap and a Shared heap over host memory keep their type, an Automatic one of
    /// those is a Placement heap too.
    #[test]
    fn automatic_heaps_are_made_placement_heaps() {
        let p = Arc::new(FakePlatform::new());
        let device = p.device();
        let blocks = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
        let size = blocks + 0x10_0000 + RGFX_CURSOR_SPAN;
        let region = Arc::new(reims_graphics::region::Region::new(size as usize).unwrap());
        let mut dec = GraphicsDecoder::<FakePlatform>::new(Arc::clone(&p), Arc::clone(&region));
        region.grant(1, blocks, 0x2_0000);
        dec.set_context(Some(1));
        let one = 1u64 << RGFX_CONTEXT_OBJ_ID_SHIFT;
        let heap = |obj_id: u64, heap_type: i32| RgfxHeapCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_HEAP_CREATE,
                size: size_of::<RgfxHeapCreate>() as u32,
                seqno: 0,
            },
            obj_id,
            gpu_address: 0x2_0000_0000 + (obj_id & 0xff) * 0x10_0000,
            size_bytes: 0x1_0000,
            options: 0,
            heap_type,
        };
        let over = |obj_id: u64, heap_type: i32| RgfxHeapOverHostMemory {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_HEAP_OVER_HOST_MEMORY,
                size: size_of::<RgfxHeapOverHostMemory>() as u32,
                seqno: 0,
            },
            obj_id,
            gpu_address: 0x3_0000_0000 + (obj_id & 0xff) * 0x10_0000,
            region_offset: blocks + (obj_id & 0xff) * 0x1_0000 - 0x3_0000,
            size_bytes: 0x1_0000,
            heap_type,
            ..RgfxHeapOverHostMemory::default()
        };
        let made_type = |message: &[u8], dec: &mut GraphicsDecoder<FakePlatform>| {
            let mark = p.call_log().mark();
            assert_eq!(dec.decode(&device, message).unwrap(), Ok(()));
            let calls = p.call_log().since(mark);
            let Some(Arg::Val(desc)) = calls[0].args.last() else {
                panic!("no descriptor: {calls:?}");
            };
            let at = desc.find("heap_type: ").unwrap() + "heap_type: ".len();
            desc[at..].split(',').next().unwrap().to_owned()
        };
        assert_eq!(made_type(heap(one | 1, 0).as_bytes(), &mut dec), "1");
        assert_eq!(made_type(heap(one | 2, 1).as_bytes(), &mut dec), "1");
        assert_eq!(made_type(over(one | 3, 0).as_bytes(), &mut dec), "1");
        assert_eq!(made_type(over(one | 4, 1).as_bytes(), &mut dec), "1");
    }

    /// A queue create whose tail holds `RGFX_QUEUE_UNSHARED` makes the platform's unshared queue,
    /// and one without it (no tail, a tail of none, or of other flags) the plain `newCommandQueue`,
    /// each stored under its id, and then its surface timeline and its device caches' end event,
    /// the first queue's followed by the caches' transfer queue, unshared, and its event, and their
    /// read-back queue, plain, and its end event. A queue the platform makes nothing of, either
    /// way, is refused, and stores and makes nothing more.
    #[test]
    fn an_unshared_queue_create_makes_the_platforms_unshared_queue() {
        use reims_graphics::wire::{
            RGFX_QUEUE_QUIET_SCHEDULED, RGFX_QUEUE_UNSHARED, RgfxQueueCreateFlags,
        };
        use reims_vgpu_core::testkit::impls::device::UNSHARED_QUEUE;

        let p = Arc::new(FakePlatform::new());
        let device = p.device();
        let region = Arc::new(reims_graphics::region::Region::new(0x1_0000).unwrap());
        let mut dec = GraphicsDecoder::<FakePlatform>::new(Arc::clone(&p), Arc::clone(&region));
        let create = |obj_id: u64, flags: Option<u32>| {
            let mut bytes = RgfxQueueCreate {
                header: RgfxMsgHeader {
                    r#type: RGFX_CMD_QUEUE_CREATE,
                    size: 0,
                    seqno: 0,
                },
                obj_id,
                submit_offset: 0x2_0000,
                complete_offset: 0x3_0000,
                submit_data_size: 0x1_0000,
                complete_data_size: 0x1_0000,
                priority: 0,
                vector: 0,
            }
            .as_bytes()
            .to_vec();
            if let Some(flags) = flags {
                let tail = RgfxQueueCreateFlags {
                    flags,
                    reserved_04: 0,
                };
                bytes.extend_from_slice(tail.as_bytes());
            }
            let size = bytes.len() as u32;
            bytes[4..8].copy_from_slice(&size.to_le_bytes());
            bytes
        };
        // What the decode answers, and the platform calls it made.
        let decode = |dec: &mut GraphicsDecoder<FakePlatform>, bytes: &[u8]| {
            let mark = p.call_log().mark();
            let outcome = dec.decode(&device, bytes).unwrap();
            let calls: Vec<&str> = p.call_log().since(mark).iter().map(|c| c.sel).collect();
            (outcome, calls)
        };
        // A made queue's calls: the queue, its surface timeline and its end event.
        let made = |queue: &'static str| vec![queue, "newSharedEvent", "newEvent"];
        let plain = made("newCommandQueue");

        let unshared = Some(RGFX_QUEUE_UNSHARED);
        let mut first = made(UNSHARED_QUEUE);
        first.extend([UNSHARED_QUEUE, "newEvent", "newCommandQueue", "newEvent"]);
        assert_eq!(decode(&mut dec, &create(1, unshared)), (Ok(()), first));
        let both = Some(RGFX_QUEUE_UNSHARED | RGFX_QUEUE_QUIET_SCHEDULED);
        assert_eq!(
            decode(&mut dec, &create(2, both)),
            (Ok(()), made(UNSHARED_QUEUE))
        );
        assert_eq!(decode(&mut dec, &create(3, None)), (Ok(()), plain.clone()));
        assert_eq!(
            decode(&mut dec, &create(4, Some(0))),
            (Ok(()), plain.clone())
        );
        let quiet = Some(RGFX_QUEUE_QUIET_SCHEDULED);
        assert_eq!(decode(&mut dec, &create(5, quiet)), (Ok(()), plain));
        for obj_id in 1..=5 {
            assert!(
                matches!(dec.named(obj_id), Some(ObjHandle::CommandQueue(_))),
                "queue {obj_id}"
            );
        }

        // Each call's nil refuses its create, before any event is made.
        p.script().push(UNSHARED_QUEUE, Step::Nil);
        assert_eq!(
            decode(&mut dec, &create(6, unshared)),
            (Err(DecodeError::Refused), vec![UNSHARED_QUEUE])
        );
        p.script().push("newCommandQueue", Step::Nil);
        assert_eq!(
            decode(&mut dec, &create(7, None)),
            (Err(DecodeError::Refused), vec!["newCommandQueue"])
        );
        assert!(dec.named(6).is_none() && dec.named(7).is_none());
    }

    /// A sampler create is answered with the sampler's `gpuResourceID`, the id the platform
    /// numbers it by, at the create's seqno, the sampler stored under its id first; a create the
    /// platform makes nothing of is refused with 0 and stores nothing, and a short one is an
    /// invalid argument.
    #[test]
    fn a_sampler_create_answers_the_platforms_id() {
        let p = Arc::new(FakePlatform::new());
        let device = p.device();
        let region = Arc::new(reims_graphics::region::Region::new(0x1_0000).unwrap());
        let mut dec = GraphicsDecoder::<FakePlatform>::new(Arc::clone(&p), Arc::clone(&region));
        let create = |obj_id, seqno| RgfxSamplerCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_SAMPLER_CREATE,
                size: size_of::<RgfxSamplerCreate>() as u32,
                seqno,
            },
            obj_id,
            max_anisotropy: 1,
            flags: 0b101,
            lod_max_clamp_bits: f32::MAX.to_bits(),
            ..RgfxSamplerCreate::default()
        };
        let first = dec
            .sampler_state(&device, create(21, 4).as_bytes())
            .unwrap()
            .unwrap();
        assert_eq!(first.header.r#type, RGFX_EVT_SAMPLER_STATE);
        assert_eq!(first.header.size as usize, size_of::<RgfxSamplerState>());
        assert_eq!((first.header.seqno, first.status), (4, RGFX_STATUS_OK));
        assert_ne!(first.gpu_resource_id, 0);
        assert!(dec.named(21).is_some(), "stored before its answer");
        let second = dec
            .sampler_state(&device, create(22, 5).as_bytes())
            .unwrap()
            .unwrap();
        assert_eq!(second.status, RGFX_STATUS_OK);
        assert_ne!(second.gpu_resource_id, first.gpu_resource_id);
        assert!(dec.named(21).is_some() && dec.named(22).is_some());
        // A create of the same id takes its place.
        dec.end_message();
        let third = dec
            .sampler_state(&device, create(21, 8).as_bytes())
            .unwrap()
            .unwrap();
        let Some(ObjHandle::Sampler(stored)) = dec.named(21) else {
            panic!("no sampler under 21");
        };
        assert_eq!(stored.gpu_resource_id(), third.gpu_resource_id);

        p.script().push("newSamplerStateWithDescriptor:", Step::Nil);
        let refused = dec
            .sampler_state(&device, create(23, 6).as_bytes())
            .unwrap()
            .unwrap();
        assert_eq!(
            (
                refused.header.seqno,
                refused.status,
                refused.gpu_resource_id
            ),
            (6, RGFX_STATUS_REFUSED, 0)
        );
        assert!(dec.named(23).is_none());
        let short = create(24, 7);
        let short = &short.as_bytes()[..size_of::<RgfxMsgHeader>() + 8];
        let invalid = dec.sampler_state(&device, short).unwrap().unwrap();
        assert_eq!(
            (invalid.header.seqno, invalid.status),
            (7, RGFX_STATUS_INVALID_ARGUMENT)
        );
        assert!(dec.sampler_state(&device, &[0u8; 8]).unwrap().is_none());
    }

    /// A heap texture size query is answered at its seqno with the device's size and alignment
    /// for the shape, and makes nothing; one the device raises at is refused with both 0, and a
    /// query too short for its shape is an invalid argument.
    #[test]
    fn heap_texture_sizes_are_the_devices() {
        let p = Arc::new(FakePlatform::new());
        let device = p.device();
        let region = Arc::new(reims_graphics::region::Region::new(0x1_0000).unwrap());
        let dec = GraphicsDecoder::<FakePlatform>::new(Arc::clone(&p), Arc::clone(&region));
        let query = RgfxHeapTextureSizeAndAlign {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_HEAP_TEXTURE_SIZE_AND_ALIGN,
                size: size_of::<RgfxHeapTextureSizeAndAlign>() as u32,
                seqno: 7,
            },
            obj_id: (1u64 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 5,
            desc: RgfxTextureDescriptor {
                pixel_format: 80,
                texture_type: 2,
                width: 100,
                height: 100,
                depth: 1,
                mipmap_level_count: 1,
                array_length: 1,
                sample_count: 1,
                ..RgfxTextureDescriptor::default()
            },
        };
        let mark = p.call_log().mark();
        let answer = dec
            .heap_texture_size_and_align(&device, query.as_bytes())
            .unwrap();
        // The fake's size: 4 bytes a texel, at its 16 KiB alignment.
        assert_eq!(
            answer,
            RgfxSizeAndAlign {
                header: RgfxMsgHeader {
                    r#type: RGFX_EVT_SIZE_AND_ALIGN,
                    size: size_of::<RgfxSizeAndAlign>() as u32,
                    seqno: 7,
                },
                status: RGFX_STATUS_OK,
                reserved_14: 0,
                size: 0xc000,
                align: 0x4000,
            }
        );
        let calls = p.call_log().since(mark);
        assert!(
            calls
                .iter()
                .any(|c| c.sel == "heapTextureSizeAndAlignWithDescriptor:"),
            "{calls:?}"
        );
        assert!(dec.named(query.obj_id).is_none());

        p.script().push(
            "heapTextureSizeAndAlignWithDescriptor:",
            Step::raise("NSInvalidArgumentException", "x"),
        );
        let refused = dec
            .heap_texture_size_and_align(&device, query.as_bytes())
            .unwrap();
        assert_eq!(
            (
                refused.header.seqno,
                refused.status,
                refused.size,
                refused.align
            ),
            (7, RGFX_STATUS_REFUSED, 0, 0)
        );

        let short = &query.as_bytes()[..size_of::<RgfxMsgHeader>() + 8];
        let invalid = dec.heap_texture_size_and_align(&device, short).unwrap();
        assert_eq!(
            (invalid.header.seqno, invalid.status),
            (7, RGFX_STATUS_INVALID_ARGUMENT)
        );
        assert!(
            dec.heap_texture_size_and_align(&device, &query.as_bytes()[..8])
                .is_none()
        );
    }

    /// A message names what its view sees, through the entries it found last ([`Recent`]): an
    /// object it destroys is named no more, and one it makes again under the same id is the new
    /// one, as are two ids at one slot of the cache. One another thread destroys meanwhile is
    /// still named, as the view sees it, and outlives the message's naming of it; the next
    /// message sees the destroy.
    #[test]
    fn a_message_names_what_its_own_destroys_and_makes_left() {
        use reims_vgpu_core::platform::device::GpuDevice as _;
        let p = Arc::new(FakePlatform::new());
        let device = p.device();
        let region = Arc::new(reims_graphics::region::Region::new(0x10000).unwrap());
        let mut dec = GraphicsDecoder::<FakePlatform>::new(Arc::clone(&p), region);
        dec.set_context(Some(1));
        let fence = || ObjHandle::Fence(device.new_fence().unwrap().unwrap());
        let event = || ObjHandle::Event(device.new_event(OnFault::Return).unwrap().unwrap());
        let is_fence = |h: Option<&ObjHandle<FakePlatform>>| matches!(h, Some(ObjHandle::Fence(_)));
        let is_event = |h: Option<&ObjHandle<FakePlatform>>| matches!(h, Some(ObjHandle::Event(_)));
        let a = (1 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 5;
        let b = a + RECENT as u64;

        // Found, then found again (from the cache).
        assert_eq!(dec.insert(a, fence()).unwrap(), Ok(()));
        assert!(is_fence(dec.named(a)));
        assert!(is_fence(dec.named(a)));
        // Its own destroy: named no more.
        dec.remove(a);
        assert!(dec.named(a).is_none());
        // Made again under the same id: the new object.
        assert_eq!(dec.insert(a, event()).unwrap(), Ok(()));
        assert!(is_event(dec.named(a)));
        // Another id at the same slot, before and after its make, and the first again.
        assert!(dec.named(b).is_none());
        assert_eq!(dec.insert(b, fence()).unwrap(), Ok(()));
        assert!(is_fence(dec.named(b)));
        assert!(is_event(dec.named(a)));
        assert!(is_fence(dec.named(b)));

        // A destroy on another thread's handle (ring 0's), while this message names `a`: the view
        // still sees it, and keeps it until the message ends.
        let mut ring0 = dec.sender(Some(1), Sinks::default());
        std::thread::scope(|s| {
            s.spawn(move || {
                ring0.remove(a);
                ring0.remove(b);
                ring0.end_message();
            });
        });
        assert!(is_event(dec.named(a)));
        assert!(is_fence(dec.named(b)));
        dec.end_message();
        // The next message sees the destroys (each answer read through, where there is one).
        assert!(!is_event(dec.named(a)) && dec.named(a).is_none());
        assert!(!is_fence(dec.named(b)) && dec.named(b).is_none());
    }

    /// A context's close sets each of its shared events past every value on the CPU, and signals
    /// its plain events past every value with one command buffer of the decoder's own unshared
    /// queue, made at the first close that has plain events and kept: nothing is left to set or
    /// signal them, so a command buffer of the context's waiting on one finishes and lets go of
    /// what it holds. A context with no event calls nothing, and so does a reset after the close;
    /// a reset sets those of the contexts it drops.
    #[test]
    fn a_gone_context_s_events_are_set_past_every_value() {
        use reims_vgpu_core::testkit::Call;
        use reims_vgpu_core::testkit::impls::device::UNSHARED_QUEUE;

        let p = Arc::new(FakePlatform::new());
        let device = p.device();
        let mut dec = GraphicsDecoder::<FakePlatform>::new(
            Arc::clone(&p),
            Arc::new(reims_graphics::region::Region::new(0x10000).unwrap()),
        );
        let obj = |context: u64, n: u64| context << RGFX_CONTEXT_OBJ_ID_SHIFT | n;
        let create = |ty: u32, obj_id: u64| {
            RgfxObjectCreate {
                header: RgfxMsgHeader {
                    r#type: ty,
                    size: size_of::<RgfxObjectCreate>() as u32,
                    seqno: 0,
                },
                obj_id,
            }
            .as_bytes()
            .to_vec()
        };
        let event = |obj_id| create(RGFX_CMD_EVENT_CREATE, obj_id);
        let fence = |obj_id| create(RGFX_CMD_FENCE_CREATE, obj_id);
        let shared = |obj_id: u64| {
            RgfxSharedEventCreate {
                header: RgfxMsgHeader {
                    r#type: RGFX_CMD_SHARED_EVENT_CREATE,
                    size: size_of::<RgfxSharedEventCreate>() as u32,
                    seqno: 0,
                },
                obj_id,
                initial_value: 3,
            }
            .as_bytes()
            .to_vec()
        };
        let made = |dec: &mut GraphicsDecoder<FakePlatform>, context: u32, messages: &[Vec<u8>]| {
            dec.set_context(Some(context));
            for m in messages {
                assert_eq!(dec.decode(&device, m).unwrap(), Ok(()));
            }
            dec.set_context(None);
        };
        let id_of = |dec: &GraphicsDecoder<FakePlatform>, obj_id: u64| match dec.named(obj_id) {
            Some(ObjHandle::Event(e)) => Arg::obj(e).as_obj().unwrap().0,
            Some(ObjHandle::SharedEvent(e)) => Arg::obj(e).as_obj().unwrap().0,
            _ => panic!("{obj_id:#x} is no event"),
        };
        let sets = |calls: &[Call]| -> Vec<(u64, u64)> {
            (calls.iter())
                .filter(|c| c.sel == "setSignaledValue:")
                .map(|c| (c.obj.0, c.args[0].as_u64().unwrap()))
                .collect()
        };
        // In the order the table lets go of them, which is not the order made.
        let signals = |calls: &[Call]| -> Vec<(u64, u64)> {
            let mut signals: Vec<(u64, u64)> = (calls.iter())
                .filter(|c| c.sel == "encodeSignalEvent:value:")
                .map(|c| (c.args[0].as_obj().unwrap().0, c.args[1].as_u64().unwrap()))
                .collect();
            signals.sort_unstable();
            signals
        };
        let release = [
            "setSignaledValue:",
            UNSHARED_QUEUE,
            "commandBuffer",
            "encodeSignalEvent:value:",
            "commit",
        ];
        let releasing = |calls: &[Call]| -> Vec<&'static str> {
            (calls.iter())
                .map(|c| c.sel)
                .filter(|s| release.contains(s))
                .collect()
        };

        // Context 1: two plain events, a shared one and a fence; context 2: a plain event and a
        // shared one; context 3: a fence alone.
        made(
            &mut dec,
            1,
            &[
                event(obj(1, 1)),
                event(obj(1, 2)),
                shared(obj(1, 3)),
                fence(obj(1, 4)),
            ],
        );
        made(&mut dec, 2, &[event(obj(2, 1)), shared(obj(2, 2))]);
        made(&mut dec, 3, &[fence(obj(3, 1))]);
        let (e11, e12) = (id_of(&dec, obj(1, 1)), id_of(&dec, obj(1, 2)));
        let (s13, e21, s22) = (
            id_of(&dec, obj(1, 3)),
            id_of(&dec, obj(2, 1)),
            id_of(&dec, obj(2, 2)),
        );

        // The lookups above hold views of the contexts' tables open, as a walk does: the close
        // sets the events all the same, though the views keep the entries taken out.
        let mark = p.call_log().mark();
        assert_eq!(dec.drop_context(&device, 1), 4);
        let calls = p.call_log().since(mark);
        assert_eq!(sets(&calls), [(s13, u64::MAX)]);
        let mut both = [(e11, u64::MAX), (e12, u64::MAX)];
        both.sort_unstable();
        assert_eq!(signals(&calls), both);
        assert_eq!(
            releasing(&calls),
            [
                "setSignaledValue:",
                UNSHARED_QUEUE,
                "commandBuffer",
                "encodeSignalEvent:value:",
                "encodeSignalEvent:value:",
                "commit",
            ]
        );

        // The queue is made once.
        let mark = p.call_log().mark();
        assert_eq!(dec.drop_context(&device, 2), 2);
        let calls = p.call_log().since(mark);
        assert_eq!(sets(&calls), [(s22, u64::MAX)]);
        assert_eq!(signals(&calls), [(e21, u64::MAX)]);
        assert_eq!(
            releasing(&calls),
            [
                "setSignaledValue:",
                "commandBuffer",
                "encodeSignalEvent:value:",
                "commit",
            ]
        );

        // A context with no event, and a reset after the closes, set nothing.
        let mark = p.call_log().mark();
        assert_eq!(dec.drop_context(&device, 3), 1);
        assert_eq!(dec.reset(&device).1, 0);
        assert!(releasing(&p.call_log().since(mark)).is_empty());

        // A reset drops a context's events as its close does.
        made(&mut dec, 4, &[event(obj(4, 1)), shared(obj(4, 2))]);
        let (e41, s42) = (id_of(&dec, obj(4, 1)), id_of(&dec, obj(4, 2)));
        let mark = p.call_log().mark();
        assert_eq!(dec.reset(&device), (vec![4], 2));
        let calls = p.call_log().since(mark);
        assert_eq!(sets(&calls), [(s42, u64::MAX)]);
        assert_eq!(signals(&calls), [(e41, u64::MAX)]);
    }

    /// A create of each state object calls the platform and holds its handle;
    /// an object destroy drops it; an unknown command and a short message are
    /// reported, not carried out.
    #[test]
    fn creates_and_destroys_the_state_objects() {
        let p = Arc::new(FakePlatform::new());
        let device = p.device();
        let mut dec = GraphicsDecoder::<FakePlatform>::new(
            Arc::clone(&p),
            Arc::new(reims_graphics::region::Region::new(0x10000).unwrap()),
        );

        let obj = |ty: u32, obj_id: u64| RgfxObjectCreate {
            header: RgfxMsgHeader {
                r#type: ty,
                size: size_of::<RgfxObjectCreate>() as u32,
                seqno: 0,
            },
            obj_id,
        };

        // An event, a fence and a shared event, each by its own command.
        assert_eq!(
            dec.decode(&device, obj(RGFX_CMD_EVENT_CREATE, 10).as_bytes())
                .unwrap(),
            Ok(())
        );
        assert_eq!(
            dec.decode(&device, obj(RGFX_CMD_FENCE_CREATE, 11).as_bytes())
                .unwrap(),
            Ok(())
        );
        let shared = RgfxSharedEventCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_SHARED_EVENT_CREATE,
                size: size_of::<RgfxSharedEventCreate>() as u32,
                seqno: 0,
            },
            obj_id: 12,
            initial_value: 0,
        };
        assert_eq!(dec.decode(&device, shared.as_bytes()).unwrap(), Ok(()));

        // A command queue and its ring placement.
        let queue = RgfxQueueCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_QUEUE_CREATE,
                size: size_of::<RgfxQueueCreate>() as u32,
                seqno: 0,
            },
            obj_id: 13,
            submit_offset: 0x2_0000,
            complete_offset: 0x3_0000,
            submit_data_size: 0x1_0000,
            complete_data_size: 0x1_0000,
            priority: 1,
            vector: 1,
        };
        assert_eq!(dec.decode(&device, queue.as_bytes()).unwrap(), Ok(()));

        // A library from its AIR bytes, which follow the header.
        let air = [1u8, 2, 3, 4, 5, 6];
        let mut lib_msg = RgfxLibraryCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_LIBRARY_CREATE,
                size: (RGFX_LIBRARY_BYTES_OFFSET as usize + air.len()).next_multiple_of(8) as u32,
                seqno: 0,
            },
            obj_id: 14,
            byte_length: air.len() as u64,
            ..RgfxLibraryCreate::default()
        }
        .as_bytes()
        .to_vec();
        lib_msg.extend_from_slice(&air);
        assert_eq!(dec.decode(&device, &lib_msg).unwrap(), Ok(()));

        // A sampler and a depth/stencil state, each from a full descriptor.
        let sampler = RgfxSamplerCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_SAMPLER_CREATE,
                size: size_of::<RgfxSamplerCreate>() as u32,
                seqno: 0,
            },
            obj_id: 15,
            max_anisotropy: 1,
            flags: 0b101,
            lod_max_clamp_bits: f32::MAX.to_bits(),
            ..RgfxSamplerCreate::default()
        };
        assert_eq!(dec.decode(&device, sampler.as_bytes()).unwrap(), Ok(()));
        let ds = RgfxDepthStencilCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_DEPTH_STENCIL_CREATE,
                size: size_of::<RgfxDepthStencilCreate>() as u32,
                seqno: 0,
            },
            obj_id: 16,
            depth_compare: 1,
            depth_write_enabled: 1,
            front: RgfxStencilDescriptor {
                read_mask: 0xff,
                write_mask: 0xff,
                ..RgfxStencilDescriptor::default()
            },
            back: RgfxStencilDescriptor::default(),
        };
        assert_eq!(dec.decode(&device, ds.as_bytes()).unwrap(), Ok(()));

        // A device-placed buffer at a guest-picked GPU address.
        let buffer = RgfxBufferCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_BUFFER_CREATE,
                size: size_of::<RgfxBufferCreate>() as u32,
                seqno: 0,
            },
            obj_id: 17,
            gpu_address: 0x4_0000,
            length: 65536,
            options: 0,
            ..RgfxBufferCreate::default()
        };
        assert_eq!(dec.decode(&device, buffer.as_bytes()).unwrap(), Ok(()));

        // The kext gives this context (0) the chunk of resource ids 65 to 128.
        let grant = RgfxResourceIdChunk {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_RESOURCE_ID_CHUNK,
                size: size_of::<RgfxResourceIdChunk>() as u32,
                seqno: 0,
            },
            context: 0,
            chunk: 1,
        };
        assert_eq!(dec.decode(&device, grant.as_bytes()).unwrap(), Ok(()));

        // A device-placed render-target texture of a non-default shape, with its
        // resource id pinned and the identity swizzle.
        let texture = RgfxTextureCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_TEXTURE_CREATE,
                size: size_of::<RgfxTextureCreate>() as u32,
                seqno: 0,
            },
            obj_id: 18,
            gpu_address: 0x5_0000,
            resource_id: 0x42,
            heap_obj_id: 0,
            heap_offset: 0,
            desc: RgfxTextureDescriptor {
                pixel_format: 80, // BGRA8Unorm
                texture_type: 2,  // 2D
                width: 1920,
                height: 1200,
                depth: 1,
                mipmap_level_count: 1,
                array_length: 1,
                sample_count: 1,
                usage: 4, // RenderTarget
                options: 0,
                swizzle: 0x0403_0201, // identity (red lowest)
                reserved_2c: 0,
            },
        };
        assert_eq!(dec.decode(&device, texture.as_bytes()).unwrap(), Ok(()));

        // A texture over the buffer just made, read from an offset.
        let tex_from_buf = RgfxTextureFromBuffer {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_TEXTURE_FROM_BUFFER,
                size: size_of::<RgfxTextureFromBuffer>() as u32,
                seqno: 0,
            },
            obj_id: 19,
            resource_id: 0x43,
            buffer_obj_id: 17,
            offset: 0,
            bytes_per_row: 1920 * 4,
            desc: RgfxTextureDescriptor {
                pixel_format: 80,
                texture_type: 2,
                width: 1920,
                height: 1200,
                depth: 1,
                mipmap_level_count: 1,
                array_length: 1,
                sample_count: 1,
                usage: 1, // ShaderRead
                options: 0,
                swizzle: 0,
                reserved_2c: 0,
            },
        };
        assert_eq!(
            dec.decode(&device, tex_from_buf.as_bytes()).unwrap(),
            Ok(())
        );

        // A texture over a buffer the host never made is a bad reference, not a
        // refusal, and makes no object.
        let bad = RgfxTextureFromBuffer {
            obj_id: 20,
            buffer_obj_id: 999,
            ..tex_from_buf
        };
        assert_eq!(
            dec.decode(&device, bad.as_bytes()).unwrap(),
            Err(DecodeError::BadReference)
        );

        // A device-placed placement heap at a guest-picked address.
        let heap = RgfxHeapCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_HEAP_CREATE,
                size: size_of::<RgfxHeapCreate>() as u32,
                seqno: 0,
            },
            obj_id: 21,
            gpu_address: 0x6_0000,
            size_bytes: 1 << 20,
            options: 0,
            heap_type: 1, // Placement
        };
        assert_eq!(dec.decode(&device, heap.as_bytes()).unwrap(), Ok(()));

        // A heap-placed texture at an offset in the heap just made.
        let heap_tex = RgfxTextureCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_TEXTURE_CREATE,
                size: size_of::<RgfxTextureCreate>() as u32,
                seqno: 0,
            },
            obj_id: 22,
            gpu_address: 0,
            resource_id: 0x44,
            heap_obj_id: 21,
            heap_offset: 0,
            desc: RgfxTextureDescriptor {
                pixel_format: 80,
                texture_type: 2,
                width: 256,
                height: 256,
                depth: 1,
                mipmap_level_count: 1,
                array_length: 1,
                sample_count: 1,
                usage: 1,
                options: 0,
                swizzle: 0,
                reserved_2c: 0,
            },
        };
        assert_eq!(dec.decode(&device, heap_tex.as_bytes()).unwrap(), Ok(()));

        // A buffer at an offset in the same heap.
        let heap_buf = RgfxBufferAtOffset {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_BUFFER_AT_OFFSET,
                size: size_of::<RgfxBufferAtOffset>() as u32,
                seqno: 0,
            },
            obj_id: 23,
            heap_obj_id: 21,
            offset: 0x1_0000,
            length: 4096,
            options: 0,
            reserved_34: 0,
        };
        assert_eq!(dec.decode(&device, heap_buf.as_bytes()).unwrap(), Ok(()));

        assert_eq!(dec.object_count(), 13);
        let log = p.call_log();
        // The guest's event, and the queue's device caches' end event, the caches' transfer
        // event and their read-back queue's end event, made with the queue.
        assert_eq!(log.calls_to("newEvent").len(), 4);
        assert_eq!(log.calls_to("newFence").len(), 1);
        // The guest's shared event, and the queue's surface timeline.
        assert_eq!(log.calls_to("newSharedEvent").len(), 2);
        // The guest's queue, and the caches' read-back queue, made with it.
        assert_eq!(log.calls_to("newCommandQueue").len(), 2);
        assert_eq!(log.calls_to("newLibraryWithData:error:").len(), 1);
        assert_eq!(log.calls_to("newSamplerStateWithDescriptor:").len(), 1);
        assert_eq!(log.calls_to("newDepthStencilStateWithDescriptor:").len(), 1);
        assert_eq!(log.calls_to("newBufferWithDescriptor:").len(), 1);
        assert_eq!(log.calls_to("newTextureWithDescriptor:").len(), 1);
        assert_eq!(
            log.calls_to("newTextureWithDescriptor:offset:bytesPerRow:")
                .len(),
            1
        );
        assert_eq!(log.calls_to("newHeapWithDescriptor:").len(), 1);
        assert_eq!(log.calls_to("newTextureWithDescriptor:offset:").len(), 1);
        assert_eq!(log.calls_to("newBufferWithLength:options:offset:").len(), 1);

        // Destroying the event drops its handle; the others stay.
        let destroy = RgfxObjectDestroy {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_OBJECT_DESTROY,
                size: size_of::<RgfxObjectDestroy>() as u32,
                seqno: 0,
            },
            obj_id: 10,
        };
        assert_eq!(dec.decode(&device, destroy.as_bytes()).unwrap(), Ok(()));
        assert_eq!(dec.object_count(), 12);
        // Destroying an id the host never made is not an error.
        assert_eq!(dec.decode(&device, destroy.as_bytes()).unwrap(), Ok(()));

        // An unknown command and a short message are reported.
        let unknown = RgfxMsgHeader {
            r#type: 0xffff,
            size: 16,
            seqno: 0,
        };
        assert_eq!(
            dec.decode(&device, unknown.as_bytes()).unwrap(),
            Err(DecodeError::UnknownCommand)
        );
        assert_eq!(
            dec.decode(&device, &[0u8; 4]).unwrap(),
            Err(DecodeError::Malformed)
        );

        // A texture over IOSurface 0x77's plane 0, in the guest's copy of it in
        // the buffer above: the host names it by the surface's id until its
        // destroy. A later plane-1 texture of the surface leaves the name alone.
        let tex_from_surface = RgfxTextureFromIOSurface {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_TEXTURE_FROM_IOSURFACE,
                size: size_of::<RgfxTextureFromIOSurface>() as u32,
                seqno: 0,
            },
            obj_id: 30,
            resource_id: 0x45,
            buffer_obj_id: 17,
            offset: 0,
            bytes_per_row: 64 * 4,
            iosurface_id: 0x77,
            plane: 0,
            desc: RgfxTextureDescriptor {
                width: 64,
                height: 64,
                usage: 5, // ShaderRead | RenderTarget
                ..tex_from_buf.desc
            },
            writers_offset: RGFX_SURFACE_NO_WRITERS,
        };
        assert!(dec.surface_texture(0x77).is_none());
        assert_eq!(
            dec.decode(&device, tex_from_surface.as_bytes()).unwrap(),
            Ok(())
        );
        assert!(dec.surface_texture(0x77).is_some());
        let plane1 = RgfxTextureFromIOSurface {
            obj_id: 31,
            plane: 1,
            ..tex_from_surface
        };
        assert_eq!(dec.decode(&device, plane1.as_bytes()).unwrap(), Ok(()));
        let destroy_plane1 = RgfxObjectDestroy {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_OBJECT_DESTROY,
                size: size_of::<RgfxObjectDestroy>() as u32,
                seqno: 0,
            },
            obj_id: 31,
        };
        assert_eq!(
            dec.decode(&device, destroy_plane1.as_bytes()).unwrap(),
            Ok(())
        );
        assert!(dec.surface_texture(0x77).is_some());
        // Another texture over the surface (another process's, say) is the one shown. A
        // present holds its own reference: its destroy leaves that reference drawable, and
        // the surface named by the older texture until that one goes too.
        let first = format!("{:?}", dec.surface_texture(0x77).unwrap());
        let second = RgfxTextureFromIOSurface {
            obj_id: 33,
            ..tex_from_surface
        };
        assert_eq!(dec.decode(&device, second.as_bytes()).unwrap(), Ok(()));
        let shown = dec.surface_texture(0x77).unwrap();
        assert_ne!(format!("{shown:?}"), first);
        let destroy_second = RgfxObjectDestroy {
            obj_id: 33,
            ..destroy_plane1
        };
        assert_eq!(
            dec.decode(&device, destroy_second.as_bytes()).unwrap(),
            Ok(())
        );
        assert_eq!(format!("{:?}", dec.surface_texture(0x77).unwrap()), first);
        assert_eq!(
            reims_vgpu_core::platform::resource::Texture::width(&shown),
            64
        );
        drop(shown);
        let destroy_surface = RgfxObjectDestroy {
            obj_id: 30,
            ..destroy_plane1
        };
        assert_eq!(
            dec.decode(&device, destroy_surface.as_bytes()).unwrap(),
            Ok(())
        );
        assert!(dec.surface_texture(0x77).is_none());
        // An id made again over another surface names only that one.
        let again = RgfxTextureFromIOSurface {
            obj_id: 34,
            ..tex_from_surface
        };
        assert_eq!(dec.decode(&device, again.as_bytes()).unwrap(), Ok(()));
        let moved = RgfxTextureFromIOSurface {
            obj_id: 34,
            iosurface_id: 0x78,
            ..tex_from_surface
        };
        assert_eq!(dec.decode(&device, moved.as_bytes()).unwrap(), Ok(()));
        assert!(dec.surface_texture(0x77).is_none());
        assert!(dec.surface_texture(0x78).is_some());
        let destroy_moved = RgfxObjectDestroy {
            obj_id: 34,
            ..destroy_plane1
        };
        assert_eq!(
            dec.decode(&device, destroy_moved.as_bytes()).unwrap(),
            Ok(())
        );
        assert!(dec.surface_texture(0x78).is_none());
        // Over a buffer the host never made: a bad reference, and no name.
        let bad_surface = RgfxTextureFromIOSurface {
            obj_id: 32,
            buffer_obj_id: 999,
            ..tex_from_surface
        };
        assert_eq!(
            dec.decode(&device, bad_surface.as_bytes()).unwrap(),
            Err(DecodeError::BadReference)
        );
        assert!(dec.surface_texture(0x77).is_none());
    }

    /// A texture create is carried out only where its resource id lies in a chunk its own
    /// context holds (`RESOURCE_ID_CHUNK`, until the context's close). An id of 0, one at or
    /// past the limit, one in no context's chunk and one in another context's are refused
    /// before any platform call, so none reaches the platform's own limit, and the decoder goes
    /// on.
    #[test]
    fn refuses_texture_ids_outside_the_context_chunks() {
        let p = Arc::new(FakePlatform::new());
        let device = p.device();
        let mut dec = GraphicsDecoder::<FakePlatform>::new(
            Arc::clone(&p),
            Arc::new(Region::new(0x10000).unwrap()),
        );
        let obj = |context: u64, n: u64| (context << RGFX_CONTEXT_OBJ_ID_SHIFT) | n;
        let grant = |context: u32, chunk: u32| RgfxResourceIdChunk {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_RESOURCE_ID_CHUNK,
                size: size_of::<RgfxResourceIdChunk>() as u32,
                seqno: 0,
            },
            context,
            chunk,
        };
        let desc = RgfxTextureDescriptor {
            pixel_format: 80,
            texture_type: 2,
            width: 64,
            height: 64,
            depth: 1,
            mipmap_level_count: 1,
            array_length: 1,
            sample_count: 1,
            usage: 5,
            ..RgfxTextureDescriptor::default()
        };
        let texture = |obj_id: u64, resource_id: u64| RgfxTextureCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_TEXTURE_CREATE,
                size: size_of::<RgfxTextureCreate>() as u32,
                seqno: 0,
            },
            obj_id,
            resource_id,
            desc,
            ..RgfxTextureCreate::default()
        };
        let made = |dec: &mut GraphicsDecoder<FakePlatform>, obj_id: u64, resource_id: u64| {
            dec.decode(&device, texture(obj_id, resource_id).as_bytes())
                .unwrap()
        };

        // Context 1 holds chunk 0 (ids 1 to 64), and context 2 the last chunk (ids 16257 to
        // 16320). A chunk past the pool is no chunk.
        assert_eq!(dec.decode(&device, grant(1, 0).as_bytes()).unwrap(), Ok(()));
        assert_eq!(
            dec.decode(&device, grant(2, RGFX_RESOURCE_ID_CHUNKS - 1).as_bytes())
                .unwrap(),
            Ok(())
        );
        assert_eq!(
            dec.decode(&device, grant(1, RGFX_RESOURCE_ID_CHUNKS).as_bytes())
                .unwrap(),
            Err(DecodeError::BadResourceId)
        );

        // Each context's ids, at both ends of its chunk.
        assert_eq!(made(&mut dec, obj(1, 1), 1), Ok(()));
        assert_eq!(made(&mut dec, obj(1, 2), 64), Ok(()));
        assert_eq!(made(&mut dec, obj(2, 1), 16257), Ok(()));
        assert_eq!(made(&mut dec, obj(2, 2), 16320), Ok(()));

        // 0; the next chunk's first id, which no context holds; another context's ids; the ids
        // past the last chunk, below the limit; the limit; and far past it.
        for (obj_id, id) in [
            (obj(1, 3), 0),
            (obj(1, 4), 65),
            (obj(1, 5), 16257),
            (obj(2, 3), 1),
            (obj(2, 4), 16321),
            (obj(2, 5), RGFX_RESOURCE_ID_LIMIT),
            (obj(2, 6), 1 << 20),
            (obj(1, 6), u64::MAX),
        ] {
            assert_eq!(
                made(&mut dec, obj_id, id),
                Err(DecodeError::BadResourceId),
                "id {id}"
            );
        }

        // A texture over a buffer and one over an IOSurface are checked the same way, and a
        // refused surface texture leaves the surface unnamed.
        let buffer = RgfxBufferCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_BUFFER_CREATE,
                size: size_of::<RgfxBufferCreate>() as u32,
                seqno: 0,
            },
            obj_id: obj(1, 10),
            length: 65536,
            ..RgfxBufferCreate::default()
        };
        assert_eq!(dec.decode(&device, buffer.as_bytes()).unwrap(), Ok(()));
        let from_buffer = |obj_id: u64, resource_id: u64| RgfxTextureFromBuffer {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_TEXTURE_FROM_BUFFER,
                size: size_of::<RgfxTextureFromBuffer>() as u32,
                seqno: 0,
            },
            obj_id,
            resource_id,
            buffer_obj_id: obj(1, 10),
            offset: 0,
            bytes_per_row: 64 * 4,
            desc,
        };
        assert_eq!(
            dec.decode(&device, from_buffer(obj(1, 11), 70).as_bytes())
                .unwrap(),
            Err(DecodeError::BadResourceId)
        );
        assert_eq!(
            dec.decode(&device, from_buffer(obj(1, 11), 3).as_bytes())
                .unwrap(),
            Ok(())
        );
        let from_surface = RgfxTextureFromIOSurface {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_TEXTURE_FROM_IOSURFACE,
                size: size_of::<RgfxTextureFromIOSurface>() as u32,
                seqno: 0,
            },
            obj_id: obj(1, 12),
            resource_id: 16300,
            buffer_obj_id: obj(1, 10),
            offset: 0,
            bytes_per_row: 64 * 4,
            iosurface_id: 0x55,
            plane: 0,
            desc,
            writers_offset: RGFX_SURFACE_NO_WRITERS,
        };
        assert_eq!(
            dec.decode(&device, from_surface.as_bytes()).unwrap(),
            Err(DecodeError::BadResourceId)
        );
        assert!(dec.surface_texture(0x55).is_none());

        // Only the textures in their contexts' chunks reached the platform.
        let log = p.call_log();
        assert_eq!(log.calls_to("newTextureWithDescriptor:").len(), 4);
        assert_eq!(
            log.calls_to("newTextureWithDescriptor:offset:bytesPerRow:")
                .len(),
            1
        );

        // Context 1's chunk goes at its close: its ids are refused to the next context until
        // the kext gives it the chunk. Context 2's stays.
        dec.drop_context(&device, 1);
        assert_eq!(
            made(&mut dec, obj(3, 1), 5),
            Err(DecodeError::BadResourceId)
        );
        assert_eq!(dec.decode(&device, grant(3, 0).as_bytes()).unwrap(), Ok(()));
        assert_eq!(made(&mut dec, obj(3, 1), 5), Ok(()));
        assert_eq!(made(&mut dec, obj(2, 7), 16319), Ok(()));
    }

    /// A `RESOURCE_ID_SMALL_CHUNK` gives its context the 16 ids of its small chunk, from the
    /// same pool as `RESOURCE_ID_CHUNK`'s 64: chunk `k` is small chunks `4k` to `4k + 3`, and a
    /// grant of either kind gives the ids it names to its context, and no others. A small chunk
    /// past the pool is none, and a context's small chunks go at its close as its chunks do. A
    /// type past it is unknown, as `RESOURCE_ID_SMALL_CHUNK` is to a host from before
    /// [`RGFX_FEATURE_SMALL_ID_CHUNKS`] (read from 8ecfc5abb's `decode`: no arm takes it), which
    /// the kext posts it to only where the host offers the feature
    /// (`guest/kext/tests/id_chunks_test.cpp`).
    #[test]
    fn a_small_chunk_holds_sixteen_ids_of_the_one_pool() {
        let p = Arc::new(FakePlatform::new());
        let device = p.device();
        let mut dec = GraphicsDecoder::<FakePlatform>::new(
            Arc::clone(&p),
            Arc::new(Region::new(0x10000).unwrap()),
        );
        let obj = |context: u64, n: u64| (context << RGFX_CONTEXT_OBJ_ID_SHIFT) | n;
        let grant = |r#type: u32, context: u32, chunk: u32| RgfxResourceIdChunk {
            header: RgfxMsgHeader {
                r#type,
                size: size_of::<RgfxResourceIdChunk>() as u32,
                seqno: 0,
            },
            context,
            chunk,
        };
        let small =
            |context: u32, chunk: u32| grant(RGFX_CMD_RESOURCE_ID_SMALL_CHUNK, context, chunk);
        let texture = |obj_id: u64, resource_id: u64| RgfxTextureCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_TEXTURE_CREATE,
                size: size_of::<RgfxTextureCreate>() as u32,
                seqno: 0,
            },
            obj_id,
            resource_id,
            desc: RgfxTextureDescriptor {
                pixel_format: 80,
                texture_type: 2,
                width: 16,
                height: 16,
                depth: 1,
                mipmap_level_count: 1,
                array_length: 1,
                sample_count: 1,
                usage: 5,
                ..RgfxTextureDescriptor::default()
            },
            ..RgfxTextureCreate::default()
        };
        let mut n = 0;
        let mut made = |dec: &mut GraphicsDecoder<FakePlatform>, context: u64, resource_id: u64| {
            n += 1;
            dec.decode(&device, texture(obj(context, n), resource_id).as_bytes())
                .unwrap()
        };

        // Context 1 holds small chunk 5, ids 81 to 96, and none either side.
        assert_eq!(dec.decode(&device, small(1, 5).as_bytes()).unwrap(), Ok(()));
        assert_eq!(made(&mut dec, 1, 81), Ok(()));
        assert_eq!(made(&mut dec, 1, 96), Ok(()));
        for id in [80, 97] {
            assert_eq!(
                made(&mut dec, 1, id),
                Err(DecodeError::BadResourceId),
                "id {id}"
            );
        }
        // Context 2 holds chunk 2, ids 129 to 192: small chunks 8 to 11.
        let chunk = grant(RGFX_CMD_RESOURCE_ID_CHUNK, 2, 2);
        assert_eq!(dec.decode(&device, chunk.as_bytes()).unwrap(), Ok(()));
        assert_eq!(made(&mut dec, 2, 129), Ok(()));
        assert_eq!(made(&mut dec, 2, 192), Ok(()));
        assert_eq!(made(&mut dec, 1, 130), Err(DecodeError::BadResourceId));
        assert_eq!(made(&mut dec, 2, 96), Err(DecodeError::BadResourceId));
        // Context 3 holds the last small chunk, ids 16305 to 16320; one past it is none.
        let last = RGFX_RESOURCE_ID_SMALL_CHUNKS - 1;
        assert_eq!(
            dec.decode(&device, small(3, last).as_bytes()).unwrap(),
            Ok(())
        );
        assert_eq!(made(&mut dec, 3, 16305), Ok(()));
        assert_eq!(made(&mut dec, 3, 16320), Ok(()));
        assert_eq!(made(&mut dec, 3, 16321), Err(DecodeError::BadResourceId));
        assert_eq!(
            dec.decode(&device, small(3, last + 1).as_bytes()).unwrap(),
            Err(DecodeError::BadResourceId)
        );
        assert_eq!(
            dec.decode(&device, small(3, u32::MAX).as_bytes()).unwrap(),
            Err(DecodeError::BadResourceId)
        );
        // A small chunk of chunk 2's goes to context 4, as the kext gives it once context 2's
        // close is answered; context 2 keeps the other three.
        assert_eq!(dec.decode(&device, small(4, 9).as_bytes()).unwrap(), Ok(()));
        assert_eq!(made(&mut dec, 4, 145), Ok(()));
        assert_eq!(made(&mut dec, 2, 160), Err(DecodeError::BadResourceId));
        assert_eq!(made(&mut dec, 2, 161), Ok(()));
        // Context 1's small chunk goes at its close.
        dec.drop_context(&device, 1);
        assert_eq!(made(&mut dec, 5, 82), Err(DecodeError::BadResourceId));
        assert_eq!(dec.decode(&device, small(5, 5).as_bytes()).unwrap(), Ok(()));
        assert_eq!(made(&mut dec, 5, 82), Ok(()));

        // A grant one byte short, and a type past it, which the decoder does not know.
        let whole = small(6, 0);
        let short = &whole.as_bytes()[..size_of::<RgfxResourceIdChunk>() - 1];
        assert_eq!(
            dec.decode(&device, short).unwrap(),
            Err(DecodeError::Malformed)
        );
        let unknown = grant(RGFX_CMD_RESOURCE_ID_SMALL_CHUNK + 1, 6, 0);
        assert_eq!(
            dec.decode(&device, unknown.as_bytes()).unwrap(),
            Err(DecodeError::UnknownCommand)
        );
        assert_eq!(made(&mut dec, 6, 1), Err(DecodeError::BadResourceId));
    }

    /// The objects table's order, which a context's objects drop in at its close, is a function
    /// of their ids and the history of creates and destroys, the same in every run: its hasher
    /// is fixed. Two decoders that made the same buffers hold them in one order.
    #[test]
    fn a_contexts_objects_drop_in_the_same_order_every_run() {
        let order = || {
            let p = Arc::new(FakePlatform::new());
            let device = p.device();
            let mut dec = GraphicsDecoder::<FakePlatform>::new(
                Arc::clone(&p),
                Arc::new(Region::new(0x10000).unwrap()),
            );
            for n in 1..=64 {
                let buffer = RgfxBufferCreate {
                    header: RgfxMsgHeader {
                        r#type: RGFX_CMD_BUFFER_CREATE,
                        size: size_of::<RgfxBufferCreate>() as u32,
                        seqno: 0,
                    },
                    obj_id: (1 << RGFX_CONTEXT_OBJ_ID_SHIFT) | n,
                    length: 64,
                    ..RgfxBufferCreate::default()
                };
                assert_eq!(dec.decode(&device, buffer.as_bytes()).unwrap(), Ok(()));
            }
            dec.object_ids()
        };
        let first = order();
        assert_eq!(first.len(), 64);
        assert_eq!(first, order());
    }

    /// A texture over a buffer the platform makes none of (Metal's nil, which names no reason) is
    /// refused with a warning naming every argument of its create: the descriptor, the offset and
    /// rows, the buffer and its length, a linear texture's bytes between slices and an IOSurface
    /// texture's surface and plane. A texture made logs nothing.
    #[test]
    fn a_refused_texture_over_a_buffer_logs_its_arguments() {
        let p = Arc::new(FakePlatform::new());
        let device = p.device();
        let mut dec = GraphicsDecoder::<FakePlatform>::new(
            Arc::clone(&p),
            Arc::new(Region::new(0x10000).unwrap()),
        );
        let obj = |n: u64| (1 << RGFX_CONTEXT_OBJ_ID_SHIFT) | n;
        let grant = RgfxResourceIdChunk {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_RESOURCE_ID_CHUNK,
                size: size_of::<RgfxResourceIdChunk>() as u32,
                seqno: 0,
            },
            context: 1,
            chunk: 0,
        };
        assert_eq!(dec.decode(&device, grant.as_bytes()).unwrap(), Ok(()));
        let buffer = RgfxBufferCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_BUFFER_CREATE,
                size: size_of::<RgfxBufferCreate>() as u32,
                seqno: 0,
            },
            obj_id: obj(10),
            length: 65536,
            ..RgfxBufferCreate::default()
        };
        assert_eq!(dec.decode(&device, buffer.as_bytes()).unwrap(), Ok(()));
        let desc = RgfxTextureDescriptor {
            pixel_format: 80,
            texture_type: 2,
            width: 64,
            height: 48,
            depth: 1,
            mipmap_level_count: 1,
            array_length: 1,
            sample_count: 1,
            usage: 5,
            options: 0x10,
            swizzle: 0x0504_0302,
            ..RgfxTextureDescriptor::default()
        };
        let from_buffer = RgfxTextureFromBuffer {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_TEXTURE_FROM_BUFFER,
                size: size_of::<RgfxTextureFromBuffer>() as u32,
                seqno: 0,
            },
            obj_id: obj(11),
            resource_id: 1,
            buffer_obj_id: obj(10),
            offset: 256,
            bytes_per_row: 512,
            desc,
        };
        let linear = RgfxLinearTextureFromBuffer {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_LINEAR_TEXTURE_FROM_BUFFER,
                size: size_of::<RgfxLinearTextureFromBuffer>() as u32,
                seqno: 0,
            },
            obj_id: obj(12),
            resource_id: 2,
            buffer_obj_id: obj(10),
            offset: 0,
            bytes_per_row: 256,
            bytes_per_image: 12288,
            desc,
        };
        let from_surface = RgfxTextureFromIOSurface {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_TEXTURE_FROM_IOSURFACE,
                size: size_of::<RgfxTextureFromIOSurface>() as u32,
                seqno: 0,
            },
            obj_id: obj(13),
            resource_id: 3,
            buffer_obj_id: obj(10),
            offset: 0,
            bytes_per_row: 256,
            iosurface_id: 0x2b,
            plane: 1,
            desc,
            writers_offset: RGFX_SURFACE_NO_WRITERS,
        };
        let logs = reims_vgpu_core::testkit::logs::capture();
        let script = p.script();
        script.push("newTextureWithDescriptor:offset:bytesPerRow:", Step::Nil);
        script.push(
            "newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:",
            Step::Nil,
        );
        script.push("newTextureWithDescriptor:iosurface:plane:", Step::Nil);
        for message in [
            from_buffer.as_bytes(),
            linear.as_bytes(),
            from_surface.as_bytes(),
        ] {
            assert_eq!(
                dec.decode(&device, message).unwrap(),
                Err(DecodeError::Refused)
            );
        }
        assert!(dec.surface_texture(0x2b).is_none());
        let shape = "pixel format 80, type 2, 64 x 48 x 1, 1 levels, array length 1, 1 samples, \
                     usage 0x5, resource options 0x10, swizzle 0x05040302";
        let warned = |object: u64, rest: &str| {
            (
                log::Level::Warn,
                "reims_vgpu::graphics".to_owned(),
                format!(
                    "a texture (object {:#x}) over buffer {:#x} of 65536 bytes was refused: \
                     {shape}, {rest}",
                    obj(object),
                    obj(10)
                ),
            )
        };
        // The logger is the process's: only this test's refusals, not another test's warnings.
        let lines: Vec<_> = logs
            .take()
            .into_iter()
            .filter(|(level, _, line)| *level == log::Level::Warn && line.contains(shape))
            .collect();
        assert_eq!(
            lines,
            [
                warned(11, "at offset 256, 512 bytes a row"),
                warned(12, "at offset 0, 256 bytes a row, 12288 bytes an image"),
                warned(13, "at offset 0, 256 bytes a row, IOSurface 0x2b plane 1"),
            ]
        );

        // Made, each logs nothing.
        for message in [
            RgfxTextureFromBuffer {
                obj_id: obj(14),
                ..from_buffer
            }
            .as_bytes(),
            RgfxTextureFromIOSurface {
                obj_id: obj(15),
                ..from_surface
            }
            .as_bytes(),
        ] {
            assert_eq!(dec.decode(&device, message).unwrap(), Ok(()));
        }
        assert!(
            logs.take()
                .iter()
                .all(|(level, ..)| *level != log::Level::Warn)
        );
    }

    /// A `TEXTURE_FROM_IOSURFACE` carries the surface's facts after its struct where its size
    /// covers them, and none at the struct's size, as before them; a size between the two, or a
    /// message shorter than its size, is malformed and reaches no platform call. A refused
    /// texture's warning names the facts it carried.
    #[test]
    fn a_surface_texture_reads_its_facts_by_the_message_size() {
        use reims_graphics::wire::{
            RGFX_SURFACE_FACTS_OFFSET, RGFX_YCBCR_MATRIX_ITU_R_2020, RgfxSurfaceFacts,
        };
        let p = Arc::new(FakePlatform::new());
        let device = p.device();
        let mut dec = GraphicsDecoder::<FakePlatform>::new(
            Arc::clone(&p),
            Arc::new(Region::new(0x10000).unwrap()),
        );
        let obj = |n: u64| (1 << RGFX_CONTEXT_OBJ_ID_SHIFT) | n;
        let grant = RgfxResourceIdChunk {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_RESOURCE_ID_CHUNK,
                size: size_of::<RgfxResourceIdChunk>() as u32,
                seqno: 0,
            },
            context: 1,
            chunk: 0,
        };
        assert_eq!(dec.decode(&device, grant.as_bytes()).unwrap(), Ok(()));
        let buffer = RgfxBufferCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_BUFFER_CREATE,
                size: size_of::<RgfxBufferCreate>() as u32,
                seqno: 0,
            },
            obj_id: obj(10),
            length: 65536,
            ..RgfxBufferCreate::default()
        };
        assert_eq!(dec.decode(&device, buffer.as_bytes()).unwrap(), Ok(()));
        let facts = RgfxSurfaceFacts {
            surface_pixel_format: 0x7834_3230, // 'x420'
            plane_count: 2,
            width: 64,
            height: 48,
            chroma_offset: 16384,
            chroma_bytes_per_row: 256,
            ycbcr_matrix: RGFX_YCBCR_MATRIX_ITU_R_2020,
            reserved_24: 0,
        };
        // A texture over plane 0 of the copy, its message `size` bytes: the struct, then the
        // facts, cut to `len` bytes.
        let message = |n: u64, size: usize, len: usize| {
            let m = RgfxTextureFromIOSurface {
                header: RgfxMsgHeader {
                    r#type: RGFX_CMD_TEXTURE_FROM_IOSURFACE,
                    size: size as u32,
                    seqno: 0,
                },
                obj_id: obj(n),
                resource_id: n,
                buffer_obj_id: obj(10),
                offset: 0,
                bytes_per_row: 256,
                iosurface_id: 0x2c,
                plane: 0,
                desc: RgfxTextureDescriptor {
                    pixel_format: 505,
                    texture_type: 2,
                    width: 64,
                    height: 48,
                    depth: 1,
                    mipmap_level_count: 1,
                    array_length: 1,
                    sample_count: 1,
                    usage: 1,
                    ..RgfxTextureDescriptor::default()
                },
                writers_offset: RGFX_SURFACE_NO_WRITERS,
            };
            let mut bytes = m.as_bytes().to_vec();
            bytes.extend_from_slice(facts.as_bytes());
            bytes.truncate(len);
            bytes
        };
        let start = RGFX_SURFACE_FACTS_OFFSET as usize;
        let whole = start + size_of::<RgfxSurfaceFacts>();
        let made = || {
            p.call_log()
                .calls_to("newTextureWithDescriptor:iosurface:plane:")
                .len()
        };

        // With the facts, and without them (the struct's size, its trailing bytes ignored).
        for (n, size) in [(11, whole), (12, start)] {
            assert_eq!(
                dec.decode(&device, &message(n, size, whole)).unwrap(),
                Ok(()),
                "size {size}"
            );
        }
        assert_eq!(made(), 2);

        // A size between the two, and a size that covers the facts on a message without them.
        for (n, size, len) in [(13, start + 8, whole), (14, whole, start)] {
            assert_eq!(
                dec.decode(&device, &message(n, size, len)).unwrap(),
                Err(DecodeError::Malformed),
                "size {size}, {len} bytes"
            );
        }
        assert_eq!(made(), 2, "no platform call for a malformed message");

        // A refused texture's warning names the facts.
        let logs = reims_vgpu_core::testkit::logs::capture();
        p.script()
            .push("newTextureWithDescriptor:iosurface:plane:", Step::Nil);
        assert_eq!(
            dec.decode(&device, &message(15, whole, whole)).unwrap(),
            Err(DecodeError::Refused)
        );
        let warned: Vec<_> = logs
            .take()
            .into_iter()
            .filter(|(level, ..)| *level == log::Level::Warn)
            .map(|(.., line)| line)
            .collect();
        assert_eq!(warned.len(), 1, "{warned:?}");
        assert!(
            warned[0].ends_with(
                ", at offset 0, 256 bytes a row, IOSurface 0x2c plane 0 (pixel format 'x420', 2 \
                 planes, 64 x 48, plane 1 at offset 16384, 256 bytes a row, YCbCr matrix 3)"
            ),
            "{}",
            warned[0]
        );
    }

    /// Every `TEXTURE_FROM_IOSURFACE` is made by the platform's IOSurface creator over its buffer
    /// (`newTextureWithDescriptor:iosurface:plane:` on the buffer), never by the buffer's own
    /// creator: the plane's index, offset and rows, whether the surface has planes, and the facts
    /// the message carried reach it, each wire matrix as its value. A message of the struct alone
    /// reaches it as before the facts: of a surface without planes, with no facts. A matrix the
    /// wire names no value for is malformed, and reaches no platform call.
    #[test]
    fn a_surface_texture_s_facts_reach_the_platform() {
        use reims_graphics::wire::{RGFX_SURFACE_FACTS_OFFSET, RgfxSurfaceFacts};
        let p = Arc::new(FakePlatform::new());
        let device = p.device();
        let mut dec = GraphicsDecoder::<FakePlatform>::new(
            Arc::clone(&p),
            Arc::new(Region::new(0x10000).unwrap()),
        );
        let obj = |n: u64| (1 << RGFX_CONTEXT_OBJ_ID_SHIFT) | n;
        let grant = RgfxResourceIdChunk {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_RESOURCE_ID_CHUNK,
                size: size_of::<RgfxResourceIdChunk>() as u32,
                seqno: 0,
            },
            context: 1,
            chunk: 0,
        };
        assert_eq!(dec.decode(&device, grant.as_bytes()).unwrap(), Ok(()));
        let buffer = RgfxBufferCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_BUFFER_CREATE,
                size: size_of::<RgfxBufferCreate>() as u32,
                seqno: 0,
            },
            obj_id: obj(10),
            length: 65536,
            ..RgfxBufferCreate::default()
        };
        assert_eq!(dec.decode(&device, buffer.as_bytes()).unwrap(), Ok(()));
        // An 'x420' surface of 64 x 48, plane 1 at 16384 with rows of 256 bytes, as the bundle
        // lays its copy out.
        let facts = |ycbcr_matrix: u32| RgfxSurfaceFacts {
            surface_pixel_format: 0x7834_3230,
            plane_count: 2,
            width: 64,
            height: 48,
            chroma_offset: 16384,
            chroma_bytes_per_row: 256,
            ycbcr_matrix,
            reserved_24: 0,
        };
        // A 505 texture over plane `plane` of the copy, at `offset`, carrying `facts` where given.
        let message = |n: u64, plane: u32, offset: u64, facts: Option<RgfxSurfaceFacts>| {
            let size = RGFX_SURFACE_FACTS_OFFSET as usize
                + facts.map_or(0, |_| size_of::<RgfxSurfaceFacts>());
            let m = RgfxTextureFromIOSurface {
                header: RgfxMsgHeader {
                    r#type: RGFX_CMD_TEXTURE_FROM_IOSURFACE,
                    size: size as u32,
                    seqno: 0,
                },
                obj_id: obj(n),
                resource_id: n,
                buffer_obj_id: obj(10),
                offset,
                bytes_per_row: 256,
                iosurface_id: 0x2d,
                plane,
                desc: RgfxTextureDescriptor {
                    pixel_format: 505,
                    texture_type: 2,
                    width: 64,
                    height: 48,
                    depth: 1,
                    mipmap_level_count: 1,
                    array_length: 1,
                    sample_count: 1,
                    usage: 1,
                    ..RgfxTextureDescriptor::default()
                },
                writers_offset: RGFX_SURFACE_NO_WRITERS,
            };
            let mut bytes = m.as_bytes().to_vec();
            if let Some(facts) = facts {
                bytes.extend_from_slice(facts.as_bytes());
            }
            bytes
        };
        let sel = "newTextureWithDescriptor:iosurface:plane:";
        // The arguments after the descriptor of the platform's last call of `sel`.
        let last = || {
            let calls = p.call_log().calls_to(sel);
            let call = calls.last().expect("a call");
            assert_eq!(call.class, "MTLBuffer");
            call.args[1..].to_vec()
        };
        let reached = |plane: u64, offset: u64, planar: bool, facts: Option<SurfaceFacts>| {
            vec![
                Arg::U64(plane),
                Arg::U64(offset),
                Arg::U64(256),
                Arg::Bool(planar),
                Arg::val(&facts),
            ]
        };
        let ours = |ycbcr_matrix| SurfaceFacts {
            pixel_format: 0x7834_3230,
            plane_count: 2,
            width: 64,
            height: 48,
            chroma_offset: 16384,
            chroma_bytes_per_row: 256,
            ycbcr_matrix,
        };

        // Each matrix of the wire, over plane 0.
        for (n, (wire, matrix)) in [
            (RGFX_YCBCR_MATRIX_NONE, None),
            (RGFX_YCBCR_MATRIX_ITU_R_601_4, Some(SurfaceMatrix::Itu601)),
            (RGFX_YCBCR_MATRIX_ITU_R_709_2, Some(SurfaceMatrix::Itu709)),
            (RGFX_YCBCR_MATRIX_ITU_R_2020, Some(SurfaceMatrix::Itu2020)),
            (
                RGFX_YCBCR_MATRIX_SMPTE_240M_1995,
                Some(SurfaceMatrix::Smpte240M1995),
            ),
            (RGFX_YCBCR_MATRIX_DCI_P3, Some(SurfaceMatrix::DciP3)),
            (RGFX_YCBCR_MATRIX_P3_D65, Some(SurfaceMatrix::P3D65)),
            (RGFX_YCBCR_MATRIX_OTHER, Some(SurfaceMatrix::Other)),
        ]
        .into_iter()
        .enumerate()
        {
            let n = 11 + n as u64;
            let m = message(n, 0, 0, Some(facts(wire)));
            assert_eq!(dec.decode(&device, &m).unwrap(), Ok(()), "matrix {wire}");
            assert_eq!(last(), reached(0, 0, true, Some(ours(matrix))), "{wire}");
        }
        assert!(dec.surface_texture(0x2d).is_some());

        // Plane 1, at its offset in the copy.
        let m = message(20, 1, 16384, Some(facts(RGFX_YCBCR_MATRIX_ITU_R_709_2)));
        assert_eq!(dec.decode(&device, &m).unwrap(), Ok(()));
        let facts_709 = Some(ours(Some(SurfaceMatrix::Itu709)));
        assert_eq!(last(), reached(1, 16384, true, facts_709));

        // Of a surface without planes.
        let bgra = RgfxSurfaceFacts {
            surface_pixel_format: u32::from_be_bytes(*b"BGRA"),
            plane_count: 0,
            chroma_offset: 0,
            chroma_bytes_per_row: 0,
            ..facts(RGFX_YCBCR_MATRIX_NONE)
        };
        assert_eq!(
            dec.decode(&device, &message(21, 0, 0, Some(bgra))).unwrap(),
            Ok(())
        );
        let none = SurfaceFacts {
            pixel_format: u32::from_be_bytes(*b"BGRA"),
            plane_count: 0,
            chroma_offset: 0,
            chroma_bytes_per_row: 0,
            ..ours(None)
        };
        assert_eq!(last(), reached(0, 0, false, Some(none)));

        // The struct alone: no facts, of a surface without planes.
        assert_eq!(
            dec.decode(&device, &message(22, 0, 0, None)).unwrap(),
            Ok(())
        );
        assert_eq!(last(), reached(0, 0, false, None));

        // A matrix past the wire's values.
        let before = p.call_log().calls_to(sel).len();
        let m = message(23, 0, 0, Some(facts(RGFX_YCBCR_MATRIX_OTHER + 1)));
        assert_eq!(
            dec.decode(&device, &m).unwrap(),
            Err(DecodeError::Malformed)
        );
        assert_eq!(p.call_log().calls_to(sel).len(), before);

        // The buffer's own creator made none of them.
        assert!(
            p.call_log()
                .calls_to("newTextureWithDescriptor:offset:bytesPerRow:")
                .is_empty()
        );
    }

    /// A decoder of a fake platform, its reflection replies, and library 1 made.
    pub(super) fn pipeline_decoder() -> (
        Arc<FakePlatform>,
        GraphicsDecoder<FakePlatform>,
        Arc<reims_vgpu_core::sync::Mutex<Vec<RgfxPipelineReflection>>>,
    ) {
        pipeline_decoder_over(Arc::new(
            reims_graphics::region::Region::new(0x10000).unwrap(),
        ))
    }

    /// [`pipeline_decoder`] over `region`.
    pub(super) fn pipeline_decoder_over(
        region: Arc<reims_graphics::region::Region>,
    ) -> (
        Arc<FakePlatform>,
        GraphicsDecoder<FakePlatform>,
        Arc<reims_vgpu_core::sync::Mutex<Vec<RgfxPipelineReflection>>>,
    ) {
        let p = Arc::new(FakePlatform::new());
        let device = p.device();
        let mut dec = GraphicsDecoder::<FakePlatform>::new(Arc::clone(&p), region);
        let replies = Arc::new(reims_vgpu_core::sync::Mutex::new(Vec::new()));
        let sink_replies = Arc::clone(&replies);
        dec.set_reflection_sink(Arc::new(move |r: PipelineReply| {
            sink_replies.lock().push(r.reflection);
        }));
        let air = [1u8, 2, 3, 4];
        let mut lib_msg = RgfxLibraryCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_LIBRARY_CREATE,
                size: (RGFX_LIBRARY_BYTES_OFFSET as usize + air.len()).next_multiple_of(8) as u32,
                seqno: 0,
            },
            obj_id: 1,
            byte_length: air.len() as u64,
            ..RgfxLibraryCreate::default()
        }
        .as_bytes()
        .to_vec();
        lib_msg.extend_from_slice(&air);
        assert_eq!(dec.decode(&device, &lib_msg).unwrap(), Ok(()));
        (p, dec, replies)
    }

    /// The colour attachments [`whole_render_create`] sets: 0 BGRA8Unorm, blending off, writing
    /// red and alpha; 1 RGBA16Float, every factor and operation its own; 2 R32Float, unused
    /// blending.
    fn whole_render_colors() -> [RgfxColorAttachmentDesc; 3] {
        [
            RgfxColorAttachmentDesc {
                pixel_format: 80,
                write_mask: 0x9,
                ..RgfxColorAttachmentDesc::default()
            },
            RgfxColorAttachmentDesc {
                pixel_format: 115,
                blending_enabled: 1,
                source_rgb_blend_factor: 4,
                destination_rgb_blend_factor: 5,
                rgb_blend_operation: 1,
                source_alpha_blend_factor: 2,
                destination_alpha_blend_factor: 7,
                alpha_blend_operation: 3,
                write_mask: 0xf,
                reserved_24: 0,
            },
            RgfxColorAttachmentDesc {
                pixel_format: 55,
                write_mask: 0x8,
                ..RgfxColorAttachmentDesc::default()
            },
        ]
    }

    /// A `RENDER_PIPELINE_CREATE` of pipeline `obj_id` from library 1's `vmain` and, with
    /// `fragment`, its `fmain`, carrying a descriptor that sets every field to a value of its own:
    /// three colour attachments, a vertex descriptor of two layouts and three attributes, depth
    /// Depth32Float, stencil Stencil8, 4 samples, every flag, topology triangle, amplification 2.
    pub(super) fn whole_render_create(obj_id: u64, fragment: bool) -> Vec<u8> {
        let head = size_of::<RgfxRenderPipelineCreate>();
        let layouts = [
            RgfxVertexBufferLayout {
                buffer_index: 0,
                stride: 12,
                step_function: 1,
                step_rate: 1,
            },
            RgfxVertexBufferLayout {
                buffer_index: 3,
                stride: 16,
                step_function: 2,
                step_rate: 4,
            },
        ];
        let attributes = [
            RgfxVertexAttribute {
                attribute_index: 0,
                format: 29,
                offset: 0,
                buffer_index: 0,
            },
            RgfxVertexAttribute {
                attribute_index: 1,
                format: 9,
                offset: 8,
                buffer_index: 0,
            },
            RgfxVertexAttribute {
                attribute_index: 5,
                format: 31,
                offset: 4,
                buffer_index: 3,
            },
        ];
        let (vn, fnm) = (b"vmain".as_slice(), b"fmain".as_slice());
        let layouts_at = head;
        let attributes_at = layouts_at + size_of_val(&layouts);
        let vn_at = attributes_at + size_of_val(&attributes);
        let fn_at = vn_at + vn.len();
        let total = (fn_at + fnm.len()).next_multiple_of(8);
        let mut colors = [RgfxColorAttachmentDesc::default(); 8];
        colors[..3].copy_from_slice(&whole_render_colors());
        let m = RgfxRenderPipelineCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_RENDER_PIPELINE_CREATE,
                size: total as u32,
                seqno: 9,
            },
            obj_id,
            vertex_library_obj_id: 1,
            fragment_library_obj_id: if fragment { 1 } else { 0 },
            vertex_name_offset: vn_at as u32,
            vertex_name_len: vn.len() as u32,
            fragment_name_offset: if fragment { fn_at as u32 } else { 0 },
            fragment_name_len: if fragment { fnm.len() as u32 } else { 0 },
            vertex_layouts_offset: layouts_at as u32,
            vertex_layout_count: 2,
            vertex_attributes_offset: attributes_at as u32,
            vertex_attribute_count: 3,
            raster_sample_count: 4,
            color_attachment_count: 3,
            depth_pixel_format: 252,
            stencil_pixel_format: 253,
            flags: RGFX_RENDER_PIPELINE_ALPHA_TO_COVERAGE
                | RGFX_RENDER_PIPELINE_ALPHA_TO_ONE
                | RGFX_RENDER_PIPELINE_RASTERIZATION_ENABLED
                | RGFX_RENDER_PIPELINE_SUPPORT_ICB,
            input_primitive_topology: 3,
            max_vertex_amplification_count: 2,
            colors,
            ..RgfxRenderPipelineCreate::default()
        };
        let mut bytes = m.as_bytes().to_vec();
        bytes.extend_from_slice(layouts.as_bytes());
        bytes.extend_from_slice(attributes.as_bytes());
        bytes.extend_from_slice(vn);
        bytes.extend_from_slice(fnm);
        bytes.resize(total, 0);
        bytes
    }

    /// What [`whole_render_create`] carries, as the platform's value takes it.
    fn whole_render_fields() -> RenderFields {
        let layout = |buffer_index, stride, step_function, step_rate| BufferLayoutDesc {
            buffer_index,
            stride,
            step_function,
            step_rate,
        };
        let attribute = |attribute_index, format, offset, buffer_index| AttributeDesc {
            attribute_index,
            format,
            offset,
            buffer_index,
        };
        RenderFields {
            vertex_descriptor: Some(VertexDesc {
                layouts: vec![layout(0, 12, 1, 1), layout(3, 16, 2, 4)],
                attributes: vec![
                    attribute(0, 29, 0, 0),
                    attribute(1, 9, 8, 0),
                    attribute(5, 31, 4, 3),
                ],
            }),
            colors: vec![
                (
                    0,
                    ColorAttachmentDesc {
                        pixel_format: 80,
                        blending_enabled: false,
                        source_rgb_blend_factor: 0,
                        destination_rgb_blend_factor: 0,
                        rgb_blend_operation: 0,
                        source_alpha_blend_factor: 0,
                        destination_alpha_blend_factor: 0,
                        alpha_blend_operation: 0,
                        write_mask: 0x9,
                    },
                ),
                (
                    1,
                    ColorAttachmentDesc {
                        pixel_format: 115,
                        blending_enabled: true,
                        source_rgb_blend_factor: 4,
                        destination_rgb_blend_factor: 5,
                        rgb_blend_operation: 1,
                        source_alpha_blend_factor: 2,
                        destination_alpha_blend_factor: 7,
                        alpha_blend_operation: 3,
                        write_mask: 0xf,
                    },
                ),
                (
                    2,
                    ColorAttachmentDesc {
                        pixel_format: 55,
                        blending_enabled: false,
                        source_rgb_blend_factor: 0,
                        destination_rgb_blend_factor: 0,
                        rgb_blend_operation: 0,
                        source_alpha_blend_factor: 0,
                        destination_alpha_blend_factor: 0,
                        alpha_blend_operation: 0,
                        write_mask: 0x8,
                    },
                ),
            ],
            depth_pixel_format: 252,
            stencil_pixel_format: 253,
            raster_sample_count: 4,
            alpha_to_coverage: true,
            alpha_to_one: true,
            rasterization_enabled: true,
            input_primitive_topology: 3,
            max_vertex_amplification_count: 2,
            support_indirect_command_buffers: true,
        }
    }

    /// Every field of a render create reaches the platform's creator, read from the wire as the
    /// value holds it: the colour attachments in use with their blend state and write masks, the
    /// vertex descriptor's layouts and attributes, the depth and stencil formats, the sample
    /// count, the four flags, the topology and the amplification count; a create naming library
    /// 0 and no name for its fragment function has none
    #[test]
    fn a_render_create_carries_its_whole_descriptor() {
        let message = whole_render_create(100, true);
        let (m, _) = RgfxRenderPipelineCreate::read_from_prefix(&message).unwrap();
        assert_eq!(render_fields(&message, &m), Ok(whole_render_fields()));
        let (p, mut dec, replies) = pipeline_decoder();
        let device = p.device();
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        assert_eq!(replies.lock()[0].status, RGFX_PIPELINE_OK);
        let calls = p
            .call_log()
            .calls_to("newRenderPipelineStateWithDescriptor:error:");
        let Arg::Val(logged) = &calls[0].args[0] else {
            panic!("{:?}", calls[0].args);
        };
        let fields = whole_render_fields();
        for part in [
            format!("vertex_descriptor: {:?}", fields.vertex_descriptor),
            format!("colors: {:?}", fields.colors),
            "depth_pixel_format: 252, stencil_pixel_format: 253, raster_sample_count: 4, \
             alpha_to_coverage: true, alpha_to_one: true, rasterization_enabled: true, \
             input_primitive_topology: 3, max_vertex_amplification_count: 2, \
             support_indirect_command_buffers: true"
                .to_owned(),
        ] {
            assert!(logged.contains(&part), "{logged}\nlacks {part}");
        }
        assert!(logged.contains("fragment: Some("), "{logged}");
        // No fragment function: none reaches the creator, and the pipeline is made.
        assert_eq!(
            dec.decode(&device, &whole_render_create(101, false))
                .unwrap(),
            Ok(())
        );
        assert_eq!(replies.lock()[1].status, RGFX_PIPELINE_OK);
        let calls = p
            .call_log()
            .calls_to("newRenderPipelineStateWithDescriptor:error:");
        let Arg::Val(logged) = &calls[1].args[0] else {
            panic!("{:?}", calls[1].args);
        };
        assert!(logged.contains("fragment: None"), "{logged}");
    }

    /// A `MESH_RENDER_PIPELINE_CREATE` of pipeline `obj_id` from library 1's `mmain` and, with
    /// `object` and `fragment`, its `omain` and `fmain`: one BGRA8Unorm target, rasterizing where
    /// it has a fragment function, a mesh threadgroup of at most 32 threads, a payload of 256 bytes
    /// and the mesh threadgroup size required 4 x 1 x 1.
    pub(super) fn mesh_create(obj_id: u64, object: bool, fragment: bool) -> Vec<u8> {
        let head = size_of::<RgfxMeshRenderPipelineCreate>();
        let mut tail = Vec::new();
        let mut name = |n: &[u8], on: bool| -> (u64, u32, u32) {
            if !on {
                return (0, 0, 0);
            }
            let at = (head + tail.len()) as u32;
            tail.extend_from_slice(n);
            (1, at, n.len() as u32)
        };
        let (object_library, object_at, object_len) = name(b"omain", object);
        let (mesh_library, mesh_at, mesh_len) = name(b"mmain", true);
        let (fragment_library, fragment_at, fragment_len) = name(b"fmain", fragment);
        let total = (head + tail.len()).next_multiple_of(8);
        let mut colors = [RgfxColorAttachmentDesc::default(); 8];
        colors[0].pixel_format = 80;
        colors[0].write_mask = 0xf;
        let m = RgfxMeshRenderPipelineCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_MESH_RENDER_PIPELINE_CREATE,
                size: total as u32,
                seqno: 4,
            },
            obj_id,
            object_library_obj_id: object_library,
            mesh_library_obj_id: mesh_library,
            fragment_library_obj_id: fragment_library,
            object_name_offset: object_at,
            object_name_len: object_len,
            mesh_name_offset: mesh_at,
            mesh_name_len: mesh_len,
            fragment_name_offset: fragment_at,
            fragment_name_len: fragment_len,
            raster_sample_count: 1,
            color_attachment_count: 1,
            flags: if fragment {
                RGFX_RENDER_PIPELINE_RASTERIZATION_ENABLED
            } else {
                0
            },
            max_total_threads_per_mesh_threadgroup: 32,
            payload_memory_length: 256,
            required_threads_per_mesh_threadgroup: [4, 1, 1],
            colors,
            ..RgfxMeshRenderPipelineCreate::default()
        };
        let mut bytes = m.as_bytes().to_vec();
        bytes.extend_from_slice(&tail);
        bytes.resize(total, 0);
        bytes
    }

    /// A mesh render pipeline create reaches the platform's mesh creator with its descriptor, and
    /// is answered with its reflection of the mesh kind and, after it, the state's mesh limits,
    /// the reply's size counting them; an unbounded grid is the wire's highest. A create naming
    /// no mesh function is malformed; one the platform refuses is a refused reply of the mesh
    /// kind, with no limits after it
    #[test]
    fn a_mesh_render_create_is_answered_with_its_mesh_limits() {
        use reims_vgpu_core::testkit::faults::Fault;
        let (p, mut dec, _) = pipeline_decoder();
        let device = p.device();
        let replies = Arc::new(reims_vgpu_core::sync::Mutex::new(Vec::new()));
        let sink_replies = Arc::clone(&replies);
        dec.set_reflection_sink(Arc::new(move |r: PipelineReply| {
            sink_replies.lock().push(r.bytes());
        }));
        assert_eq!(
            dec.decode(&device, &mesh_create(100, true, true)).unwrap(),
            Ok(())
        );
        let calls = p
            .call_log()
            .calls_to("newRenderPipelineStateWithMeshDescriptor:options:reflection:error:");
        let Arg::Val(logged) = &calls[0].args[0] else {
            panic!("{:?}", calls[0].args);
        };
        for part in [
            "object: Some(",
            "fragment: Some(",
            "max_total_threads_per_mesh_threadgroup: 32",
            "payload_memory_length: 256",
            "rasterization_enabled: true",
        ] {
            assert!(
                logged.contains(part),
                "{logged}
lacks {part}"
            );
        }
        let reply = replies.lock()[0].clone();
        assert_eq!(
            reply.len(),
            size_of::<RgfxPipelineReflection>() + size_of::<RgfxMeshPipelineReflection>()
        );
        let (r, _) = RgfxPipelineReflection::read_from_prefix(&reply).unwrap();
        assert_eq!(r.header.size as usize, reply.len());
        assert_eq!(
            (r.obj_id, r.status, r.kind),
            (100, RGFX_PIPELINE_OK, RGFX_PIPELINE_KIND_MESH)
        );
        let (limits, _) = RgfxMeshPipelineReflection::read_from_prefix(
            &reply[reims_graphics::wire::RGFX_PIPELINE_REFLECTION_MESH_OFFSET as usize..],
        )
        .unwrap();
        assert_eq!(
            limits,
            RgfxMeshPipelineReflection {
                max_total_threads_per_object_threadgroup: 1024,
                max_total_threads_per_mesh_threadgroup: 32,
                object_thread_execution_width: 32,
                mesh_thread_execution_width: 32,
                max_total_threadgroups_per_mesh_grid: u32::MAX,
                reserved_14: 0,
            }
        );

        // No object stage: the fake answers no object limits, as an x86 Mac's Metal does.
        assert_eq!(
            dec.decode(&device, &mesh_create(101, false, false))
                .unwrap(),
            Ok(())
        );
        let reply = replies.lock()[1].clone();
        let (limits, _) = RgfxMeshPipelineReflection::read_from_prefix(
            &reply[reims_graphics::wire::RGFX_PIPELINE_REFLECTION_MESH_OFFSET as usize..],
        )
        .unwrap();
        assert_eq!(
            (
                limits.max_total_threads_per_object_threadgroup,
                limits.object_thread_execution_width
            ),
            (0, 0)
        );

        let mut nameless = mesh_create(102, false, true);
        edit_head(&mut nameless, |m: &mut RgfxMeshRenderPipelineCreate| {
            (m.mesh_library_obj_id, m.mesh_name_len) = (0, 0);
        });
        assert_eq!(
            dec.decode(&device, &nameless).unwrap(),
            Err(DecodeError::Malformed)
        );

        p.faults().at(
            "MTLDevice",
            "newRenderPipelineStateWithMeshDescriptor:options:reflection:error:",
            0,
            Fault::Crash,
        );
        assert_eq!(
            dec.decode(&device, &mesh_create(103, true, true)).unwrap(),
            Err(DecodeError::Refused)
        );
        let reply = replies.lock().last().unwrap().clone();
        assert_eq!(reply.len(), size_of::<RgfxPipelineReflection>());
        let (r, _) = RgfxPipelineReflection::read_from_prefix(&reply).unwrap();
        assert_eq!(
            (r.obj_id, r.status, r.kind, r.header.size as usize),
            (
                103,
                RGFX_PIPELINE_REFUSED,
                RGFX_PIPELINE_KIND_MESH,
                reply.len()
            )
        );
    }

    /// A pipeline create the decoder carries out at once (not on the device's compile pool) that
    /// the platform ends the process at is refused there: the create answers
    /// [`DecodeError::Refused`] with a refused reply, nothing is stored, and the next is made
    #[test]
    fn a_compile_the_platform_refuses_in_the_decoder_is_a_refused_reply() {
        use reims_vgpu_core::testkit::faults::Fault;
        let (p, mut dec, replies) = pipeline_decoder();
        let device = p.device();
        let faults = p.faults();
        faults.at(
            "MTLDevice",
            "newRenderPipelineStateWithDescriptor:error:",
            0,
            Fault::Crash,
        );
        assert_eq!(
            dec.decode(&device, &whole_render_create(100, true))
                .unwrap(),
            Err(DecodeError::Refused)
        );
        faults.at("MTLLibrary", "newFunctionWithName:", 0, Fault::Crash);
        assert_eq!(
            dec.decode(&device, &whole_render_create(101, false))
                .unwrap(),
            Err(DecodeError::Refused)
        );
        assert_eq!(dec.object_count(), 1, "only the library");
        assert_eq!(
            dec.decode(&device, &whole_render_create(102, true))
                .unwrap(),
            Ok(())
        );
        let statuses: Vec<_> = replies
            .lock()
            .iter()
            .map(|r| (r.obj_id, r.status))
            .collect();
        assert_eq!(
            statuses,
            [
                (100, RGFX_PIPELINE_REFUSED),
                (101, RGFX_PIPELINE_REFUSED),
                (102, RGFX_PIPELINE_OK)
            ]
        );
    }

    /// A render create with more than eight colour attachments, or a vertex buffer or attribute
    /// index past 30, is malformed: no platform call, and a refused reply
    #[test]
    fn a_render_create_past_metal_s_arrays_is_malformed() {
        let (p, mut dec, replies) = pipeline_decoder();
        let device = p.device();
        let whole = whole_render_create(100, true);
        let edit = |f: &dyn Fn(&mut RgfxRenderPipelineCreate)| {
            let (mut m, _) = RgfxRenderPipelineCreate::read_from_prefix(&whole).unwrap();
            f(&mut m);
            let mut bytes = whole.clone();
            bytes[..size_of::<RgfxRenderPipelineCreate>()].copy_from_slice(m.as_bytes());
            bytes
        };
        let layouts_at = size_of::<RgfxRenderPipelineCreate>();
        let mut past_buffer = whole.clone();
        past_buffer[layouts_at..layouts_at + 4].copy_from_slice(&31u32.to_ne_bytes());
        let attributes_at = layouts_at + 2 * size_of::<RgfxVertexBufferLayout>();
        let mut past_attribute = whole.clone();
        past_attribute[attributes_at..attributes_at + 4].copy_from_slice(&31u32.to_ne_bytes());
        let m = p.call_log().mark();
        for bad in [
            edit(&|m| m.color_attachment_count = 9),
            edit(&|m| m.vertex_attribute_count = 99),
            past_buffer,
            past_attribute,
        ] {
            assert_eq!(
                dec.decode(&device, &bad).unwrap(),
                Err(DecodeError::Malformed)
            );
        }
        assert!(
            !p.call_log()
                .sels_since(m)
                .contains(&"newRenderPipelineStateWithDescriptor:error:")
        );
        assert!(
            replies
                .lock()
                .iter()
                .all(|r| r.status == RGFX_PIPELINE_REFUSED)
        );
        assert_eq!(replies.lock().len(), 4);
    }

    /// Every field of a compute create reaches the platform's creator: the threadgroup-size
    /// multiple, the maximum threads per threadgroup, the maximum call stack depth and indirect
    /// command buffer support
    #[test]
    fn a_compute_create_carries_its_whole_descriptor() {
        let (p, mut dec, replies) = pipeline_decoder();
        let device = p.device();
        let head = size_of::<RgfxComputePipelineCreate>() as u32;
        let name = b"cmain".as_slice();
        let total = (head as usize + name.len()).next_multiple_of(8);
        let m = RgfxComputePipelineCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_COMPUTE_PIPELINE_CREATE,
                size: total as u32,
                seqno: 3,
            },
            obj_id: 100,
            function_library_obj_id: 1,
            name_offset: head,
            name_len: name.len() as u32,
            flags: RGFX_COMPUTE_PIPELINE_TG_MULTIPLE_OF_EXEC_WIDTH
                | RGFX_COMPUTE_PIPELINE_SUPPORT_ICB,
            max_total_threads_per_threadgroup: 256,
            max_call_stack_depth: 7,
            ..RgfxComputePipelineCreate::default()
        };
        let mut bytes = m.as_bytes().to_vec();
        bytes.extend_from_slice(name);
        bytes.resize(total, 0);
        assert_eq!(dec.decode(&device, &bytes).unwrap(), Ok(()));
        assert_eq!(replies.lock()[0].status, RGFX_PIPELINE_OK);
        let calls = p
            .call_log()
            .calls_to("newComputePipelineStateWithDescriptor:options:reflection:error:");
        let Arg::Val(logged) = &calls[0].args[0] else {
            panic!("{:?}", calls[0].args);
        };
        let part = "threadgroup_size_is_multiple_of_execution_width: true, \
                    max_total_threads_per_threadgroup: 256, max_call_stack_depth: 7, \
                    support_indirect_command_buffers: true, stage_input: None";
        assert!(logged.contains(part), "{logged}");
    }

    /// A linked function [`append_linked`] appends: its name, list, `MTLFunctionOptions`, group
    /// name (empty for none), constants and specialized name (empty for none).
    type LinkedSpec<'a> = (
        &'a str,
        u32,
        u32,
        &'a str,
        &'a [RgfxFunctionConstant],
        &'a str,
    );

    /// Appends to `bytes` the linked functions `specs` name ([`LinkedSpec`]), each of library
    /// `library`, then their records, and answers the records' offset and count; the message's
    /// size is set again, a multiple of 8.
    fn append_linked(bytes: &mut Vec<u8>, library: u64, specs: &[LinkedSpec<'_>]) -> (u32, u32) {
        let mut records = Vec::new();
        for &(name, list, options, group, constants, specialized) in specs {
            let name_offset = bytes.len() as u32;
            bytes.extend_from_slice(name.as_bytes());
            let group_offset = bytes.len() as u32;
            bytes.extend_from_slice(group.as_bytes());
            let specialized_offset = bytes.len() as u32;
            bytes.extend_from_slice(specialized.as_bytes());
            let constants_offset = bytes.len() as u32;
            bytes.extend_from_slice(constants.as_bytes());
            records.push(RgfxLinkedFunction {
                library_obj_id: library,
                name_offset,
                name_len: name.len() as u32,
                constants_offset,
                constant_count: constants.len() as u32,
                list,
                options,
                group_offset: if group.is_empty() { 0 } else { group_offset },
                group_len: group.len() as u32,
                specialized_name_offset: if specialized.is_empty() {
                    0
                } else {
                    specialized_offset
                },
                specialized_name_len: specialized.len() as u32,
            });
        }
        bytes.resize(bytes.len().next_multiple_of(8), 0);
        let at = bytes.len() as u32;
        bytes.extend_from_slice(records.as_bytes());
        let size = bytes.len() as u32;
        bytes[4..8].copy_from_slice(&size.to_le_bytes());
        (at, specs.len() as u32)
    }

    /// `message` with its fixed head of `T` edited by `f`.
    fn edit_head<T: FromBytes + IntoBytes + KnownLayout + Immutable>(
        message: &mut [u8],
        f: impl FnOnce(&mut T),
    ) {
        let (mut m, _) = T::read_from_prefix(message).unwrap();
        f(&mut m);
        message[..size_of::<T>()].copy_from_slice(m.as_bytes());
    }

    /// A compute create of library 1's `cmain`, with no field past its function set.
    pub(super) fn compute_create(obj_id: u64) -> Vec<u8> {
        let head = size_of::<RgfxComputePipelineCreate>() as u32;
        let name = b"cmain".as_slice();
        let total = (head as usize + name.len()).next_multiple_of(8);
        let m = RgfxComputePipelineCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_COMPUTE_PIPELINE_CREATE,
                size: total as u32,
                seqno: 4,
            },
            obj_id,
            function_library_obj_id: 1,
            name_offset: head,
            name_len: name.len() as u32,
            max_call_stack_depth: 1,
            ..RgfxComputePipelineCreate::default()
        };
        let mut bytes = m.as_bytes().to_vec();
        bytes.extend_from_slice(name);
        bytes.resize(total, 0);
        bytes
    }

    /// The `ObjId` numbers in `text` after `from`, in order.
    fn ids_after(text: &str, from: &str) -> Vec<u64> {
        let rest = &text[text
            .find(from)
            .unwrap_or_else(|| panic!("{text}\nlacks {from}"))..];
        rest.split("ObjId(")
            .skip(1)
            .map(|p| p[..p.find(')').unwrap()].parse().unwrap())
            .collect()
    }

    /// A compute create's linked functions reach the platform's creator in their lists, each
    /// made from its library with its constants, and with its options where it has any (a binary
    /// function compiled to a binary); its stage-input descriptor reaches it too, a dynamic
    /// stride as Metal's `MTLBufferLayoutStrideDynamic`
    #[test]
    fn a_compute_create_carries_its_linked_functions_and_stage_input() {
        let (p, mut dec, replies) = pipeline_decoder();
        let device = p.device();
        let mut bytes = compute_create(100);
        let constant = RgfxFunctionConstant {
            index: 2,
            data_type: 29,
            value: [7; 32],
        };
        let (linked_offset, linked_count) = append_linked(
            &mut bytes,
            1,
            &[
                ("fa", RGFX_LINKED_FUNCTIONS, 0, "", &[], ""),
                ("fb", RGFX_LINKED_BINARY_FUNCTIONS, 1, "", &[], ""),
                ("ga", RGFX_LINKED_GROUP, 0, "g", &[], ""),
                ("gb", RGFX_LINKED_GROUP, 0, "g", &[], ""),
                ("fp", RGFX_LINKED_PRIVATE_FUNCTIONS, 0, "", &[constant], ""),
            ],
        );
        let layouts_at = bytes.len() as u32;
        bytes.extend_from_slice(
            RgfxVertexBufferLayout {
                buffer_index: 1,
                stride: u32::MAX,
                step_function: 5,
                step_rate: 1,
            }
            .as_bytes(),
        );
        let attributes_at = bytes.len() as u32;
        bytes.extend_from_slice(
            RgfxVertexAttribute {
                attribute_index: 0,
                format: 36,
                offset: 0,
                buffer_index: 1,
            }
            .as_bytes(),
        );
        let size = bytes.len() as u32;
        edit_head(&mut bytes, |m: &mut RgfxComputePipelineCreate| {
            m.header.size = size;
            m.linked_offset = linked_offset;
            m.linked_count = linked_count;
            m.stage_input_layouts_offset = layouts_at;
            m.stage_input_layout_count = 1;
            m.stage_input_attributes_offset = attributes_at;
            m.stage_input_attribute_count = 1;
            m.stage_input_index_type = 1;
            m.stage_input_index_buffer_index = 2;
        });
        let mark = p.call_log().mark();
        assert_eq!(dec.decode(&device, &bytes).unwrap(), Ok(()));
        assert_eq!(replies.lock()[0].status, RGFX_PIPELINE_OK);
        // Each function made in the wire's order, the binary one with its options.
        let made: Vec<_> = p
            .call_log()
            .since(mark)
            .into_iter()
            .filter(|c| c.sel.starts_with("newFunction"))
            .map(|c| (c.sel, c.args.clone()))
            .collect();
        let sels: Vec<_> = made.iter().map(|(sel, _)| *sel).collect();
        assert_eq!(
            sels,
            [
                "newFunctionWithName:",
                "newFunctionWithName:",
                "newFunctionWithDescriptor:error:",
                "newFunctionWithName:",
                "newFunctionWithName:",
                "newFunctionWithName:constantValues:error:",
            ]
        );
        assert_eq!(made[2].1[..2], [Arg::Str("fb".to_owned()), Arg::U64(1)]);
        assert_eq!(
            made[5].1[..3],
            [Arg::Str("fp".to_owned()), Arg::U64(2), Arg::U64(29)]
        );
        let calls = p
            .call_log()
            .calls_to("newComputePipelineStateWithDescriptor:options:reflection:error:");
        let Arg::Val(logged) = &calls[0].args[0] else {
            panic!("{:?}", calls[0].args);
        };
        // The kernel, then fa, fb, ga, gb and fp, each in its own list.
        let [kernel] = ids_after(logged, "function: ")[..1] else {
            unreachable!()
        };
        let lists = [
            "functions: [",
            "binary_functions: [",
            "groups: [(\"g\", [",
            "private_functions: [",
        ];
        let firsts: Vec<_> = lists.iter().map(|l| ids_after(logged, l)[0]).collect();
        assert!(kernel < firsts[0], "{logged}");
        assert!(firsts.windows(2).all(|w| w[0] < w[1]), "{logged}");
        assert!(logged.contains("groups: [(\"g\", [ObjId("), "{logged}");
        assert_eq!(
            ids_after(logged, "groups: [").len(),
            3,
            "two in g, then fp: {logged}"
        );
        let stage = "stage_input: Some(StageInputDesc { layouts: [BufferLayoutDesc { buffer_index: \
                     1, stride: 18446744073709551615, step_function: 5, step_rate: 1 }], \
                     attributes: [AttributeDesc { attribute_index: 0, format: 36, offset: 0, \
                     buffer_index: 1 }], index_type: 1, index_buffer_index: 2 })";
        assert!(logged.contains(stage), "{logged}");
    }

    /// A render create's vertex and fragment linked functions reach the platform's creator, each
    /// stage's its own
    #[test]
    fn a_render_create_carries_its_stages_linked_functions() {
        let (p, mut dec, replies) = pipeline_decoder();
        let device = p.device();
        let mut bytes = whole_render_create(100, true);
        let vertex = append_linked(
            &mut bytes,
            1,
            &[("va", RGFX_LINKED_FUNCTIONS, 0, "", &[], "")],
        );
        let fragment = append_linked(
            &mut bytes,
            1,
            &[("fa", RGFX_LINKED_PRIVATE_FUNCTIONS, 0, "", &[], "")],
        );
        edit_head(&mut bytes, |m: &mut RgfxRenderPipelineCreate| {
            (m.vertex_linked_offset, m.vertex_linked_count) = vertex;
            (m.fragment_linked_offset, m.fragment_linked_count) = fragment;
        });
        assert_eq!(dec.decode(&device, &bytes).unwrap(), Ok(()));
        assert_eq!(replies.lock()[0].status, RGFX_PIPELINE_OK);
        let calls = p
            .call_log()
            .calls_to("newRenderPipelineStateWithDescriptor:error:");
        let Arg::Val(logged) = &calls[0].args[0] else {
            panic!("{:?}", calls[0].args);
        };
        let vertex_part =
            &logged[logged.find("vertex_linked").unwrap()..logged.find("fragment_linked").unwrap()];
        assert_eq!(ids_after(vertex_part, "functions: [").len(), 1, "{logged}");
        assert!(vertex_part.contains("private_functions: []"), "{logged}");
        let fragment_part = &logged[logged.find("fragment_linked").unwrap()..];
        assert!(
            fragment_part.contains(
                "functions: [], binary_functions: [], groups: [], private_functions: [ObjId("
            ),
            "{logged}"
        );
    }

    /// A linked function naming no library is a bad reference and one in no list Metal has is
    /// malformed, neither reaching the platform; a linked function the platform ends the process
    /// at refuses the create
    #[test]
    fn a_create_s_bad_linked_function_makes_nothing() {
        use reims_vgpu_core::testkit::faults::Fault;
        let (p, mut dec, replies) = pipeline_decoder();
        let device = p.device();
        let with = |obj_id, library, list| {
            let mut bytes = compute_create(obj_id);
            let linked = append_linked(&mut bytes, library, &[("fa", list, 1, "", &[], "")]);
            edit_head(&mut bytes, |m: &mut RgfxComputePipelineCreate| {
                (m.linked_offset, m.linked_count) = linked;
            });
            bytes
        };
        let mark = p.call_log().mark();
        assert_eq!(
            dec.decode(&device, &with(100, 77, RGFX_LINKED_FUNCTIONS))
                .unwrap(),
            Err(DecodeError::BadReference)
        );
        assert_eq!(
            dec.decode(&device, &with(101, 1, 9)).unwrap(),
            Err(DecodeError::Malformed)
        );
        assert!(
            !p.call_log()
                .sels_since(mark)
                .iter()
                .any(|s| s.starts_with("newComputePipelineState"))
        );
        p.faults().at(
            "MTLLibrary",
            "newFunctionWithDescriptor:error:",
            0,
            Fault::Crash,
        );
        assert_eq!(
            dec.decode(&device, &with(102, 1, RGFX_LINKED_BINARY_FUNCTIONS))
                .unwrap(),
            Err(DecodeError::Refused)
        );
        let statuses: Vec<_> = replies
            .lock()
            .iter()
            .map(|r| (r.obj_id, r.status))
            .collect();
        assert_eq!(
            statuses,
            [
                (100, RGFX_PIPELINE_REFUSED),
                (101, RGFX_PIPELINE_REFUSED),
                (102, RGFX_PIPELINE_REFUSED)
            ]
        );
        assert_eq!(dec.object_count(), 1, "only the library");
    }

    /// A linked function the guest gave a specialized name is made by descriptor with that name,
    /// though its options are 0, and one with neither by name alone
    #[test]
    fn a_linked_function_s_specialized_name_reaches_the_platform() {
        let (p, mut dec, replies) = pipeline_decoder();
        let device = p.device();
        let mut bytes = compute_create(100);
        let linked = append_linked(
            &mut bytes,
            1,
            &[
                ("fp", RGFX_LINKED_PRIVATE_FUNCTIONS, 0, "", &[], "custom_fn"),
                ("fa", RGFX_LINKED_FUNCTIONS, 0, "", &[], ""),
            ],
        );
        edit_head(&mut bytes, |m: &mut RgfxComputePipelineCreate| {
            (m.linked_offset, m.linked_count) = linked;
        });
        let mark = p.call_log().mark();
        assert_eq!(dec.decode(&device, &bytes).unwrap(), Ok(()));
        assert_eq!(replies.lock()[0].status, RGFX_PIPELINE_OK);
        let made: Vec<_> = p
            .call_log()
            .since(mark)
            .into_iter()
            .filter(|c| c.sel.starts_with("newFunction"))
            .map(|c| (c.sel, c.args))
            .collect();
        assert_eq!(
            made[1..],
            [
                (
                    "newFunctionWithDescriptor:error:",
                    vec![
                        Arg::Str("fp".to_owned()),
                        Arg::U64(0),
                        Arg::Str("custom_fn".to_owned())
                    ]
                ),
                ("newFunctionWithName:", vec![Arg::Str("fa".to_owned())]),
            ]
        );
    }

    /// On Metal, from the wire: a kernel that calls `linked_add_one`, which its library only
    /// declares, is made with the callee library's `linked_add_two` linked as a private function
    /// made with the specialized name `linked_add_one`, and with `linked_add_one` itself; it is
    /// refused with `linked_add_two` linked under its own name, and with `linked_add_one` made
    /// under another
    #[cfg(target_os = "macos")]
    #[test]
    fn on_metal_a_linked_function_s_specialized_name_resolves_its_call() {
        a_linked_function_s_specialized_name_resolves_its_call::<reims_vgpu_metal::MetalPlatform>();
    }

    /// [`on_metal_a_linked_function_s_specialized_name_resolves_its_call`] on the Vulkan
    /// platform, whose answers are Metal's
    #[cfg(feature = "vulkan")]
    #[test]
    fn on_vulkan_a_linked_function_s_specialized_name_resolves_its_call() {
        a_linked_function_s_specialized_name_resolves_its_call::<reims_vgpu_vulkan::VulkanPlatform>(
        );
    }

    /// The cases of [`on_metal_a_linked_function_s_specialized_name_resolves_its_call`] on `P`.
    fn a_linked_function_s_specialized_name_resolves_its_call<P: crate::platform::ApiPlatform>() {
        let platform = P::process();
        let device = platform.metal_device(None).unwrap();
        let mut dec = GraphicsDecoder::<P>::new(
            Arc::clone(&platform),
            Arc::new(reims_graphics::region::Region::new(0x10000).unwrap()),
        );
        let replies = Arc::new(reims_vgpu_core::sync::Mutex::new(Vec::new()));
        let sink_replies = Arc::clone(&replies);
        dec.set_reflection_sink(Arc::new(move |r: PipelineReply| {
            sink_replies
                .lock()
                .push((r.reflection.obj_id, r.reflection.status));
        }));
        let caller =
            include_bytes!("../../../../reims-vgpu-vulkan/metallibs/linked_caller.metallib");
        let callee =
            include_bytes!("../../../../reims-vgpu-vulkan/metallibs/linked_callee.metallib");
        for (obj_id, bytes) in [(1, caller.as_slice()), (2, callee.as_slice())] {
            let mut m = RgfxLibraryCreate {
                header: RgfxMsgHeader {
                    r#type: RGFX_CMD_LIBRARY_CREATE,
                    size: (RGFX_LIBRARY_BYTES_OFFSET as usize + bytes.len()).next_multiple_of(8)
                        as u32,
                    seqno: obj_id,
                },
                obj_id,
                byte_length: bytes.len() as u64,
                ..RgfxLibraryCreate::default()
            }
            .as_bytes()
            .to_vec();
            m.extend_from_slice(bytes);
            m.resize(m.len().next_multiple_of(8), 0);
            assert_eq!(dec.decode(&device, &m).unwrap(), Ok(()));
        }
        let create = |obj_id: u64, linked: &str, specialized: &str| {
            let mut bytes = compute_create(obj_id);
            let name = b"linked_kernel";
            let at = bytes.len() as u32;
            bytes.extend_from_slice(name);
            bytes.resize(bytes.len().next_multiple_of(8), 0);
            let tail = append_linked(
                &mut bytes,
                2,
                &[(
                    linked,
                    RGFX_LINKED_PRIVATE_FUNCTIONS,
                    0,
                    "",
                    &[],
                    specialized,
                )],
            );
            edit_head(&mut bytes, |m: &mut RgfxComputePipelineCreate| {
                (m.name_offset, m.name_len) = (at, name.len() as u32);
                (m.linked_offset, m.linked_count) = tail;
            });
            bytes
        };
        let cases = [
            (100, "linked_add_two", "linked_add_one", RGFX_PIPELINE_OK),
            (101, "linked_add_one", "", RGFX_PIPELINE_OK),
            (102, "linked_add_two", "", RGFX_PIPELINE_REFUSED),
            (103, "linked_add_one", "custom_fn", RGFX_PIPELINE_REFUSED),
        ];
        for (obj_id, linked, specialized, _) in cases {
            let made = dec
                .decode(&device, &create(obj_id, linked, specialized))
                .unwrap();
            assert!(
                made == Ok(()) || made == Err(DecodeError::Refused),
                "{linked} as {specialized:?}: {made:?}"
            );
        }
        let got: Vec<_> = replies.lock().clone();
        let want: Vec<_> = cases
            .iter()
            .map(|&(obj_id, _, _, status)| (obj_id, status))
            .collect();
        assert_eq!(got, want);
    }

    /// The shell tests' stitchable functions (guest/bundle/tests/stitchable.metal): `fused`, a
    /// stitched graph of two of them, is `(a + b) * c`, which `stitch_caller` writes of its inputs.
    #[cfg(target_os = "macos")]
    const STITCHABLE: &str = include_str!("../../../../../guest/bundle/tests/stitchable.metal");

    /// `stitch_caller`'s inputs, and what `fused` makes of them.
    #[cfg(any(target_os = "macos", feature = "vulkan"))]
    const STITCH_INPUTS: [f32; 3] = [2.0, 3.0, 4.0];

    /// What the guest sends of a stitched library, made on this Mac as the guest's route makes it
    /// (the bundle's `RgfxStitchOverShadows`): the caller's library, compiled by Metal, and the
    /// library Metal's stitcher makes of the graph `fused` over that library's functions, each the
    /// bytes Metal answers for it (`libraryDataContents`); and Metal's own device and libraries.
    #[cfg(target_os = "macos")]
    struct Stitched {
        device: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>>,
        caller: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLLibrary>>,
        stitched: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLLibrary>>,
        caller_bytes: Vec<u8>,
        stitched_bytes: Vec<u8>,
    }

    /// [`Stitched`], made on this Mac's default device.
    #[cfg(target_os = "macos")]
    fn stitched_on_metal() -> Stitched {
        use objc2::rc::Retained;
        use objc2::runtime::ProtocolObject;
        use objc2::{AnyThread as _, msg_send};
        use objc2_foundation::{NSArray, NSData, NSString};
        use objc2_metal::{
            MTLCompileOptions, MTLCreateSystemDefaultDevice, MTLDevice as _,
            MTLFunctionStitchingAttribute, MTLFunctionStitchingFunctionNode,
            MTLFunctionStitchingGraph, MTLFunctionStitchingInputNode, MTLFunctionStitchingNode,
            MTLLibrary, MTLStitchedLibraryDescriptor,
        };
        let bytes = |library: &ProtocolObject<dyn MTLLibrary>| {
            // SAFETY: Metal's library answers its bytes to this selector (measured: macOS 27).
            let data: Retained<NSData> = unsafe { msg_send![library, libraryDataContents] };
            data.to_vec()
        };
        let device = MTLCreateSystemDefaultDevice().expect("a Metal device");
        let caller = device
            .newLibraryWithSource_options_error(
                &NSString::from_str(STITCHABLE),
                Some(&MTLCompileOptions::new()),
            )
            .unwrap();
        let input = |index| {
            let node = MTLFunctionStitchingInputNode::alloc();
            // SAFETY: an argument index of the graph's three.
            let node = unsafe { MTLFunctionStitchingInputNode::initWithArgumentIndex(node, index) };
            ProtocolObject::<dyn MTLFunctionStitchingNode>::from_retained(node)
        };
        let call =
            |name: &str, arguments: &[Retained<ProtocolObject<dyn MTLFunctionStitchingNode>>]| {
                MTLFunctionStitchingFunctionNode::initWithName_arguments_controlDependencies(
                    MTLFunctionStitchingFunctionNode::alloc(),
                    &NSString::from_str(name),
                    &NSArray::from_retained_slice(arguments),
                    &NSArray::new(),
                )
            };
        let sum = call("stitch_add", &[input(0), input(1)]);
        let product = call(
            "stitch_mul",
            &[ProtocolObject::from_retained(sum.clone()), input(2)],
        );
        let graph = MTLFunctionStitchingGraph::initWithFunctionName_nodes_outputNode_attributes(
            MTLFunctionStitchingGraph::alloc(),
            &NSString::from_str("fused"),
            &NSArray::from_retained_slice(&[sum]),
            Some(&product),
            &NSArray::<ProtocolObject<dyn MTLFunctionStitchingAttribute>>::new(),
        );
        let function = |name: &str| {
            caller
                .newFunctionWithName(&NSString::from_str(name))
                .unwrap()
        };
        let descriptor = MTLStitchedLibraryDescriptor::new();
        descriptor.setFunctionGraphs(&NSArray::from_retained_slice(&[graph]));
        // SAFETY: the functions are of a library of this device.
        unsafe {
            descriptor.setFunctions(&NSArray::from_retained_slice(&[
                function("stitch_add"),
                function("stitch_mul"),
            ]))
        };
        let stitched = device
            .newLibraryWithStitchedDescriptor_error(&descriptor)
            .unwrap();
        Stitched {
            caller_bytes: bytes(&caller),
            stitched_bytes: bytes(&stitched),
            device,
            caller,
            stitched,
        }
    }

    /// Where a compute pipeline links `fused` from: one of the lists of `MTLLinkedFunctions`, or
    /// none.
    #[cfg(any(target_os = "macos", feature = "vulkan"))]
    const STITCH_LINKS: [Option<u32>; 3] = [
        Some(RGFX_LINKED_FUNCTIONS),
        Some(RGFX_LINKED_PRIVATE_FUNCTIONS),
        None,
    ];

    /// Metal's own answers: for each of [`STITCH_LINKS`], whether Metal makes `stitch_caller`'s
    /// pipeline with `fused` linked there, and what the pipeline writes of [`STITCH_INPUTS`].
    #[cfg(target_os = "macos")]
    fn metal_s_own_stitched_links(s: &Stitched) -> Vec<Option<f32>> {
        use objc2_foundation::{NSArray, NSString};
        use objc2_metal::{
            MTLBuffer as _, MTLCommandBuffer as _, MTLCommandEncoder as _, MTLCommandQueue as _,
            MTLComputeCommandEncoder as _, MTLComputePipelineDescriptor, MTLDevice as _,
            MTLLibrary as _, MTLLinkedFunctions, MTLPipelineOption, MTLResourceOptions, MTLSize,
        };
        let kernel = s
            .caller
            .newFunctionWithName(&NSString::from_str("stitch_caller"))
            .unwrap();
        let fused = s
            .stitched
            .newFunctionWithName(&NSString::from_str("fused"))
            .unwrap();
        STITCH_LINKS
            .iter()
            .map(|&list| {
                let d = MTLComputePipelineDescriptor::new();
                d.setComputeFunction(Some(&kernel));
                let linked = MTLLinkedFunctions::new();
                let one = NSArray::from_retained_slice(std::slice::from_ref(&fused));
                match list {
                    Some(RGFX_LINKED_FUNCTIONS) => linked.setFunctions(Some(&one)),
                    Some(_) => linked.setPrivateFunctions(Some(&one)),
                    None => {}
                }
                d.setLinkedFunctions(Some(&linked));
                let state = s
                    .device
                    .newComputePipelineStateWithDescriptor_options_reflection_error(
                        &d,
                        MTLPipelineOption::None,
                        None,
                    )
                    .ok()?;
                let input = STITCH_INPUTS.map(f32::to_le_bytes).concat();
                // SAFETY: `input` holds the length given.
                let inputs = unsafe {
                    s.device.newBufferWithBytes_length_options(
                        std::ptr::NonNull::new(input.as_ptr().cast_mut().cast()).unwrap(),
                        input.len(),
                        MTLResourceOptions::StorageModeShared,
                    )
                }
                .unwrap();
                let output = s
                    .device
                    .newBufferWithLength_options(4, MTLResourceOptions::StorageModeShared)
                    .unwrap();
                let queue = s.device.newCommandQueue().unwrap();
                let cb = queue.commandBuffer().unwrap();
                let enc = cb.computeCommandEncoder().unwrap();
                enc.setComputePipelineState(&state);
                // SAFETY: the kernel's two buffers, at its indices.
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&output), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&inputs), 0, 1);
                }
                let one = MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                };
                enc.dispatchThreadgroups_threadsPerThreadgroup(one, one);
                enc.endEncoding();
                cb.commit();
                cb.waitUntilCompleted();
                // SAFETY: the GPU wrote the buffer's 4 bytes, and is done.
                Some(unsafe { output.contents().cast::<f32>().read() })
            })
            .collect()
    }

    /// What the guest's stitched library gets through the decoder on `P`: the two library creates
    /// the guest sends (`caller`'s bytes, then `stitched`'s), then for each of [`STITCH_LINKS`] a
    /// pipeline create of `stitch_caller` linking `fused` from the stitched library, and what each
    /// pipeline made writes of [`STITCH_INPUTS`], run on the platform; `None` for one refused.
    #[cfg(any(target_os = "macos", feature = "vulkan"))]
    fn stitched_links_through_the_decoder<P: crate::platform::ApiPlatform>(
        caller: &[u8],
        stitched: &[u8],
    ) -> Vec<Option<f32>> {
        use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
        use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder as _};
        use reims_vgpu_core::platform::device::GpuDevice as _;
        use reims_vgpu_core::platform::resource::Buffer as _;
        use reims_vgpu_core::platform::types::{MtlSize, NsRange};
        use reims_vgpu_core::platform::values::ComputePassDesc;
        use reims_vgpu_core::raise::OnFault;
        let platform = P::process();
        let device = platform.metal_device(None).unwrap();
        let mut dec = GraphicsDecoder::<P>::new(
            Arc::clone(&platform),
            Arc::new(reims_graphics::region::Region::new(0x10000).unwrap()),
        );
        let replies = Arc::new(reims_vgpu_core::sync::Mutex::new(Vec::new()));
        let sink_replies = Arc::clone(&replies);
        dec.set_reflection_sink(Arc::new(move |r: PipelineReply| {
            sink_replies
                .lock()
                .push((r.reflection.obj_id, r.reflection.status));
        }));
        for (obj_id, bytes) in [(1, caller), (2, stitched)] {
            let mut m = RgfxLibraryCreate {
                header: RgfxMsgHeader {
                    r#type: RGFX_CMD_LIBRARY_CREATE,
                    size: (RGFX_LIBRARY_BYTES_OFFSET as usize + bytes.len()).next_multiple_of(8)
                        as u32,
                    seqno: obj_id,
                },
                obj_id,
                byte_length: bytes.len() as u64,
                ..RgfxLibraryCreate::default()
            }
            .as_bytes()
            .to_vec();
            m.extend_from_slice(bytes);
            m.resize(m.len().next_multiple_of(8), 0);
            assert_eq!(dec.decode(&device, &m).unwrap(), Ok(()));
        }
        let mut made = Vec::new();
        for (n, &list) in STITCH_LINKS.iter().enumerate() {
            let obj_id = 100 + n as u64;
            let mut bytes = compute_create(obj_id);
            let name = b"stitch_caller";
            let at = bytes.len() as u32;
            bytes.extend_from_slice(name);
            bytes.resize(bytes.len().next_multiple_of(8), 0);
            let specs: Vec<LinkedSpec<'_>> = list
                .map(|list| ("fused", list, 0, "", &[][..], ""))
                .into_iter()
                .collect();
            let tail = append_linked(&mut bytes, 2, &specs);
            edit_head(&mut bytes, |m: &mut RgfxComputePipelineCreate| {
                (m.name_offset, m.name_len) = (at, name.len() as u32);
                (m.linked_offset, m.linked_count) = tail;
            });
            let decoded = dec.decode(&device, &bytes).unwrap();
            assert!(
                decoded == Ok(()) || decoded == Err(DecodeError::Refused),
                "{list:?}: {decoded:?}"
            );
            let status = replies
                .lock()
                .pop()
                .map(|(id, status)| (id == obj_id, status));
            let Some(ObjHandle::ComputePipeline(pipeline)) = dec.object(obj_id) else {
                assert_eq!(status, Some((true, RGFX_PIPELINE_REFUSED)), "{list:?}");
                made.push(None);
                continue;
            };
            assert_eq!(status, Some((true, RGFX_PIPELINE_OK)), "{list:?}");
            let buffer = |len| {
                device
                    .new_buffer_of_length(OnFault::Return, len, 0)
                    .unwrap()
                    .unwrap()
            };
            let (output, inputs) = (buffer(4), buffer(12));
            // SAFETY: the buffer is Shared and holds 12 bytes; the GPU has not started.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    STITCH_INPUTS.as_ptr(),
                    inputs.contents().as_ptr::<f32>(),
                    3,
                )
            };
            let queue = device.new_command_queue().unwrap();
            let cb = queue.command_buffer(OnFault::Return).unwrap().unwrap();
            let enc = cb
                .compute_encoder_for_pass(&ComputePassDesc::default())
                .unwrap()
                .unwrap();
            let one = MtlSize::new(1, 1, 1);
            for cmd in [
                ComputeCmd::SetComputePipelineState {
                    state: Some(pipeline),
                },
                ComputeCmd::SetBuffers {
                    buffers: &[Some(&output), Some(&inputs)],
                    offsets: &[0, 0],
                    range: NsRange {
                        location: 0,
                        length: 2,
                    },
                },
                ComputeCmd::DispatchThreadgroups {
                    threadgroups_per_grid: one,
                    threads_per_threadgroup: one,
                },
                ComputeCmd::EndEncoding,
            ] {
                enc.encode(OnFault::Return, cmd).unwrap();
            }
            cb.commit(OnFault::Return).unwrap();
            cb.wait_until_completed(OnFault::Return).unwrap();
            // SAFETY: the GPU wrote the buffer's 4 bytes, and is done.
            made.push(Some(unsafe { output.contents().as_ptr::<f32>().read() }));
        }
        made
    }

    /// The Vulkan crate's metallibs of the shell tests' stitchable functions and of the library
    /// Metal's stitcher makes of them (its build.rs, `stitched`), which a host without Metal takes.
    #[cfg(any(target_os = "macos", feature = "vulkan"))]
    const COMMITTED_STITCHED: [&[u8]; 2] = [
        include_bytes!("../../../../reims-vgpu-vulkan/metallibs/stitchable.metallib"),
        include_bytes!("../../../../reims-vgpu-vulkan/metallibs/stitched.metallib"),
    ];

    /// On Metal, a guest's stitched library is a library like any other: the library Metal's
    /// stitcher made in the guest, sent as its bytes, gives a function `stitch_caller`'s pipeline
    /// links from either list, as Metal's own does, and the pipeline writes what Metal's writes;
    /// with nothing linked, it is refused, as Metal refuses it. The Vulkan crate's committed
    /// stitched metallibs get the same answers
    #[cfg(target_os = "macos")]
    #[test]
    fn on_metal_a_stitched_function_links_and_computes_as_metal_s() {
        use reims_vgpu_metal::MetalPlatform;
        let s = stitched_on_metal();
        let metal = metal_s_own_stitched_links(&s);
        assert_eq!(metal, [Some(20.0), Some(20.0), None]);
        assert_eq!(
            stitched_links_through_the_decoder::<MetalPlatform>(&s.caller_bytes, &s.stitched_bytes),
            metal
        );
        let [caller, stitched] = COMMITTED_STITCHED;
        assert_eq!(
            stitched_links_through_the_decoder::<MetalPlatform>(caller, stitched),
            metal
        );
    }

    /// On the Vulkan platform, a guest's stitched library gets Metal's answers
    /// (`on_metal_a_stitched_function_links_and_computes_as_metal_s`): the runtime links the
    /// function Metal's stitcher made like any other, from either list, and the pipeline writes
    /// 20; with nothing linked, it is refused. The one linked signature the runtime still refuses
    /// (Unsupported, Feature::Linking), a linked-constants signature on a module that still
    /// declares function constants, is not this library's: it declares none
    #[cfg(feature = "vulkan")]
    #[test]
    fn on_vulkan_a_stitched_function_links_and_computes_as_metal_s() {
        let [caller, stitched] = COMMITTED_STITCHED;
        assert_eq!(
            stitched_links_through_the_decoder::<reims_vgpu_vulkan::VulkanPlatform>(
                caller, stitched
            ),
            [Some(20.0), Some(20.0), None]
        );
    }

    /// The Vulkan crate's metallib of the readers of a texture (its `ycbcr_probe.metal`), which a
    /// host without Metal takes as a Mac does.
    #[cfg(any(target_os = "macos", feature = "vulkan"))]
    const PROBE_LIBRARY: &[u8] =
        include_bytes!("../../../../reims-vgpu-vulkan/metallibs/ycbcr_probe.metallib");

    /// What `kernel` of [`PROBE_LIBRARY`] reads of `texture` on `P`'s `device`, into floats: a
    /// texel for each of a grid of `grid`, with the sampler `sampler` and the probe bytes `probe`
    /// where given.
    #[cfg(any(target_os = "macos", feature = "vulkan"))]
    fn kernel_reads<P: Platform>(
        device: &P::Device,
        kernel: &str,
        texture: &P::Texture,
        grid: (u64, u64),
        (sampler, probe): (Option<&P::SamplerState>, Option<&[u8]>),
    ) -> Vec<[f32; 4]> {
        use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
        use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder as _};
        use reims_vgpu_core::platform::types::{MtlSize, NsRange};
        let _pool = reims_vgpu_core::rt::Pool::here();
        let library = device.new_library(PROBE_LIBRARY).unwrap().unwrap().unwrap();
        let f = library.new_function(OnFault::Die, kernel).unwrap().unwrap();
        let state = device
            .new_compute_pipeline_state_for_function(Some(&f))
            .unwrap()
            .unwrap();
        let shared = |length: u64| {
            device
                .new_buffer_of_length(OnFault::Die, length, 0)
                .unwrap()
                .unwrap()
        };
        let length = grid.0 * grid.1 * 16;
        let out = shared(length);
        let probe = probe.map(|bytes| {
            let b = shared(bytes.len() as u64);
            // SAFETY: a Shared buffer of `bytes.len()` bytes, which no command uses yet.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    b.contents().as_ptr::<u8>(),
                    bytes.len(),
                )
            };
            b
        });
        let queue = device.new_command_queue().unwrap();
        let cb = queue.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
        let buffers = [Some(&out), probe.as_ref()];
        let bound = if probe.is_some() { 2 } else { 1 };
        let samplers = [sampler];
        let mut cmds = vec![
            ComputeCmd::SetComputePipelineState {
                state: Some(&state),
            },
            ComputeCmd::SetTexture {
                texture: Some(texture),
                index: 0,
            },
            ComputeCmd::SetBuffers {
                buffers: &buffers[..bound],
                offsets: &[0, 0][..bound],
                range: NsRange::new(0, bound as u64),
            },
        ];
        if sampler.is_some() {
            cmds.push(ComputeCmd::SetSamplerStates {
                samplers: &samplers,
                range: NsRange::new(0, 1),
            });
        }
        cmds.push(ComputeCmd::DispatchThreads {
            threads_per_grid: MtlSize::new(grid.0, grid.1, 1),
            threads_per_threadgroup: MtlSize::new(8, 8, 1),
        });
        cmds.push(ComputeCmd::EndEncoding);
        for cmd in cmds {
            e.encode(OnFault::Die, cmd).unwrap();
        }
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert!(cb.error().is_none(), "{:?}", cb.error());
        // SAFETY: a Shared buffer's bytes, read after the command buffer completed.
        let bytes =
            unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), length as usize) };
        bytes
            .chunks(16)
            .map(|t| {
                std::array::from_fn(|i| f32::from_le_bytes(t[i * 4..][..4].try_into().unwrap()))
            })
            .collect()
    }

    /// A decoder on `P`'s process platform, the platform's default device, the guest's texture
    /// ids kept on it as a graphics device keeps them, and context 1's first chunk of resource ids
    /// granted. Each test runs in a process of its own, so this is the first use of the device's
    /// ids.
    #[cfg(any(target_os = "macos", feature = "vulkan"))]
    fn surface_decoder<P: crate::platform::ApiPlatform>() -> (GraphicsDecoder<P>, P::Device) {
        let platform = P::process();
        let device = platform.metal_device(None).unwrap();
        crate::graphics::reserve_guest_texture_ids(platform.as_ref(), &device);
        let mut dec = GraphicsDecoder::<P>::new(
            Arc::clone(&platform),
            Arc::new(Region::new(0x10000).unwrap()),
        );
        let grant = RgfxResourceIdChunk {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_RESOURCE_ID_CHUNK,
                size: size_of::<RgfxResourceIdChunk>() as u32,
                seqno: 0,
            },
            context: 1,
            chunk: 0,
        };
        assert_eq!(dec.decode(&device, grant.as_bytes()).unwrap(), Ok(()));
        (dec, device)
    }

    /// Context 1's object `n`.
    #[cfg(any(target_os = "macos", feature = "vulkan"))]
    const fn context_obj(n: u64) -> u64 {
        (1 << RGFX_CONTEXT_OBJ_ID_SHIFT) | n
    }

    /// A Shared buffer of the decoder's, object `n`, holding `bytes` from its first.
    #[cfg(any(target_os = "macos", feature = "vulkan"))]
    fn buffer_holding<P: Platform>(
        dec: &mut GraphicsDecoder<P>,
        device: &P::Device,
        n: u64,
        bytes: &[u8],
    ) {
        let create = RgfxBufferCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_BUFFER_CREATE,
                size: size_of::<RgfxBufferCreate>() as u32,
                seqno: 0,
            },
            obj_id: context_obj(n),
            length: (bytes.len() as u64).next_multiple_of(0x4000),
            ..RgfxBufferCreate::default()
        };
        assert_eq!(dec.decode(device, create.as_bytes()).unwrap(), Ok(()));
        let Some(ObjHandle::Buffer(b)) = dec.object(context_obj(n)) else {
            panic!("no buffer {n}");
        };
        // SAFETY: a Shared buffer of at least `bytes.len()` bytes, which no command uses yet.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), b.contents().as_ptr::<u8>(), bytes.len())
        };
    }

    /// A 2D texture descriptor of the wire: `format`, `size`, one level, usage `ShaderRead`,
    /// options 0 (Shared), the identity swizzle, as WindowServer asks for its surfaces' textures.
    #[cfg(any(target_os = "macos", feature = "vulkan"))]
    fn surface_desc(format: u32, (width, height): (u64, u64)) -> RgfxTextureDescriptor {
        RgfxTextureDescriptor {
            pixel_format: format,
            texture_type: 2,
            width: width as u32,
            height: height as u32,
            depth: 1,
            mipmap_level_count: 1,
            array_length: 1,
            sample_count: 1,
            usage: 1,
            ..RgfxTextureDescriptor::default()
        }
    }

    /// The `TEXTURE_FROM_IOSURFACE` of texture `n` (resource id `n`) over plane 0 of buffer 10,
    /// at offset 0 with rows `bytes_per_row` apart, of `desc`, carrying `facts`.
    #[cfg(any(target_os = "macos", feature = "vulkan"))]
    fn surface_texture_message(
        n: u64,
        desc: RgfxTextureDescriptor,
        bytes_per_row: u64,
        facts: &reims_graphics::wire::RgfxSurfaceFacts,
    ) -> Vec<u8> {
        use reims_graphics::wire::{RGFX_SURFACE_FACTS_OFFSET, RgfxSurfaceFacts};
        let m = RgfxTextureFromIOSurface {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_TEXTURE_FROM_IOSURFACE,
                size: (RGFX_SURFACE_FACTS_OFFSET as usize + size_of::<RgfxSurfaceFacts>()) as u32,
                seqno: 0,
            },
            obj_id: context_obj(n),
            resource_id: n,
            buffer_obj_id: context_obj(10),
            offset: 0,
            bytes_per_row,
            iosurface_id: 0x40 + n as u32,
            plane: 0,
            desc,
            writers_offset: RGFX_SURFACE_NO_WRITERS,
        };
        let mut bytes = m.as_bytes().to_vec();
        bytes.extend_from_slice(facts.as_bytes());
        bytes
    }

    /// What the guest's texture `n` of the decoder answers: its format, type, width, height,
    /// usage, resource options and storage mode, and the bits of every texel `read()` reads of it
    /// into floats.
    #[cfg(any(target_os = "macos", feature = "vulkan"))]
    type TextureAnswer = ([u64; 7], Vec<[u32; 4]>);

    /// [`TextureAnswer`] of `texture`, of `size`, on `device`.
    #[cfg(any(target_os = "macos", feature = "vulkan"))]
    fn texture_answer<P: Platform>(
        device: &P::Device,
        texture: &P::Texture,
        size: (u64, u64),
    ) -> TextureAnswer {
        use reims_vgpu_core::platform::resource::{GpuResource as _, Texture as _};
        let t = texture;
        let getters = [
            t.pixel_format(),
            t.texture_type(),
            t.width(),
            t.height(),
            t.usage(),
            t.resource_options(),
            t.storage_mode(),
        ];
        let read = kernel_reads::<P>(device, "read_texels_float", t, size, (None, None));
        (getters, read.iter().map(|t| t.map(f32::to_bits)).collect())
    }

    /// The pin's textures: 16 x 8, of BGRA8 (80) over a 'BGRA' surface and of RGBA16F (115) over
    /// an 'RGhA' one, with the surfaces' OSTypes and bytes a texel.
    #[cfg(any(target_os = "macos", feature = "vulkan"))]
    const PIN_CASES: [(u32, &[u8; 4], u64); 2] = [(80, b"BGRA", 4), (115, b"RGhA", 8)];

    /// The pin's size, and its rows: the bundle's copy of a surface without planes lays its rows
    /// 256 bytes apart (`alignUp(IOSurfaceGetBytesPerRow, 256)`).
    #[cfg(any(target_os = "macos", feature = "vulkan"))]
    const PIN_SIZE: (u64, u64) = (16, 8);
    #[cfg(any(target_os = "macos", feature = "vulkan"))]
    const PIN_ROW: u64 = 256;

    /// The pin's bytes: each a value of its place, every row whole, so a half's bits are no NaN
    /// (RGBA16F's exponents stay below 31).
    #[cfg(any(target_os = "macos", feature = "vulkan"))]
    fn pin_bytes() -> Vec<u8> {
        (0..PIN_ROW * PIN_SIZE.1)
            .map(|i| ((i * 37 + i / 7) % 0x7b) as u8)
            .collect()
    }

    /// What a texture of each of [`PIN_CASES`] over a surface without planes answers through
    /// the decoder on `P`, both ways over the guest's copy of the surface, at offset 0 with rows
    /// [`PIN_ROW`] apart, options 0 (Shared): by today's buffer creator (`TEXTURE_FROM_BUFFER`),
    /// and by the IOSurface creator (`TEXTURE_FROM_IOSURFACE`, its facts saying no planes).
    #[cfg(any(target_os = "macos", feature = "vulkan"))]
    fn surface_textures_without_planes<P: crate::platform::ApiPlatform>() -> Vec<[TextureAnswer; 2]>
    {
        use reims_graphics::wire::RgfxSurfaceFacts;
        let (mut dec, device) = surface_decoder::<P>();
        buffer_holding(&mut dec, &device, 10, &pin_bytes());
        let answer = |dec: &GraphicsDecoder<P>, n: u64| {
            let Some(ObjHandle::Texture(t)) = dec.object(context_obj(n)) else {
                panic!("no texture {n}");
            };
            texture_answer::<P>(&device, t, PIN_SIZE)
        };
        let mut answers = Vec::new();
        for (k, &(format, ostype, texel)) in PIN_CASES.iter().enumerate() {
            let (n, desc) = (11 + 2 * k as u64, surface_desc(format, PIN_SIZE));
            let from_buffer = RgfxTextureFromBuffer {
                header: RgfxMsgHeader {
                    r#type: RGFX_CMD_TEXTURE_FROM_BUFFER,
                    size: size_of::<RgfxTextureFromBuffer>() as u32,
                    seqno: 0,
                },
                obj_id: context_obj(n),
                resource_id: n,
                buffer_obj_id: context_obj(10),
                offset: 0,
                bytes_per_row: PIN_ROW,
                desc,
            };
            assert_eq!(dec.decode(&device, from_buffer.as_bytes()).unwrap(), Ok(()));
            let facts = RgfxSurfaceFacts {
                surface_pixel_format: u32::from_be_bytes(*ostype),
                plane_count: 0,
                width: PIN_SIZE.0 as u32,
                height: PIN_SIZE.1 as u32,
                ..RgfxSurfaceFacts::default()
            };
            assert!(PIN_SIZE.0 * texel <= PIN_ROW);
            let m = surface_texture_message(n + 1, desc, PIN_ROW, &facts);
            assert_eq!(dec.decode(&device, &m).unwrap(), Ok(()), "{format}");
            answers.push([answer(&dec, n), answer(&dec, n + 1)]);
        }
        answers
    }

    /// The 505 texture's size (WindowServer's is 3840 x 2160; the layout is the same at any
    /// size), and the bundle's copy of its 'x420' surface: plane 0 at 0, plane 1 at
    /// [`YCBCR_CHROMA_AT`], each row `alignUp(IOSurfaceGetBytesPerRowOfPlane(k), 256)` apart, and
    /// each plane at a multiple of 256.
    #[cfg(any(target_os = "macos", feature = "vulkan"))]
    const YCBCR_SIZE: (u64, u64) = (16, 8);
    #[cfg(any(target_os = "macos", feature = "vulkan"))]
    const YCBCR_ROW: u64 = 256;
    #[cfg(any(target_os = "macos", feature = "vulkan"))]
    const YCBCR_CHROMA_AT: u64 = YCBCR_ROW * YCBCR_SIZE.1;

    /// The bundle's copy of the 505 texture's surface: 10-bit luma codes in the top bits of 16-bit
    /// samples, then interleaved Cb and Cr codes, one pair per 2 x 2 block, of a fixed noise.
    #[cfg(any(target_os = "macos", feature = "vulkan"))]
    fn ycbcr_bytes() -> Vec<u8> {
        let mut seed = 0x2545_f491_4f6c_dd1d_u64;
        let mut code = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 20) % 1024) as u16
        };
        let (width, height) = YCBCR_SIZE;
        let mut bytes = vec![0u8; (YCBCR_CHROMA_AT + YCBCR_ROW * height / 2) as usize];
        let mut put = |at: u64, code: u16| {
            let at = at as usize;
            bytes[at..at + 2].copy_from_slice(&(code << 6).to_le_bytes());
        };
        for y in 0..height {
            for x in 0..width {
                put(y * YCBCR_ROW + x * 2, code());
            }
        }
        for y in 0..height / 2 {
            for x in 0..width / 2 {
                let at = YCBCR_CHROMA_AT + y * YCBCR_ROW + x * 4;
                put(at, code());
                put(at + 2, code());
            }
        }
        bytes
    }

    /// The probe WindowServer's composite is like: a linear sample in normalized coordinates of an
    /// 8 x 5 grid of positions from (0.375, 0.375) texels, 1.875 apart, which meet each weight
    /// between two texels of a multiple of 1/8 and the edges.
    #[cfg(any(target_os = "macos", feature = "vulkan"))]
    const YCBCR_GRID: (u64, u64) = (8, 5);

    /// [`YCBCR_GRID`]'s bytes (`Probe` in `ycbcr_probe.metal`).
    #[cfg(any(target_os = "macos", feature = "vulkan"))]
    fn ycbcr_probe() -> Vec<u8> {
        let floats = [0.375f32, 0.375, 1.875, 1.875].map(f32::to_le_bytes);
        let words = [YCBCR_GRID.0 as u32, YCBCR_GRID.1 as u32, 1, 0].map(u32::to_le_bytes);
        [floats.concat(), words.concat()].concat()
    }

    /// The linear, clamp-to-edge sampler of normalized coordinates the probe samples through.
    #[cfg(any(target_os = "macos", feature = "vulkan"))]
    fn linear_sampler<P: Platform>(device: &P::Device) -> P::SamplerState {
        let desc = SamplerDesc {
            min_filter: Some(1),
            mag_filter: Some(1),
            s_address_mode: Some(0),
            t_address_mode: Some(0),
            normalized_coordinates: Some(true),
            ..SamplerDesc::default()
        };
        device.new_sampler_state(&desc).unwrap().unwrap()
    }

    /// What a texture of 505 over the bundle's copy of an 'x420' surface of BT.2020
    /// ([`ycbcr_bytes`]) samples through the probe ([`ycbcr_probe`]) through the decoder on `P`,
    /// with the surface's facts in its `TEXTURE_FROM_IOSURFACE`; `None` where the create is
    /// refused.
    #[cfg(any(target_os = "macos", feature = "vulkan"))]
    fn ycbcr_through_the_decoder<P: crate::platform::ApiPlatform>() -> Option<Vec<[f32; 4]>> {
        use reims_graphics::wire::{RGFX_YCBCR_MATRIX_ITU_R_2020, RgfxSurfaceFacts};
        let (mut dec, device) = surface_decoder::<P>();
        buffer_holding(&mut dec, &device, 10, &ycbcr_bytes());
        let facts = RgfxSurfaceFacts {
            surface_pixel_format: u32::from_be_bytes(*b"x420"),
            plane_count: 2,
            width: YCBCR_SIZE.0 as u32,
            height: YCBCR_SIZE.1 as u32,
            chroma_offset: YCBCR_CHROMA_AT,
            chroma_bytes_per_row: YCBCR_ROW,
            ycbcr_matrix: RGFX_YCBCR_MATRIX_ITU_R_2020,
            reserved_24: 0,
        };
        let m = surface_texture_message(11, surface_desc(505, YCBCR_SIZE), YCBCR_ROW, &facts);
        match dec.decode(&device, &m).unwrap() {
            Ok(()) => {}
            Err(DecodeError::Refused) => return None,
            Err(e) => panic!("{e:?}"),
        }
        let Some(ObjHandle::Texture(t)) = dec.object(context_obj(11)) else {
            panic!("no texture");
        };
        let sampler = linear_sampler::<P>(&device);
        let probe = ycbcr_probe();
        let read = (Some(&sampler), Some(&probe[..]));
        Some(kernel_reads::<P>(
            &device,
            "sample_texels_float",
            t,
            YCBCR_GRID,
            read,
        ))
    }

    /// Metal's answer to [`ycbcr_through_the_decoder`]'s probe of the same bytes over a real
    /// 'x420' surface of BT.2020 (`on_metal_a_505_texture_over_an_x420_surface_samples_as_recorded`,
    /// an Apple M3 Max under macOS 27.0.1): each sample's red, green and blue, as their bits,
    /// in the grid's rows; every alpha is 1.
    #[cfg(any(target_os = "macos", feature = "vulkan"))]
    const METAL_505: [[u32; 3]; 40] = [
        [0x3f800000, 0x3f16c000, 0x00000000],
        [0x3e1b8000, 0x3f0d0000, 0x3e398000],
        [0x3e88c000, 0x3eec4000, 0x3db10000],
        [0x3f800000, 0x3f14e000, 0x3f27e000],
        [0x3f53e000, 0x3cb80000, 0x3ed20000],
        [0x3f480000, 0x3dbf0000, 0x3dfb0000],
        [0x3f65e000, 0x3ef44000, 0x3f0b8000],
        [0x3f6d6000, 0x3f6c8000, 0x3f130000],
        [0x3f3ec000, 0x00000000, 0x3e1b8000],
        [0x3f0e0000, 0x3e8e4000, 0x3ecdc000],
        [0x3f22a000, 0x3f65a000, 0x3f672000],
        [0x3e980000, 0x3e320000, 0x3f050000],
        [0x3f1e4000, 0x3e9b0000, 0x3f0b0000],
        [0x3f6e6000, 0x3f368000, 0x3f2a2000],
        [0x3f16c000, 0x3df50000, 0x3e960000],
        [0x3f452000, 0x3f0c2000, 0x3dc50000],
        [0x3f788000, 0x3ec14000, 0x3f4b8000],
        [0x3f7c0000, 0x3e91c000, 0x3f07e000],
        [0x3ef34000, 0x3ebe0000, 0x3f154000],
        [0x3f0a8000, 0x3ec24000, 0x3f1fc000],
        [0x3f104000, 0x3f04e000, 0x3f3e4000],
        [0x3dc30000, 0x3f274000, 0x3f36a000],
        [0x3e3e8000, 0x3e960000, 0x3f366000],
        [0x3f7fa000, 0x3ed68000, 0x3f2fe000],
        [0x3f6b4000, 0x3e438000, 0x3f31a000],
        [0x3f498000, 0x3d8b0000, 0x3f17e000],
        [0x3f460000, 0x3eb3c000, 0x3f5da000],
        [0x3edb0000, 0x3e938000, 0x3eb60000],
        [0x3eb48000, 0x3f2e4000, 0x3f208000],
        [0x3ccc0000, 0x3f08a000, 0x3f58c000],
        [0x3eaf0000, 0x3f29c000, 0x3f474000],
        [0x3f000000, 0x3f2f4000, 0x3f236000],
        [0x3f0d8000, 0x00000000, 0x3f466000],
        [0x3f444000, 0x3dea0000, 0x3f800000],
        [0x3f518000, 0x3ea74000, 0x3f800000],
        [0x3e1f0000, 0x3e6d0000, 0x3eb38000],
        [0x00000000, 0x3eb90000, 0x3e7f0000],
        [0x3d3c0000, 0x3e9c0000, 0x3f026000],
        [0x3e924000, 0x3f036000, 0x3e790000],
        [0x3e728000, 0x3f800000, 0x3f194000],
    ];

    /// On Metal, a texture over a surface's plane of BGRA8 or RGBA16F answers alike by the
    /// buffer's creator over the guest's copy of a surface without planes, and by Metal's own
    /// IOSurface creator over a real surface of the same bytes and rows, as the Metal platform's
    /// IOSurface route over the copy (the buffer's creator) does: so its answers stay the
    /// buffer's on the Vulkan platform too (`on_vulkan_surface_textures_without_planes_are_the_buffer_s`).
    #[cfg(target_os = "macos")]
    #[test]
    fn on_metal_surface_textures_without_planes_are_the_buffer_s() {
        use reims_vgpu_core::platform::os::surface::{Surface as _, SurfaceSpec};
        use reims_vgpu_metal::MetalPlatform;

        use crate::platform::ApiPlatform as _;
        let answers = surface_textures_without_planes::<MetalPlatform>();
        let platform = MetalPlatform::process();
        let device = platform.metal_device(None).unwrap();
        for (&(format, ostype, texel), [from_buffer, from_surface]) in
            PIN_CASES.iter().zip(&answers)
        {
            assert_eq!(from_buffer, from_surface, "{format}");
            let spec = SurfaceSpec {
                pixel_format: u32::from_be_bytes(*ostype),
                alloc_size: 0x4000,
                width: PIN_SIZE.0,
                height: PIN_SIZE.1,
                bytes_per_element: texel,
                bytes_per_row: PIN_ROW,
                element_width: 1,
                element_height: 1,
                ..SurfaceSpec::default()
            };
            let surface = platform.surfaces().create(&spec).unwrap();
            assert_eq!(surface.bytes_per_row(), PIN_ROW);
            let bytes = pin_bytes();
            // SAFETY: the surface's allocation holds its rows, which no command uses yet.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    surface.base_address().as_ptr::<u8>(),
                    bytes.len(),
                )
            };
            let desc = surface_desc(format, PIN_SIZE);
            let descriptor = platform.texture_descriptor(&TextureDesc {
                resource_index: None,
                ..texture_desc(&desc, 0)
            });
            let own = device
                .new_texture_on_surface(&descriptor, Some(&surface), 0)
                .unwrap()
                .unwrap();
            let metal = texture_answer::<MetalPlatform>(&device, &own, PIN_SIZE);
            assert_eq!(&metal, from_buffer, "{format}");
        }
    }

    /// On the Vulkan platform, a texture of BGRA8 or RGBA16F over the guest's copy of a surface
    /// without planes answers by the runtime's IOSurface creator what it answers by the buffer's
    /// creator, as Metal's own two creators answer alike
    /// (`on_metal_surface_textures_without_planes_are_the_buffer_s`)
    #[cfg(feature = "vulkan")]
    #[test]
    fn on_vulkan_surface_textures_without_planes_are_the_buffer_s() {
        let answers = surface_textures_without_planes::<reims_vgpu_vulkan::VulkanPlatform>();
        for (&(format, ..), [from_buffer, from_surface]) in PIN_CASES.iter().zip(&answers) {
            assert_eq!(from_buffer, from_surface, "{format}");
        }
    }

    /// What a `SAMPLER_CREATE` costs the host, through the decoder as the guest's waited create
    /// reaches it ([`GraphicsDecoder::sampler_state`]), with 1, 64 and 1000 of the context's
    /// samplers alive, and its `OBJECT_DESTROY` after it, so as many stay alive: each create a
    /// descriptor of its own, as the bundle's state cache sends one only for a descriptor it has
    /// not made. A measurement, run by hand (`--run-ignored only --no-capture`); it prints the
    /// mean, median and 99th percentile of each. `RVG_SAMPLER_BENCH_LIVE` (comma-separated) sets
    /// other live counts.
    #[cfg(any(target_os = "macos", feature = "vulkan"))]
    fn sampler_create_cost<P: crate::platform::ApiPlatform>() {
        use reims_vgpu_core::platform::os::Clock as _;
        let (mut dec, device) = surface_decoder::<P>();
        let platform = P::process();
        let now = || platform.clock().now_mono();
        let create = |obj_id, lod_max: f32| RgfxSamplerCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_SAMPLER_CREATE,
                size: size_of::<RgfxSamplerCreate>() as u32,
                seqno: 1,
            },
            obj_id,
            min_filter: 1,
            mag_filter: 1,
            max_anisotropy: 1,
            flags: 0b001,
            lod_max_clamp_bits: lod_max.to_bits(),
            ..RgfxSamplerCreate::default()
        };
        let destroy = |obj_id| RgfxObjectDestroy {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_OBJECT_DESTROY,
                size: size_of::<RgfxObjectDestroy>() as u32,
                seqno: 0,
            },
            obj_id,
        };
        let stats = |mut us: Vec<f64>| {
            us.sort_by(f64::total_cmp);
            let mean = us.iter().sum::<f64>() / us.len() as f64;
            (mean, us[us.len() / 2], us[us.len() * 99 / 100])
        };
        let mut lod = 1000.0f32;
        let mut next = 1u64;
        let mut alive = std::collections::VecDeque::new();
        const ROUNDS: usize = 2000;
        let counts: Vec<usize> = std::env::var("RVG_SAMPLER_BENCH_LIVE")
            .map(|v| v.split(',').map(|n| n.trim().parse().unwrap()).collect())
            .unwrap_or_else(|_| vec![1, 64, 1000]);
        for live in counts {
            while alive.len() < live {
                lod -= 0.25;
                let answer = dec.sampler_state(&device, create(context_obj(next), lod).as_bytes());
                assert_eq!(answer.unwrap().unwrap().status, RGFX_STATUS_OK);
                dec.end_message();
                alive.push_back(next);
                next += 1;
            }
            let (mut made, mut dropped) = (Vec::new(), Vec::new());
            for _ in 0..ROUNDS {
                lod -= 0.25;
                let message = create(context_obj(next), lod);
                let t0 = now();
                let answer = dec.sampler_state(&device, message.as_bytes());
                let t1 = now();
                assert_eq!(answer.unwrap().unwrap().status, RGFX_STATUS_OK);
                dec.end_message();
                alive.push_back(next);
                next += 1;
                let gone = destroy(context_obj(alive.pop_front().unwrap()));
                let t2 = now();
                assert_eq!(dec.decode(&device, gone.as_bytes()).unwrap(), Ok(()));
                let t3 = now();
                dec.end_message();
                made.push(t1.saturating_duration_since(t0).as_secs_f64() * 1e6);
                dropped.push(t3.saturating_duration_since(t2).as_secs_f64() * 1e6);
            }
            let (m, p50, p99) = stats(made);
            let (dm, dp50, dp99) = stats(dropped);
            println!(
                "sampler_create_cost: {live} alive: create {m:.2} us (p50 {p50:.2}, p99 {p99:.2}), \
                 destroy {dm:.2} us (p50 {dp50:.2}, p99 {dp99:.2}), {ROUNDS} rounds"
            );
        }
    }

    /// [`sampler_create_cost`] on the Vulkan platform.
    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "a measurement, run by hand"]
    fn on_vulkan_sampler_create_cost_by_live_samplers() {
        sampler_create_cost::<reims_vgpu_vulkan::VulkanPlatform>();
    }

    /// [`sampler_create_cost`] on Metal.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "a measurement, run by hand"]
    fn on_metal_sampler_create_cost_by_live_samplers() {
        sampler_create_cost::<reims_vgpu_metal::MetalPlatform>();
    }

    /// On Metal, a texture of 505 over plane 0 of a real 'x420' surface of BT.2020 holding the
    /// bundle's copy's bytes, at its planes' offsets and rows, samples through a linear sampler
    /// as recorded ([`METAL_505`]), every alpha 1. Through the decoder the Metal platform refuses
    /// the create: Metal makes no texture over a surface's plane from a buffer's bytes, and the
    /// host holds no surface, only the guest's copy of it.
    #[cfg(target_os = "macos")]
    #[test]
    fn on_metal_a_505_texture_over_an_x420_surface_samples_as_recorded() {
        use std::ffi::c_void;

        use reims_vgpu_core::platform::os::surface::{Surface as _, SurfacePlaneSpec, SurfaceSpec};
        use reims_vgpu_metal::MetalPlatform;

        use crate::platform::ApiPlatform as _;
        // The guest's route first, which keeps the guest's texture ids on the device before any
        // texture is made on it.
        assert_eq!(ycbcr_through_the_decoder::<MetalPlatform>(), None);
        #[link(name = "IOSurface", kind = "framework")]
        unsafe extern "C" {
            fn IOSurfaceSetValue(buffer: *const c_void, key: *const c_void, value: *const c_void);
        }
        #[link(name = "CoreFoundation", kind = "framework")]
        unsafe extern "C" {
            fn CFStringCreateWithBytes(
                alloc: *const c_void,
                bytes: *const u8,
                num_bytes: isize,
                encoding: u32,
                is_external_representation: u8,
            ) -> *const c_void;
            fn CFRelease(cf: *const c_void);
        }
        // A UTF-8 CFString of `s`, +1.
        let cf_string = |s: &str| {
            // SAFETY: `s`'s bytes, live for the call, which copies them.
            unsafe {
                CFStringCreateWithBytes(
                    std::ptr::null(),
                    s.as_ptr(),
                    s.len() as isize,
                    0x0800_0100,
                    0,
                )
            }
        };
        let platform = MetalPlatform::process();
        let device = platform.metal_device(None).unwrap();
        let (width, height) = YCBCR_SIZE;
        let plane = |offset, (width, height): (u64, u64), element| SurfacePlaneSpec {
            base: offset,
            offset,
            width,
            height,
            size: YCBCR_ROW * height,
            bytes_per_row: YCBCR_ROW,
            bytes_per_element: element,
            element_width: 1,
            element_height: 1,
            ..SurfacePlaneSpec::default()
        };
        let spec = SurfaceSpec {
            planes: Some(vec![
                plane(0, YCBCR_SIZE, 2),
                plane(YCBCR_CHROMA_AT, (width / 2, height / 2), 4),
            ]),
            pixel_format: u32::from_be_bytes(*b"x420"),
            alloc_size: 0x4000,
            width,
            height,
            bytes_per_element: 2,
            bytes_per_row: YCBCR_ROW,
            element_width: 1,
            element_height: 1,
            ..SurfaceSpec::default()
        };
        let surface = platform.surfaces().create(&spec).unwrap();
        assert_eq!(surface.plane_count(), 2);
        let ptr = std::ptr::with_exposed_provenance::<c_void>(surface.id().0 as usize);
        let (key, value) = (cf_string("IOSurfaceYCbCrMatrix"), cf_string("ITU_R_2020"));
        // SAFETY: the surface's IOSurfaceRef (its handle's id), live while `surface` is, and two
        // CFStrings, released after the call, which retains what it keeps.
        unsafe {
            IOSurfaceSetValue(ptr, key, value);
            CFRelease(key);
            CFRelease(value);
        }
        let bytes = ycbcr_bytes();
        // SAFETY: the surface's allocation holds both planes, which no command uses yet.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                surface.base_address().as_ptr::<u8>(),
                bytes.len(),
            )
        };
        let desc = surface_desc(505, YCBCR_SIZE);
        let descriptor = platform.texture_descriptor(&TextureDesc {
            resource_index: None,
            ..texture_desc(&desc, 0)
        });
        let own = device
            .new_texture_on_surface(&descriptor, Some(&surface), 0)
            .unwrap()
            .unwrap();
        let sampler = linear_sampler::<MetalPlatform>(&device);
        let probe = ycbcr_probe();
        let read = (Some(&sampler), Some(&probe[..]));
        let metal =
            kernel_reads::<MetalPlatform>(&device, "sample_texels_float", &own, YCBCR_GRID, read);
        let bits: Vec<[u32; 3]> = metal
            .iter()
            .map(|t| [t[0], t[1], t[2]].map(f32::to_bits))
            .collect();
        eprintln!("METAL_505 = {bits:#010x?}");
        assert!(metal.iter().all(|t| t[3] == 1.0), "{metal:?}");
        assert_eq!(bits, METAL_505);
    }

    /// On the Vulkan platform, the 505 texture WindowServer composites the wallpaper's video
    /// frames through, over the bundle's copy of its 'x420' surface of BT.2020 with the surface's
    /// facts, is made, and its linear samples lie within 1/2048 (one step of the format's grid)
    /// of Metal's over the real surface ([`METAL_505`]), every alpha 1. What the runtime announces
    /// of the sampling is printed: on a native device nothing of [`reims_vgpu_vulkan::answer::DIES`]
    /// is relayed but what a native device relays (`DeviceSampling` among them); on MoltenVK,
    /// Metal's own sampler's filter of the converted texels is `YCbCrFilterNotExact`'s, relayed.
    #[cfg(feature = "vulkan")]
    #[test]
    fn on_vulkan_a_505_texture_over_the_bundle_s_copy_samples_as_metal_s() {
        use reims_vgpu_vulkan::answer::{self, native};
        let before = (answer::relayed_counts(), native::counts());
        let ours = ycbcr_through_the_decoder::<reims_vgpu_vulkan::VulkanPlatform>()
            .expect("the 505 texture is made");
        // The kinds whose count grew from `before` to `after`.
        fn grew<K: PartialEq + Copy>(after: Vec<(K, u64)>, before: &[(K, u64)]) -> Vec<K> {
            (after.into_iter())
                .filter(|(k, n)| before.iter().all(|(b, m)| b != k || m < n))
                .map(|(k, _)| k)
                .collect()
        }
        let relayed = grew(answer::relayed_counts(), &before.0);
        let natively = grew(native::counts(), &before.1);
        let differences: Vec<f32> = (ours.iter().zip(&METAL_505))
            .flat_map(|(o, m)| (0..3).map(move |c| (o[c] - f32::from_bits(m[c])).abs()))
            .collect();
        let worst = differences.iter().copied().fold(0.0_f32, f32::max);
        let exact = differences.iter().filter(|&&d| d == 0.0).count();
        eprintln!(
            "placement {:?}: relayed of DIES {relayed:?}, relayed natively {natively:?}; {exact} \
             of {} channels Metal's, the farthest {} steps of 1/2048 off",
            native::placement(),
            differences.len(),
            worst * 2048.0
        );
        assert!(ours.iter().all(|t| t[3] == 1.0), "{ours:?}");
        assert!(worst <= 1.0 / 2048.0, "{} steps of 1/2048", worst * 2048.0);
        match native::placement() {
            native::Placement::Native => assert!(relayed.is_empty(), "{relayed:?}"),
            native::Placement::OnMetal => assert!(
                (relayed.iter()).all(|k| format!("{k:?}") == "YCbCrFilterNotExact"),
                "{relayed:?}"
            ),
        }
    }

    /// A render and a compute pipeline create each resolve their functions from a
    /// library, call the right platform creator, store the handle and emit a
    /// reflection reply keyed to the create's seqno; a pipeline that names a
    /// non-library is a bad reference and makes nothing.
    #[test]
    fn render_and_compute_pipeline_creates() {
        use reims_vgpu_core::sync::Mutex;

        let p = Arc::new(FakePlatform::new());
        let device = p.device();
        let mut dec = GraphicsDecoder::<FakePlatform>::new(
            Arc::clone(&p),
            Arc::new(reims_graphics::region::Region::new(0x10000).unwrap()),
        );

        let replies = Arc::new(Mutex::new(Vec::<RgfxPipelineReflection>::new()));
        let sink_replies = Arc::clone(&replies);
        dec.set_reflection_sink(Arc::new(move |r: PipelineReply| {
            sink_replies.lock().push(r.reflection);
        }));

        // A library the pipelines draw their functions from.
        let air = [1u8, 2, 3, 4];
        let mut lib_msg = RgfxLibraryCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_LIBRARY_CREATE,
                size: (RGFX_LIBRARY_BYTES_OFFSET as usize + air.len()).next_multiple_of(8) as u32,
                seqno: 0,
            },
            obj_id: 1,
            byte_length: air.len() as u64,
            ..RgfxLibraryCreate::default()
        }
        .as_bytes()
        .to_vec();
        lib_msg.extend_from_slice(&air);
        assert_eq!(dec.decode(&device, &lib_msg).unwrap(), Ok(()));

        // A render pipeline: vertex and fragment names in the tail, colour
        // attachment 0 BGRA8Unorm with source-over blend.
        let render_msg = |obj_id: u64, seqno: u64, vlib: u64, flib: u64| {
            let head = size_of::<RgfxRenderPipelineCreate>() as u32;
            let (vn, fnm) = (b"vmain".as_slice(), b"fmain".as_slice());
            let mut colors = [RgfxColorAttachmentDesc::default(); 8];
            colors[0] = RgfxColorAttachmentDesc {
                pixel_format: 80,
                blending_enabled: 1,
                source_rgb_blend_factor: 1,
                destination_rgb_blend_factor: 5,
                rgb_blend_operation: 0,
                source_alpha_blend_factor: 1,
                destination_alpha_blend_factor: 5,
                alpha_blend_operation: 0,
                write_mask: 0xf,
                reserved_24: 0,
            };
            let total = (head as usize + vn.len() + fnm.len()).next_multiple_of(8);
            let m = RgfxRenderPipelineCreate {
                header: RgfxMsgHeader {
                    r#type: RGFX_CMD_RENDER_PIPELINE_CREATE,
                    size: total as u32,
                    seqno,
                },
                obj_id,
                vertex_library_obj_id: vlib,
                fragment_library_obj_id: flib,
                vertex_name_offset: head,
                vertex_name_len: vn.len() as u32,
                fragment_name_offset: head + vn.len() as u32,
                fragment_name_len: fnm.len() as u32,
                raster_sample_count: 1,
                color_attachment_count: 1,
                colors,
                ..RgfxRenderPipelineCreate::default()
            };
            let mut bytes = m.as_bytes().to_vec();
            bytes.extend_from_slice(vn);
            bytes.extend_from_slice(fnm);
            bytes.resize(total, 0);
            bytes
        };

        let m = p.call_log().mark();
        assert_eq!(
            dec.decode(&device, &render_msg(100, 7, 1, 1)).unwrap(),
            Ok(())
        );
        assert_eq!(
            p.call_log()
                .sels_since(m)
                .iter()
                .filter(|s| **s == "newRenderPipelineStateWithDescriptor:error:")
                .count(),
            1
        );
        {
            let r = replies.lock();
            assert_eq!(r.len(), 1);
            assert_eq!(r[0].header.seqno, 7);
            assert_eq!(r[0].obj_id, 100);
            assert_eq!(r[0].status, RGFX_PIPELINE_OK);
            assert_eq!(r[0].kind, RGFX_PIPELINE_KIND_RENDER);
            assert_eq!(r[0].max_total_threads_per_threadgroup, 1024);
        }

        // A compute pipeline from a function in the same library.
        let compute_msg = |obj_id: u64, seqno: u64, lib: u64| {
            let head = size_of::<RgfxComputePipelineCreate>() as u32;
            let name = b"cmain".as_slice();
            let total = (head as usize + name.len()).next_multiple_of(8);
            let m = RgfxComputePipelineCreate {
                header: RgfxMsgHeader {
                    r#type: RGFX_CMD_COMPUTE_PIPELINE_CREATE,
                    size: total as u32,
                    seqno,
                },
                obj_id,
                function_library_obj_id: lib,
                name_offset: head,
                name_len: name.len() as u32,
                ..RgfxComputePipelineCreate::default()
            };
            let mut bytes = m.as_bytes().to_vec();
            bytes.extend_from_slice(name);
            bytes.resize(total, 0);
            bytes
        };

        let m = p.call_log().mark();
        assert_eq!(
            dec.decode(&device, &compute_msg(101, 8, 1)).unwrap(),
            Ok(())
        );
        assert_eq!(
            p.call_log()
                .sels_since(m)
                .iter()
                .filter(|s| **s == "newComputePipelineStateWithDescriptor:options:reflection:error:")
                .count(),
            1
        );
        {
            let r = replies.lock();
            assert_eq!(r.len(), 2);
            assert_eq!(r[1].obj_id, 101);
            assert_eq!(r[1].kind, RGFX_PIPELINE_KIND_COMPUTE);
            assert_eq!(r[1].thread_execution_width, 32);
        }

        // Both pipelines are held alongside the library.
        assert_eq!(dec.object_count(), 3);

        // A render create that names a non-library is a bad reference: no handle, and a
        // refusal for its reply, which its process waits for.
        assert_eq!(
            dec.decode(&device, &render_msg(102, 9, 999, 999)).unwrap(),
            Err(DecodeError::BadReference)
        );
        assert_eq!(dec.object_count(), 3);
        let r = replies.lock();
        assert_eq!(r.len(), 3);
        assert_eq!((r[2].obj_id, r[2].header.seqno), (102, 9));
        assert_eq!(
            (r[2].status, r[2].kind),
            (RGFX_PIPELINE_REFUSED, RGFX_PIPELINE_KIND_RENDER)
        );
    }
}
