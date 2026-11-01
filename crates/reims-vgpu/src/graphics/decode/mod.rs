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
//! the exec the same way ([`bad_reference`], [`op_fault`]).
//!
//! [`super::GraphicsControl`] drives it from the device's control thread, one
//! message at a time, and gives it the sinks that post onto the control
//! completion ring. The object-resolution API below ([`GraphicsDecoder::require_buffer`]
//! and the rest) is the whole contract the encoder-op decoders call; some of its
//! resolvers are for ops the first op slice does not emit yet, and so are not
//! reached, but they are kept whole (project rule "Unread is not dead").

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use reims_graphics::region::Region;
use reims_graphics::wire::{
    RGFX_CMD_BUFFER_AT_OFFSET, RGFX_CMD_BUFFER_CREATE, RGFX_CMD_BUFFER_DID_MODIFY_RANGE,
    RGFX_CMD_BUFFER_OVER_HOST_MEMORY, RGFX_CMD_COMMAND_BUFFER, RGFX_CMD_COMPUTE_PIPELINE_CREATE,
    RGFX_CMD_DEPTH_STENCIL_CREATE, RGFX_CMD_EVENT_CREATE, RGFX_CMD_FENCE_CREATE,
    RGFX_CMD_HEAP_CREATE, RGFX_CMD_HEAP_OVER_HOST_MEMORY, RGFX_CMD_LIBRARY_CREATE,
    RGFX_CMD_LIBRARY_FROM_REGION, RGFX_CMD_LINEAR_TEXTURE_FROM_BUFFER,
    RGFX_CMD_MANAGED_TEXTURE_CREATE, RGFX_CMD_OBJECT_DESTROY, RGFX_CMD_QUEUE_CREATE,
    RGFX_CMD_RENDER_PIPELINE_CREATE, RGFX_CMD_RESOURCE_ID_CHUNK, RGFX_CMD_SAMPLER_CREATE,
    RGFX_CMD_SHARED_EVENT_CREATE, RGFX_CMD_SHARED_EVENT_SET_VALUE, RGFX_CMD_TEXTURE_CREATE,
    RGFX_CMD_TEXTURE_FROM_BUFFER, RGFX_CMD_TEXTURE_FROM_IOSURFACE, RGFX_CMD_TEXTURE_REPLACE_REGION,
    RGFX_CMD_TEXTURE_VIEW, RGFX_COMPUTE_PIPELINE_SUPPORT_ICB,
    RGFX_COMPUTE_PIPELINE_TG_MULTIPLE_OF_EXEC_WIDTH, RGFX_CONTEXT_ALLOCATIONS_OFFSET,
    RGFX_CONTEXT_OBJ_ID_SHIFT, RGFX_CURSOR_SPAN, RGFX_EVT_PIPELINE_REFLECTION,
    RGFX_EVT_SAMPLER_STATE, RGFX_EVT_SIZE_AND_ALIGN, RGFX_LINKED_BINARY_FUNCTIONS,
    RGFX_LINKED_FUNCTIONS, RGFX_LINKED_GROUP, RGFX_LINKED_PRIVATE_FUNCTIONS,
    RGFX_PIPELINE_KIND_COMPUTE, RGFX_PIPELINE_KIND_RENDER, RGFX_PIPELINE_OK,
    RGFX_PIPELINE_REFLECT_SUPPORT_ICB, RGFX_PIPELINE_REFLECT_THREADGROUP_MATCHES_TILE,
    RGFX_PIPELINE_REFUSED, RGFX_RENDER_PIPELINE_ALPHA_TO_COVERAGE,
    RGFX_RENDER_PIPELINE_ALPHA_TO_ONE, RGFX_RENDER_PIPELINE_RASTERIZATION_ENABLED,
    RGFX_RENDER_PIPELINE_SUPPORT_ICB, RGFX_RESOURCE_ID_CHUNK, RGFX_RESOURCE_ID_CHUNKS,
    RGFX_RESOURCE_ID_LIMIT, RGFX_STATUS_INVALID_ARGUMENT, RGFX_STATUS_OK, RGFX_STATUS_REFUSED,
    RGFX_SURFACE_NO_WRITERS, RgfxBufferAtOffset, RgfxBufferCreate, RgfxBufferOverHostMemory,
    RgfxCbComplete, RgfxColorAttachmentDesc, RgfxComputePipelineCreate, RgfxDepthStencilCreate,
    RgfxEventSignalled, RgfxFunctionConstant, RgfxHeapCreate, RgfxHeapOverHostMemory,
    RgfxHeapTextureSizeAndAlign, RgfxLinearTextureFromBuffer, RgfxLinkedFunction, RgfxMsgHeader,
    RgfxObjectCreate, RgfxObjectDestroy, RgfxPipelineReflection, RgfxQueueCreate,
    RgfxRenderPipelineCreate, RgfxResourceIdChunk, RgfxSamplerCreate, RgfxSamplerState,
    RgfxSharedEventCreate, RgfxSharedEventSetValue, RgfxSizeAndAlign, RgfxStencilDescriptor,
    RgfxSurfaceWriter, RgfxSurfaceWriters, RgfxTextureCreate, RgfxTextureDescriptor,
    RgfxTextureFromBuffer, RgfxTextureFromIOSurface, RgfxVertexAttribute, RgfxVertexBufferLayout,
};
use reims_vgpu_core::host::HostPtr;
use reims_vgpu_core::platform::device::GpuDevice;
use reims_vgpu_core::platform::event::SharedEvent;
use reims_vgpu_core::platform::resource::{
    Buffer, ComputePipeline, FunctionConstant, Heap, Library, RenderPipeline, SamplerState,
};
use reims_vgpu_core::platform::types::Swizzle;
use reims_vgpu_core::platform::values::{
    AttributeDesc, BufferDesc, BufferLayoutDesc, ColorAttachmentDesc, DepthStencilDesc,
    FullComputePipelineDesc, FullRenderPipelineDesc, HeapDesc, LinkedFunctionsDesc, OptionSet,
    SamplerDesc, StageInputDesc, StencilDesc, TextureDesc, VertexDesc,
};
use reims_vgpu_core::platform::{Handle, HostRange, Platform};
use reims_vgpu_core::raise::{OnFault, Raise, RaiseResult, refusals};
use reims_vgpu_core::sync::Mutex;
use zerocopy::{FromBytes, Immutable, KnownLayout};

/// Metal's page on Apple silicon, to which a Shared heap's memory is rounded up: the guest's
/// block holds it whole ([`RGFX_CMD_HEAP_OVER_HOST_MEMORY`]).
const HEAP_PAGE: u64 = 0x4000;

mod argument;
mod blit;
mod cmdbuf;
mod compute;
mod library;
mod managed;
mod render;
mod view;
mod waits;

#[cfg(test)]
pub(crate) use library::tests::{metallib as test_metallib, sha as test_sha};
#[cfg(test)]
pub(crate) use library::verified_key as test_key;
use library::{LibraryCache, LibraryKey, LibraryRef, LibrarySlot};
pub(crate) use library::{
    LibraryJob, LibraryStart, MadeLibrary, RELEASED_BUDGET, complete, make_library,
};
use managed::{ManagedCopy, STORAGE_MODE_MANAGED, storage_mode};

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

/// A sink the decoder calls with a pipeline create's reflection reply once the
/// pipeline is made (or refused). The ring loop sets one that writes the reply to
/// the queue's completion ring, keyed to the create by its `seqno`; until it is
/// set, the reply is built and dropped.
pub(crate) type ReflectionSink = Arc<dyn Fn(RgfxPipelineReflection) + Send + Sync>;

/// An answer the host gives once a context's command buffers have finished (a `CONTEXT_CLOSE`'s,
/// a `BLOCK_RELEASE`'s).
pub(crate) type DrainAnswer = Box<dyn FnOnce() + Send>;

/// The command buffers one context committed since its last drain began: each holds a clone
/// until its completion handler has run, and the answers waiting on them are given when the last
/// clone goes ([`GraphicsDecoder::drain_context`]).
#[derive(Default)]
pub(crate) struct Drain {
    /// Given when the drain drops.
    answers: Mutex<Vec<DrainAnswer>>,
}

impl Drop for Drain {
    fn drop(&mut self) {
        for answer in self.answers.get_mut().drain(..) {
            answer();
        }
    }
}

/// The contexts whose work failed off its op's thread: a command buffer's completion the
/// platform refused on its own thread ([`CompletedInfo::refused`]), where the paravirt device ends
/// the process. The decoder refuses each one's later work, its command buffers and creates, until
/// its close: only its destroys are carried out (docs/graphics-kext/design.md, "Guest input the
/// host refuses"). Shared with the command buffers' completed handlers, which mark a context.
///
/// [`CompletedInfo::refused`]: reims_vgpu_core::platform::command::CompletedInfo::refused
#[derive(Default)]
pub(crate) struct FailedContexts(Mutex<HashSet<u32>>);

impl FailedContexts {
    /// Marks `context` failed; true where it was not yet.
    pub(crate) fn fail(&self, context: u32) -> bool {
        self.0.lock().insert(context)
    }

    /// Whether `context` failed.
    pub(crate) fn has(&self, context: u32) -> bool {
        self.0.lock().contains(&context)
    }

    /// Forgets `context`, at its close.
    fn forget(&self, context: u32) {
        self.0.lock().remove(&context);
    }
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
    // No create makes an indirect command buffer yet (the guest produces one on a queue, an op
    // the first slice does not emit), so this is matched by `require_indirect_command_buffer` but
    // not yet constructed; kept whole for when that op lands (project rule "Unread is not dead").
    #[allow(dead_code)]
    Icb(P::Icb),
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
    /// (`RESOURCE_ID_CHUNK`): 0, at or past [`RGFX_RESOURCE_ID_LIMIT`], or another
    /// context's. The platform is not called, so no id the host's Metal ends the
    /// process for reaches it.
    BadResourceId,
    /// The message's context failed off its op's thread ([`FailedContexts`]): its later work is
    /// refused.
    ContextFailed,
}

/// The stage-3 decoder for one device: the `obj_id` → handle table and the
/// dispatch from a wire message to a `Platform` call.
pub struct GraphicsDecoder<P: Platform> {
    /// The platform whose device the decoder drives, held for the descriptor
    /// builders ([`Platform::texture_descriptor`]) some per-device creates need.
    platform: Arc<P>,
    objects: HashMap<u64, ObjHandle<P>>,
    /// The textures over each IOSurface's plane 0, by the surface's global id:
    /// every `TEXTURE_FROM_IOSURFACE` naming the surface whose texture the table
    /// holds, in the order made, so a destroy of one leaves the others (another
    /// process's, or an older one's) to name the surface. A present names its
    /// surface by that id and shows the latest.
    surfaces: HashMap<u32, Vec<u64>>,
    /// Where in the region each of those textures' copy keeps its
    /// [`RgfxSurfaceWriters`], by the texture's id, where its message gave one.
    surface_writers: HashMap<u64, u64>,
    /// The bytes of each library whose `LIBRARY_FROM_REGION` chunks have not yet
    /// reached its length, by the library's id.
    library_chunks: HashMap<u64, Vec<u8>>,
    /// The libraries being made, by id, until each is stored or cancelled (`library`).
    making_libraries: HashMap<u64, Arc<LibrarySlot<P>>>,
    /// The libraries made from keyed bytes, by key, across every process (`library`).
    library_cache: LibraryCache<P>,
    /// The key of each id the cache's entries are named by.
    cached_libraries: HashMap<u64, LibraryKey>,
    /// The context holding each chunk of the texture resource ids, by chunk, as
    /// `RESOURCE_ID_CHUNK` gave it, until its `CONTEXT_CLOSE`: every process's
    /// textures share the one device's ids, so a texture create's id must lie in
    /// a chunk of its own context ([`Self::holds_resource_id`]).
    id_chunks: Box<[Option<u32>]>,
    /// Where a command buffer's completion record goes once its work finishes;
    /// `None` builds and drops it (the ring loop sets a sink later).
    completion_sink: Option<CompletionSink>,
    /// Where the shared events a command buffer signalled go once its work finishes; `None`
    /// builds and drops them (the ring loop sets a sink later).
    signal_sink: Option<SignalSink>,
    /// The watch over command buffers that wait on events, which logs the ones that wait long
    /// (diagnostics only: [`waits`]).
    waits: waits::WaitWatch<P>,
    /// Where a pipeline create's reflection reply goes; `None` builds and drops
    /// it (the ring loop sets a sink later).
    reflection_sink: Option<ReflectionSink>,
    /// The context (the process) of the message being carried out, whose objects alone it may
    /// name ([`Self::named`]); `None` names any (the decoder's own tests).
    context: Option<u32>,
    /// The region span (offset, length) each object lies over, for those over the region's
    /// memory: a buffer or heap over host memory, and what is placed in one (a buffer at an
    /// offset, a texture in the heap or over the buffer), with its parent's span. A block is not
    /// given out again while an object lies in it ([`Self::span_in_use`]).
    spans: HashMap<u64, (u64, u64)>,
    /// Each context's command buffers in flight since its last drain began
    /// ([`Self::drain_context`]).
    drains: HashMap<u32, Arc<Drain>>,
    /// The contexts whose work failed off its op's thread, whose later work is refused.
    failed: Arc<FailedContexts>,
    /// The CPU copy of each `Managed` buffer and texture ([`managed`]).
    managed: HashMap<u64, ManagedCopy<P>>,
    /// Whether a `Managed` resource is one copy, over its block ([`Self::set_unified_memory`]).
    unified_memory: bool,
    /// The device's BAR 2 region: the host memory the over-host-memory buffers
    /// and heaps are placed over. Held so it stays mapped while the decoder's
    /// objects live; declared last so the object table drops before it (a buffer
    /// also keeps its own clone alive in its no-copy deallocator).
    region: Arc<Region>,
}

impl<P: Platform> GraphicsDecoder<P> {
    /// A new decoder for `platform` over its `region`, with no objects.
    pub fn new(platform: Arc<P>, region: Arc<Region>) -> Self {
        GraphicsDecoder {
            waits: waits::WaitWatch::new(Arc::clone(&platform)),
            platform,
            objects: HashMap::new(),
            surfaces: HashMap::new(),
            surface_writers: HashMap::new(),
            library_chunks: HashMap::new(),
            making_libraries: HashMap::new(),
            library_cache: LibraryCache::new(RELEASED_BUDGET),
            cached_libraries: HashMap::new(),
            id_chunks: vec![None; RGFX_RESOURCE_ID_CHUNKS as usize].into_boxed_slice(),
            completion_sink: None,
            signal_sink: None,
            reflection_sink: None,
            context: None,
            spans: HashMap::new(),
            drains: HashMap::new(),
            failed: Arc::default(),
            managed: HashMap::new(),
            unified_memory: false,
            region,
        }
    }

    /// Sets the sink the decoder hands each command buffer's completion record to
    /// once its work finishes (the ring loop's, writing the queue's completion
    /// ring).
    pub(crate) fn set_completion_sink(&mut self, sink: CompletionSink) {
        self.completion_sink = Some(sink);
    }

    /// Sets the sink the decoder hands each shared event a command buffer signalled to, once the
    /// buffer's work finishes (the ring loop's, writing the queue's completion ring).
    pub(crate) fn set_signal_sink(&mut self, sink: SignalSink) {
        self.signal_sink = Some(sink);
    }

    /// Sets the sink the decoder hands each pipeline create's reflection reply to
    /// (the ring loop's, writing the queue's completion ring).
    pub(crate) fn set_reflection_sink(&mut self, sink: ReflectionSink) {
        self.reflection_sink = Some(sink);
    }

    /// Sets the context of the messages carried out next: the process that sent them, whose
    /// objects alone they may name. On ring 0 it is the context of the object a message creates or
    /// destroys, which the kext checked is the sender's; on a queue's ring, the queue's.
    pub(crate) fn set_context(&mut self, context: Option<u32>) {
        self.context = context;
    }

    /// The contexts whose work failed off its op's thread, shared ([`FailedContexts`]).
    pub(crate) fn failed_contexts(&self) -> Arc<FailedContexts> {
        Arc::clone(&self.failed)
    }

    /// Whether the current context's work failed off its op's thread ([`FailedContexts`]), so
    /// its later work is refused.
    pub(super) fn context_failed(&self) -> bool {
        self.context.is_some_and(|context| self.failed.has(context))
    }

    /// Whether `obj_id` is the current context's to name (any where none is set).
    fn owned(&self, obj_id: u64) -> bool {
        self.context
            .is_none_or(|context| obj_id >> RGFX_CONTEXT_OBJ_ID_SHIFT == u64::from(context))
    }

    /// The object `obj_id` names, where it is the current context's: a process names only its own
    /// objects, and an id of another's resolves to nothing (a bad reference where it is named).
    fn named(&self, obj_id: u64) -> Option<&ObjHandle<P>> {
        if self.owned(obj_id) {
            self.objects.get(&obj_id)
        } else {
            None
        }
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

    /// The objects the table holds (for the tests).
    #[cfg(any(test, feature = "testkit"))]
    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    /// The texture a present of IOSurface `iosurface_id` shows ([`Self::surface_present`]).
    #[cfg(test)]
    pub(crate) fn surface_texture(&self, iosurface_id: u32) -> Option<P::Texture> {
        self.surface_present(iosurface_id)
            .map(|(texture, _, _)| texture)
    }

    /// What a present of IOSurface `iosurface_id` shows: the latest texture a guest process made
    /// over its plane 0 with `TEXTURE_FROM_IOSURFACE` and has not destroyed, and the command
    /// buffers committed that write that texture's copy, which the present waits for: the slots
    /// of its [`RgfxSurfaceWriters`] that name a queue the decoder holds (a slot naming another,
    /// or none, has nothing to wait for); and the context that made the texture. The caller holds
    /// its own reference to the texture: a destroy of it after this leaves it drawable.
    pub(crate) fn surface_present(
        &self,
        iosurface_id: u32,
    ) -> Option<(P::Texture, Vec<RgfxSurfaceWriter>, u32)> {
        let obj_id = *self.surfaces.get(&iosurface_id)?.last()?;
        let Some(ObjHandle::Texture(texture)) = self.objects.get(&obj_id) else {
            return None;
        };
        let writers = self
            .surface_writers
            .get(&obj_id)
            .and_then(|&at| self.region.surface_writers(at))
            .map(|record| {
                record
                    .writers
                    .into_iter()
                    .filter(|w| {
                        w.seqno != 0
                            && matches!(
                                self.objects.get(&w.queue_obj_id),
                                Some(ObjHandle::CommandQueue(_))
                            )
                    })
                    .collect()
            })
            .unwrap_or_default();
        let owner = (obj_id >> RGFX_CONTEXT_OBJ_ID_SHIFT) as u32;
        Some((texture.clone_ref(), writers, owner))
    }

    /// Forgets texture `obj_id` as a name of the surfaces it lies over.
    fn forget_surface_texture(&mut self, obj_id: u64) {
        self.surfaces.retain(|_, textures| {
            textures.retain(|t| *t != obj_id);
            !textures.is_empty()
        });
        self.surface_writers.remove(&obj_id);
    }

    /// Forgets the surface names of textures the table no longer holds: a caller
    /// that drops objects other than by `OBJECT_DESTROY` (a context's whole id
    /// range at its close) calls it, so a later object that takes a dropped id is
    /// never taken for the surface's texture.
    pub(crate) fn forget_dropped_surfaces(&mut self) {
        let objects = &self.objects;
        self.surface_writers
            .retain(|t, _| matches!(objects.get(t), Some(ObjHandle::Texture(_))));
        self.surfaces.retain(|_, textures| {
            textures.retain(|t| matches!(objects.get(t), Some(ObjHandle::Texture(_))));
            !textures.is_empty()
        });
    }

    /// Drops every object of context `context`: those whose id lies in its range,
    /// `context << RGFX_CONTEXT_OBJ_ID_SHIFT` and up, below the next context's (its process closed
    /// the device or died), and its chunks of the texture resource ids. How many objects were
    /// dropped. Each object is let go of in a refusal scope of its own ([`drop_refused`]), so a
    /// release the platform dies at is refused, logged, and the close goes on; the context's
    /// failure ([`FailedContexts`]) is forgotten with it.
    pub(crate) fn drop_context(&mut self, context: u32) -> usize {
        for holder in self.id_chunks.iter_mut() {
            if *holder == Some(context) {
                *holder = None;
            }
        }
        self.failed.forget(context);
        let ours = |id: &u64| (id >> RGFX_CONTEXT_OBJ_ID_SHIFT) == u64::from(context);
        let objects: Vec<u64> = self.objects.keys().copied().filter(ours).collect();
        for &id in &objects {
            if let Some(handle) = self.objects.remove(&id) {
                drop_refused(id, handle);
            }
        }
        self.spans.retain(|id, _| !ours(id));
        let managed: Vec<u64> = self.managed.keys().copied().filter(ours).collect();
        for id in managed {
            if let Some(copy) = self.managed.remove(&id) {
                drop_refused(id, copy);
            }
        }
        let libraries: Vec<u64> = (self.making_libraries.keys())
            .chain(self.library_chunks.keys())
            .chain(self.cached_libraries.keys())
            .copied()
            .filter(ours)
            .collect();
        for id in libraries {
            if let Err(refusal) = refusals::refusing(|| self.forget_library(id)) {
                refused_release(id, &refusal);
            }
        }
        objects.len()
    }

    /// Whether texture resource id `id` lies in a chunk that the context of object
    /// `obj_id` holds: never 0, and always below [`RGFX_RESOURCE_ID_LIMIT`].
    fn holds_resource_id(&self, obj_id: u64, id: u64) -> bool {
        if id == 0 || id >= RGFX_RESOURCE_ID_LIMIT {
            return false;
        }
        let context = obj_id >> RGFX_CONTEXT_OBJ_ID_SHIFT;
        self.id_chunks
            .get(((id - 1) / RGFX_RESOURCE_ID_CHUNK) as usize)
            .copied()
            .flatten()
            .is_some_and(|holder| u64::from(holder) == context)
    }

    /// Gives context `context` chunk `chunk` of the texture resource ids, as a
    /// `RESOURCE_ID_CHUNK` does.
    #[cfg(test)]
    pub(crate) fn hold_resource_ids(&mut self, context: u32, chunk: u32) {
        self.id_chunks[chunk as usize] = Some(context);
    }

    /// A `SAMPLER_CREATE`'s answer (`SAMPLER_STATE`): the sampler made under the message's id,
    /// with its `gpuResourceID`, the id the platform numbers it by, which the guest's sampler
    /// answers and its argument buffers hold; [`RGFX_STATUS_REFUSED`] with 0 where the platform
    /// makes none, and [`RGFX_STATUS_INVALID_ARGUMENT`] for a short create. `None` for a message
    /// too short to carry its seqno, which no one can be answered by.
    pub(crate) fn sampler_state(
        &mut self,
        device: &P::Device,
        message: &[u8],
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
        // A failed context's later work is refused ([`FailedContexts`]).
        if self.context_failed() {
            return Ok(Some(answer(RGFX_STATUS_REFUSED, 0)));
        }
        Ok(Some(match self.make_sampler(device, message)? {
            Ok(id) => answer(RGFX_STATUS_OK, id),
            Err(DecodeError::Malformed) => answer(RGFX_STATUS_INVALID_ARGUMENT, 0),
            Err(_) => answer(RGFX_STATUS_REFUSED, 0),
        }))
    }

    /// The sampler of a `SAMPLER_CREATE`, stored under its id: its `gpuResourceID`.
    fn make_sampler(
        &mut self,
        device: &P::Device,
        message: &[u8],
    ) -> RaiseResult<Result<u64, DecodeError>> {
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
        let Some(sampler) = device.new_sampler_state(&desc)? else {
            return Ok(Err(DecodeError::Refused));
        };
        let id = sampler.gpu_resource_id();
        Ok(self
            .insert(m.obj_id, ObjHandle::Sampler(sampler))?
            .map(|()| id))
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
        // A failed context's later work is refused ([`FailedContexts`]).
        if self.context_failed() {
            return Some(answer(RGFX_STATUS_REFUSED, 0, 0));
        }
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

    /// Carries out one wire message against `device`: a create inserts the new
    /// handle (keyed by the guest-picked `obj_id`), a destroy drops it. The
    /// inner `Result` says whether the message was carried out; a platform raise
    /// propagates through the outer `RaiseResult`.
    pub fn decode(
        &mut self,
        device: &P::Device,
        message: &[u8],
    ) -> RaiseResult<Result<(), DecodeError>> {
        let Some(header) = read::<RgfxMsgHeader>(message) else {
            return Ok(Err(DecodeError::Malformed));
        };
        // A failed context's later work is refused: only its destroys are carried out.
        if header.r#type != RGFX_CMD_OBJECT_DESTROY && self.context_failed() {
            return Ok(Err(DecodeError::ContextFailed));
        }
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
                self.spans.remove(&m.obj_id);
                if let Some(copy) = self.managed.remove(&m.obj_id) {
                    drop_refused(m.obj_id, copy);
                }
                if let Some(handle) = self.objects.remove(&m.obj_id) {
                    let texture = matches!(handle, ObjHandle::Texture(_));
                    drop_refused(m.obj_id, handle);
                    if texture {
                        self.forget_surface_texture(m.obj_id);
                    }
                }
                Ok(Ok(()))
            }
            RGFX_CMD_RESOURCE_ID_CHUNK => {
                let Some(m) = read::<RgfxResourceIdChunk>(message) else {
                    return Ok(Err(DecodeError::Malformed));
                };
                // The kext's grant: the context holds the chunk until its close. A
                // chunk past the pool is not one.
                let Some(holder) = self.id_chunks.get_mut(m.chunk as usize) else {
                    return Ok(Err(DecodeError::BadResourceId));
                };
                *holder = Some(m.context);
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
                let Some(ObjHandle::SharedEvent(event)) = self.named(m.obj_id) else {
                    return Ok(Err(DecodeError::BadReference));
                };
                event.set_signaled_value(m.value)?;
                Ok(Ok(()))
            }
            RGFX_CMD_QUEUE_CREATE => {
                let Some(m) = read::<RgfxQueueCreate>(message) else {
                    return Ok(Err(DecodeError::Malformed));
                };
                // The ring pair this message places (`submit_offset`,
                // `complete_offset`, the data sizes and the vector) is set up
                // with the ring loop in a later increment; here the queue handle
                // is made so later command buffers can name it.
                match device.new_command_queue() {
                    Some(queue) => self.insert(m.obj_id, ObjHandle::CommandQueue(queue)),
                    None => Ok(Err(DecodeError::Refused)),
                }
            }
            RGFX_CMD_LIBRARY_CREATE | RGFX_CMD_LIBRARY_FROM_REGION => {
                self.decode_library(device, message)
            }
            RGFX_CMD_SAMPLER_CREATE => Ok(self.make_sampler(device, message)?.map(|_| ())),
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
                match made {
                    Some(buffer) => self.insert_over(
                        m.obj_id,
                        ObjHandle::Buffer(buffer),
                        Some((m.region_offset, m.length)),
                    ),
                    None => Ok(Err(DecodeError::Refused)),
                }
            }
            RGFX_CMD_BUFFER_DID_MODIFY_RANGE => self.did_modify_range(message),
            RGFX_CMD_MANAGED_TEXTURE_CREATE => self.managed_texture(device, message),
            RGFX_CMD_TEXTURE_REPLACE_REGION => self.replace_region(message),
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
                let span = self.span_of(m.heap_obj_id);
                match made {
                    Some(buffer) => self.insert_over(m.obj_id, ObjHandle::Buffer(buffer), span),
                    None => Ok(Err(DecodeError::Refused)),
                }
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
                    buffer.new_texture(OnFault::Return, &descriptor, m.offset, m.bytes_per_row)?
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
                    buffer.new_linear_texture(
                        &descriptor,
                        m.offset,
                        m.bytes_per_row,
                        m.bytes_per_image,
                    )?
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
                if !self.holds_resource_id(m.obj_id, m.resource_id) {
                    return Ok(Err(DecodeError::BadResourceId));
                }
                // Placed as TEXTURE_FROM_BUFFER places a texture, over the buffer
                // the guest keeps its copy of the plane in; a buffer the host does
                // not hold is a guest protocol error.
                let desc = texture_desc(&m.desc, m.resource_id);
                let descriptor = self.platform.texture_descriptor(&desc);
                if !matches!(self.named(m.buffer_obj_id), Some(ObjHandle::Buffer(_))) {
                    return Ok(Err(DecodeError::BadReference));
                }
                let span = self.span_of(m.buffer_obj_id);
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
                    buffer.new_texture(OnFault::Return, &descriptor, m.offset, m.bytes_per_row)?
                };
                match made {
                    Some(texture) => {
                        // An id made again names only its new texture's surface.
                        self.forget_surface_texture(m.obj_id);
                        if m.plane == 0 {
                            self.surfaces
                                .entry(m.iosurface_id)
                                .or_default()
                                .push(m.obj_id);
                            if let Some(at) = writers {
                                self.surface_writers.insert(m.obj_id, at);
                            }
                        }
                        self.insert_over(m.obj_id, ObjHandle::Texture(texture), span)
                    }
                    None => Ok(Err(DecodeError::Refused)),
                }
            }
            RGFX_CMD_RENDER_PIPELINE_CREATE | RGFX_CMD_COMPUTE_PIPELINE_CREATE => {
                self.decode_pipeline(device, message)
            }
            RGFX_CMD_COMMAND_BUFFER => {
                // The buffer's context's drain, held until its completion handler has run.
                let drain = (self.context)
                    .map(|context| Arc::clone(self.drains.entry(context).or_default()));
                self.decode_command_buffer(message, drain)
            }
            _ => Ok(Err(DecodeError::UnknownCommand)),
        }
    }

    /// Inserts a new object's handle under its guest-picked id.
    fn insert(
        &mut self,
        obj_id: u64,
        handle: ObjHandle<P>,
    ) -> RaiseResult<Result<(), DecodeError>> {
        self.insert_over(obj_id, handle, None)
    }

    /// Inserts a new object's handle under its guest-picked id, with the region span it lies over
    /// (`spans`), where it lies over one.
    fn insert_over(
        &mut self,
        obj_id: u64,
        handle: ObjHandle<P>,
        span: Option<(u64, u64)>,
    ) -> RaiseResult<Result<(), DecodeError>> {
        self.store(obj_id, handle);
        if let Some(span) = span {
            self.spans.insert(obj_id, span);
        }
        Ok(Ok(()))
    }

    /// Stores `handle` under `obj_id`, in place of what the id named.
    fn store(&mut self, obj_id: u64, handle: ObjHandle<P>) {
        self.forget_library(obj_id);
        self.spans.remove(&obj_id);
        self.managed.remove(&obj_id);
        self.objects.insert(obj_id, handle);
    }

    /// The region span `obj_id`'s object lies over (`spans`), where it is one the current
    /// context may name and lies over one.
    fn span_of(&self, obj_id: u64) -> Option<(u64, u64)> {
        self.named(obj_id)
            .and_then(|_| self.spans.get(&obj_id).copied())
    }

    /// Whether an object the decoder holds lies in the `size` bytes at region offset `offset`:
    /// the block they make is not to be given out again while it does.
    pub(crate) fn span_in_use(&self, offset: u64, size: u64) -> bool {
        let end = offset.saturating_add(size);
        self.spans
            .values()
            .any(|&(at, length)| at < end && offset < at.saturating_add(length))
    }

    /// Begins a drain of `context`'s command buffers: `answer` is given once every one the decoder
    /// has committed for the context has finished its completion handler, and not later ones. The
    /// drain is returned for the caller to drop with no lock held: the answer goes then where no
    /// buffer is in flight, else on the GPU thread of the last to finish. `closed` ends the
    /// context's drains (its `CONTEXT_CLOSE`); else a new one begins for its later buffers.
    pub(crate) fn drain_context(
        &mut self,
        context: u32,
        closed: bool,
        answer: DrainAnswer,
    ) -> Arc<Drain> {
        let drain = if closed {
            self.drains.remove(&context)
        } else {
            self.drains.insert(context, Arc::default())
        }
        .unwrap_or_default();
        drain.answers.lock().push(answer);
        drain
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
                if let (Some(sink), Some(refused)) = (&self.reflection_sink, refused_reply(message))
                {
                    sink(refused);
                }
                return Ok(Err(e));
            }
        };
        let compiled = compile_refusing::<P>(device, job);
        let refused = compiled.is_refused();
        if let Some(sink) = self.reflection_sink.clone() {
            sink(compiled.reflection);
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
        if self.context_failed() {
            return Err(DecodeError::ContextFailed);
        }
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

    /// The last step of a pipeline create: stores the compiled pipeline under its id (a refused
    /// one stores nothing).
    pub(crate) fn finish_pipeline(&mut self, compiled: CompiledPipeline<P>) {
        if let Some(handle) = compiled.handle {
            self.spans.remove(&compiled.obj_id);
            self.objects.insert(compiled.obj_id, handle);
        }
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
    /// [`RGFX_PIPELINE_KIND_RENDER`] or [`RGFX_PIPELINE_KIND_COMPUTE`].
    pub(crate) fn kind(&self) -> u32 {
        match self.shape {
            PipelineShape::Render { .. } => RGFX_PIPELINE_KIND_RENDER,
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
    Compute {
        library: LibraryRef<P>,
        name: String,
        constants: Vec<FunctionConstant>,
        linked: Vec<LinkedRef<P>>,
        fields: ComputeFields,
    },
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
            refusal: Some(reason.into()),
        }
    }

    /// A made pipeline and its reflection.
    fn made(obj_id: u64, handle: ObjHandle<P>, reflection: RgfxPipelineReflection) -> Self {
        CompiledPipeline {
            obj_id,
            handle: Some(handle),
            reflection,
            refusal: None,
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

/// The reply a pipeline create refused at its op owes: a refusal, keyed to the create by its
/// seqno, for the object its header names (`None` for a message too short to name one). The
/// create's process waits for it, so a create the host will not carry out is still answered.
pub(crate) fn refused_reply(message: &[u8]) -> Option<RgfxPipelineReflection> {
    let header = read::<RgfxMsgHeader>(message)?;
    let obj_id = u64::from_le_bytes(message.get(16..24)?.try_into().ok()?);
    let kind = if header.r#type == RGFX_CMD_RENDER_PIPELINE_CREATE {
        RGFX_PIPELINE_KIND_RENDER
    } else {
        RGFX_PIPELINE_KIND_COMPUTE
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
/// nil-allowed reference (`0` is `None`, a non-zero miss is an exec fault). Each
/// returns a fresh counted reference ([`Handle::clone_ref`]) the caller holds for
/// the encode.
macro_rules! resolvers {
    ($( ($req:ident, $get:ident, $variant:ident, $assoc:ident, $what:literal) ),* $(,)?) => {
        // The whole object-resolution API: some resolvers are for ops the first slice does not
        // emit yet, so they are not reached but are kept whole (project rule "Unread is not dead").
        #[allow(dead_code)]
        impl<P: Platform> GraphicsDecoder<P> {
            $(
                #[doc = concat!("The ", $what, " `id` names; an exec fault when `id` is 0 or names no ", $what, ".")]
                pub(crate) fn $req(&self, id: u64) -> RaiseResult<P::$assoc> {
                    match self.named(id) {
                        Some(ObjHandle::$variant(h)) => Ok(h.clone_ref()),
                        _ => Err(bad_reference()),
                    }
                }
                #[doc = concat!("The ", $what, " `id` names, `None` for 0; an exec fault when a non-zero `id` names no ", $what, ".")]
                pub(crate) fn $get(&self, id: u64) -> RaiseResult<Option<P::$assoc>> {
                    if id == 0 {
                        return Ok(None);
                    }
                    match self.named(id) {
                        Some(ObjHandle::$variant(h)) => Ok(Some(h.clone_ref())),
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
        RGFX_CMD_HEAP_TEXTURE_SIZE_AND_ALIGN, RGFX_LIBRARY_BYTES_OFFSET, RgfxLibraryCreate,
        RgfxLibraryFromRegion,
    };
    use reims_vgpu_core::testkit::{Arg, FakePlatform, Step};
    use zerocopy::IntoBytes;

    use super::*;

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

    /// A sampler create is answered with the sampler's `gpuResourceID`, the id the platform
    /// numbers it by, at the create's seqno, and the sampler is stored under its id; a create the
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
        assert!(dec.named(21).is_some());
        let second = dec
            .sampler_state(&device, create(22, 5).as_bytes())
            .unwrap()
            .unwrap();
        assert_eq!(second.status, RGFX_STATUS_OK);
        assert_ne!(second.gpu_resource_id, first.gpu_resource_id);

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
        assert_eq!(log.calls_to("newEvent").len(), 1);
        assert_eq!(log.calls_to("newFence").len(), 1);
        assert_eq!(log.calls_to("newSharedEvent").len(), 1);
        assert_eq!(log.calls_to("newCommandQueue").len(), 1);
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
        dec.drop_context(1);
        assert_eq!(
            made(&mut dec, obj(3, 1), 5),
            Err(DecodeError::BadResourceId)
        );
        assert_eq!(dec.decode(&device, grant(3, 0).as_bytes()).unwrap(), Ok(()));
        assert_eq!(made(&mut dec, obj(3, 1), 5), Ok(()));
        assert_eq!(made(&mut dec, obj(2, 7), 16319), Ok(()));
    }

    /// A decoder of a fake platform, its reflection replies, and library 1 made.
    fn pipeline_decoder() -> (
        Arc<FakePlatform>,
        GraphicsDecoder<FakePlatform>,
        Arc<reims_vgpu_core::sync::Mutex<Vec<RgfxPipelineReflection>>>,
    ) {
        let p = Arc::new(FakePlatform::new());
        let device = p.device();
        let mut dec = GraphicsDecoder::<FakePlatform>::new(
            Arc::clone(&p),
            Arc::new(reims_graphics::region::Region::new(0x10000).unwrap()),
        );
        let replies = Arc::new(reims_vgpu_core::sync::Mutex::new(Vec::new()));
        let sink_replies = Arc::clone(&replies);
        dec.set_reflection_sink(Arc::new(move |r| sink_replies.lock().push(r)));
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
    fn whole_render_create(obj_id: u64, fragment: bool) -> Vec<u8> {
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
    fn compute_create(obj_id: u64) -> Vec<u8> {
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
        use crate::platform::ApiPlatform as _;
        use reims_vgpu_metal::MetalPlatform;
        let platform = MetalPlatform::process();
        let device = platform.metal_device(None).unwrap();
        let mut dec = GraphicsDecoder::<MetalPlatform>::new(
            Arc::clone(&platform),
            Arc::new(reims_graphics::region::Region::new(0x10000).unwrap()),
        );
        let replies = Arc::new(reims_vgpu_core::sync::Mutex::new(Vec::new()));
        let sink_replies = Arc::clone(&replies);
        dec.set_reflection_sink(Arc::new(move |r: RgfxPipelineReflection| {
            sink_replies.lock().push((r.obj_id, r.status));
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
    #[cfg(target_os = "macos")]
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
    #[cfg(target_os = "macos")]
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
    /// the guest sends (the caller's, then the stitched one), then for each of [`STITCH_LINKS`] a
    /// pipeline create of `stitch_caller` linking `fused` from the stitched library, and what each
    /// pipeline made writes of [`STITCH_INPUTS`], run on the platform; `None` for one refused.
    #[cfg(target_os = "macos")]
    fn stitched_links_through_the_decoder<P: crate::platform::ApiPlatform>(
        s: &Stitched,
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
        dec.set_reflection_sink(Arc::new(move |r: RgfxPipelineReflection| {
            sink_replies.lock().push((r.obj_id, r.status));
        }));
        for (obj_id, bytes) in [(1, &s.caller_bytes), (2, &s.stitched_bytes)] {
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
            let Some(ObjHandle::ComputePipeline(pipeline)) = dec.objects.get(&obj_id) else {
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

    /// On Metal, a guest's stitched library is a library like any other: the library Metal's
    /// stitcher made in the guest, sent as its bytes, gives a function `stitch_caller`'s pipeline
    /// links from either list, as Metal's own does, and the pipeline writes what Metal's writes;
    /// with nothing linked, it is refused, as Metal refuses it
    #[cfg(target_os = "macos")]
    #[test]
    fn on_metal_a_stitched_function_links_and_computes_as_metal_s() {
        let s = stitched_on_metal();
        let metal = metal_s_own_stitched_links(&s);
        assert_eq!(metal, [Some(20.0), Some(20.0), None]);
        assert_eq!(
            stitched_links_through_the_decoder::<reims_vgpu_metal::MetalPlatform>(&s),
            metal
        );
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
        dec.set_reflection_sink(Arc::new(move |r| sink_replies.lock().push(r)));

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
