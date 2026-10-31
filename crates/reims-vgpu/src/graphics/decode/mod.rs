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

use std::collections::HashMap;
use std::sync::Arc;

use reims_graphics::region::Region;
use reims_graphics::wire::{
    RGFX_CMD_BUFFER_AT_OFFSET, RGFX_CMD_BUFFER_CREATE, RGFX_CMD_BUFFER_DID_MODIFY_RANGE,
    RGFX_CMD_BUFFER_OVER_HOST_MEMORY, RGFX_CMD_COMMAND_BUFFER, RGFX_CMD_COMPUTE_PIPELINE_CREATE,
    RGFX_CMD_DEPTH_STENCIL_CREATE, RGFX_CMD_EVENT_CREATE, RGFX_CMD_FENCE_CREATE,
    RGFX_CMD_HEAP_CREATE, RGFX_CMD_HEAP_OVER_HOST_MEMORY, RGFX_CMD_LIBRARY_CREATE,
    RGFX_CMD_LIBRARY_FROM_REGION, RGFX_CMD_MANAGED_TEXTURE_CREATE, RGFX_CMD_OBJECT_DESTROY,
    RGFX_CMD_QUEUE_CREATE, RGFX_CMD_RENDER_PIPELINE_CREATE, RGFX_CMD_RESOURCE_ID_CHUNK,
    RGFX_CMD_SAMPLER_CREATE, RGFX_CMD_SHARED_EVENT_CREATE, RGFX_CMD_TEXTURE_CREATE,
    RGFX_CMD_TEXTURE_FROM_BUFFER, RGFX_CMD_TEXTURE_FROM_IOSURFACE, RGFX_CMD_TEXTURE_REPLACE_REGION,
    RGFX_CONTEXT_ALLOCATIONS_OFFSET, RGFX_CONTEXT_OBJ_ID_SHIFT, RGFX_CURSOR_SPAN,
    RGFX_EVT_PIPELINE_REFLECTION, RGFX_PIPELINE_KIND_COMPUTE, RGFX_PIPELINE_KIND_RENDER,
    RGFX_PIPELINE_OK, RGFX_PIPELINE_REFLECT_SUPPORT_ICB,
    RGFX_PIPELINE_REFLECT_THREADGROUP_MATCHES_TILE, RGFX_PIPELINE_REFUSED, RGFX_RESOURCE_ID_CHUNK,
    RGFX_RESOURCE_ID_CHUNKS, RGFX_RESOURCE_ID_LIMIT, RgfxBufferAtOffset, RgfxBufferCreate,
    RgfxBufferOverHostMemory, RgfxCbComplete, RgfxColorAttachmentDesc, RgfxComputePipelineCreate,
    RgfxDepthStencilCreate, RgfxFunctionConstant, RgfxHeapCreate, RgfxHeapOverHostMemory,
    RgfxMsgHeader, RgfxObjectCreate, RgfxObjectDestroy, RgfxPipelineReflection, RgfxQueueCreate,
    RgfxRenderPipelineCreate, RgfxResourceIdChunk, RgfxSamplerCreate, RgfxSharedEventCreate,
    RgfxStencilDescriptor, RgfxTextureCreate, RgfxTextureDescriptor, RgfxTextureFromBuffer,
    RgfxTextureFromIOSurface,
};
use reims_vgpu_core::platform::device::GpuDevice;
use reims_vgpu_core::platform::resource::{
    Buffer, ComputePipeline, FunctionConstant, Heap, Library, RenderPipeline,
};
use reims_vgpu_core::platform::types::{ResourceAddressRange, Swizzle};
use reims_vgpu_core::platform::values::{
    BufferDesc, DepthStencilDesc, HeapDesc, OptionSet, RenderPipelineDesc, SamplerDesc,
    StencilDesc, TextureDesc,
};
use reims_vgpu_core::platform::{Handle, Platform};
use reims_vgpu_core::raise::{OnFault, Raise, RaiseResult, refusals};
use reims_vgpu_core::sync::Mutex;
use zerocopy::FromBytes;

mod blit;
mod cmdbuf;
mod compute;
mod library;
mod managed;
mod render;

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
            platform,
            objects: HashMap::new(),
            surfaces: HashMap::new(),
            library_chunks: HashMap::new(),
            making_libraries: HashMap::new(),
            library_cache: LibraryCache::new(RELEASED_BUDGET),
            cached_libraries: HashMap::new(),
            id_chunks: vec![None; RGFX_RESOURCE_ID_CHUNKS as usize].into_boxed_slice(),
            completion_sink: None,
            reflection_sink: None,
            context: None,
            spans: HashMap::new(),
            drains: HashMap::new(),
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

    /// The texture over plane 0 of the IOSurface whose global id is
    /// `iosurface_id`: the latest one a guest process made with
    /// `TEXTURE_FROM_IOSURFACE` and has not destroyed. Its pixels are what that
    /// process's command buffers wrote, as of their completion. The caller holds
    /// its own reference: a destroy of the texture after this leaves it drawable.
    #[allow(dead_code)] // the present path reads it
    pub(crate) fn surface_texture(&self, iosurface_id: u32) -> Option<P::Texture> {
        match self.objects.get(self.surfaces.get(&iosurface_id)?.last()?) {
            Some(ObjHandle::Texture(texture)) => Some(texture.clone_ref()),
            _ => None,
        }
    }

    /// Forgets texture `obj_id` as a name of the surfaces it lies over.
    fn forget_surface_texture(&mut self, obj_id: u64) {
        self.surfaces.retain(|_, textures| {
            textures.retain(|t| *t != obj_id);
            !textures.is_empty()
        });
    }

    /// Forgets the surface names of textures the table no longer holds: a caller
    /// that drops objects other than by `OBJECT_DESTROY` (a context's whole id
    /// range at its close) calls it, so a later object that takes a dropped id is
    /// never taken for the surface's texture.
    pub(crate) fn forget_dropped_surfaces(&mut self) {
        let objects = &self.objects;
        self.surfaces.retain(|_, textures| {
            textures.retain(|t| matches!(objects.get(t), Some(ObjHandle::Texture(_))));
            !textures.is_empty()
        });
    }

    /// Drops every object of context `context`: those whose id lies in its range,
    /// `context << RGFX_CONTEXT_OBJ_ID_SHIFT` and up, below the next context's (its process closed
    /// the device or died), and its chunks of the texture resource ids. How many objects were
    /// dropped.
    pub(crate) fn drop_context(&mut self, context: u32) -> usize {
        for holder in self.id_chunks.iter_mut() {
            if *holder == Some(context) {
                *holder = None;
            }
        }
        let ours = |id: &u64| (id >> RGFX_CONTEXT_OBJ_ID_SHIFT) == u64::from(context);
        let before = self.objects.len();
        self.objects.retain(|id, _| !ours(id));
        self.spans.retain(|id, _| !ours(id));
        self.managed.retain(|id, _| !ours(id));
        let libraries: Vec<u64> = (self.making_libraries.keys())
            .chain(self.library_chunks.keys())
            .chain(self.cached_libraries.keys())
            .copied()
            .filter(ours)
            .collect();
        for id in libraries {
            self.forget_library(id);
        }
        before - self.objects.len()
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
        match header.r#type {
            RGFX_CMD_OBJECT_DESTROY => {
                let Some(m) = read::<RgfxObjectDestroy>(message) else {
                    return Ok(Err(DecodeError::Malformed));
                };
                // A missing id is not an error: the guest may destroy an object
                // the host never made (a create that the platform refused).
                self.forget_library(m.obj_id);
                self.spans.remove(&m.obj_id);
                self.managed.remove(&m.obj_id);
                if matches!(self.objects.remove(&m.obj_id), Some(ObjHandle::Texture(_))) {
                    self.forget_surface_texture(m.obj_id);
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
                // `initial_value` is honoured once a shared event's value can be
                // set from the host; a new shared event starts at 0 today.
                match device.new_shared_event() {
                    Some(event) => self.insert(m.obj_id, ObjHandle::SharedEvent(event)),
                    None => Ok(Err(DecodeError::Refused)),
                }
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
            RGFX_CMD_SAMPLER_CREATE => {
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
                match device.new_sampler_state(&desc)? {
                    Some(sampler) => self.insert(m.obj_id, ObjHandle::Sampler(sampler)),
                    None => Ok(Err(DecodeError::Refused)),
                }
            }
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
                let made =
                    self.buffer_over_block(device, m.region_offset, m.length, m.options.into())?;
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
            RGFX_CMD_HEAP_CREATE => {
                let Some(m) = read::<RgfxHeapCreate>(message) else {
                    return Ok(Err(DecodeError::Malformed));
                };
                // A device-placed heap at a guest-picked GPU address (0 lets the
                // host choose). `heap_type` widens to the `MTLHeapType`. Heaps
                // over host memory land with that region's decode.
                let desc = HeapDesc {
                    size: m.size_bytes,
                    heap_type: m.heap_type as i64,
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
                // A Shared heap placed over the guest's own memory in the device's
                // region, `size_bytes` at `region_offset`: its address range is the
                // host VA of that block, as the detached-heap path builds a heap
                // over host pages. A span in no block of the sender's is a guest
                // protocol error. The region stays mapped while the decoder holds it, and the
                // heap (in the object table) drops before the region (field order).
                if !self.in_own_block(m.region_offset, m.size_bytes) {
                    return Ok(Err(DecodeError::BadReference));
                }
                // SAFETY: the block lies inside the region (checked above), one
                // allocation, so the offset is in bounds.
                let address = unsafe { self.region.base().add(m.region_offset as usize) } as u64;
                let range = ResourceAddressRange {
                    address,
                    length: m.size_bytes,
                };
                let ranges = self.platform.new_address_range_array(&[range])?;
                let desc = HeapDesc {
                    size: m.size_bytes,
                    heap_type: m.heap_type as i64,
                    options: vec![OptionSet::ResourceOptions(m.options.into())],
                    address_ranges: Some(&ranges),
                    pinned_gpu_address: (m.gpu_address != 0).then_some(m.gpu_address),
                };
                match device.new_heap(&desc)?.heap {
                    Some(heap) => self.insert_over(
                        m.obj_id,
                        ObjHandle::Heap(heap),
                        Some((m.region_offset, m.size_bytes)),
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
                let made = {
                    let Some(ObjHandle::Buffer(buffer)) = self.named(m.buffer_obj_id) else {
                        return Ok(Err(DecodeError::BadReference));
                    };
                    buffer.new_texture(OnFault::Return, &descriptor, m.offset, m.bytes_per_row)?
                };
                let span = self.span_of(m.buffer_obj_id);
                match made {
                    Some(texture) => {
                        // An id made again names only its new texture's surface.
                        self.forget_surface_texture(m.obj_id);
                        if m.plane == 0 {
                            self.surfaces
                                .entry(m.iosurface_id)
                                .or_default()
                                .push(m.obj_id);
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
        let (obj_id, seqno, kind) = (job.obj_id, job.seqno, job.kind());
        let compiled = match refusing(|| compile_pipeline::<P>(device, job)) {
            Ok(compiled) => compiled,
            Err(raise) => CompiledPipeline::refused(obj_id, seqno, kind, raise.to_string()),
        };
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
    /// the table, its entry points, constants and descriptor fields read from the message, so the
    /// compile needs nothing of the decoder's. A library reference that names no library is a
    /// guest protocol error, as the other creates treat one ([`DecodeError::BadReference`]); a
    /// span the message does not cover, or a name that is not UTF-8, is
    /// [`DecodeError::Malformed`]. Neither is answered.
    ///
    /// The render form carries the vertex and fragment functions, colour attachment 0's pixel
    /// format and the one source-over blend [`RenderPipelineDesc`] can express, and a label; the
    /// rest of the wire descriptor (colour attachments 1..8, arbitrary blend state,
    /// depth/stencil formats, the raster sample count, the vertex descriptor and the pipeline
    /// flags) has no place in that value and is not applied. The compute form applies only the
    /// function; its flags and limits are not applied. Each function is specialised by the
    /// index-keyed constants the wire carries for it. The full path is Metal's
    /// serialized-descriptor deserializer, which is the Apple lane's; carrying the whole
    /// descriptor on the wire keeps the guest ABI stable for when the platform value grows.
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
            return Ok(PipelineJob {
                obj_id: m.obj_id,
                seqno: m.header.seqno,
                shape: PipelineShape::Compute {
                    library,
                    name: name.to_owned(),
                    constants,
                },
            });
        }
        let m = read::<RgfxRenderPipelineCreate>(message).ok_or(DecodeError::Malformed)?;
        // Both function libraries (which may be the same).
        let vertex_library = library(m.vertex_library_obj_id)?;
        let fragment_library = library(m.fragment_library_obj_id)?;
        let (Some(vertex_name), Some(fragment_name)) = (
            str_tail(message, m.vertex_name_offset, m.vertex_name_len),
            str_tail(message, m.fragment_name_offset, m.fragment_name_len),
        ) else {
            return Err(DecodeError::Malformed);
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
        let (Some(vertex_constants), Some(fragment_constants)) = (
            constants_tail(message, m.vertex_constants_offset, m.vertex_constant_count),
            constants_tail(
                message,
                m.fragment_constants_offset,
                m.fragment_constant_count,
            ),
        ) else {
            return Err(DecodeError::Malformed);
        };
        Ok(PipelineJob {
            obj_id: m.obj_id,
            seqno: m.header.seqno,
            shape: PipelineShape::Render {
                vertex_library,
                vertex_name: vertex_name.to_owned(),
                vertex_constants,
                fragment_library,
                fragment_name: fragment_name.to_owned(),
                fragment_constants,
                label,
                color0_pixel_format: m.colors[0].pixel_format.into(),
                blend_over: is_source_over(&m.colors[0]),
            },
        })
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
        fragment_library: LibraryRef<P>,
        fragment_name: String,
        fragment_constants: Vec<FunctionConstant>,
        label: Option<String>,
        color0_pixel_format: u64,
        blend_over: bool,
    },
    Compute {
        library: LibraryRef<P>,
        name: String,
        constants: Vec<FunctionConstant>,
    },
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
            fragment_library,
            fragment_name,
            fragment_constants,
            label,
            color0_pixel_format,
            blend_over,
        } => {
            // A library being made is waited for here; one the platform refused refuses the
            // pipeline.
            let Some(vertex_library) = vertex_library.resolve() else {
                return Ok(refused(no_library("vertex")));
            };
            let Some(fragment_library) = fragment_library.resolve() else {
                return Ok(refused(no_library("fragment")));
            };
            let Some(vertex) = specialised::<P>(&vertex_library, &vertex_name, &vertex_constants)?
            else {
                return Ok(refused(no_function(&vertex_name)));
            };
            let Some(fragment) =
                specialised::<P>(&fragment_library, &fragment_name, &fragment_constants)?
            else {
                return Ok(refused(no_function(&fragment_name)));
            };
            let desc = RenderPipelineDesc::<P> {
                label: label.as_deref(),
                vertex: &vertex,
                fragment: &fragment,
                color0_pixel_format,
                blend_over,
                resource_index: None,
            };
            // `new_render_pipeline_state` returns a plain `Result` (no path reaches it under a
            // handler in the Apple lane); a nil state or an `NSError` refuses.
            Ok(match device.new_render_pipeline_state(&desc) {
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
        } => {
            let Some(library) = library.resolve() else {
                return Ok(refused(no_library("kernel")));
            };
            let Some(function) = specialised::<P>(&library, &name, &constants)? else {
                return Ok(refused(no_function(&name)));
            };
            Ok(
                match device.new_compute_pipeline_state_for_function(Some(&function)) {
                    Ok(Some(pipeline)) => {
                        let reflection = compute_reflection::<P>(&pipeline, seqno, obj_id);
                        CompiledPipeline::made(
                            obj_id,
                            ObjHandle::ComputePipeline(pipeline),
                            reflection,
                        )
                    }
                    Ok(None) => refused("the platform answered nil".to_owned()),
                    Err(e) => refused(format!("the platform answered an error: {}", e.description)),
                },
            )
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

fn str_tail(message: &[u8], offset: u32, len: u32) -> Option<&str> {
    let start = offset as usize;
    let end = start.checked_add(len as usize)?;
    std::str::from_utf8(message.get(start..end)?).ok()
}

/// Whether a colour attachment's blend state is the premultiplied source-over
/// blend the platform's [`RenderPipelineDesc`] value can express: blending on,
/// both source factors `One`, both destination factors `OneMinusSourceAlpha`,
/// both operations `Add`. Any other blend state cannot travel through that value
/// and is dropped (the render decoder's note).
fn is_source_over(c: &RgfxColorAttachmentDesc) -> bool {
    /// `MTLBlendFactorOne`.
    const ONE: u32 = 1;
    /// `MTLBlendFactorOneMinusSourceAlpha`.
    const ONE_MINUS_SRC_ALPHA: u32 = 5;
    /// `MTLBlendOperationAdd`.
    const ADD: u32 = 0;
    c.blending_enabled != 0
        && c.source_rgb_blend_factor == ONE
        && c.source_alpha_blend_factor == ONE
        && c.destination_rgb_blend_factor == ONE_MINUS_SRC_ALPHA
        && c.destination_alpha_blend_factor == ONE_MINUS_SRC_ALPHA
        && c.rgb_blend_operation == ADD
        && c.alpha_blend_operation == ADD
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
        RGFX_LIBRARY_BYTES_OFFSET, RgfxLibraryCreate, RgfxLibraryFromRegion,
    };
    use reims_vgpu_core::testkit::{Arg, FakePlatform};
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
                .filter(|s| **s == "newComputePipelineStateWithFunction:error:")
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
