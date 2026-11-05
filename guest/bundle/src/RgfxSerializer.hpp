/*
 * RgfxSerializer: turns the guest's Metal resource create/destroy calls into the
 * reims-graphics wire messages and writes them onto a submission ring
 * (docs/graphics-kext/design.md, "Stage 3 wire protocol: the Metal-call stream";
 * the host decoder it mirrors is crates/reims-vgpu/src/graphics/decode.rs).
 *
 * This is the first serialiser slice: the object lifecycle whose wire structs are
 * fixed in crates/reims-graphics/src/wire.rs (reims_graphics_wire.h) — command
 * queue, heap, buffer, texture, sampler, depth/stencil, library, event, shared
 * event, fence, and OBJECT_DESTROY. The guest picks each ObjId, GPU address and
 * resource id (RgfxIdSpace); the host pins them. Lifecycle messages go on the
 * control ring (ring 0); each is fire-and-forget (no reply) in this slice.
 *
 * Not in this slice (flagged, not guessed):
 *  - COMMAND_BUFFER encoder-op stream: the op schema is being finalised host-side.
 *  - RENDER/COMPUTE_PIPELINE_CREATE: opcodes exist but the descriptor structs are
 *    "later", and the creates block for a reflection reply.
 *
 * Our own code; depends only on the committed wire header and the two helpers
 * above. No Metal SPI here — the Metal device shell (built later) translates each
 * MTLDescriptor into these plain fields and calls us.
 */

#ifndef RGFX_SERIALIZER_HPP_
#define RGFX_SERIALIZER_HPP_

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "RgfxKnownPipelines.hpp"
#include "RgfxObjectModel.hpp"
#include "RgfxRingWriter.hpp"
#include "reims_graphics_wire.h"

class RgfxIcbSlotTable;

/// A sampler's state, as MTLSamplerDescriptor gives it. The LOD clamps are
/// floats here; the serialiser stores their bits on the wire (so it stays
/// all-integer), as RgfxSamplerCreate asks.
struct RgfxSamplerParams {
	uint32_t min_filter;
	uint32_t mag_filter;
	uint32_t mip_filter;
	uint32_t s_address_mode;
	uint32_t t_address_mode;
	uint32_t r_address_mode;
	uint32_t border_color;
	uint32_t compare_function;
	uint32_t max_anisotropy;
	float lod_min_clamp;
	float lod_max_clamp;
	uint32_t flags; // bit 0 normalized coords, bit 1 LOD average, bit 2 argument-buffer usable
};

/// A function reference a pipeline draws from: the library it lives in, its
/// entry-point name, and the function constants that specialise it (`constants`
/// may be null when `constant_count` is 0). The name is a NUL-terminated C
/// string; the serialiser copies `strlen(entry_point)` bytes to the wire.
struct RgfxFunctionRef {
	RgfxObjId library;
	const char *entry_point;
	const RgfxFunctionConstant *constants;
	uint32_t constant_count;
};

/// One function a pipeline stage links (MTLLinkedFunctions): the function, the list
/// it is in (an RGFX_LINKED_*), the MTLFunctionOptions it was made with, for
/// RGFX_LINKED_GROUP the group's name (null otherwise), and the specializedName it was
/// made with (null for none).
struct RgfxLinkedFunctionRef {
	RgfxFunctionRef function;
	uint32_t list;
	uint32_t options;
	const char *group;
	const char *specialized_name;
};

/// A render pipeline's descriptor, as MTLRenderPipelineDescriptor gives it, in
/// plain fields the serialiser packs into RgfxRenderPipelineCreate. `colors[0..
/// color_attachment_count)` are in use; the vertex descriptor's `layouts` and
/// `attributes`, and each stage's linked functions, may be null when their counts
/// are 0; `label` may be null.
struct RgfxRenderPipelineParams {
	RgfxFunctionRef vertex;
	RgfxFunctionRef fragment;
	const char *label;
	uint32_t raster_sample_count;
	uint32_t color_attachment_count; // <= 8
	RgfxColorAttachmentDesc colors[8];
	uint32_t depth_pixel_format;
	uint32_t stencil_pixel_format;
	const RgfxVertexBufferLayout *vertex_layouts;
	uint32_t vertex_layout_count;
	const RgfxVertexAttribute *vertex_attributes;
	uint32_t vertex_attribute_count;
	uint32_t flags; // RGFX_RENDER_PIPELINE_*
	uint32_t input_primitive_topology;
	uint32_t max_vertex_amplification_count;
	const RgfxLinkedFunctionRef *vertex_linked;
	uint32_t vertex_linked_count;
	const RgfxLinkedFunctionRef *fragment_linked;
	uint32_t fragment_linked_count;
};

/// A compute pipeline's descriptor in plain fields the serialiser packs into
/// RgfxComputePipelineCreate. The stage-input descriptor's `layouts` and
/// `attributes` (their step functions MTLStepFunction, their formats
/// MTLAttributeFormat) and the linked functions may be null when their counts are 0.
struct RgfxComputePipelineParams {
	RgfxFunctionRef function;
	uint32_t flags; // RGFX_COMPUTE_PIPELINE_*
	uint32_t max_total_threads_per_threadgroup; // 0 leaves Metal's
	uint32_t max_call_stack_depth;
	const RgfxVertexBufferLayout *stage_input_layouts;
	uint32_t stage_input_layout_count;
	const RgfxVertexAttribute *stage_input_attributes;
	uint32_t stage_input_attribute_count;
	uint32_t stage_input_index_type;
	uint32_t stage_input_index_buffer_index;
	const RgfxLinkedFunctionRef *linked;
	uint32_t linked_count;
};

/// What a create did. `obj_id`, `gpu_address` and `resource_id` are the values
/// the guest picked (valid whether or not the message reached the ring, so a
/// caller that retries keeps the same handle); `gpu_address`/`resource_id` are 0
/// where the create has none. `seqno` is the ring seqno when `posted`. `posted`
/// is false when the ring had no room — the caller retries the create. `status` is
/// the device's COMPLETE status for a create the device answers through the ring's
/// forward (a library's), kRgfxNoAnswer where none came back; 0 for the others.
/// A texture create with no resource id left (RgfxIdSpace::allocResourceId) posts
/// nothing and answers every field 0: the caller makes no texture.
/// A `TEXTURE_FROM_IOSURFACE` as the bundle sends it: the struct, then the surface's facts at
/// RGFX_SURFACE_FACTS_OFFSET.
struct RgfxSurfaceTextureMessage {
	RgfxTextureFromIOSurface texture;
	RgfxSurfaceFacts facts;
};

static_assert(offsetof(RgfxSurfaceTextureMessage, facts) == RGFX_SURFACE_FACTS_OFFSET,
              "the facts follow the struct, where the host reads them");
static_assert(sizeof(RgfxSurfaceTextureMessage) % 8 == 0, "a ring message is whole words");

/// What one flushIcbWrites() posted: its ICB_WRITE messages, their bytes, and how long their
/// posts took, in nanoseconds.
struct RgfxIcbFlush {
	uint32_t messages;
	uint64_t bytes;
	uint64_t nanoseconds;
};

struct RgfxCreated {
	RgfxObjId obj_id;
	uint64_t gpu_address;
	uint64_t resource_id;
	uint64_t seqno;
	bool posted;
	uint32_t status;
};

/// A pipeline create's post against the process's known pipelines (RgfxKnownPipelines): where
/// the route defers answers (RgfxRingWriter::defers), the create is keyed, and one whose key is
/// known is posted without waiting for its answer.
struct RgfxPipelinePost {
	/// In: the process's known pipelines; null keys nothing, and every create waits.
	RgfxKnownPipelines *known = nullptr;
	/// Out: whether the create was keyed, and its key, under which its answer is learnt.
	bool keyed = false;
	RgfxPipelineKey key = {};
	/// Out: whether it was posted without waiting, its own answer to come later; `reflection`
	/// is then the equal create's, which the new state takes at once.
	bool deferred = false;
	RgfxPipelineReflection reflection = {};
};

class RgfxSerializer {
public:
	/// Serialises onto `control`, picking ids from `ids`. Both outlive the
	/// serialiser. `control` is the device's ring 0.
	RgfxSerializer(RgfxRingWriter &control, RgfxIdSpace &ids);

	// --- queues ---------------------------------------------------------------

	/// A command queue and its ring pair. The ring offsets, sizes, priority and
	/// vector are the caller's (the region allocator that places the rings is a
	/// later piece); the queue's own ObjId is picked here.
	RgfxCreated createQueue(uint64_t submit_offset, uint64_t complete_offset,
	                        uint32_t submit_data_size, uint32_t complete_data_size,
	                        uint32_t priority, uint32_t vector);

	/// Where queues open: `open` hands the device the queue a createQueue() describes (the
	/// accelerator kext's QUEUE_CREATE), and answers whether the device took it. With one
	/// set, createQueue() posts nothing on the ring. Called with `context`.
	typedef bool (*QueueOpenFn)(void *context, const RgfxQueueCreate &queue);
	void setQueueOpener(QueueOpenFn open, void *context);

	/// Where queues close: `close` hands the device the end of the queue `obj_id` (the
	/// accelerator kext's QUEUE_DESTROY, which stops watching its completion ring and posts its
	/// destroy). With one set, destroyQueue() posts nothing on the ring. Called with `context`.
	typedef void (*QueueCloseFn)(void *context, RgfxObjId obj_id);
	void setQueueCloser(QueueCloseFn close, void *context);

	/// Ends the queue `obj_id` on the device: through the closer where one is set, else an
	/// OBJECT_DESTROY on the ring. The device takes neither of its rings after it.
	void destroyQueue(RgfxObjId obj_id);

	/// Where the device's answers to queries come from: `ask` hands the device one whole
	/// message (its header's `type` and `size` stamped) on the control ring and waits for its
	/// answer, which it copies to `reply` (at most `capacity` bytes, its length in
	/// `*reply_size`): in the bundle, the accelerator kext's submit, which stamps the seqno.
	/// False where nothing came back. Called with `context`. With none set, a query has no
	/// answer.
	typedef bool (*AskFn)(void *context, const void *message, uint32_t size, void *reply,
	                      uint32_t capacity, uint32_t *reply_size);
	void setAsker(AskFn ask, void *context);

	// --- queries --------------------------------------------------------------

	/// The size and alignment the host's device gives a texture of shape `desc` in a heap
	/// (HEAP_TEXTURE_SIZE_AND_ALIGN, under a new id that names nothing): true with them in
	/// `*size` and `*align` where the device answered, false where no answer came back or the
	/// device refused the shape. The host's device answers a shape alike each time, so each
	/// answered shape is asked once and its answer kept; a refused one is asked again.
	bool heapTextureSizeAndAlign(const RgfxTextureDescriptor &desc, uint64_t *size,
	                             uint64_t *align);

	/// ARGUMENT_LAYOUT of `count` argument descriptors, in their order, under a new id that
	/// names nothing: the device writes the layout into the `reply_length` bytes at
	/// `reply_offset`, a block of the caller's, and answers its COMPLETE. Answers its status
	/// (RGFX_STATUS_OK; RGFX_STATUS_TOO_SMALL, with the bytes needed written; a refusal), or
	/// kRgfxNoAnswer where no answer came back.
	uint32_t argumentLayoutOfDescriptors(const RgfxArgumentDescriptor *descriptors, uint32_t count,
	                                     uint64_t reply_offset, uint64_t reply_length);

	/// ARGUMENT_LAYOUT of the argument buffer function `entry_point` of library `library`
	/// takes at `buffer_index`, specialised by `constant_count` constants where there are any;
	/// answered as argumentLayoutOfDescriptors.
	uint32_t argumentLayoutOfFunction(RgfxObjId library, const char *entry_point,
	                                  const RgfxFunctionConstant *constants,
	                                  uint32_t constant_count, uint32_t buffer_index,
	                                  uint64_t reply_offset, uint64_t reply_length);

	// --- heaps ----------------------------------------------------------------

	/// A device heap of `size_bytes`. `options` is MTLResourceOptions, `heap_type`
	/// MTLHeapType. A GPU address of `size_bytes` aligned to `align` is picked. This and the
	/// three creates after it post nothing, and answer an obj_id and a gpu_address of 0, where
	/// the context's range has no room left for the address.
	RgfxCreated createHeap(uint64_t size_bytes, uint32_t options, int32_t heap_type,
	                       uint64_t align = kDefaultAddressAlign);

	/// A Shared heap over the region block at `region_offset`.
	RgfxCreated createHeapOverHostMemory(uint64_t size_bytes, uint32_t options, int32_t heap_type,
	                                     uint64_t region_offset,
	                                     uint64_t align = kDefaultAddressAlign);

	/// A Shared heap over the guest RAM of the pages block of `block_size` bytes at
	/// `block_offset`, whose pages are one run (HEAP_OVER_PAGES, where the device took
	/// RGFX_FEATURE_HEAP_OVER_PAGES).
	RgfxCreated createHeapOverPages(uint64_t size_bytes, uint32_t options, int32_t heap_type,
	                                uint64_t block_offset, uint64_t block_size,
	                                uint64_t align = kDefaultAddressAlign);

	// --- buffers --------------------------------------------------------------

	/// A device buffer of `length` bytes at a picked GPU address.
	RgfxCreated createBuffer(uint64_t length, uint32_t options,
	                         uint64_t align = kDefaultAddressAlign);

	/// A Shared buffer over the region block at `region_offset`.
	RgfxCreated createBufferOverHostMemory(uint64_t length, uint32_t options,
	                                       uint64_t region_offset,
	                                       uint64_t align = kDefaultAddressAlign);

	/// A Shared buffer of `length` bytes over the IOSurface pages of the shared block at
	/// `block_offset`, whose part its holders map is `block_size` bytes. No GPU address: the
	/// create names none.
	RgfxCreated createBufferOverSurfacePages(uint64_t length, uint32_t options,
	                                         uint64_t block_offset, uint64_t block_size);

	/// A Shared buffer of `length` bytes at `offset` in the view of the pages block of
	/// `block_size` bytes at `block_offset` (RgfxRegion::acquirePages), `offset` the app
	/// pointer's in its first page, at a GPU address that keeps it, as Metal's no-copy buffer's
	/// address keeps its pointer's offset in its page. The answer's `gpu_address` is the address
	/// picked, given back through releaseAddress; the buffer's is that plus `offset`.
	RgfxCreated createBufferOverPages(uint64_t length, uint32_t options, uint64_t block_offset,
	                                  uint64_t block_size, uint64_t offset);

	/// A buffer at `offset` in the heap `heap`.
	RgfxCreated createBufferAtOffset(RgfxObjId heap, uint64_t offset, uint64_t length,
	                                 uint32_t options);

	/// A Managed buffer of `length` bytes at a picked GPU address, its CPU copy the bytes at
	/// `offset` in the heap over host memory `heap` (`BUFFER_MANAGED_AT_OFFSET`, where the
	/// device took RGFX_FEATURE_MANAGED_AT_OFFSET).
	RgfxCreated createBufferManagedAtOffset(RgfxObjId heap, uint64_t offset, uint64_t length,
	                                        uint32_t options,
	                                        uint64_t align = kDefaultAddressAlign);

	/// A Managed buffer of `length` bytes at a picked GPU address, its CPU copy the first `length`
	/// bytes of the Shared buffer `cpu_buffer` over the process's memory (`MANAGED_BUFFER_CREATE`,
	/// where the device took RGFX_FEATURE_MANAGED_BUFFER_SLABS).
	RgfxCreated createManagedBuffer(RgfxObjId cpu_buffer, uint64_t length, uint32_t options,
	                                uint64_t align = kDefaultAddressAlign);

	/// The process wrote `length` bytes at `location` of Managed buffer `buffer`'s CPU copy
	/// (`didModifyRange:`): every command buffer committed after it sees them. False where the
	/// message was not posted.
	bool bufferDidModifyRange(RgfxObjId buffer, uint64_t location, uint64_t length);

	// --- textures -------------------------------------------------------------

	/// Gives a texture's resource id back as the texture goes (RgfxIdSpace::freeResourceId).
	void releaseResourceId(uint64_t resource_id);

	/// Gives the GPU address a create picked back, once the object's destroy is posted
	/// (RgfxIdSpace::freeAddress).
	void releaseAddress(uint64_t gpu_address);

	/// Whether the device never got the create `created` describes (its message went nowhere:
	/// the kext refused it, or the ring had no room), so no object of that id exists there: the
	/// caller makes none, as Metal answers nil for an object it cannot make. Gives back the GPU
	/// address and resource id it picked then, as no destroy will follow.
	bool refused(const RgfxCreated &created);

	/// A texture of shape `desc`. When `heap` is RGFX_OBJ_NONE the texture is
	/// device-placed at a picked GPU address, whose range holds `span` bytes (its
	/// allocatedSize), as a buffer's holds its length; otherwise it lies at
	/// `heap_offset` in `heap`. A resource id is picked either way.
	RgfxCreated createTexture(const RgfxTextureDescriptor &desc, RgfxObjId heap = RGFX_OBJ_NONE,
	                          uint64_t heap_offset = 0, uint64_t align = kDefaultAddressAlign,
	                          uint64_t span = 1);

	/// A Managed texture of shape `desc` at a picked GPU address, whose CPU copy is the Shared
	/// buffer `cpu_buffer` over a block, of blocks `bytes_per_block` bytes and `block_width` by
	/// `block_height` texels (RgfxManagedLayout.hpp), `flags` RGFX_MANAGED_TEXTURE_LINEAR or 0,
	/// the address's range `span` bytes (its allocatedSize). A resource id is picked.
	RgfxCreated createManagedTexture(const RgfxTextureDescriptor &desc, RgfxObjId cpu_buffer,
	                                 uint32_t bytes_per_block, uint32_t block_width,
	                                 uint32_t block_height, uint32_t flags,
	                                 uint64_t align = kDefaultAddressAlign, uint64_t span = 1);

	/// The process wrote the texels of `size` at `origin` of `level` and `slice` of Managed
	/// texture `texture`'s CPU copy (`replaceRegion:`): every command buffer committed after it
	/// sees them. False where the message was not posted.
	bool textureReplaceRegion(RgfxObjId texture, uint32_t level, uint32_t slice,
	                          const uint32_t origin[3], const uint32_t size[3]);

	/// A view of a texture (RGFX_CMD_TEXTURE_VIEW): `view`'s parent, format, flags, type, levels,
	/// slices and swizzle, with its id and resource id picked; `view`'s own are not read.
	RgfxCreated createTextureView(const RgfxTextureView &view);

	/// A texture of shape `desc` over the buffer `buffer` from `offset`, each row
	/// `bytes_per_row` apart. A resource id is picked.
	RgfxCreated createTextureFromBuffer(RgfxObjId buffer, uint64_t offset, uint64_t bytes_per_row,
	                                    const RgfxTextureDescriptor &desc);

	/// As createTextureFromBuffer, by newLinearTextureWithDescriptor:offset:bytesPerRow:
	/// bytesPerImage:'s form (RGFX_CMD_LINEAR_TEXTURE_FROM_BUFFER), which takes a 2D array, with
	/// the bytes between its slices the guest asked, `bytes_per_image`.
	RgfxCreated createLinearTextureFromBuffer(RgfxObjId buffer, uint64_t offset,
	                                          uint64_t bytes_per_row, uint64_t bytes_per_image,
	                                          const RgfxTextureDescriptor &desc);

	/// A texture of shape `desc` over plane `plane` of the IOSurface `iosurface_id`,
	/// lying over the buffer `buffer` (this process's copy of the surface) from
	/// `offset`, each row `bytes_per_row` apart, the copy's writers (RgfxSurfaceWriters)
	/// `writers_offset` bytes into the buffer (RGFX_SURFACE_NO_WRITERS for none), the
	/// surface's `facts` after it (RgfxSurfaceTextureMessage). A resource id is picked.
	RgfxCreated createTextureFromIOSurface(RgfxObjId buffer, uint64_t offset,
	                                       uint64_t bytes_per_row, uint32_t iosurface_id,
	                                       uint32_t plane, const RgfxTextureDescriptor &desc,
	                                       uint64_t writers_offset,
	                                       const RgfxSurfaceFacts &facts);

	// --- state objects --------------------------------------------------------

	/// A sampler state. With an asker set, the device answers the create (SAMPLER_STATE):
	/// `status` is its status, and `resource_id` the sampler's gpuResourceID, the id the host's
	/// device numbers it by; `posted` is false where no answer came back. Without one the create
	/// goes on the ring unanswered, with no resource id.
	RgfxCreated createSampler(const RgfxSamplerParams &params);

	/// A depth/stencil state.
	RgfxCreated createDepthStencil(uint32_t depth_compare, bool depth_write_enabled,
	                               const RgfxStencilDescriptor &front,
	                               const RgfxStencilDescriptor &back);

	// --- shaders --------------------------------------------------------------

	/// A library from `byte_length` bytes of .metallib or AIR at `bytes`,
	/// forwarded unchanged, with their content key where `key` (32 bytes) is not
	/// null. `posted` is false when the message (its header plus the bytes, rounded
	/// to a multiple of 8) does not fit the ring; `status` is the device's answer.
	RgfxCreated createLibrary(const void *bytes, uint64_t byte_length,
	                          const uint8_t *key = nullptr);

	/// A LIBRARY_LOOKUP of the library whose `slice_size` bytes have the content
	/// key `key` (32 bytes), under a new id: `status` RGFX_STATUS_OK where the device
	/// holds it, and the id names it; any other where it does not, and the caller
	/// makes the library from its bytes (kRgfxNoAnswer where no answer came back).
	RgfxCreated lookupLibrary(const uint8_t *key, uint64_t slice_size);

	/// A new object id for a library streamed in chunks (postLibraryChunk), which
	/// every chunk of that library names.
	RgfxObjId allocLibraryId();

	/// One LIBRARY_FROM_REGION chunk of library `obj_id`: the `byte_length` bytes
	/// at `chunk_offset` of its `total_length`, which the caller copied to
	/// `region_offset` in the device's region. The host writes `ack_value` at
	/// `ack_offset` once it has copied them out. Returns whether the message posted
	/// (false when the ring has no room). `key` (32 bytes) is the library's content
	/// key, or null. The library's last chunk is answered once the library is made:
	/// `*out_status` (when not null) is the answer, or kRgfxNoAnswer.
	bool postLibraryChunk(RgfxObjId obj_id, uint64_t region_offset, uint64_t byte_length,
	                      uint64_t total_length, uint64_t chunk_offset, uint64_t ack_offset,
	                      uint64_t ack_value, const uint8_t *key = nullptr,
	                      uint32_t *out_status = nullptr);

	// --- pipelines ------------------------------------------------------------

	/// A render pipeline from `params`. The functions, label and vertex
	/// descriptor are packed into the message tail; the colour attachments are
	/// inline. The create is answered by the host with an RgfxPipelineReflection
	/// carrying the returned `seqno`, read off the queue's completion ring (the
	/// reply-ring reader is a later transport piece; `RgfxCreated::seqno` is the
	/// key the reply is matched by). `posted` is false when the message does not
	/// fit the ring. With `post`, the create is posted against the known pipelines
	/// (RgfxPipelinePost).
	RgfxCreated createRenderPipeline(const RgfxRenderPipelineParams &params,
	                                 RgfxPipelinePost *post = nullptr);

	/// A compute pipeline from `params`, answered as createRenderPipeline is.
	RgfxCreated createComputePipeline(const RgfxComputePipelineParams &params,
	                                  RgfxPipelinePost *post = nullptr);

	/// Takes the answers the route keeps for the process's deferred pipeline creates, waiting
	/// first for the one at `seqno` (RgfxRingWriter::awaitAnswer): false where it keeps none.
	bool awaitAnswer(uint64_t seqno) { return m_control.awaitAnswer(seqno); }

	// --- sync objects ---------------------------------------------------------

	/// An event.
	RgfxCreated createEvent();
	/// A fence.
	RgfxCreated createFence();
	/// A shared event starting at `initial_value`.
	RgfxCreated createSharedEvent(uint64_t initial_value);
	/// The CPU set shared event `event` to `value` (`setSignaledValue:`); the host's only rises.
	/// False where the message was not posted.
	bool setSharedEventValue(RgfxObjId event, uint64_t value);

	// --- indirect command buffers ---------------------------------------------

	/// An indirect command buffer of the descriptor `desc` carries (its header and obj_id are
	/// stamped here, and the obj_id picked). With an asker set, the device answers the create
	/// (ICB_STATE): `status` is its status, and `resource_id` the buffer's gpuResourceID, the id
	/// the host's device numbers it by; `posted` is false where no answer came back. Without one
	/// the create goes on the ring unanswered, with no resource id.
	RgfxCreated createIcb(const RgfxIcbCreate &desc);

	/// One CPU encode into the indirect command buffer `icb`: a record of `op` (an
	/// RGFX_ICB_OP_*) on command `index` (a reset range's location), its body the `size` bytes
	/// at `body` (a multiple of 8, 0 for none). It waits with the device's other records, in
	/// call order, until flushIcbWrites() or a destroy() posts them, or until its buffer's
	/// message would pass the largest the control ring takes, when that message goes at once.
	void icbRecord(RgfxObjId icb, uint16_t op, uint32_t index, const void *body, uint16_t size);

	/// Posts every waiting write: the slots written since the last flush (ICB_SLOTS, their runs,
	/// RgfxIcbSlotTable.hpp), then the records (ICB_WRITE, one message for each buffer's run of
	/// records), so the device carries them out before any command buffer the process posts after
	/// this returns. One load where none waits.
	RgfxIcbFlush flushIcbWrites();

	/// The slots `slots` were written since the last flush: they go on the list the next flush
	/// takes, once a flush cycle (as their `pending` flag turns on).
	void icbSlotsPending(RgfxIcbSlotTable *slots);

	/// Posts what `slots` were written since the last flush, now, ahead of the records that
	/// follow: their buffer turned to records (RgfxIcbSlotTable::toRecords).
	void flushIcbSlots(RgfxIcbSlotTable *slots);

	/// `slots` go, their buffer's destroy posted: off the list, and their bytes given back.
	void forgetIcbSlots(RgfxIcbSlotTable *slots);

	/// The most bytes of slots the process's buffers hold together: under a tenth of the region
	/// room an app has (about 400 MiB), so slots never starve its Shared buffers. A buffer whose
	/// slots would pass it takes records.
	static constexpr uint64_t kIcbSlotBytes = 32ull << 20;

	/// Room for `bytes` more of slots inside kIcbSlotBytes: taken, where it fits.
	bool reserveIcbSlotBytes(uint64_t bytes);
	/// Gives back `bytes` reserveIcbSlotBytes took.
	void releaseIcbSlotBytes(uint64_t bytes);

	// --- video ----------------------------------------------------------------

	/// A video session of codec `codec` (RGFX_VIDEO_CODEC_*) in direction `direction`
	/// (RGFX_VIDEO_DECODE or RGFX_VIDEO_ENCODE), which video encoders name. Not answered.
	RgfxCreated createVideoSession(uint32_t codec, uint32_t direction);

	// --- destroy --------------------------------------------------------------

	/// Drops the host handle for `obj_id` (an object of any kind, a queue
	/// included: the host handle table knows the kind), after the waiting ICB records, which may
	/// name it, then runs `after` (when set): what the host object lay over (its GPU address, its
	/// resource id, its heap's bytes) goes back once the host is told to drop it. Where the
	/// destroy does not reach the ring (the kext refused it, as it does where its control ring
	/// stays full), the host still holds the object: the destroy and its `after` wait, behind any
	/// destroy waiting before them, and go out at the next destroy() or postWaitingDestroys(), so
	/// nothing reuses what the object lay over until the host has dropped it. Returns whether
	/// this destroy reached the ring now.
	bool destroy(RgfxObjId obj_id, std::function<void()> after = {});

	/// Posts the destroys waiting (destroy()), in the order they were made, each one's `after`
	/// run once it is on the ring, until one is refused again. One load where none waits.
	/// Returns whether none waits now.
	bool postWaitingDestroys();

	/// The default alignment for a picked GPU address when the caller names none.
	/// The caller passes a caps-derived alignment where Metal demands one; this
	/// is a safe floor for the first slice.
	static constexpr uint64_t kDefaultAddressAlign = 256;

private:
	RgfxRingWriter &m_control;
	RgfxIdSpace &m_ids;
	QueueOpenFn m_queueOpen = nullptr;
	void *m_queueOpenContext = nullptr;
	QueueCloseFn m_queueClose = nullptr;
	void *m_queueCloseContext = nullptr;
	AskFn m_ask = nullptr;
	void *m_askContext = nullptr;

	// heapTextureSizeAndAlign's answers, by the shape's bytes, under m_sizesLock.
	typedef std::array<uint8_t, sizeof(RgfxTextureDescriptor)> SizeKey;
	struct SizeKeyHash {
		size_t operator()(const SizeKey &key) const;
	};
	struct SizeAnswer {
		uint64_t size;
		uint64_t align;
	};
	std::mutex m_sizesLock;
	std::unordered_map<SizeKey, SizeAnswer, SizeKeyHash> m_sizes;

	// The destroys the ring did not take (destroy()), in the order they were made, under
	// m_destroyLock, each posted under it so they reach the ring in that order; m_destroysWaiting
	// says whether any waits.
	struct WaitingDestroy {
		RgfxObjId obj;
		std::function<void()> after;
	};
	std::mutex m_destroyLock;
	std::atomic<bool> m_destroysWaiting{false};
	std::deque<WaitingDestroy> m_destroys;
	// Posts OBJECT_DESTROY of `obj_id`: whether it reached the ring.
	bool postDestroy(RgfxObjId obj_id);

	// The ICB records waiting for flushIcbWrites(), under m_icbLock: one ICB_WRITE message for
	// each buffer's run of records, in the order the runs began. m_icbPending is set as a record
	// is added, and cleared only once a flush has posted them all, so a flush that finds it
	// clear has nothing to wait for, and one that finds it set waits for a flush in progress.
	struct IcbMessage {
		RgfxObjId icb;
		std::vector<uint8_t> bytes; // the RgfxIcbWrite, then the records
	};
	std::mutex m_icbLock;
	std::atomic<bool> m_icbPending{false};
	std::vector<IcbMessage> m_icbMessages;
	// The slots written since the last flush, under m_icbLock (icbSlotsPending), and the bytes
	// of slots the process's buffers hold.
	std::vector<RgfxIcbSlotTable *> m_icbSlots;
	std::atomic<uint64_t> m_icbSlotBytes{0};
	// The largest message the control ring takes: a forwarding writer's whole size, a ring's
	// half (RgfxRingWriter::fits).
	uint32_t icbMessageCap() const;
	// Posts `message` (under m_icbLock), adding to `*flush`.
	void postIcbMessage(IcbMessage &message, RgfxIcbFlush *flush);
	// Posts `runs` as ICB_SLOTS, as few as the ring's cap allows (under m_icbLock), adding to
	// `*flush` where given.
	void postIcbSlots(const std::vector<RgfxIcbSlotRun> &runs, RgfxIcbFlush *flush);
};

#endif // RGFX_SERIALIZER_HPP_
