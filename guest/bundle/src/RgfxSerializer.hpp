/*
 * RgfxSerializer: turns the guest's Metal resource create/destroy calls into the
 * reims-graphics wire messages and writes them onto a submission ring
 * (docs/graphics-kext/design.md, "Stage 3 wire protocol: the Metal-call stream";
 * the host decoder it mirrors is crates/reims-vgpu/src/graphics/decode.rs).
 *
 * This is the first serialiser slice: the object lifecycle whose wire structs are
 * fixed in crates/reims-vgpu-graphics/src/wire.rs (reims_graphics_wire.h) — command
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

/// The MTLResourceOptions bits a heap or buffer create carries to the host: the CPU cache mode
/// (bits 0-3), the storage mode (4-7), the hazard tracking mode (8-9), and Metal's private bit 24,
/// which the host takes as Metal does. Any other bit is a hint from the caller to the device's
/// driver, which the bundle is, and the bundle takes none; the host's runtime refuses a buffer
/// with a bit Metal gives no meaning (measured, macOS 26.6.1 on an RTX 5080 host: the GL
/// renderer's no-copy buffers, made with bit 20, 0x100000, by GLDBufferRec::load). A buffer
/// answers its three modes alone, as the x86 Macs' do (RgfxBuffer's resourceOptions).
constexpr uint32_t kRgfxHostResourceOptions = 0xfu | (0xfu << 4) | (0x3u << 8) | (1u << 24);

/// `options` as a create carries them to the host: the bits kRgfxHostResourceOptions names.
constexpr uint32_t RgfxHostResourceOptions(uint32_t options)
{
	return options & kRgfxHostResourceOptions;
}

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
	uint32_t reduction_mode; // MTLSamplerReductionMode, SAMPLER_CREATE's extension
	float lod_bias;          // lodBias, SAMPLER_CREATE's extension
};

/// SAMPLER_CREATE as the serialiser sends it: the create and its extension
/// (RGFX_SAMPLER_CREATE_EXT_OFFSET), the reduction mode and the LOD bias.
struct RgfxSamplerCreateWhole {
	RgfxSamplerCreate create;
	RgfxSamplerCreateExt ext;
};
static_assert(offsetof(RgfxSamplerCreateWhole, ext) == RGFX_SAMPLER_CREATE_EXT_OFFSET &&
                  sizeof(RgfxSamplerCreateWhole) ==
                      RGFX_SAMPLER_CREATE_EXT_OFFSET + sizeof(RgfxSamplerCreateExt),
              "SAMPLER_CREATE and its extension");

/// MANAGED_BUFFER_CREATE as the serialiser sends it: the create and the bytes of its CPU copy
/// that read zero (RGFX_MANAGED_BUFFER_ZERO_OFFSET), from `zero_from` to the length; a
/// `zero_from` of the length says nothing of them.
struct RgfxManagedBufferCreateWhole {
	RgfxManagedBufferCreate create;
	RgfxManagedBufferZero zero;
};
static_assert(offsetof(RgfxManagedBufferCreateWhole, zero) == RGFX_MANAGED_BUFFER_ZERO_OFFSET &&
                  sizeof(RgfxManagedBufferCreateWhole) ==
                      RGFX_MANAGED_BUFFER_ZERO_OFFSET + sizeof(RgfxManagedBufferZero),
              "MANAGED_BUFFER_CREATE and its zero bytes");

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

/// One dynamic library a compute pipeline's kernel links (RgfxPipelineLibrary): dynamic library
/// `library`, preloaded where `install_name` is null, else the library that install name loads.
struct RgfxPipelineLibraryRef {
	RgfxObjId library;
	const char *install_name;
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
	/// The descriptor's private state, sent after the head with RGFX_RENDER_PIPELINE_PRIVATE_STATE;
	/// null for none.
	const RgfxRenderPipelinePrivateState *private_state;
	/// The tessellation state, each Metal's raw value (tessellationFactorScaleEnabled is
	/// RGFX_RENDER_PIPELINE_TESSELLATION_FACTOR_SCALE in `flags`).
	uint32_t tessellation_partition_mode;
	uint32_t max_tessellation_factor;
	uint32_t tessellation_factor_format;
	uint32_t tessellation_control_point_index_type;
	uint32_t tessellation_factor_step_function;
	uint32_t tessellation_output_winding_order;
	uint32_t max_vertex_call_stack_depth;
	uint32_t max_fragment_call_stack_depth;
};

/// A mesh render pipeline's descriptor, as MTLMeshRenderPipelineDescriptor gives it, in plain
/// fields the serialiser packs into RgfxMeshRenderPipelineCreate: the object and fragment
/// functions may name none (library 0, null name); `colors[0..color_attachment_count)` are in use;
/// each stage's linked functions may be null when their counts are 0; `label` may be null. A
/// maximum, payload length or required size of 0 leaves Metal's.
struct RgfxMeshRenderPipelineParams {
	RgfxFunctionRef object;
	RgfxFunctionRef mesh;
	RgfxFunctionRef fragment;
	const char *label;
	uint32_t raster_sample_count;
	uint32_t color_attachment_count; // <= 8
	RgfxColorAttachmentDesc colors[8];
	uint32_t depth_pixel_format;
	uint32_t stencil_pixel_format;
	uint32_t flags; // RGFX_RENDER_PIPELINE_* and RGFX_MESH_PIPELINE_*
	uint32_t max_total_threads_per_object_threadgroup;
	uint32_t max_total_threads_per_mesh_threadgroup;
	uint32_t max_total_threadgroups_per_mesh_grid;
	uint32_t payload_memory_length;
	uint32_t max_vertex_amplification_count;
	uint32_t required_threads_per_object_threadgroup[3];
	uint32_t required_threads_per_mesh_threadgroup[3];
	const RgfxLinkedFunctionRef *object_linked;
	uint32_t object_linked_count;
	const RgfxLinkedFunctionRef *mesh_linked;
	uint32_t mesh_linked_count;
	const RgfxLinkedFunctionRef *fragment_linked;
	uint32_t fragment_linked_count;
};

/// A compute pipeline's descriptor in plain fields the serialiser packs into
/// RgfxComputePipelineCreate. The stage-input descriptor's `layouts` and
/// `attributes` (their step functions MTLStepFunction, their formats
/// MTLAttributeFormat), the linked functions and the dynamic libraries may be null when their
/// counts are 0.
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
	const RgfxPipelineLibraryRef *libraries; // where the host makes dynamic libraries alone
	uint32_t library_count;
};

/// What a create did. `obj_id`, `gpu_address` and `resource_id` are the values
/// the guest picked (valid whether or not the message reached the ring, so a
/// caller that retries keeps the same handle); `gpu_address`/`resource_id` are 0
/// where the create has none. `seqno` is the ring seqno when `posted`. `posted`
/// is false when the ring had no room — the caller retries the create. `status` is
/// the device's COMPLETE status for a create the device answers through the ring's
/// forward (a library's), kRgfxNoAnswer where none came back; 0 for the others.
/// A texture create with no resource id left (RgfxIdSpace::allocResourceId), or with no GPU
/// address left for one that takes an address, posts nothing and answers every field 0: the caller
/// makes no texture.
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

/// A rasterization rate map's descriptor (MTLRasterizationRateMapDescriptor) as RATE_MAP_CREATE
/// and RATE_MAP_RESET carry it: its screen size, its private mutability where set, and its layers
/// from index 0, each its sample counts and its samples, horizontal then vertical.
struct RgfxRateMapShape {
	struct Layer {
		uint32_t horizontal_count = 0;
		uint32_t vertical_count = 0;
		std::vector<float> samples; // horizontal_count horizontal ones, then the vertical ones
	};
	uint64_t screen_width = 0;
	uint64_t screen_height = 0;
	uint64_t screen_depth = 0;
	bool mutability_set = false;
	uint32_t mutability = 0;
	std::vector<Layer> layers;
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

	/// Where shared events cross processes (the accelerator kext's shared event ports,
	/// RgfxSharedEventUserClient.hpp): `share` names the process's shared event `event` for other
	/// processes and answers its port, a send right of the caller's, 0 where it gave none; `take`
	/// makes the process's new shared event `event` of the one `port` names, a send right the
	/// caller holds, answering the event's value in `*value`, false where it made none. Called
	/// with `context`. With none set, no event crosses.
	typedef uint32_t (*EventShareFn)(void *context, RgfxObjId event);
	typedef bool (*EventTakeFn)(void *context, uint32_t port, RgfxObjId event, uint64_t *value);
	void setEventSharer(EventShareFn share, EventTakeFn take, void *context);
	/// Whether events cross processes here: a sharer is set.
	bool sharesEvents() const { return m_eventShare != nullptr; }
	/// Whether render pipeline creates carry their descriptors' private state: the host took
	/// RGFX_FEATURE_PRIVATE_RENDER_STATE. Set once, before the first create; off by default.
	void setCarriesPrivateRenderState(bool carries) { m_privateRenderState = carries; }
	bool carriesPrivateRenderState() const { return m_privateRenderState; }
	/// Whether a texture's didModifyData reaches the host (textureDidModifyData): the host took
	/// RGFX_FEATURE_TEXTURE_DID_MODIFY_DATA. Set once, before the first texture; off by default.
	void setCarriesTextureDidModifyData(bool carries) { m_textureDidModifyData = carries; }
	bool carriesTextureDidModifyData() const { return m_textureDidModifyData; }
	/// Whether the host hears of a no-copy buffer's identity leaving with no message
	/// (bufferEscaped): the host took RGFX_FEATURE_BUFFER_ESCAPED. Set once, before the first
	/// buffer; off by default, and then no buffer notes its escape (RgfxBuffer.h, RgfxNoteEscape).
	void setCarriesBufferEscaped(bool carries) { m_bufferEscaped = carries; }
	bool carriesBufferEscaped() const { return m_bufferEscaped; }
	/// Whether the host makes rasterization rate maps (createRateMap, resetRateMap and
	/// queryRateMap): it took RGFX_FEATURE_RATE_MAPS. Set once, before the first map; off by
	/// default.
	void setCarriesRateMaps(bool carries) { m_rateMaps = carries; }
	bool carriesRateMaps() const { return m_rateMaps; }
	/// Whether render and compute pipeline creates ask for their pipeline keys
	/// (RGFX_*_PIPELINE_ANSWER_KEY) and are posted against the answers the kext keeps for every
	/// process (RgfxRingWriter::tryPostShared): the host took RGFX_FEATURE_PIPELINE_KEYS and the
	/// route shares answers. Set once, before the first create; off by default.
	void setSharesPipelines(bool shares) { m_sharesPipelines = shares; }
	bool sharesPipelines() const { return m_sharesPipelines; }
	/// Keeps library `obj_id`'s content key (32 bytes, RgfxLibraryKey.hpp), which the keys of the
	/// pipelines made of it name (rgfxSharedPipelineKey), until its destroy.
	void setLibraryKey(RgfxObjId obj_id, const uint8_t *key);
	/// Whether the host makes a compute pipeline's function handles and visible function tables
	/// (createFunctionHandle, createFunctionTable and setFunctionTable): it took
	/// RGFX_FEATURE_FUNCTION_POINTERS. Set once, before the first pipeline; off by default.
	void setCarriesFunctionPointers(bool carries) { m_functionPointers = carries; }
	bool carriesFunctionPointers() const { return m_functionPointers; }
	/// Whether the host makes dynamic libraries (createDynamicLibrary) and links them into a
	/// compute pipeline (RgfxComputePipelineParams::libraries): it took
	/// RGFX_FEATURE_DYNAMIC_LIBRARIES. Set once, before the first library; off by default.
	void setCarriesDynamicLibraries(bool carries) { m_dynamicLibraries = carries; }
	bool carriesDynamicLibraries() const { return m_dynamicLibraries; }
	/// Names shared event `event` for other processes: its port, 0 where none was given.
	uint32_t shareEvent(RgfxObjId event);
	/// The process's new shared event made of the one `port` names: its id in `obj_id` and the
	/// event's value in `*value`, `posted` false where none was made (the id then names nothing).
	RgfxCreated takeSharedEvent(uint32_t port, uint64_t *value);

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

	/// A buffer at `offset` in the heap `heap`. No one waits for it: where the control ring
	/// batches (RgfxRingWriter::postLater), it is held to go with the messages around it, before
	/// anything posted after it and before any command buffer committed after it
	/// (postHeldCreates), and `seqno` is 0.
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
	/// where the device took RGFX_FEATURE_MANAGED_BUFFER_SLABS), whose bytes from `zeroFrom` to
	/// the length read zero as it is posted (RgfxManagedBufferZero): the host's GPU copy, where
	/// it keeps one, starts with zeros there rather than the CPU copy's bytes. A `zeroFrom` of
	/// `length` or more says nothing of them (sent as `length`). The pick holds `offset` bytes
	/// more, and the buffer's address is the pick plus `offset`, as a no-copy buffer's over the
	/// app's pages keeps its pointer's offset in its page (createBufferOverPages). The answer's
	/// `gpu_address` is the address picked, given back through releaseAddress. No one waits for
	/// it: held where the control ring batches, as createBufferAtOffset's buffer is.
	RgfxCreated createManagedBuffer(RgfxObjId cpu_buffer, uint64_t length, uint32_t options,
	                                uint64_t zeroFrom, uint64_t align = kDefaultAddressAlign,
	                                uint64_t offset = 0);

	/// The process wrote `length` bytes at `location` of Managed buffer `buffer`'s CPU copy
	/// (`didModifyRange:`): every command buffer committed after it sees them. Held where the
	/// control ring batches, as createBufferAtOffset's buffer is, so a commit posts it first.
	/// False where the message was neither held nor posted.
	bool bufferDidModifyRange(RgfxObjId buffer, uint64_t location, uint64_t length);

	/// The process changed the bytes texture `texture` lies over (`didModifyData`, SPI): a
	/// mipmapped texture over a buffer takes them again at its next use in a command buffer
	/// committed after it. False where the message was not posted, and where the host does not
	/// carry it (carriesTextureDidModifyData), which posts nothing.
	bool textureDidModifyData(RgfxObjId texture);

	/// The identity of buffer `buffer`, a no-copy buffer the host reads in place, leaves the
	/// process's Metal with no message, by the act `kind` (RGFX_BUFFER_ESCAPED_*): a message no one
	/// waits for (postUnwaited), held or posted before anything posted after it and before any
	/// command buffer committed after it, so the host has it before any command buffer that could
	/// use the identity. False where it was neither held nor posted, and where the host does not
	/// carry it (carriesBufferEscaped), which posts nothing.
	bool bufferEscaped(RgfxObjId buffer, uint32_t kind);

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
	/// `heap_offset` in `heap`. A resource id is picked either way. No one waits for it: held
	/// where the control ring batches, as createBufferAtOffset's buffer is.
	RgfxCreated createTexture(const RgfxTextureDescriptor &desc, RgfxObjId heap = RGFX_OBJ_NONE,
	                          uint64_t heap_offset = 0, uint64_t align = kDefaultAddressAlign,
	                          uint64_t span = 1);

	/// A Managed texture of shape `desc` at a picked GPU address, whose CPU copy is the Shared
	/// buffer `cpu_buffer` over a block, of blocks `bytes_per_block` bytes and `block_width` by
	/// `block_height` texels (RgfxManagedLayout.hpp), `flags` RGFX_MANAGED_TEXTURE_LINEAR or 0,
	/// the address's range `span` bytes (its allocatedSize). A resource id is picked. No one waits
	/// for it: held where the control ring batches, as createBufferAtOffset's buffer is.
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
	/// slices and swizzle, with its id and resource id picked; `view`'s own are not read. No one
	/// waits for it: held where the control ring batches, as createBufferAtOffset's buffer is.
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

	/// A rasterization rate map of `shape` (RATE_MAP_CREATE), asked of the device, whose answer,
	/// the map's state (RATE_MAP_STATE), lands in `*state`: `status` is its status, `posted`
	/// false where no answer came back. No message goes where the host makes no maps
	/// (carriesRateMaps), where no asker is set, or where the message is past what the kext takes
	/// of a lifecycle message (half the control ring): `status` is then kRgfxNoAnswer.
	RgfxCreated createRateMap(const RgfxRateMapShape &shape, RgfxRateMapState *state);

	/// Compiles map `map` again from `shape` (RATE_MAP_RESET, `resetUsingDescriptor:`), its new
	/// state in `*state`. The device's status, or kRgfxNoAnswer, as createRateMap.
	uint32_t resetRateMap(RgfxObjId map, const RgfxRateMapShape &shape, RgfxRateMapState *state);

	/// One of map `map`'s answers that takes an argument (RATE_MAP_QUERY, `query` an
	/// RGFX_RATE_MAP_QUERY_*, `at` its layer or offset, `buffer` a copy's buffer, `x` and `y` a
	/// coordinate), its values in `values`. The device's status, or kRgfxNoAnswer, as
	/// createRateMap; `values` is zeroed where it is not RGFX_STATUS_OK.
	uint32_t queryRateMap(RgfxObjId map, uint32_t query, uint64_t at, RgfxObjId buffer, float x,
	                      float y, uint64_t values[3]);

	/// The handle of `function` (FUNCTION_HANDLE_CREATE), one of the functions compute pipeline
	/// `pipeline` links, asked of the device, whose answer (FUNCTION_POINTER_STATE) lands in
	/// `*state`: `status` is its status (RGFX_STATUS_NOT_FOUND where the pipeline links no such
	/// function, Metal's nil), `posted` false where no answer came back. No message goes where the
	/// host makes no function pointers (carriesFunctionPointers), where no asker is set, or where
	/// the message is past what the kext takes of a lifecycle message: `status` is then
	/// kRgfxNoAnswer.
	RgfxCreated createFunctionHandle(RgfxObjId pipeline, const RgfxLinkedFunctionRef &function,
	                                 RgfxFunctionPointerState *state);

	/// A visible function table of `count` entries of compute pipeline `pipeline`
	/// (FUNCTION_TABLE_CREATE), asked as createFunctionHandle: the state holds the table's
	/// gpuResourceID, allocatedSize and resourceOptions.
	RgfxCreated createFunctionTable(RgfxObjId pipeline, uint64_t count,
	                                RgfxFunctionPointerState *state);

	/// Sets `count` of table `table`'s entries from `first` to the handles `handles` names (0 a
	/// nil entry) through FUNCTION_TABLE_SET messages, which the device does not answer and
	/// carries out before any command buffer the process posts after them: as many as the kext's
	/// lifecycle messages take, each of a run of the entries. False where one did not go (the
	/// entries of the runs after it are not set either), or where the host makes no function
	/// pointers.
	bool setFunctionTable(RgfxObjId table, uint64_t first, const RgfxObjId *handles,
	                      uint64_t count);

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

	/// A dynamic library of library `library` (DYNAMIC_LIBRARY_CREATE): `status` is the
	/// device's answer (RGFX_STATUS_OK where it made it), `posted` false where the ring had no
	/// room. No message goes where the host makes no dynamic libraries
	/// (carriesDynamicLibraries): `status` is then kRgfxNoAnswer.
	RgfxCreated createDynamicLibrary(RgfxObjId library);

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

	/// A mesh render pipeline from `params` (RGFX_CMD_MESH_RENDER_PIPELINE_CREATE, where the
	/// host makes mesh pipelines: RGFX_FEATURE_MESH_PIPELINES), answered as createRenderPipeline
	/// is, the mesh stages' limits after the reflection. Every create waits for its answer.
	RgfxCreated createMeshRenderPipeline(const RgfxMeshRenderPipelineParams &params);

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

	/// Posts every waiting write: the slots written since the last flush (ICB_SLOTS and
	/// ICB_SLOTS_IN_PAGES, their runs, RgfxIcbSlotTable.hpp), then the records (ICB_WRITE, one message for each buffer's run of
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
	///
	/// Where the route batches (RgfxRingWriter::batches), the destroy is held to go with others in
	/// one call instead (RgfxRingWriter::postLater), unless `atOnce`, and `after` runs at once:
	/// whatever reuses what it gives back, a create or a block's free, is posted after the held
	/// destroys, so the host still drops the object first. A destroy the host orders other
	/// contexts' messages by goes `atOnce`: a texture over an IOSurface, which a present of the
	/// surface waits for (docs/graphics-kext/design.md, "Destroys and heap-placed buffers go
	/// together").
	bool destroy(RgfxObjId obj_id, std::function<void()> after = {}, bool atOnce = false);

	/// Posts the destroys waiting (destroy()), in the order they were made, each one's `after`
	/// run once it is on the ring, until one is refused again. One load where none waits.
	/// Returns whether none waits now.
	bool postWaitingDestroys();

	/// Posts what the control ring holds to go together (RgfxRingWriter::postLater), as a command
	/// buffer's commit does before the buffer goes on its ring, which the host reads after the
	/// control ring's messages posted before it: whether nothing but destroys stays held, so no
	/// create the buffer may name, nor a Managed buffer's range it may read. Destroys left held
	/// wait for a later post, as refused ones do.
	bool postHeldCreates();

	/// The control ring's writer, where a command buffer's commit may hand its publish to the held
	/// messages instead of posting them first (RgfxPublishVia): the route takes it
	/// (RgfxRingWriter::publishesQueues) and nothing else goes between them, no encode into an
	/// indirect command buffer nor a refused destroy waiting (flushIcbWrites,
	/// postWaitingDestroys). Else null, and the commit posts them first (postHeldCreates).
	RgfxRingWriter *publishVia() const
	{
		return m_control.publishesQueues() && !m_icbPending.load(std::memory_order_acquire) &&
		               !m_destroysWaiting.load(std::memory_order_acquire)
		           ? &m_control
		           : nullptr;
	}

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
	EventShareFn m_eventShare = nullptr;
	EventTakeFn m_eventTake = nullptr;
	bool m_privateRenderState = false;
	bool m_textureDidModifyData = false;
	bool m_bufferEscaped = false;
	bool m_rateMaps = false;
	bool m_functionPointers = false;
	bool m_dynamicLibraries = false;
	bool m_sharesPipelines = false;
	void *m_eventContext = nullptr;

	// Each keyed library's content key (setLibraryKey), by its id, under m_libraryKeysLock.
	std::mutex m_libraryKeysLock;
	std::unordered_map<RgfxObjId, RgfxLibraryContentKey> m_libraryKeys;
	// Posts the pipeline create of `type`, the `size` bytes at `message`, into `out`: against
	// `post`'s known pipelines, then the kext's answers of every process, where it can.
	void postPipeline(uint32_t type, const uint8_t *message, uint32_t size, RgfxCreated &out,
	                  RgfxPipelinePost *post);

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

	// Posts a message no one waits for that names no block (a create of one, a Managed buffer's
	// modified range): where the route batches (RgfxRingWriter::postLater), held to go with the
	// messages around it, before anything posted after it and before any command buffer
	// committed after it (postHeldCreates), `*out_seqno` left as it is; else at once, as tryPost
	// posts it. Whether it is held or posted.
	bool postUnwaited(uint32_t type, const void *message, uint32_t size, uint64_t *out_seqno);

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
	// Posts `runs` as messages of `type` (ICB_SLOTS of RgfxIcbSlotRun, ICB_SLOTS_IN_PAGES of
	// RgfxIcbSlotPagesRun), as few as the ring's cap allows (under m_icbLock), adding to `*flush`
	// where given.
	template <typename Run>
	void postIcbSlots(uint32_t type, const std::vector<Run> &runs, RgfxIcbFlush *flush);
	// Posts the runs of both kinds (postIcbSlots), the region's first.
	void postIcbSlots(const std::vector<RgfxIcbSlotRun> &runs,
	                  const std::vector<RgfxIcbSlotPagesRun> &pagesRuns, RgfxIcbFlush *flush);
};

#endif // RGFX_SERIALIZER_HPP_
