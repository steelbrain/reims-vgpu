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
 *  - TEXTURE_FROM_IOSURFACE: opcode exists but no wire struct yet.
 *
 * Our own code; depends only on the committed wire header and the two helpers
 * above. No Metal SPI here — the Metal device shell (built later) translates each
 * MTLDescriptor into these plain fields and calls us.
 */

#ifndef RGFX_SERIALIZER_HPP_
#define RGFX_SERIALIZER_HPP_

#include <cstdint>

#include "RgfxObjectModel.hpp"
#include "RgfxRingWriter.hpp"
#include "reims_graphics_wire.h"

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

/// What a create did. `obj_id`, `gpu_address` and `resource_id` are the values
/// the guest picked (valid whether or not the message reached the ring, so a
/// caller that retries keeps the same handle); `gpu_address`/`resource_id` are 0
/// where the create has none. `seqno` is the ring seqno when `posted`. `posted`
/// is false when the ring had no room — the caller retries the create.
struct RgfxCreated {
	RgfxObjId obj_id;
	uint64_t gpu_address;
	uint64_t resource_id;
	uint64_t seqno;
	bool posted;
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

	// --- heaps ----------------------------------------------------------------

	/// A device heap of `size_bytes`. `options` is MTLResourceOptions, `heap_type`
	/// MTLHeapType. A GPU address of `size_bytes` aligned to `align` is picked.
	RgfxCreated createHeap(uint64_t size_bytes, uint32_t options, int32_t heap_type,
	                       uint64_t align = kDefaultAddressAlign);

	/// A Shared heap over the region block at `region_offset`.
	RgfxCreated createHeapOverHostMemory(uint64_t size_bytes, uint32_t options, int32_t heap_type,
	                                     uint64_t region_offset,
	                                     uint64_t align = kDefaultAddressAlign);

	// --- buffers --------------------------------------------------------------

	/// A device buffer of `length` bytes at a picked GPU address.
	RgfxCreated createBuffer(uint64_t length, uint32_t options,
	                         uint64_t align = kDefaultAddressAlign);

	/// A Shared buffer over the region block at `region_offset`.
	RgfxCreated createBufferOverHostMemory(uint64_t length, uint32_t options,
	                                       uint64_t region_offset,
	                                       uint64_t align = kDefaultAddressAlign);

	/// A buffer at `offset` in the heap `heap`.
	RgfxCreated createBufferAtOffset(RgfxObjId heap, uint64_t offset, uint64_t length,
	                                 uint32_t options);

	// --- textures -------------------------------------------------------------

	/// A texture of shape `desc`. When `heap` is RGFX_OBJ_NONE the texture is
	/// device-placed at a picked GPU address; otherwise it lies at `heap_offset`
	/// in `heap`. A resource id is picked either way.
	RgfxCreated createTexture(const RgfxTextureDescriptor &desc, RgfxObjId heap = RGFX_OBJ_NONE,
	                          uint64_t heap_offset = 0, uint64_t align = kDefaultAddressAlign);

	/// A texture of shape `desc` over the buffer `buffer` from `offset`, each row
	/// `bytes_per_row` apart. A resource id is picked.
	RgfxCreated createTextureFromBuffer(RgfxObjId buffer, uint64_t offset, uint64_t bytes_per_row,
	                                    const RgfxTextureDescriptor &desc);

	// --- state objects --------------------------------------------------------

	/// A sampler state.
	RgfxCreated createSampler(const RgfxSamplerParams &params);

	/// A depth/stencil state.
	RgfxCreated createDepthStencil(uint32_t depth_compare, bool depth_write_enabled,
	                               const RgfxStencilDescriptor &front,
	                               const RgfxStencilDescriptor &back);

	// --- shaders --------------------------------------------------------------

	/// A library from `byte_length` bytes of .metallib or AIR at `bytes`,
	/// forwarded unchanged. `posted` is false when the message (its header plus
	/// the bytes, rounded to a multiple of 8) does not fit the ring.
	RgfxCreated createLibrary(const void *bytes, uint64_t byte_length);

	// --- sync objects ---------------------------------------------------------

	/// An event.
	RgfxCreated createEvent();
	/// A fence.
	RgfxCreated createFence();
	/// A shared event starting at `initial_value`.
	RgfxCreated createSharedEvent(uint64_t initial_value);

	// --- destroy --------------------------------------------------------------

	/// Drops the host handle for `obj_id` (an object of any kind, a queue
	/// included: the host handle table knows the kind). Returns whether the
	/// message reached the ring.
	bool destroy(RgfxObjId obj_id);

	/// The default alignment for a picked GPU address when the caller names none.
	/// The caller passes a caps-derived alignment where Metal demands one; this
	/// is a safe floor for the first slice.
	static const uint64_t kDefaultAddressAlign = 256;

private:
	RgfxRingWriter &m_control;
	RgfxIdSpace &m_ids;
};

#endif // RGFX_SERIALIZER_HPP_
