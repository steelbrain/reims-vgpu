/*
 * RgfxSerializer: building each object-lifecycle wire message and posting it on
 * the control ring (RgfxSerializer.hpp).
 */

#include "RgfxSerializer.hpp"

#include "RgfxIcbSlotTable.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

/// The f32 bits of `value`, as the wire stores a float field.
uint32_t floatBits(float value)
{
	uint32_t bits;
	std::memcpy(&bits, &value, sizeof(bits));
	return bits;
}

/// `n` rounded up to a multiple of 8, as every ring message's size is.
uint64_t roundUp8(uint64_t n)
{
	return (n + 7) & ~static_cast<uint64_t>(7);
}

} // namespace

RgfxSerializer::RgfxSerializer(RgfxRingWriter &control, RgfxIdSpace &ids)
    : m_control(control), m_ids(ids)
{
}

RgfxCreated RgfxSerializer::createQueue(uint64_t submit_offset, uint64_t complete_offset,
                                        uint32_t submit_data_size, uint32_t complete_data_size,
                                        uint32_t priority, uint32_t vector)
{
	RgfxCreated out = {};
	out.obj_id = m_ids.allocObjId();

	RgfxQueueCreate msg = {};
	msg.obj_id = out.obj_id;
	msg.submit_offset = submit_offset;
	msg.complete_offset = complete_offset;
	msg.submit_data_size = submit_data_size;
	msg.complete_data_size = complete_data_size;
	msg.priority = priority;
	msg.vector = vector;
	if (m_queueOpen != nullptr) {
		out.posted = m_queueOpen(m_queueOpenContext, msg);
		return out;
	}
	out.posted = m_control.tryPost(RGFX_CMD_QUEUE_CREATE, &msg, sizeof(msg), &out.seqno);
	return out;
}

void RgfxSerializer::setQueueOpener(QueueOpenFn open, void *context)
{
	m_queueOpen = open;
	m_queueOpenContext = context;
}

void RgfxSerializer::setQueueCloser(QueueCloseFn close, void *context)
{
	m_queueClose = close;
	m_queueCloseContext = context;
}

void RgfxSerializer::setAsker(AskFn ask, void *context)
{
	m_ask = ask;
	m_askContext = context;
}

size_t RgfxSerializer::SizeKeyHash::operator()(const SizeKey &key) const
{
	uint64_t h = 1469598103934665603ull; // FNV-1a
	for (uint8_t b : key) {
		h ^= b;
		h *= 1099511628211ull;
	}
	return static_cast<size_t>(h);
}

namespace {
// The most shapes heapTextureSizeAndAlign keeps answers of; past it they are dropped and asked
// again. An app sizes a few shapes, over and over.
constexpr size_t kKeptSizes = 4096;
} // namespace

bool RgfxSerializer::heapTextureSizeAndAlign(const RgfxTextureDescriptor &desc, uint64_t *size,
                                             uint64_t *align)
{
	if (m_ask == nullptr)
		return false;
	SizeKey key;
	std::memcpy(key.data(), &desc, sizeof(desc));
	{
		std::lock_guard<std::mutex> guard(m_sizesLock);
		auto it = m_sizes.find(key);
		if (it != m_sizes.end()) {
			*size = it->second.size;
			*align = it->second.align;
			return true;
		}
	}
	RgfxHeapTextureSizeAndAlign msg = {};
	msg.header.type = RGFX_CMD_HEAP_TEXTURE_SIZE_AND_ALIGN;
	msg.header.size = sizeof(msg);
	msg.obj_id = m_ids.allocObjId();
	msg.desc = desc;
	RgfxSizeAndAlign answer = {};
	uint32_t length = 0;
	if (!m_ask(m_askContext, &msg, sizeof(msg), &answer, sizeof(answer), &length) ||
	    length < sizeof(answer) || answer.header.type != RGFX_EVT_SIZE_AND_ALIGN ||
	    answer.status != RGFX_STATUS_OK)
		return false;
	*size = answer.size;
	*align = answer.align;
	std::lock_guard<std::mutex> guard(m_sizesLock);
	if (m_sizes.size() >= kKeptSizes)
		m_sizes.clear();
	m_sizes[key] = {answer.size, answer.align};
	return true;
}

// One ARGUMENT_LAYOUT through `ask`: the head, then its tail (the descriptors, or the name and
// the constants), its offsets set as `fill` appends them. Answers the COMPLETE's status, or
// kRgfxNoAnswer.
template <typename Fill>
static uint32_t askArgumentLayout(RgfxSerializer::AskFn ask, void *askContext, RgfxIdSpace &ids,
                                  uint32_t max_size, uint32_t source, uint64_t reply_offset,
                                  uint64_t reply_length, Fill fill)
{
	if (ask == nullptr)
		return kRgfxNoAnswer;
	const uint32_t head = sizeof(RgfxArgumentLayoutQuery);
	std::vector<uint8_t> tail;
	auto append = [&](const void *data, size_t n) -> uint32_t {
		uint32_t off = static_cast<uint32_t>(head + tail.size());
		if (n) {
			const uint8_t *b = static_cast<const uint8_t *>(data);
			tail.insert(tail.end(), b, b + n);
		}
		return off;
	};
	RgfxArgumentLayoutQuery msg = {};
	msg.obj_id = ids.allocObjId();
	msg.reply_offset = reply_offset;
	msg.reply_length = reply_length;
	msg.source = source;
	fill(msg, append);
	const uint64_t total = (uint64_t(head) + tail.size() + 7) & ~uint64_t(7);
	if (total > max_size)
		return kRgfxNoAnswer;
	msg.header.type = RGFX_CMD_ARGUMENT_LAYOUT;
	msg.header.size = static_cast<uint32_t>(total);
	std::vector<uint8_t> buf(static_cast<size_t>(total), 0);
	std::memcpy(buf.data(), &msg, sizeof(msg));
	if (!tail.empty())
		std::memcpy(buf.data() + head, tail.data(), tail.size());
	RgfxComplete answer = {};
	uint32_t length = 0;
	if (!ask(askContext, buf.data(), static_cast<uint32_t>(total), &answer, sizeof(answer),
	         &length) ||
	    length < sizeof(answer) || answer.header.type != RGFX_EVT_COMPLETE)
		return kRgfxNoAnswer;
	return answer.status;
}

uint32_t RgfxSerializer::argumentLayoutOfDescriptors(const RgfxArgumentDescriptor *descriptors,
                                                     uint32_t count, uint64_t reply_offset,
                                                     uint64_t reply_length)
{
	return askArgumentLayout(
	    m_ask, m_askContext, m_ids, m_control.dataSize(), RGFX_ARGUMENT_SOURCE_DESCRIPTORS,
	    reply_offset, reply_length, [&](RgfxArgumentLayoutQuery &msg, auto &append) {
		    msg.count = count;
		    msg.descriptors_offset = append(descriptors, size_t(count) * sizeof(*descriptors));
	    });
}

uint32_t RgfxSerializer::argumentLayoutOfFunction(RgfxObjId library, const char *entry_point,
                                                  const RgfxFunctionConstant *constants,
                                                  uint32_t constant_count, uint32_t buffer_index,
                                                  uint64_t reply_offset, uint64_t reply_length)
{
	return askArgumentLayout(
	    m_ask, m_askContext, m_ids, m_control.dataSize(), RGFX_ARGUMENT_SOURCE_FUNCTION,
	    reply_offset, reply_length, [&](RgfxArgumentLayoutQuery &msg, auto &append) {
		    const uint32_t len =
		        entry_point ? static_cast<uint32_t>(std::strlen(entry_point)) : 0;
		    msg.library_obj_id = library;
		    msg.buffer_index = buffer_index;
		    msg.name_offset = append(entry_point, len);
		    msg.name_len = len;
		    msg.constants_offset =
		        append(constants, size_t(constant_count) * sizeof(RgfxFunctionConstant));
		    msg.constant_count = constant_count;
	    });
}

void RgfxSerializer::destroyQueue(RgfxObjId obj_id)
{
	if (m_queueClose != nullptr) {
		m_queueClose(m_queueCloseContext, obj_id);
		return;
	}
	destroy(obj_id);
}

RgfxCreated RgfxSerializer::createHeap(uint64_t size_bytes, uint32_t options, int32_t heap_type,
                                       uint64_t align)
{
	RgfxCreated out = {};
	// No address left for it: no object, as Metal answers one it cannot place.
	out.gpu_address = m_ids.allocAddress(size_bytes, align);
	if (out.gpu_address == 0)
		return out;
	out.obj_id = m_ids.allocObjId();

	RgfxHeapCreate msg = {};
	msg.obj_id = out.obj_id;
	msg.gpu_address = out.gpu_address;
	msg.size_bytes = size_bytes;
	msg.options = options;
	msg.heap_type = heap_type;
	out.posted = m_control.tryPost(RGFX_CMD_HEAP_CREATE, &msg, sizeof(msg), &out.seqno);
	return out;
}

RgfxCreated RgfxSerializer::createHeapOverHostMemory(uint64_t size_bytes, uint32_t options,
                                                     int32_t heap_type, uint64_t region_offset,
                                                     uint64_t align)
{
	RgfxCreated out = {};
	// No address left for it: no object, as Metal answers one it cannot place.
	out.gpu_address = m_ids.allocAddress(size_bytes, align);
	if (out.gpu_address == 0)
		return out;
	out.obj_id = m_ids.allocObjId();

	RgfxHeapOverHostMemory msg = {};
	msg.obj_id = out.obj_id;
	msg.gpu_address = out.gpu_address;
	msg.region_offset = region_offset;
	msg.size_bytes = size_bytes;
	msg.options = options;
	msg.heap_type = heap_type;
	out.posted =
	    m_control.tryPost(RGFX_CMD_HEAP_OVER_HOST_MEMORY, &msg, sizeof(msg), &out.seqno);
	return out;
}

RgfxCreated RgfxSerializer::createHeapOverPages(uint64_t size_bytes, uint32_t options,
                                                int32_t heap_type, uint64_t block_offset,
                                                uint64_t block_size, uint64_t align)
{
	RgfxCreated out = {};
	// No address left for it: no object, as for a heap over host memory.
	out.gpu_address = m_ids.allocAddress(size_bytes, align);
	if (out.gpu_address == 0)
		return out;
	out.obj_id = m_ids.allocObjId();

	RgfxHeapOverPages msg = {};
	msg.obj_id = out.obj_id;
	msg.gpu_address = out.gpu_address;
	msg.block_offset = block_offset;
	msg.block_size = block_size;
	msg.size_bytes = size_bytes;
	msg.options = options;
	msg.heap_type = heap_type;
	out.posted = m_control.tryPost(RGFX_CMD_HEAP_OVER_PAGES, &msg, sizeof(msg), &out.seqno);
	return out;
}

RgfxCreated RgfxSerializer::createBuffer(uint64_t length, uint32_t options, uint64_t align)
{
	RgfxCreated out = {};
	// No address left for it: no object, as Metal answers one it cannot place.
	out.gpu_address = m_ids.allocAddress(length, align);
	if (out.gpu_address == 0)
		return out;
	out.obj_id = m_ids.allocObjId();

	RgfxBufferCreate msg = {};
	msg.obj_id = out.obj_id;
	msg.gpu_address = out.gpu_address;
	msg.length = length;
	msg.options = options;
	out.posted = m_control.tryPost(RGFX_CMD_BUFFER_CREATE, &msg, sizeof(msg), &out.seqno);
	return out;
}

RgfxCreated RgfxSerializer::createBufferOverHostMemory(uint64_t length, uint32_t options,
                                                       uint64_t region_offset, uint64_t align)
{
	RgfxCreated out = {};
	// No address left for it: no object, as Metal answers one it cannot place.
	out.gpu_address = m_ids.allocAddress(length, align);
	if (out.gpu_address == 0)
		return out;
	out.obj_id = m_ids.allocObjId();

	RgfxBufferOverHostMemory msg = {};
	msg.obj_id = out.obj_id;
	msg.gpu_address = out.gpu_address;
	msg.region_offset = region_offset;
	msg.length = length;
	msg.options = options;
	out.posted =
	    m_control.tryPost(RGFX_CMD_BUFFER_OVER_HOST_MEMORY, &msg, sizeof(msg), &out.seqno);
	return out;
}

RgfxCreated RgfxSerializer::createBufferOverSurfacePages(uint64_t length, uint32_t options,
                                                         uint64_t block_offset,
                                                         uint64_t block_size)
{
	RgfxCreated out = {};
	out.obj_id = m_ids.allocObjId();

	RgfxBufferOverSurfacePages msg = {};
	msg.obj_id = out.obj_id;
	msg.block_offset = block_offset;
	msg.block_size = block_size;
	msg.length = length;
	msg.options = options;
	out.posted =
	    m_control.tryPost(RGFX_CMD_BUFFER_OVER_SURFACE_PAGES, &msg, sizeof(msg), &out.seqno);
	return out;
}

RgfxCreated RgfxSerializer::createBufferManagedAtOffset(RgfxObjId heap, uint64_t offset,
                                                        uint64_t length, uint32_t options,
                                                        uint64_t align)
{
	RgfxCreated out = {};
	// The GPU copy's address; no address left for it: no object, as for a buffer over host
	// memory.
	out.gpu_address = m_ids.allocAddress(length, align);
	if (out.gpu_address == 0)
		return out;
	out.obj_id = m_ids.allocObjId();

	RgfxBufferManagedAtOffset msg = {};
	msg.obj_id = out.obj_id;
	msg.gpu_address = out.gpu_address;
	msg.heap_obj_id = heap;
	msg.offset = offset;
	msg.length = length;
	msg.options = options;
	out.posted =
	    m_control.tryPost(RGFX_CMD_BUFFER_MANAGED_AT_OFFSET, &msg, sizeof(msg), &out.seqno);
	return out;
}

RgfxCreated RgfxSerializer::createManagedBuffer(RgfxObjId cpu_buffer, uint64_t length,
                                                uint32_t options, uint64_t align)
{
	RgfxCreated out = {};
	// The GPU copy's address; no address left for it: no object, as for a buffer over host
	// memory.
	out.gpu_address = m_ids.allocAddress(length, align);
	if (out.gpu_address == 0)
		return out;
	out.obj_id = m_ids.allocObjId();

	RgfxManagedBufferCreate msg = {};
	msg.obj_id = out.obj_id;
	msg.gpu_address = out.gpu_address;
	msg.cpu_buffer_obj_id = cpu_buffer;
	msg.length = length;
	msg.options = options;
	out.posted = m_control.tryPost(RGFX_CMD_MANAGED_BUFFER_CREATE, &msg, sizeof(msg), &out.seqno);
	return out;
}

RgfxCreated RgfxSerializer::createBufferOverPages(uint64_t length, uint32_t options,
                                                  uint64_t block_offset, uint64_t block_size,
                                                  uint64_t offset)
{
	RgfxCreated out = {};
	// The pick holds the span from the page the buffer starts in. No address left for it: no
	// object, as Metal answers one it cannot place.
	if (length > UINT64_MAX - offset)
		return out;
	out.gpu_address = m_ids.allocAddress(length + offset, kDefaultAddressAlign);
	if (out.gpu_address == 0)
		return out;
	out.obj_id = m_ids.allocObjId();

	RgfxBufferOverPages msg = {};
	msg.obj_id = out.obj_id;
	msg.gpu_address = out.gpu_address + offset;
	msg.block_offset = block_offset;
	msg.block_size = block_size;
	msg.offset = offset;
	msg.length = length;
	msg.options = options;
	out.posted = m_control.tryPost(RGFX_CMD_BUFFER_OVER_PAGES, &msg, sizeof(msg), &out.seqno);
	return out;
}

RgfxCreated RgfxSerializer::createBufferAtOffset(RgfxObjId heap, uint64_t offset, uint64_t length,
                                                 uint32_t options)
{
	RgfxCreated out = {};
	out.obj_id = m_ids.allocObjId();
	// A heap-placed buffer's address is the heap's plus the offset; the host
	// derives it from the heap it pinned, so none is sent.

	RgfxBufferAtOffset msg = {};
	msg.obj_id = out.obj_id;
	msg.heap_obj_id = heap;
	msg.offset = offset;
	msg.length = length;
	msg.options = options;
	out.posted = m_control.tryPost(RGFX_CMD_BUFFER_AT_OFFSET, &msg, sizeof(msg), &out.seqno);
	return out;
}

void RgfxSerializer::releaseResourceId(uint64_t resource_id)
{
	m_ids.freeResourceId(resource_id);
}

void RgfxSerializer::releaseAddress(uint64_t gpu_address)
{
	m_ids.freeAddress(gpu_address);
}

bool RgfxSerializer::refused(const RgfxCreated &created)
{
	if (created.posted)
		return false;
	m_ids.freeAddress(created.gpu_address);
	m_ids.freeResourceId(created.resource_id);
	return true;
}

RgfxCreated RgfxSerializer::createTexture(const RgfxTextureDescriptor &desc, RgfxObjId heap,
                                          uint64_t heap_offset, uint64_t align, uint64_t span)
{
	RgfxCreated out = {};
	out.resource_id = m_ids.allocResourceId();
	if (out.resource_id == 0)
		return out;
	out.obj_id = m_ids.allocObjId();

	RgfxTextureCreate msg = {};
	msg.obj_id = out.obj_id;
	msg.resource_id = out.resource_id;
	msg.desc = desc;
	if (heap == RGFX_OBJ_NONE) {
		// Device-placed: pick a GPU address, its range the texture's bytes; heap_obj_id stays 0.
		out.gpu_address = m_ids.allocAddress(std::max<uint64_t>(span, 1), align);
		msg.gpu_address = out.gpu_address;
	} else {
		msg.heap_obj_id = heap;
		msg.heap_offset = heap_offset;
	}
	out.posted = m_control.tryPost(RGFX_CMD_TEXTURE_CREATE, &msg, sizeof(msg), &out.seqno);
	return out;
}

bool RgfxSerializer::bufferDidModifyRange(RgfxObjId buffer, uint64_t location, uint64_t length)
{
	RgfxBufferDidModifyRange msg = {};
	msg.obj_id = buffer;
	msg.location = location;
	msg.length = length;
	uint64_t seqno = 0;
	return m_control.tryPost(RGFX_CMD_BUFFER_DID_MODIFY_RANGE, &msg, sizeof(msg), &seqno);
}

RgfxCreated RgfxSerializer::createManagedTexture(const RgfxTextureDescriptor &desc,
                                                 RgfxObjId cpu_buffer, uint32_t bytes_per_block,
                                                 uint32_t block_width, uint32_t block_height,
                                                 uint32_t flags, uint64_t align, uint64_t span)
{
	RgfxCreated out = {};
	out.resource_id = m_ids.allocResourceId();
	if (out.resource_id == 0)
		return out;
	out.obj_id = m_ids.allocObjId();
	out.gpu_address = m_ids.allocAddress(std::max<uint64_t>(span, 1), align);

	RgfxManagedTextureCreate msg = {};
	msg.obj_id = out.obj_id;
	msg.gpu_address = out.gpu_address;
	msg.resource_id = out.resource_id;
	msg.cpu_buffer_obj_id = cpu_buffer;
	msg.bytes_per_block = bytes_per_block;
	msg.block_width = block_width;
	msg.block_height = block_height;
	msg.flags = flags;
	msg.desc = desc;
	out.posted =
	    m_control.tryPost(RGFX_CMD_MANAGED_TEXTURE_CREATE, &msg, sizeof(msg), &out.seqno);
	return out;
}

bool RgfxSerializer::textureReplaceRegion(RgfxObjId texture, uint32_t level, uint32_t slice,
                                          const uint32_t origin[3], const uint32_t size[3])
{
	RgfxTextureReplaceRegion msg = {};
	msg.obj_id = texture;
	msg.level = level;
	msg.slice = slice;
	for (int i = 0; i < 3; i++) {
		msg.origin[i] = origin[i];
		msg.size[i] = size[i];
	}
	uint64_t seqno = 0;
	return m_control.tryPost(RGFX_CMD_TEXTURE_REPLACE_REGION, &msg, sizeof(msg), &seqno);
}

RgfxCreated RgfxSerializer::createTextureFromBuffer(RgfxObjId buffer, uint64_t offset,
                                                    uint64_t bytes_per_row,
                                                    const RgfxTextureDescriptor &desc)
{
	RgfxCreated out = {};
	out.resource_id = m_ids.allocResourceId();
	if (out.resource_id == 0)
		return out;
	out.obj_id = m_ids.allocObjId();

	RgfxTextureFromBuffer msg = {};
	msg.obj_id = out.obj_id;
	msg.resource_id = out.resource_id;
	msg.buffer_obj_id = buffer;
	msg.offset = offset;
	msg.bytes_per_row = bytes_per_row;
	msg.desc = desc;
	out.posted =
	    m_control.tryPost(RGFX_CMD_TEXTURE_FROM_BUFFER, &msg, sizeof(msg), &out.seqno);
	return out;
}

RgfxCreated RgfxSerializer::createLinearTextureFromBuffer(RgfxObjId buffer, uint64_t offset,
                                                          uint64_t bytes_per_row,
                                                          uint64_t bytes_per_image,
                                                          const RgfxTextureDescriptor &desc)
{
	RgfxCreated out = {};
	out.resource_id = m_ids.allocResourceId();
	if (out.resource_id == 0)
		return out;
	out.obj_id = m_ids.allocObjId();

	RgfxLinearTextureFromBuffer msg = {};
	msg.obj_id = out.obj_id;
	msg.resource_id = out.resource_id;
	msg.buffer_obj_id = buffer;
	msg.offset = offset;
	msg.bytes_per_row = bytes_per_row;
	msg.bytes_per_image = bytes_per_image;
	msg.desc = desc;
	out.posted =
	    m_control.tryPost(RGFX_CMD_LINEAR_TEXTURE_FROM_BUFFER, &msg, sizeof(msg), &out.seqno);
	return out;
}

RgfxCreated RgfxSerializer::createTextureView(const RgfxTextureView &view)
{
	RgfxCreated out = {};
	out.resource_id = m_ids.allocResourceId();
	if (out.resource_id == 0)
		return out;
	out.obj_id = m_ids.allocObjId();

	RgfxTextureView msg = view;
	msg.header = {};
	msg.obj_id = out.obj_id;
	msg.resource_id = out.resource_id;
	out.posted = m_control.tryPost(RGFX_CMD_TEXTURE_VIEW, &msg, sizeof(msg), &out.seqno);
	return out;
}

RgfxCreated RgfxSerializer::createTextureFromIOSurface(RgfxObjId buffer, uint64_t offset,
                                                       uint64_t bytes_per_row,
                                                       uint32_t iosurface_id, uint32_t plane,
                                                       const RgfxTextureDescriptor &desc,
                                                       uint64_t writers_offset,
                                                       const RgfxSurfaceFacts &facts)
{
	RgfxCreated out = {};
	out.resource_id = m_ids.allocResourceId();
	if (out.resource_id == 0)
		return out;
	out.obj_id = m_ids.allocObjId();

	RgfxSurfaceTextureMessage msg = {};
	msg.texture.obj_id = out.obj_id;
	msg.texture.resource_id = out.resource_id;
	msg.texture.buffer_obj_id = buffer;
	msg.texture.offset = offset;
	msg.texture.bytes_per_row = bytes_per_row;
	msg.texture.iosurface_id = iosurface_id;
	msg.texture.plane = plane;
	msg.texture.desc = desc;
	msg.texture.writers_offset = writers_offset;
	msg.facts = facts;
	out.posted =
	    m_control.tryPost(RGFX_CMD_TEXTURE_FROM_IOSURFACE, &msg, sizeof(msg), &out.seqno);
	return out;
}

RgfxCreated RgfxSerializer::createSampler(const RgfxSamplerParams &params)
{
	RgfxCreated out = {};
	out.obj_id = m_ids.allocObjId();

	RgfxSamplerCreate msg = {};
	msg.obj_id = out.obj_id;
	msg.min_filter = params.min_filter;
	msg.mag_filter = params.mag_filter;
	msg.mip_filter = params.mip_filter;
	msg.s_address_mode = params.s_address_mode;
	msg.t_address_mode = params.t_address_mode;
	msg.r_address_mode = params.r_address_mode;
	msg.border_color = params.border_color;
	msg.compare_function = params.compare_function;
	msg.max_anisotropy = params.max_anisotropy;
	msg.lod_min_clamp_bits = floatBits(params.lod_min_clamp);
	msg.lod_max_clamp_bits = floatBits(params.lod_max_clamp);
	msg.flags = params.flags;
	if (m_ask == nullptr) {
		out.posted = m_control.tryPost(RGFX_CMD_SAMPLER_CREATE, &msg, sizeof(msg), &out.seqno);
		return out;
	}
	msg.header.type = RGFX_CMD_SAMPLER_CREATE;
	msg.header.size = sizeof(msg);
	RgfxSamplerState answer = {};
	uint32_t length = 0;
	out.posted = m_ask(m_askContext, &msg, sizeof(msg), &answer, sizeof(answer), &length) &&
	             length >= sizeof(answer) && answer.header.type == RGFX_EVT_SAMPLER_STATE;
	out.status = out.posted ? answer.status : kRgfxNoAnswer;
	out.seqno = out.posted ? answer.header.seqno : 0;
	out.resource_id = out.posted && answer.status == RGFX_STATUS_OK ? answer.gpu_resource_id : 0;
	return out;
}

RgfxCreated RgfxSerializer::createIcb(const RgfxIcbCreate &desc)
{
	RgfxCreated out = {};
	out.obj_id = m_ids.allocObjId();

	RgfxIcbCreate msg = desc;
	msg.obj_id = out.obj_id;
	if (m_ask == nullptr) {
		out.posted = m_control.tryPost(RGFX_CMD_ICB_CREATE, &msg, sizeof(msg), &out.seqno);
		return out;
	}
	msg.header.type = RGFX_CMD_ICB_CREATE;
	msg.header.size = sizeof(msg);
	RgfxIcbState answer = {};
	uint32_t length = 0;
	out.posted = m_ask(m_askContext, &msg, sizeof(msg), &answer, sizeof(answer), &length) &&
	             length >= sizeof(answer) && answer.header.type == RGFX_EVT_ICB_STATE;
	out.status = out.posted ? answer.status : kRgfxNoAnswer;
	out.seqno = out.posted ? answer.header.seqno : 0;
	out.resource_id = out.posted && answer.status == RGFX_STATUS_OK ? answer.gpu_resource_id : 0;
	return out;
}

void RgfxSerializer::icbRecord(RgfxObjId icb, uint16_t op, uint32_t index, const void *body,
                               uint16_t size)
{
	const uint32_t recordBytes = sizeof(RgfxIcbRecord) + size;
	const uint32_t cap = icbMessageCap();
	std::lock_guard<std::mutex> lock(m_icbLock);
	// The buffer's run of records, most often the last one begun.
	IcbMessage *message = nullptr;
	for (size_t i = m_icbMessages.size(); i-- > 0;) {
		if (m_icbMessages[i].icb == icb) {
			message = &m_icbMessages[i];
			break;
		}
	}
	if (message != nullptr && message->bytes.size() + recordBytes > cap) {
		// Its message is full: it goes now, and a new one takes the record. The records of other
		// buffers wait: no record names another buffer's commands.
		postIcbMessage(*message, nullptr);
		message->bytes.resize(sizeof(RgfxIcbWrite));
	}
	if (message == nullptr) {
		m_icbMessages.push_back({icb, std::vector<uint8_t>(sizeof(RgfxIcbWrite))});
		message = &m_icbMessages.back();
		message->bytes.reserve(1024);
	}
	const size_t at = message->bytes.size();
	message->bytes.resize(at + recordBytes);
	RgfxIcbRecord record = {op, static_cast<uint16_t>(recordBytes), index};
	std::memcpy(message->bytes.data() + at, &record, sizeof(record));
	if (size != 0)
		std::memcpy(message->bytes.data() + at + sizeof(record), body, size);
	m_icbPending.store(true, std::memory_order_release);
}

uint32_t RgfxSerializer::icbMessageCap() const
{
	return m_control.fits(m_control.dataSize()) ? m_control.dataSize() : m_control.dataSize() / 2;
}

void RgfxSerializer::postIcbMessage(IcbMessage &message, RgfxIcbFlush *flush)
{
	if (message.bytes.size() <= sizeof(RgfxIcbWrite))
		return;
	RgfxIcbWrite head = {};
	head.obj_id = message.icb;
	std::memcpy(message.bytes.data(), &head, sizeof(head));
	const auto start = std::chrono::steady_clock::now();
	const uint32_t size = static_cast<uint32_t>(message.bytes.size());
	if (!m_control.tryPost(RGFX_CMD_ICB_WRITE, message.bytes.data(), size, nullptr)) {
		// The kext refused it, or the ring had no room: the buffer's commands keep what they
		// had, as a lost encode.
		std::fprintf(stderr, "RGFX: ICB_WRITE of %u bytes into 0x%llx was not posted\n", size,
		             static_cast<unsigned long long>(message.icb));
	}
	if (flush != nullptr) {
		flush->messages++;
		flush->bytes += size;
		flush->nanoseconds += static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
		                                                         start)
		        .count());
	}
}

void RgfxSerializer::postIcbSlots(const std::vector<RgfxIcbSlotRun> &runs, RgfxIcbFlush *flush)
{
	const uint32_t cap = icbMessageCap();
	const size_t most = (cap - sizeof(RgfxIcbSlots)) / sizeof(RgfxIcbSlotRun);
	std::vector<uint8_t> bytes;
	for (size_t at = 0; at < runs.size(); at += most) {
		const size_t n = std::min(most, runs.size() - at);
		RgfxIcbSlots head = {};
		head.obj_id = runs[at].icb_obj_id;
		head.run_count = static_cast<uint32_t>(n);
		const uint32_t size = static_cast<uint32_t>(sizeof(head) + n * sizeof(RgfxIcbSlotRun));
		bytes.resize(size);
		std::memcpy(bytes.data(), &head, sizeof(head));
		std::memcpy(bytes.data() + sizeof(head), &runs[at], n * sizeof(RgfxIcbSlotRun));
		const auto start = std::chrono::steady_clock::now();
		if (!m_control.tryPost(RGFX_CMD_ICB_SLOTS, bytes.data(), size, nullptr)) {
			// As a lost ICB_WRITE: the commands keep what they had.
			std::fprintf(stderr, "RGFX: ICB_SLOTS of %zu runs was not posted\n", n);
		}
		if (flush != nullptr) {
			flush->messages++;
			flush->bytes += size;
			flush->nanoseconds += static_cast<uint64_t>(
			    std::chrono::duration_cast<std::chrono::nanoseconds>(
			        std::chrono::steady_clock::now() - start)
			        .count());
		}
	}
}

RgfxIcbFlush RgfxSerializer::flushIcbWrites()
{
	RgfxIcbFlush flush = {};
	if (!m_icbPending.load(std::memory_order_acquire))
		return flush;
	// The posts are made under the lock, so a flush that finds writes pending returns only once
	// they are on the ring, whichever thread posts them. Each table's flag is cleared before its
	// chunks are taken, so a write after the take puts it on the list again (RgfxIcbSlotTable.hpp).
	std::lock_guard<std::mutex> lock(m_icbLock);
	if (!m_icbSlots.empty()) {
		std::vector<RgfxIcbSlotRun> runs;
		for (RgfxIcbSlotTable *slots : m_icbSlots) {
			slots->pending.store(false, std::memory_order_seq_cst);
			slots->takeRuns(runs);
		}
		m_icbSlots.clear();
		postIcbSlots(runs, &flush);
	}
	for (IcbMessage &message : m_icbMessages)
		postIcbMessage(message, &flush);
	m_icbMessages.clear();
	m_icbPending.store(false, std::memory_order_release);
	return flush;
}

void RgfxSerializer::icbSlotsPending(RgfxIcbSlotTable *slots)
{
	std::lock_guard<std::mutex> lock(m_icbLock);
	if (slots->pending.exchange(true, std::memory_order_seq_cst))
		return;
	m_icbSlots.push_back(slots);
	m_icbPending.store(true, std::memory_order_release);
}

void RgfxSerializer::flushIcbSlots(RgfxIcbSlotTable *slots)
{
	std::lock_guard<std::mutex> lock(m_icbLock);
	std::vector<RgfxIcbSlotRun> runs;
	slots->takeRuns(runs);
	postIcbSlots(runs, nullptr);
}

void RgfxSerializer::forgetIcbSlots(RgfxIcbSlotTable *slots)
{
	{
		std::lock_guard<std::mutex> lock(m_icbLock);
		m_icbSlots.erase(std::remove(m_icbSlots.begin(), m_icbSlots.end(), slots),
		                 m_icbSlots.end());
	}
	releaseIcbSlotBytes(slots->bytes());
}

bool RgfxSerializer::reserveIcbSlotBytes(uint64_t bytes)
{
	uint64_t held = m_icbSlotBytes.load(std::memory_order_relaxed);
	do {
		if (bytes > kIcbSlotBytes - held)
			return false;
	} while (!m_icbSlotBytes.compare_exchange_weak(held, held + bytes, std::memory_order_relaxed));
	return true;
}

void RgfxSerializer::releaseIcbSlotBytes(uint64_t bytes)
{
	m_icbSlotBytes.fetch_sub(bytes, std::memory_order_relaxed);
}

RgfxCreated RgfxSerializer::createDepthStencil(uint32_t depth_compare, bool depth_write_enabled,
                                               const RgfxStencilDescriptor &front,
                                               const RgfxStencilDescriptor &back)
{
	RgfxCreated out = {};
	out.obj_id = m_ids.allocObjId();

	RgfxDepthStencilCreate msg = {};
	msg.obj_id = out.obj_id;
	msg.depth_compare = depth_compare;
	msg.depth_write_enabled = depth_write_enabled ? 1u : 0u;
	msg.front = front;
	msg.back = back;
	out.posted =
	    m_control.tryPost(RGFX_CMD_DEPTH_STENCIL_CREATE, &msg, sizeof(msg), &out.seqno);
	return out;
}

RgfxCreated RgfxSerializer::createLibrary(const void *bytes, uint64_t byte_length,
                                          const uint8_t *key)
{
	RgfxCreated out = {};
	out.obj_id = m_ids.allocObjId();

	// The message is the header struct, then the bytes, rounded up to a multiple
	// of 8. A library larger than the ring cannot be posted in one message; the
	// caller retries or (a later transport piece) places large AIR in a region
	// block.
	uint64_t total = roundUp8(RGFX_LIBRARY_BYTES_OFFSET + byte_length);
	if (total > m_control.dataSize() || total > UINT32_MAX) {
		out.posted = false;
		return out;
	}
	std::vector<uint8_t> buf(static_cast<size_t>(total), 0);
	RgfxLibraryCreate head = {};
	head.obj_id = out.obj_id;
	head.byte_length = byte_length;
	if (key) {
		head.flags = RGFX_LIBRARY_FLAG_KEYED;
		std::memcpy(head.key, key, sizeof(head.key));
	}
	std::memcpy(buf.data(), &head, sizeof(head));
	if (byte_length)
		std::memcpy(buf.data() + RGFX_LIBRARY_BYTES_OFFSET, bytes,
		            static_cast<size_t>(byte_length));
	out.posted = m_control.tryPost(RGFX_CMD_LIBRARY_CREATE, buf.data(),
	                               static_cast<uint32_t>(total), &out.seqno, &out.status);
	return out;
}

RgfxCreated RgfxSerializer::lookupLibrary(const uint8_t *key, uint64_t slice_size)
{
	RgfxCreated out = {};
	out.obj_id = m_ids.allocObjId();
	RgfxLibraryLookup m = {};
	m.obj_id = out.obj_id;
	m.slice_size = slice_size;
	std::memcpy(m.key, key, sizeof(m.key));
	out.posted =
	    m_control.tryPost(RGFX_CMD_LIBRARY_LOOKUP, &m, sizeof(m), &out.seqno, &out.status);
	return out;
}

RgfxObjId RgfxSerializer::allocLibraryId()
{
	return m_ids.allocObjId();
}

bool RgfxSerializer::postLibraryChunk(RgfxObjId obj_id, uint64_t region_offset, uint64_t byte_length,
                                      uint64_t total_length, uint64_t chunk_offset,
                                      uint64_t ack_offset, uint64_t ack_value, const uint8_t *key,
                                      uint32_t *out_status)
{
	RgfxLibraryFromRegion m = {};
	m.obj_id = obj_id;
	m.region_offset = region_offset;
	m.byte_length = byte_length;
	m.total_length = total_length;
	m.chunk_offset = chunk_offset;
	m.ack_offset = ack_offset;
	m.ack_value = ack_value;
	if (key) {
		m.flags = RGFX_LIBRARY_FLAG_KEYED;
		std::memcpy(m.key, key, sizeof(m.key));
	}
	return m_control.tryPost(RGFX_CMD_LIBRARY_FROM_REGION, &m, sizeof(m), nullptr, out_status);
}

namespace {

/// Appends `count` linked functions to a pipeline create's tail through `append`
/// (which answers the byte offset from the message start of what it appended): each
/// one's name, constants, group name and specialized name, then their
/// RgfxLinkedFunction records, whose offset it answers.
template <typename Append>
uint32_t appendLinkedFunctions(Append &append, const RgfxLinkedFunctionRef *refs, uint32_t count)
{
	std::vector<RgfxLinkedFunction> records(count);
	for (uint32_t i = 0; i < count; i++) {
		const RgfxLinkedFunctionRef &r = refs[i];
		RgfxLinkedFunction &m = records[i];
		m.library_obj_id = r.function.library;
		const uint32_t len =
		    r.function.entry_point ? static_cast<uint32_t>(std::strlen(r.function.entry_point)) : 0;
		m.name_offset = append(r.function.entry_point, len);
		m.name_len = len;
		m.constants_offset =
		    append(r.function.constants, r.function.constant_count * sizeof(RgfxFunctionConstant));
		m.constant_count = r.function.constant_count;
		m.list = r.list;
		m.options = r.options;
		if (r.list == RGFX_LINKED_GROUP && r.group != nullptr) {
			const uint32_t glen = static_cast<uint32_t>(std::strlen(r.group));
			m.group_offset = append(r.group, glen);
			m.group_len = glen;
		}
		if (r.specialized_name != nullptr) {
			const uint32_t slen = static_cast<uint32_t>(std::strlen(r.specialized_name));
			m.specialized_name_offset = append(r.specialized_name, slen);
			m.specialized_name_len = slen;
		}
	}
	return append(records.data(), count * sizeof(RgfxLinkedFunction));
}

// Posts the pipeline create of `type`, the `size` bytes at `message`, on `control` into `out`,
// against `post`'s known pipelines where it has some and the route defers answers.
void postPipeline(RgfxRingWriter &control, uint32_t type, const uint8_t *message, uint32_t size,
                  RgfxCreated &out, RgfxPipelinePost *post)
{
	if (post != nullptr && post->known != nullptr && control.defers()) {
		post->key = RgfxKnownPipelines::keyOf(type, message, size);
		post->keyed = true;
		if (post->known->find(post->key, &post->reflection)) {
			out.posted = control.tryPostKnown(type, message, size, &out.seqno, &post->deferred);
			return;
		}
	}
	out.posted = control.tryPost(type, message, size, &out.seqno);
}
} // namespace

RgfxCreated RgfxSerializer::createRenderPipeline(const RgfxRenderPipelineParams &p,
                                                 RgfxPipelinePost *post)
{
	RgfxCreated out = {};
	out.obj_id = m_ids.allocObjId();

	// The functions, label and vertex descriptor go in the message tail, each
	// addressed by a byte offset from the message start (the fixed head's size
	// plus the segment's position in the tail).
	const uint32_t head = sizeof(RgfxRenderPipelineCreate);
	std::vector<uint8_t> tail;
	auto append = [&](const void *data, size_t n) -> uint32_t {
		uint32_t off = static_cast<uint32_t>(head + tail.size());
		if (n) {
			const uint8_t *b = static_cast<const uint8_t *>(data);
			tail.insert(tail.end(), b, b + n);
		}
		return off;
	};
	auto nameLen = [](const char *s) -> uint32_t {
		return s ? static_cast<uint32_t>(std::strlen(s)) : 0;
	};

	RgfxRenderPipelineCreate msg = {};
	msg.obj_id = out.obj_id;
	msg.vertex_library_obj_id = p.vertex.library;
	msg.fragment_library_obj_id = p.fragment.library;

	uint32_t vlen = nameLen(p.vertex.entry_point);
	msg.vertex_name_offset = append(p.vertex.entry_point, vlen);
	msg.vertex_name_len = vlen;
	uint32_t flen = nameLen(p.fragment.entry_point);
	msg.fragment_name_offset = append(p.fragment.entry_point, flen);
	msg.fragment_name_len = flen;
	uint32_t llen = nameLen(p.label);
	msg.label_offset = append(p.label, llen);
	msg.label_len = llen;
	msg.vertex_constants_offset =
	    append(p.vertex.constants, p.vertex.constant_count * sizeof(RgfxFunctionConstant));
	msg.vertex_constant_count = p.vertex.constant_count;
	msg.fragment_constants_offset =
	    append(p.fragment.constants, p.fragment.constant_count * sizeof(RgfxFunctionConstant));
	msg.fragment_constant_count = p.fragment.constant_count;
	msg.vertex_layouts_offset =
	    append(p.vertex_layouts, p.vertex_layout_count * sizeof(RgfxVertexBufferLayout));
	msg.vertex_layout_count = p.vertex_layout_count;
	msg.vertex_attributes_offset =
	    append(p.vertex_attributes, p.vertex_attribute_count * sizeof(RgfxVertexAttribute));
	msg.vertex_attribute_count = p.vertex_attribute_count;
	msg.vertex_linked_offset = appendLinkedFunctions(append, p.vertex_linked, p.vertex_linked_count);
	msg.vertex_linked_count = p.vertex_linked_count;
	msg.fragment_linked_offset =
	    appendLinkedFunctions(append, p.fragment_linked, p.fragment_linked_count);
	msg.fragment_linked_count = p.fragment_linked_count;

	msg.raster_sample_count = p.raster_sample_count;
	uint32_t ncolor = p.color_attachment_count > 8 ? 8 : p.color_attachment_count;
	msg.color_attachment_count = ncolor;
	for (uint32_t i = 0; i < ncolor; i++)
		msg.colors[i] = p.colors[i];
	msg.depth_pixel_format = p.depth_pixel_format;
	msg.stencil_pixel_format = p.stencil_pixel_format;
	msg.flags = p.flags;
	msg.input_primitive_topology = p.input_primitive_topology;
	msg.max_vertex_amplification_count = p.max_vertex_amplification_count;

	uint64_t total = roundUp8(head + tail.size());
	if (total > m_control.dataSize() || total > UINT32_MAX) {
		out.posted = false;
		return out;
	}
	std::vector<uint8_t> buf(static_cast<size_t>(total), 0);
	std::memcpy(buf.data(), &msg, sizeof(msg));
	if (!tail.empty())
		std::memcpy(buf.data() + head, tail.data(), tail.size());
	postPipeline(m_control, RGFX_CMD_RENDER_PIPELINE_CREATE, buf.data(),
	             static_cast<uint32_t>(total), out, post);
	return out;
}

RgfxCreated RgfxSerializer::createComputePipeline(const RgfxComputePipelineParams &p,
                                                  RgfxPipelinePost *post)
{
	RgfxCreated out = {};
	out.obj_id = m_ids.allocObjId();

	const uint32_t head = sizeof(RgfxComputePipelineCreate);
	std::vector<uint8_t> tail;
	auto append = [&](const void *data, size_t n) -> uint32_t {
		uint32_t off = static_cast<uint32_t>(head + tail.size());
		if (n) {
			const uint8_t *b = static_cast<const uint8_t *>(data);
			tail.insert(tail.end(), b, b + n);
		}
		return off;
	};

	RgfxComputePipelineCreate msg = {};
	msg.obj_id = out.obj_id;
	msg.function_library_obj_id = p.function.library;
	uint32_t nlen = p.function.entry_point
	                    ? static_cast<uint32_t>(std::strlen(p.function.entry_point))
	                    : 0;
	msg.name_offset = append(p.function.entry_point, nlen);
	msg.name_len = nlen;
	msg.constants_offset =
	    append(p.function.constants, p.function.constant_count * sizeof(RgfxFunctionConstant));
	msg.constant_count = p.function.constant_count;
	msg.flags = p.flags;
	msg.max_total_threads_per_threadgroup = p.max_total_threads_per_threadgroup;
	msg.max_call_stack_depth = p.max_call_stack_depth;
	msg.stage_input_layouts_offset = append(
	    p.stage_input_layouts, p.stage_input_layout_count * sizeof(RgfxVertexBufferLayout));
	msg.stage_input_layout_count = p.stage_input_layout_count;
	msg.stage_input_attributes_offset = append(
	    p.stage_input_attributes, p.stage_input_attribute_count * sizeof(RgfxVertexAttribute));
	msg.stage_input_attribute_count = p.stage_input_attribute_count;
	msg.stage_input_index_type = p.stage_input_index_type;
	msg.stage_input_index_buffer_index = p.stage_input_index_buffer_index;
	msg.linked_offset = appendLinkedFunctions(append, p.linked, p.linked_count);
	msg.linked_count = p.linked_count;

	uint64_t total = roundUp8(head + tail.size());
	if (total > m_control.dataSize() || total > UINT32_MAX) {
		out.posted = false;
		return out;
	}
	std::vector<uint8_t> buf(static_cast<size_t>(total), 0);
	std::memcpy(buf.data(), &msg, sizeof(msg));
	if (!tail.empty())
		std::memcpy(buf.data() + head, tail.data(), tail.size());
	postPipeline(m_control, RGFX_CMD_COMPUTE_PIPELINE_CREATE, buf.data(),
	             static_cast<uint32_t>(total), out, post);
	return out;
}

RgfxCreated RgfxSerializer::createEvent()
{
	RgfxCreated out = {};
	out.obj_id = m_ids.allocObjId();

	RgfxObjectCreate msg = {};
	msg.obj_id = out.obj_id;
	out.posted = m_control.tryPost(RGFX_CMD_EVENT_CREATE, &msg, sizeof(msg), &out.seqno);
	return out;
}

RgfxCreated RgfxSerializer::createFence()
{
	RgfxCreated out = {};
	out.obj_id = m_ids.allocObjId();

	RgfxObjectCreate msg = {};
	msg.obj_id = out.obj_id;
	out.posted = m_control.tryPost(RGFX_CMD_FENCE_CREATE, &msg, sizeof(msg), &out.seqno);
	return out;
}

RgfxCreated RgfxSerializer::createSharedEvent(uint64_t initial_value)
{
	RgfxCreated out = {};
	out.obj_id = m_ids.allocObjId();

	RgfxSharedEventCreate msg = {};
	msg.obj_id = out.obj_id;
	msg.initial_value = initial_value;
	out.posted =
	    m_control.tryPost(RGFX_CMD_SHARED_EVENT_CREATE, &msg, sizeof(msg), &out.seqno);
	return out;
}

bool RgfxSerializer::setSharedEventValue(RgfxObjId event, uint64_t value)
{
	RgfxSharedEventSetValue msg = {};
	msg.obj_id = event;
	msg.value = value;
	uint64_t seqno = 0;
	return m_control.tryPost(RGFX_CMD_SHARED_EVENT_SET_VALUE, &msg, sizeof(msg), &seqno);
}

RgfxCreated RgfxSerializer::createVideoSession(uint32_t codec, uint32_t direction)
{
	RgfxCreated out = {};
	out.obj_id = m_ids.allocObjId();

	RgfxVideoSessionCreate msg = {};
	msg.obj_id = out.obj_id;
	msg.codec = codec;
	msg.direction = direction;
	out.posted =
	    m_control.tryPost(RGFX_CMD_VIDEO_SESSION_CREATE, &msg, sizeof(msg), &out.seqno);
	return out;
}

bool RgfxSerializer::postDestroy(RgfxObjId obj_id)
{
	RgfxObjectDestroy msg = {};
	msg.obj_id = obj_id;
	return m_control.tryPost(RGFX_CMD_OBJECT_DESTROY, &msg, sizeof(msg), nullptr);
}

bool RgfxSerializer::destroy(RgfxObjId obj_id, std::function<void()> after)
{
	// A waiting record may name the object (a pipeline or a buffer the app lets go after
	// encoding it), so the records go first.
	flushIcbWrites();
	// With none waiting, it goes at once; else behind those waiting, which go first.
	const bool behind = m_destroysWaiting.load(std::memory_order_acquire);
	if (!behind && postDestroy(obj_id)) {
		if (after)
			after();
		return true;
	}
	{
		std::lock_guard<std::mutex> lock(m_destroyLock);
		m_destroys.push_back({obj_id, std::move(after)});
		m_destroysWaiting.store(true, std::memory_order_release);
	}
	if (behind)
		return postWaitingDestroys();
	// Refused just now: it waits for the next destroy or commit.
	static std::atomic<bool> logged{false};
	if (!logged.exchange(true))
		std::fprintf(stderr,
		             "RGFX: the destroy of 0x%llx was not posted: it waits, and what the object "
		             "lies over stays the host's, until a later destroy or commit posts it\n",
		             static_cast<unsigned long long>(obj_id));
	return false;
}

bool RgfxSerializer::postWaitingDestroys()
{
	if (!m_destroysWaiting.load(std::memory_order_acquire))
		return true;
	for (;;) {
		WaitingDestroy posted;
		{
			std::lock_guard<std::mutex> lock(m_destroyLock);
			if (m_destroys.empty()) {
				m_destroysWaiting.store(false, std::memory_order_release);
				return true;
			}
			if (!postDestroy(m_destroys.front().obj))
				return false;
			posted = std::move(m_destroys.front());
			m_destroys.pop_front();
		}
		// Out of the lock: what it gives back may let go of more objects, whose destroys come
		// here.
		if (posted.after)
			posted.after();
	}
}
