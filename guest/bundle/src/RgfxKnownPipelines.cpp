/*
 * RgfxKnownPipelines: see RgfxKnownPipelines.hpp. Our own C++.
 */

#include "RgfxKnownPipelines.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <cstddef>
#include <cstring>
#include <vector>

namespace {
// Where a create's object id lies in its message: just past the header, as in every create.
constexpr uint32_t kObjIdOffset = sizeof(RgfxMsgHeader);
} // namespace

namespace {
// The bytes a pipeline key's digest starts with.
constexpr char kSharedKeyPrefix[] = "rgfx-pipeline-v1";

// Appends the library ids' offsets of the run of `count` linked functions at byte `offset` of a
// message of `size` bytes: false where the run does not lie inside it.
bool linkedIds(uint32_t size, uint32_t offset, uint32_t count, std::vector<uint32_t> &ids)
{
	const uint64_t record = sizeof(RgfxLinkedFunction);
	if (count != 0 && static_cast<uint64_t>(offset) + record * count > size)
		return false;
	for (uint32_t i = 0; i < count; i++)
		ids.push_back(static_cast<uint32_t>(offset + record * i +
		                                    offsetof(RgfxLinkedFunction, library_obj_id)));
	return true;
}
} // namespace

bool rgfxSharedPipelineKey(uint32_t type, const uint8_t *message, uint32_t size,
                           const std::function<bool(uint64_t, RgfxLibraryContentKey *)> &libraryKey,
                           RgfxPipelineKey *out)
{
	const RgfxMsgHeader header = { type, size, 0 };
	if (message == nullptr || size < sizeof(header))
		return false;
	std::vector<uint32_t> ids;
	if (header.type == RGFX_CMD_COMPUTE_PIPELINE_CREATE) {
		RgfxComputePipelineCreate m;
		if (size < sizeof(m))
			return false;
		std::memcpy(&m, message, sizeof(m));
		if ((m.flags & RGFX_COMPUTE_PIPELINE_ANSWER_KEY) == 0 || m.library_count != 0)
			return false;
		ids.push_back(offsetof(RgfxComputePipelineCreate, function_library_obj_id));
		if (!linkedIds(size, m.linked_offset, m.linked_count, ids))
			return false;
	} else if (header.type == RGFX_CMD_RENDER_PIPELINE_CREATE) {
		RgfxRenderPipelineCreate m;
		if (size < sizeof(m))
			return false;
		std::memcpy(&m, message, sizeof(m));
		if ((m.flags & RGFX_RENDER_PIPELINE_ANSWER_KEY) == 0)
			return false;
		ids.push_back(offsetof(RgfxRenderPipelineCreate, vertex_library_obj_id));
		ids.push_back(offsetof(RgfxRenderPipelineCreate, fragment_library_obj_id));
		if (!linkedIds(size, m.vertex_linked_offset, m.vertex_linked_count, ids) ||
		    !linkedIds(size, m.fragment_linked_offset, m.fragment_linked_count, ids))
			return false;
	} else {
		return false;
	}
	std::vector<uint8_t> zeroed(message, message + size);
	std::memcpy(zeroed.data(), &header, sizeof(header));
	std::memset(zeroed.data() + kObjIdOffset, 0, sizeof(uint64_t));
	std::vector<RgfxLibraryContentKey> keys(ids.size());
	for (size_t i = 0; i < ids.size(); i++) {
		uint64_t id;
		std::memcpy(&id, message + ids[i], sizeof(id));
		keys[i] = {};
		if (id != 0 && !libraryKey(id, &keys[i]))
			return false;
		std::memset(zeroed.data() + ids[i], 0, sizeof(id));
	}
	CC_SHA256_CTX ctx;
	CC_SHA256_Init(&ctx);
	CC_SHA256_Update(&ctx, kSharedKeyPrefix, sizeof(kSharedKeyPrefix) - 1);
	CC_SHA256_Update(&ctx, zeroed.data(), static_cast<CC_LONG>(zeroed.size()));
	for (const RgfxLibraryContentKey &k : keys)
		CC_SHA256_Update(&ctx, k.data(), static_cast<CC_LONG>(k.size()));
	CC_SHA256_Final(out->data(), &ctx);
	return true;
}

RgfxPipelineKey RgfxKnownPipelines::keyOf(uint32_t type, const uint8_t *message, uint32_t size)
{
	CC_SHA256_CTX ctx;
	CC_SHA256_Init(&ctx);
	CC_SHA256_Update(&ctx, &type, sizeof(type));
	CC_SHA256_Update(&ctx, &size, sizeof(size));
	// The header and the object id, which differ between equal creates, as zeros.
	const uint8_t zeros[kObjIdOffset + sizeof(uint64_t)] = {};
	const uint32_t skipped = size < sizeof(zeros) ? size : static_cast<uint32_t>(sizeof(zeros));
	CC_SHA256_Update(&ctx, zeros, skipped);
	if (size > skipped)
		CC_SHA256_Update(&ctx, message + skipped, size - skipped);
	RgfxPipelineKey key;
	CC_SHA256_Final(key.data(), &ctx);
	return key;
}

size_t RgfxKnownPipelines::KeyHash::operator()(const RgfxPipelineKey &key) const
{
	size_t h;
	std::memcpy(&h, key.data(), sizeof(h));
	return h;
}

bool RgfxKnownPipelines::find(const RgfxPipelineKey &key, RgfxPipelineReflection *out) const
{
	std::lock_guard<std::mutex> guard(m_lock);
	auto it = m_known.find(key);
	if (it == m_known.end())
		return false;
	*out = it->second;
	return true;
}

void RgfxKnownPipelines::learn(const RgfxPipelineKey &key, const RgfxPipelineReflection &reflection)
{
	if (reflection.status != RGFX_PIPELINE_OK)
		return;
	std::lock_guard<std::mutex> guard(m_lock);
	if (m_known.size() >= kMaxKnown && m_known.find(key) == m_known.end())
		return;
	m_known[key] = reflection;
}

void RgfxPipelineAnswer::take(const RgfxPipelineReflection &reflection)
{
	const bool made = reflection.status == RGFX_PIPELINE_OK;
	if (made)
		m_id.store(reflection.gpu_resource_id, std::memory_order_relaxed);
	uint32_t awaiting = kAwaiting;
	m_state.compare_exchange_strong(awaiting, made ? kAnswered : kRefused, std::memory_order_acq_rel);
}

uint64_t RgfxPipelineAnswer::resourceId(const std::function<void()> &await)
{
	if (state() == kAwaiting) {
		await();
		uint32_t awaiting = kAwaiting;
		m_state.compare_exchange_strong(awaiting, kUnanswered, std::memory_order_acq_rel);
	}
	return state() == kAnswered ? m_id.load(std::memory_order_relaxed) : 0;
}

size_t RgfxKnownPipelines::size() const
{
	std::lock_guard<std::mutex> guard(m_lock);
	return m_known.size();
}
