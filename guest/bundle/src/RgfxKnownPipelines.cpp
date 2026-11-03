/*
 * RgfxKnownPipelines: see RgfxKnownPipelines.hpp. Our own C++.
 */

#include "RgfxKnownPipelines.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <cstring>

namespace {
// Where a create's object id lies in its message: just past the header, as in every create.
constexpr uint32_t kObjIdOffset = sizeof(RgfxMsgHeader);
} // namespace

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
