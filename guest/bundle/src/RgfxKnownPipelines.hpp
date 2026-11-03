/*
 * RgfxKnownPipelines: the pipeline creates the device accepted for this process, by the whole of
 * their message, with the reflection it answered. A create whose message equals a known one's
 * (functions, libraries, constants, state, label: every byte but its object id) is posted without
 * waiting for its answer (RgfxRingWriter::tryPostKnown): its state takes the known reflection at
 * once, and only what the device gives each create alone, its gpuResourceID, waits at its first
 * read for this create's own answer. The first create of each message waits, so a refusal is known
 * at the create, as Metal's creators know theirs.
 *
 * Natively a create of a descriptor the process made before answers from Metal's caches in
 * microseconds, and the answer is a new object with a new resource ID: here too each create is a
 * new object, posted as any create, and the device gives it its own ID.
 *
 * Bounded (kMaxKnown); thread-safe. Our own C++; depends only on the committed wire header and
 * CommonCrypto.
 */

#ifndef RGFX_KNOWN_PIPELINES_HPP_
#define RGFX_KNOWN_PIPELINES_HPP_

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <unordered_map>

#include "reims_graphics_wire.h"

/// A pipeline create's key: SHA-256 over its type and its message, the header and the object id
/// zeroed.
typedef std::array<uint8_t, 32> RgfxPipelineKey;

class RgfxKnownPipelines {
public:
	/// The most messages kept; past it a new message is not kept, and its later equal creates wait
	/// as a first one does.
	static constexpr size_t kMaxKnown = 16384;

	/// The key of the create of `type` whose whole message is the `size` bytes at `message` (its
	/// header first, its object id at offset 16).
	static RgfxPipelineKey keyOf(uint32_t type, const uint8_t *message, uint32_t size);

	/// The reflection the device answered for the create of `key`, into `*out`: false where none
	/// is known.
	bool find(const RgfxPipelineKey &key, RgfxPipelineReflection *out) const;

	/// Keeps `reflection`, the device's answer to the create of `key`, where it accepted it
	/// (RGFX_PIPELINE_OK) and room is left. A refusal is never kept: an equal create waits again.
	void learn(const RgfxPipelineKey &key, const RgfxPipelineReflection &reflection);

	/// How many messages are known.
	size_t size() const;

private:
	struct KeyHash {
		size_t operator()(const RgfxPipelineKey &key) const;
	};
	mutable std::mutex m_lock;
	std::unordered_map<RgfxPipelineKey, RgfxPipelineReflection, KeyHash> m_known;
};

/// A deferred pipeline create's own answer, which its state's gpuResourceID waits for at its first
/// read: the create's reflection is the equal one's, but its resource ID is its own. Thread-safe:
/// the answer is taken on whichever thread takes it from the kext.
class RgfxPipelineAnswer {
public:
	enum State : uint32_t {
		// Posted, not yet answered.
		kAwaiting,
		// Answered: the device made the pipeline, and `resourceId` is its ID.
		kAnswered,
		// Answered with a refusal (the platform could not make the pipeline): the ID is 0, and a
		// command buffer that names the pipeline fails on the host.
		kRefused,
		// No answer came while it was awaited: the ID is 0.
		kUnanswered,
	};

	/// Takes the create's answer, `reflection`; a second answer is ignored.
	void take(const RgfxPipelineReflection &reflection);

	/// The pipeline's resource ID: while the answer is awaited, `await` is called once to take it
	/// (it waits for it); 0 where it was refused or never came.
	uint64_t resourceId(const std::function<void()> &await);

	State state() const { return static_cast<State>(m_state.load(std::memory_order_acquire)); }

private:
	std::atomic<uint32_t> m_state{kAwaiting};
	std::atomic<uint64_t> m_id{0};
};

#endif // RGFX_KNOWN_PIPELINES_HPP_
