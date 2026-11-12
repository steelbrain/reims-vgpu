/*
 * RgfxStateCache: see RgfxStateCache.h. Our own ObjC++; no Apple private SPI.
 */

#import "RgfxStateCache.h"

#import "RgfxDepthStencilState.h"
#import "RgfxSampler.h"

#include <algorithm>
#include <cstring>
#include <dispatch/dispatch.h>
#include <iterator>
#include <mutex>
#include <unordered_map>

#include "RgfxSerializer.hpp"

namespace {

constexpr size_t kKept = 64;

/// The serial queue the caches let go of the objects they push out on (RgfxStateCache.h).
dispatch_queue_t rgfxStateReleases()
{
	static dispatch_queue_t queue = dispatch_queue_create(
	    "ai.aneesiqbal.reims-graphics.state-releases", DISPATCH_QUEUE_SERIAL);
	return queue;
}

/// Lets go of the reference `ref` holds (rgfxStateReleases).
void rgfxStateRelease(void *ref)
{
	id object = (__bridge_transfer id)ref;
	(void)object;
}

/// `value`, with -0.0 as 0.0, as Metal compares a descriptor's floats.
float rgfxCanonicalFloat(float value)
{
	return value == 0.0f ? 0.0f : value;
}

/// A descriptor as the cache compares it: its state's bytes (`State`, plain words with no
/// padding) and its label, nil for none or an empty one.
template <typename State>
struct RgfxStateKey {
	State state;
	NSString *label;

	bool operator==(const RgfxStateKey &other) const
	{
		return std::memcmp(&state, &other.state, sizeof(State)) == 0 &&
		       (label == other.label || [label isEqualToString:other.label]);
	}
};

template <typename State>
struct RgfxStateKeyHash {
	size_t operator()(const RgfxStateKey<State> &key) const
	{
		uint64_t h = 1469598103934665603ull; // FNV-1a over the state's bytes
		const uint8_t *bytes = reinterpret_cast<const uint8_t *>(&key.state);
		for (size_t i = 0; i < sizeof(State); i++) {
			h ^= bytes[i];
			h *= 1099511628211ull;
		}
		return static_cast<size_t>(h ^ static_cast<uint64_t>(key.label.hash));
	}
};

/// The objects of one kind: each held weakly by its key, and the last kKept made held strongly
/// too (RgfxStateCache.h).
template <typename State>
class RgfxStateTable {
public:
	/// The object cached for `key` while it lives, else what `make` makes, cached where it
	/// answers `cache` true. `make` runs outside the lock.
	template <typename Make>
	id find(const RgfxStateKey<State> &key, Make make)
	{
		{
			std::lock_guard<std::mutex> guard(m_lock);
			auto it = m_objects.find(key);
			if (it != m_objects.end()) {
				id held = it->second;
				if (held != nil)
					return held;
			}
		}
		bool cache = false;
		id made = make(&cache);
		if (made == nil || !cache)
			return made;
		id winner = nil;
		id evicted = nil;
		{
			std::lock_guard<std::mutex> guard(m_lock);
			auto it = m_objects.find(key);
			if (it != m_objects.end())
				winner = it->second;
			if (winner == nil) {
				if (it != m_objects.end()) {
					it->second = made;
				} else {
					prune();
					m_objects.emplace(key, made);
				}
				evicted = m_kept[m_next];
				m_kept[m_next] = made;
				m_next = (m_next + 1) % kKept;
			}
		}
		// The cache's reference to the object pushed out goes to the release queue whole, so this
		// create never runs its dealloc, which posts its destroy where the app holds it no more.
		// A `made` that lost to another thread's is let go here as it leaves scope, outside the
		// lock.
		if (evicted != nil) {
			void *ref = (__bridge_retained void *)evicted;
			evicted = nil;
			dispatch_async_f(rgfxStateReleases(), ref, &rgfxStateRelease);
		}
		return winner != nil ? winner : made;
	}

private:
	// Drops the keys of objects gone, once there are twice as many keys as there were live
	// ones at the last pass, so the table stays the size of what lives.
	void prune()
	{
		if (m_objects.size() < m_pruneAt)
			return;
		for (auto it = m_objects.begin(); it != m_objects.end();) {
			id held = it->second;
			it = held == nil ? m_objects.erase(it) : std::next(it);
		}
		m_pruneAt = std::max<size_t>(2 * m_objects.size(), 4 * kKept);
	}

	std::mutex m_lock;
	std::unordered_map<RgfxStateKey<State>, __weak id, RgfxStateKeyHash<State>> m_objects;
	id m_kept[kKept];
	size_t m_next = 0;
	size_t m_pruneAt = 4 * kKept;
};

/// A sampler descriptor's state: what SAMPLER_CREATE carries, its extension's lodBias and
/// reductionMode (macOS 26) among it.
struct RgfxSamplerKeyState {
	RgfxSamplerParams params;
};
static_assert(sizeof(RgfxSamplerKeyState) == sizeof(RgfxSamplerParams), "no padding");

/// A depth/stencil descriptor's state, as DEPTH_STENCIL_CREATE carries it.
struct RgfxDepthStencilKeyState {
	uint32_t depth_compare;
	uint32_t depth_write_enabled;
	RgfxStencilDescriptor front;
	RgfxStencilDescriptor back;
};
static_assert(sizeof(RgfxDepthStencilKeyState) == 8 + 2 * sizeof(RgfxStencilDescriptor),
              "no padding");

/// `label`, or nil for an empty one, as Metal answers it.
NSString *rgfxStateLabel(NSString *label)
{
	return label.length != 0 ? [label copy] : nil;
}

} // namespace

@implementation RgfxStateCache {
	__weak id<MTLDevice> _device;
	RgfxSerializer *_serializer; // borrowed: the device owns it
	RgfxStateTable<RgfxSamplerKeyState> _samplers;
	RgfxStateTable<RgfxDepthStencilKeyState> _depthStencils;
}

+ (NSUInteger)keptCount
{
	return kKept;
}

+ (void)waitForReleases
{
	dispatch_sync(rgfxStateReleases(), ^{
	});
}

- (instancetype)initWithDevice:(id<MTLDevice>)device serializer:(RgfxSerializer *)serializer
{
	self = [super init];
	if (self) {
		_device = device;
		_serializer = serializer;
	}
	return self;
}

- (id<MTLSamplerState>)samplerWithDescriptor:(MTLSamplerDescriptor *)descriptor
{
	if (_serializer == nullptr || descriptor == nil)
		return nil;
	RgfxStateKey<RgfxSamplerKeyState> key = {};
	key.state.params = RgfxSamplerParamsOf(descriptor);
	key.state.params.lod_min_clamp = rgfxCanonicalFloat(key.state.params.lod_min_clamp);
	key.state.params.lod_max_clamp = rgfxCanonicalFloat(key.state.params.lod_max_clamp);
	key.state.params.lod_bias = rgfxCanonicalFloat(key.state.params.lod_bias);
	key.label = rgfxStateLabel(descriptor.label);
	id<MTLDevice> device = _device;
	RgfxSerializer *serializer = _serializer;
	return _samplers.find(key, [&](bool *cache) -> id {
		RgfxSampler *made = [RgfxSampler samplerWithDescriptor:descriptor
		                                                device:device
		                                            serializer:serializer];
		*cache = made != nil;
		return made;
	});
}

- (id<MTLDepthStencilState>)depthStencilStateWithDescriptor:(MTLDepthStencilDescriptor *)descriptor
{
	if (_serializer == nullptr || descriptor == nil)
		return nil;
	RgfxStateKey<RgfxDepthStencilKeyState> key = {};
	key.state.depth_compare = static_cast<uint32_t>(descriptor.depthCompareFunction);
	key.state.depth_write_enabled = descriptor.depthWriteEnabled ? 1u : 0u;
	key.state.front = RgfxStencilOf(descriptor.frontFaceStencil);
	key.state.back = RgfxStencilOf(descriptor.backFaceStencil);
	key.label = rgfxStateLabel(descriptor.label);
	id<MTLDevice> device = _device;
	RgfxSerializer *serializer = _serializer;
	return _depthStencils.find(key, [&](bool *cache) -> id {
		RgfxDepthStencilState *made =
		    [RgfxDepthStencilState depthStencilStateWithDescriptor:descriptor
		                                                    device:device
		                                                serializer:serializer];
		*cache = made != nil;
		return made;
	});
}

@end
