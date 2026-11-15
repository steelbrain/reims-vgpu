// RgfxSharedPipelines: the device's answers to the pipeline creates of every process of the guest,
// by their pipeline keys (RGFX_PIPELINE_REFLECT_KEYED in the wire header), so that an equal create
// of any process need not wait for its answer (kRgfxAccelMethodSubmit with kRgfxSubmitShared).
//
// The kext learns an entry only from the device's own answer: a made pipeline's reflection whose
// reply carries the key the host computed, from the create's bytes and the content keys of the
// libraries it holds (verified against their bytes). A process names the key of its own create,
// which the kext cannot check; a process that names another's key only takes a wrong reflection
// for its own pipeline. What the kext keeps of an answer is what an equal create's is: every field
// but the header, the object id and those the device gives each create alone (gpuResourceID,
// uniqueIdentifier, resourceIndex), which are zeroed.
//
// Ways entries a set, Sets sets, a key's set its first four bytes': a set full of other keys
// replaces the one found or learnt longest ago. The caller holds a lock around every call.
//
// At its start the kext seeds the table with the answers the device kept of keyed creates of its
// earlier runs (RGFX_CMD_PIPELINE_ANSWERS), which the device hands newest first: a seeded answer
// takes only a free entry, so a full set keeps the newer ones, and ranks below every answer this
// run learnt or found, the newer seed above the older, so a set full of seeds replaces its oldest.
// The kext forgets a key's answer where the device refuses a create the kept answer answered
// (forget), so the next equal create waits for its own, and keeps the device's own answer to every
// create it posted without waiting (answered), so a kept answer the device now answers otherwise,
// a seed of an earlier run among them, gives way to the device's for every later create.
// Header-only and free of the kernel's headers, so its native test
// (tests/shared_pipelines_test.cpp) builds it as the kext does.
//
// Our own code; it depends only on the committed wire header.
#ifndef RGFX_SHARED_PIPELINES_HPP_
#define RGFX_SHARED_PIPELINES_HPP_

#include <stdint.h>
#include <string.h>

#include "reims_graphics_wire.h"

template <uint32_t Sets, uint32_t Ways>
class RgfxSharedPipelines {
public:
	// A pipeline key's bytes.
	static constexpr uint32_t kKeyBytes = 32;
	// A keyed reply's bytes: the reflection, then the key.
	static constexpr uint32_t kKeyedReplyBytes = RGFX_PIPELINE_REFLECTION_KEY_OFFSET + kKeyBytes;
	// How many answers the table holds, and the most a start seeds.
	static constexpr uint32_t kSeeds = Sets * Ways;

	// Learns the device's answer to a pipeline create, the `length` bytes of `reply` (its
	// RGFX_EVT_PIPELINE_REFLECTION, header first), where it carries a made pipeline's key: true
	// where it was kept.
	bool learn(const uint8_t *reply, uint32_t length)
	{
		RgfxPipelineReflection r;
		if (!keepable(reply, length, &r))
			return false;
		const uint8_t *key = reply + RGFX_PIPELINE_REFLECTION_KEY_OFFSET;
		Entry *set = setOf(key);
		Entry *slot = same(set, key);
		if (slot == nullptr)
			slot = unused(set);
		if (slot == nullptr) {
			slot = &set[0];
			for (uint32_t w = 1; w < Ways; w++) {
				if (set[w].tick < slot->tick)
					slot = &set[w];
			}
		}
		keep(slot, key, r, ++m_tick);
		return true;
	}

	// Learns an answer the device kept from an earlier run, the `order`th newest it hands (0 the
	// newest), the `length` bytes of `reply` as learn takes them, where its set has a free entry
	// and its key no entry: true where it was kept. Fewer than kSeeds are ever seeded.
	bool seed(const uint8_t *reply, uint32_t length, uint32_t order)
	{
		RgfxPipelineReflection r;
		if (order >= kSeeds || !keepable(reply, length, &r))
			return false;
		const uint8_t *key = reply + RGFX_PIPELINE_REFLECTION_KEY_OFFSET;
		Entry *set = setOf(key);
		Entry *slot = same(set, key) ? nullptr : unused(set);
		if (slot == nullptr)
			return false;
		keep(slot, key, r, kSeeds - order);
		return true;
	}

	// The answer kept for the create of `key` (kKeyBytes), as the create at `seqno` of object
	// `objId` gets it, into `*out`: false where none is kept.
	bool find(const uint8_t *key, uint64_t seqno, uint64_t objId, RgfxPipelineReflection *out)
	{
		Entry *set = setOf(key);
		for (uint32_t w = 0; w < Ways; w++) {
			Entry &e = set[w];
			if (!e.used || memcmp(e.key, key, kKeyBytes) != 0)
				continue;
			e.tick = ++m_tick;
			*out = e.reflection;
			out->header.seqno = seqno;
			out->obj_id = objId;
			return true;
		}
		return false;
	}

	// Takes the device's answer to a create posted without waiting, the `length` bytes of `reply`
	// as learn takes them: where `refusedKey` is not null (the create was posted for that key's
	// kept answer, and the device did not make its pipeline) the key's answer is forgotten, and
	// else a made pipeline's keyed answer is kept as learn keeps it.
	void answered(const uint8_t *reply, uint32_t length, const uint8_t *refusedKey)
	{
		if (refusedKey != nullptr)
			forget(refusedKey);
		else
			learn(reply, length);
	}

	// Forgets the answer kept for `key` (kKeyBytes): true where one was kept.
	bool forget(const uint8_t *key)
	{
		Entry *e = same(setOf(key), key);
		if (e == nullptr)
			return false;
		e->used = false;
		return true;
	}

	// Forgets every answer: the kext's start, which may take the table from a run before.
	void clear()
	{
		memset(m_entries, 0, sizeof(m_entries));
		m_tick = kSeeds;
	}

	// How many answers are kept.
	uint32_t size() const
	{
		uint32_t n = 0;
		for (uint32_t i = 0; i < Sets * Ways; i++)
			n += m_entries[i].used ? 1 : 0;
		return n;
	}

private:
	struct Entry {
		uint8_t key[kKeyBytes];
		RgfxPipelineReflection reflection;
		uint64_t tick;
		bool used;
	};

	Entry *setOf(const uint8_t *key)
	{
		uint32_t h;
		memcpy(&h, key, sizeof(h));
		return &m_entries[(h % Sets) * Ways];
	}

	// Whether `reply`'s `length` bytes are a made pipeline's keyed answer, its reflection into
	// `*r`.
	static bool keepable(const uint8_t *reply, uint32_t length, RgfxPipelineReflection *r)
	{
		if (length < kKeyedReplyBytes)
			return false;
		memcpy(r, reply, sizeof(*r));
		return r->header.type == RGFX_EVT_PIPELINE_REFLECTION &&
		       r->header.size >= kKeyedReplyBytes && r->status == RGFX_PIPELINE_OK &&
		       (r->flags & RGFX_PIPELINE_REFLECT_KEYED) != 0 &&
		       r->kind != RGFX_PIPELINE_KIND_MESH;
	}

	// `set`'s entry of `key`, or none.
	static Entry *same(Entry *set, const uint8_t *key)
	{
		for (uint32_t w = 0; w < Ways; w++) {
			if (set[w].used && memcmp(set[w].key, key, kKeyBytes) == 0)
				return &set[w];
		}
		return nullptr;
	}

	// `set`'s first free entry, or none.
	static Entry *unused(Entry *set)
	{
		for (uint32_t w = 0; w < Ways; w++) {
			if (!set[w].used)
				return &set[w];
		}
		return nullptr;
	}

	// Keeps `r`, the answer of `key`, in `slot` at rank `tick`, as an equal create's is: every
	// field the device gives each create alone zeroed.
	static void keep(Entry *slot, const uint8_t *key, RgfxPipelineReflection r, uint64_t tick)
	{
		memcpy(slot->key, key, kKeyBytes);
		r.header.size = sizeof(r);
		r.header.seqno = 0;
		r.obj_id = 0;
		r.gpu_resource_id = 0;
		r.unique_identifier = 0;
		r.resource_index = 0;
		slot->reflection = r;
		slot->used = true;
		slot->tick = tick;
	}

	Entry m_entries[Sets * Ways] = {};
	// Every rank this run gives is above every seed's (kSeeds - order).
	uint64_t m_tick = kSeeds;
};

#endif // RGFX_SHARED_PIPELINES_HPP_
