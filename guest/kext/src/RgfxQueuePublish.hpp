// RgfxQueuePublish: a command buffer's publish on its queue's submission ring, made by the kext in
// the call that posts the messages the bundle held for the control ring before it
// (kRgfxAccelMethodSubmitBatch's publish, kRgfxKextFeatureQueuePublish), so a commit behind held
// messages makes one kext call, not two (the batch, then the queue's doorbell).
//
// The publish is the bundle's own (RgfxRingWriter::tryPost), step for step: the ring's producer
// stored, a full fence, the ring's consumer_idle read, and the queue's doorbell written where the
// host marked the ring asleep, or walking for a command buffer it answers at the doorbell, that
// buffer's word naming RGFX_DOORBELL_ASKS where the bundle asks (rgfxQueueDoorbell). The bundle
// writes the buffer and fences its stores before the call; the kext publishes only once every
// message the call carries is posted, after them, so the host sees what it saw before: the held
// messages on the control ring, then the buffer on its queue's ring.
//
// The client keeps a kernel mapping of each of its queues' submission ring headers for it
// (RgfxQueueHeaders): made at the queue's create, and let go at the queue's destroy, at the free
// of the block the ring lies in, and at the context's close, before its blocks go, so the kext
// never writes a ring the process no longer holds.
//
// Header-only and free of the kernel's headers, so the native test (tests/queue_publish_test.cpp)
// checks the rule and the table as the kext keeps them.
//
// Our own code.
#ifndef RGFX_QUEUE_PUBLISH_HPP_
#define RGFX_QUEUE_PUBLISH_HPP_

#include <stdint.h>

#include "reims_graphics_wire.h"

// kRgfxAccelMethodSubmitBatch's publish flags, its third scalar in.
enum {
	// The command buffer is one the host answers at the doorbell (its stream ends in its ask):
	// its doorbell is rung for a walking ring too.
	kRgfxPublishAnswered = 1 << 0,
	// The doorbell's word names RGFX_DOORBELL_ASKS (the device took RGFX_FEATURE_DOORBELL_ASKS),
	// for an answered buffer.
	kRgfxPublishAsks = 1 << 1,
	kRgfxPublishFlags = kRgfxPublishAnswered | kRgfxPublishAsks,
};

// Whether a ring whose consumer_idle reads `idle` is rung after a publish: asleep, or walking for a
// buffer it answers at the doorbell. A walking host comes back to the ring by itself.
inline bool rgfxQueueDoorbellWanted(uint32_t idle, bool answered)
{
	return idle == RGFX_CONSUMER_ASLEEP || (idle == RGFX_CONSUMER_WALKING && answered);
}

// The publish of `producer` on a queue's ring, `ring` its doorbell id, as rgfxQueuePublish's
// caller makes it: the producer stored, a full fence, consumer_idle read. Where its doorbell is
// wanted, true, with `*word` what the doorbell register is written. `Ring`:
//   void storeProducer(uint64_t)   a release store of the ring header's producer;
//   void fullFence();
//   uint32_t consumerIdle()        an acquire load of the ring header's consumer_idle.
template <class Ring>
inline bool rgfxQueuePublish(Ring &r, uint32_t ring, uint64_t producer, uint64_t flags,
                             uint32_t *word)
{
	r.storeProducer(producer);
	r.fullFence();
	const uint32_t idle = r.consumerIdle();
	const bool answered = (flags & kRgfxPublishAnswered) != 0;
	if (!rgfxQueueDoorbellWanted(idle, answered))
		return false;
	*word = answered && (flags & kRgfxPublishAsks) != 0 ? ring | RGFX_DOORBELL_ASKS : ring;
	return true;
}

// A queue's submission ring header as the client maps it: the queue, the block its ring lies in
// (the block's region offset), the ring's doorbell id (its region offset), the mapping, and the
// header in it.
template <class Map>
struct RgfxQueueHeader {
	uint64_t queue;
	uint64_t block;
	uint32_t ring;
	Map *map;
	volatile uint8_t *header;
};

// The client's queues' headers, at most `Capacity`, in no order. Each change and each look is
// under the client's list lock; the mappings a drop takes out are let go by the caller after it.
template <class Map, uint32_t Capacity>
struct RgfxQueueHeaders {
	uint32_t count = 0;
	RgfxQueueHeader<Map> entries[Capacity] = {};
};

// Keeps `entry`: false where the table is full or holds its queue already, the caller then
// letting its mapping go (the queue's buffers are then published by the bundle itself).
template <class Map, uint32_t Capacity>
inline bool rgfxQueueHeaderAdd(RgfxQueueHeaders<Map, Capacity> &t, const RgfxQueueHeader<Map> &entry)
{
	if (t.count == Capacity)
		return false;
	for (uint32_t i = 0; i < t.count; i++) {
		if (t.entries[i].queue == entry.queue)
			return false;
	}
	t.entries[t.count++] = entry;
	return true;
}

// The header of `queue`, or nullptr; valid until the list lock is let go.
template <class Map, uint32_t Capacity>
inline const RgfxQueueHeader<Map> *rgfxQueueHeaderFind(const RgfxQueueHeaders<Map, Capacity> &t,
                                                       uint64_t queue)
{
	for (uint32_t i = 0; i < t.count; i++) {
		if (t.entries[i].queue == queue)
			return &t.entries[i];
	}
	return nullptr;
}

// Drops the headers `drop(entry)` names, each mapping into `out` (Capacity of room) for the caller
// to let go: how many.
template <class Map, uint32_t Capacity, class Drop>
inline uint32_t rgfxQueueHeaderDropIf(RgfxQueueHeaders<Map, Capacity> &t, Drop drop, Map **out)
{
	uint32_t n = 0;
	for (uint32_t i = 0; i < t.count;) {
		if (drop(t.entries[i])) {
			out[n++] = t.entries[i].map;
			t.entries[i] = t.entries[--t.count];
			t.entries[t.count] = {};
		} else {
			i++;
		}
	}
	return n;
}

// The queue's destroy: its header, if kept.
template <class Map, uint32_t Capacity>
inline uint32_t rgfxQueueHeaderDrop(RgfxQueueHeaders<Map, Capacity> &t, uint64_t queue, Map **out)
{
	return rgfxQueueHeaderDropIf(
	    t, [queue](const RgfxQueueHeader<Map> &e) { return e.queue == queue; }, out);
}

// The free of the block at `block`: the headers of the rings in it.
template <class Map, uint32_t Capacity>
inline uint32_t rgfxQueueHeaderDropBlock(RgfxQueueHeaders<Map, Capacity> &t, uint64_t block,
                                         Map **out)
{
	return rgfxQueueHeaderDropIf(
	    t, [block](const RgfxQueueHeader<Map> &e) { return e.block == block; }, out);
}

// The context's close: every header.
template <class Map, uint32_t Capacity>
inline uint32_t rgfxQueueHeaderDropAll(RgfxQueueHeaders<Map, Capacity> &t, Map **out)
{
	return rgfxQueueHeaderDropIf(t, [](const RgfxQueueHeader<Map> &) { return true; }, out);
}

#endif // RGFX_QUEUE_PUBLISH_HPP_
