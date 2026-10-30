/*
 * shell_tests: the Metal shell's user-space check, run on the Mac's native arch
 * (guest/bundle/Makefile `test`). It exercises the NSObject-based object wrappers
 * — the part of the shell that does not need Apple's private device base — and
 * pins the ROUTING ASSERTION PATTERN the four later slices copy:
 *
 *   build a ring + serialiser (the shared tests/rgfx_test_ring.hpp harness, the
 *   same one the serialiser check uses) -> instantiate a wrapper, which routes its
 *   create through the serialiser -> read the message back off the ring with
 *   expectMessage<T> and assert the exact wire fields, plus the wrapper's getters.
 *
 * The device shell (RgfxMetalDevice) is NOT built into this binary: it subclasses
 * an Apple-private class whose symbols resolve only in the guest. Only the
 * NSObject-based wrappers are tested here.
 *
 * Exits 0 and prints a one-line summary when every check passes; aborts on the
 * first failure.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdint>
#include <cstring>

#import "RgfxBuffer.h"
#import "RgfxEvent.h"
#import "RgfxFence.h"
#import "RgfxResource.h"
#import "RgfxSharedEvent.h"

#include "RgfxObjectModel.hpp"
#include "RgfxRingWriter.hpp"
#include "RgfxSerializer.hpp"
#include "reims_graphics_wire.h"
#include "rgfx_test_ring.hpp" // CHECK, g_checks, TestRing, expectMessage

namespace {

// A ring + id space + serialiser over a plain-buffer region, the fixture every
// wrapper routing test stands up. Mirrors how a device category would reach its
// serialiser, minus the device (nil here, as a wrapper allows).
struct Fixture {
	TestRing ring;
	RgfxRingWriter writer;
	RgfxIdSpace ids;
	RgfxSerializer ser;

	Fixture()
	    : ring(0x8000), writer(ring.base(), ring.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell,
	                           nullptr),
	      ids(), ser(writer, ids)
	{
		ring.setConsumerIdle(1); // host idle, so each post rings the doorbell
	}
};

// A buffer wrapper's create routes an RGFX_CMD_BUFFER_CREATE onto the ring, and
// its getters answer from the created values. This is the pattern the resource
// slice replicates for textures/samplers/heaps.
void test_buffer_routing()
{
	Fixture fx;

	const NSUInteger length = 0x2000;
	const MTLResourceOptions options =
	    MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined;

	RgfxBuffer *buf = [RgfxBuffer bufferWithLength:length
	                                       options:options
	                                        device:nil
	                                    serializer:&fx.ser];
	CHECK(buf != nil);

	// The wrapper's getters answer from the created values.
	CHECK(buf.rgfxObjId == 1);         // first object on this device
	CHECK([buf length] == length);     // MTLBuffer.length
	CHECK([buf gpuAddress] != 0);      // a GPU address was picked
	CHECK(([buf gpuAddress] % RgfxSerializer::kDefaultAddressAlign) == 0);
	CHECK([buf gpuAddress] >= RgfxIdSpace::kDefaultAddressBase);
	CHECK([buf resourceOptions] == options);
	CHECK(buf.device == nil);          // no device in this test
	CHECK([buf serializer] == &fx.ser); // borrowed route is wired

	// contents is a usable, zeroed, length-sized CPU backing.
	void *contents = [buf contents];
	CHECK(contents != nullptr);
	uint8_t *bytes = static_cast<uint8_t *>(contents);
	CHECK(bytes[0] == 0 && bytes[length - 1] == 0);
	bytes[0] = 0xab;
	bytes[length - 1] = 0xcd;
	CHECK(bytes[0] == 0xab && bytes[length - 1] == 0xcd);

	// The create wrote exactly one BUFFER_CREATE, and its fields match the wrapper.
	auto msg = expectMessage<RgfxBufferCreate>(fx.ring, RGFX_CMD_BUFFER_CREATE);
	CHECK(msg.header.seqno == 1);
	CHECK(msg.obj_id == buf.rgfxObjId);
	CHECK(msg.gpu_address == [buf gpuAddress]);
	CHECK(msg.length == length);
	CHECK(msg.options == static_cast<uint32_t>(options));

	// didModifyRange: is a no-op for coherent Shared storage (slice 0) and must not
	// post anything or fault. (Apple deprecated the selector on macOS 27 — Managed
	// storage has no effect on Apple Silicon — but we still implement it for
	// MTLBuffer conformance and older guests, so the call is deprecation-silenced.)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
	[buf didModifyRange:NSMakeRange(0, length)];
#pragma clang diagnostic pop
	RgfxMsgHeader extra = {};
	std::vector<uint8_t> extraBytes;
	CHECK(!fx.ring.next(&extra, &extraBytes)); // nothing more on the ring
}

// The id space is monotonic across wrappers: a second buffer gets the next objId
// and a disjoint, higher GPU address, and its own BUFFER_CREATE.
void test_second_buffer_is_monotonic()
{
	Fixture fx;

	RgfxBuffer *a = [RgfxBuffer bufferWithLength:0x1000
	                                     options:MTLResourceStorageModeShared
	                                      device:nil
	                                  serializer:&fx.ser];
	RgfxBuffer *b = [RgfxBuffer bufferWithLength:0x1000
	                                     options:MTLResourceStorageModeShared
	                                      device:nil
	                                  serializer:&fx.ser];
	CHECK(a != nil && b != nil);
	CHECK(b.rgfxObjId == a.rgfxObjId + 1);
	CHECK([b gpuAddress] >= [a gpuAddress] + 0x1000);

	auto ma = expectMessage<RgfxBufferCreate>(fx.ring, RGFX_CMD_BUFFER_CREATE);
	auto mb = expectMessage<RgfxBufferCreate>(fx.ring, RGFX_CMD_BUFFER_CREATE);
	CHECK(ma.obj_id == a.rgfxObjId && mb.obj_id == b.rgfxObjId);
	CHECK(mb.header.seqno == ma.header.seqno + 1);
}

// A wrapper with no serialiser route cannot be created (a buffer must reach the
// host), and the base label getter/setter work.
void test_wrapper_guards_and_base()
{
	RgfxBuffer *none = [RgfxBuffer bufferWithLength:0x1000
	                                        options:MTLResourceStorageModeShared
	                                         device:nil
	                                     serializer:nullptr];
	CHECK(none == nil);

	Fixture fx;
	RgfxBuffer *buf = [RgfxBuffer bufferWithLength:0x1000
	                                       options:MTLResourceStorageModeShared
	                                        device:nil
	                                    serializer:&fx.ser];
	CHECK(buf.label == nil);
	buf.label = @"scratch";
	CHECK([buf.label isEqualToString:@"scratch"]);
	// RgfxBuffer is a RgfxResource and conforms to MTLBuffer.
	CHECK([buf isKindOfClass:[RgfxResource class]]);
	CHECK([buf conformsToProtocol:@protocol(MTLBuffer)]);
}

// --- sync slice (slice 4): event / fence / shared-event wrappers -------------

// An event wrapper's create routes one RGFX_CMD_EVENT_CREATE (an RgfxObjectCreate
// carrying its objId), and the wrapper carries that id, its device and route.
void test_event_routing()
{
	Fixture fx;

	RgfxEvent *ev = [RgfxEvent eventWithDevice:nil serializer:&fx.ser];
	CHECK(ev != nil);
	CHECK(ev.rgfxObjId == 1); // first object on this device
	CHECK(ev.device == nil);
	CHECK([ev serializer] == &fx.ser);
	CHECK([ev isKindOfClass:[RgfxResource class]]);
	CHECK([ev conformsToProtocol:@protocol(MTLEvent)]);

	auto msg = expectMessage<RgfxObjectCreate>(fx.ring, RGFX_CMD_EVENT_CREATE);
	CHECK(msg.header.seqno == 1);
	CHECK(msg.obj_id == ev.rgfxObjId);

	// Nothing else reached the ring.
	RgfxMsgHeader extra = {};
	std::vector<uint8_t> extraBytes;
	CHECK(!fx.ring.next(&extra, &extraBytes));

	// A null route declines the create (an event must reach the host).
	CHECK([RgfxEvent eventWithDevice:nil serializer:nullptr] == nil);
}

// A fence wrapper's create routes one RGFX_CMD_FENCE_CREATE, same shape as an
// event.
void test_fence_routing()
{
	Fixture fx;

	RgfxFence *fence = [RgfxFence fenceWithDevice:nil serializer:&fx.ser];
	CHECK(fence != nil);
	CHECK(fence.rgfxObjId == 1);
	CHECK([fence conformsToProtocol:@protocol(MTLFence)]);

	auto msg = expectMessage<RgfxObjectCreate>(fx.ring, RGFX_CMD_FENCE_CREATE);
	CHECK(msg.header.seqno == 1);
	CHECK(msg.obj_id == fence.rgfxObjId);

	CHECK([RgfxFence fenceWithDevice:nil serializer:nullptr] == nil);
}

// A shared-event wrapper's create routes one RGFX_CMD_SHARED_EVENT_CREATE carrying
// the initial value, and the wrapper's signaledValue mirror starts there.
void test_shared_event_routing()
{
	Fixture fx;

	const uint64_t initial = 0x1234;
	RgfxSharedEvent *ev = [RgfxSharedEvent sharedEventWithInitialValue:initial
	                                                           device:nil
	                                                       serializer:&fx.ser];
	CHECK(ev != nil);
	CHECK(ev.rgfxObjId == 1);
	CHECK([ev conformsToProtocol:@protocol(MTLSharedEvent)]);

	auto msg = expectMessage<RgfxSharedEventCreate>(fx.ring, RGFX_CMD_SHARED_EVENT_CREATE);
	CHECK(msg.header.seqno == 1);
	CHECK(msg.obj_id == ev.rgfxObjId);
	CHECK(msg.initial_value == initial);

	// The local value mirror starts at the initial value, and a CPU-side set
	// updates the mirror (no wire op for a set in this slice).
	CHECK(ev.signaledValue == initial);
	ev.signaledValue = initial + 7;
	CHECK(ev.signaledValue == initial + 7);

	// The set posts nothing on the ring (documented follow-up, no wire op yet), and
	// the listener/handle stubs are safe no-ops.
	RgfxMsgHeader extra = {};
	std::vector<uint8_t> extraBytes;
	CHECK(!fx.ring.next(&extra, &extraBytes));
	CHECK([ev newSharedEventHandle] == nil);
	MTLSharedEventListener *listener = [[MTLSharedEventListener alloc] init];
	[ev notifyListener:listener
	           atValue:initial + 7
	             block:^(id<MTLSharedEvent> e, uint64_t v) {
		             (void)e;
		             (void)v;
	             }]; // stub: must not fault

	CHECK([RgfxSharedEvent sharedEventWithInitialValue:0 device:nil serializer:nullptr] == nil);
}

} // namespace

int main()
{
	@autoreleasepool {
		test_buffer_routing();
		test_second_buffer_is_monotonic();
		test_wrapper_guards_and_base();
		test_event_routing();
		test_fence_routing();
		test_shared_event_routing();
	}
	std::printf("shell_tests: OK (%d checks)\n", g_checks);
	return 0;
}
