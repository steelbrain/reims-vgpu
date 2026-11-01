/*
 * resource_tests: the shell check of what the resource wrappers answer past Metal's public
 * protocols (RgfxMetalPrivate.h), which Apple's frameworks call (built into shell_tests, which
 * calls rgfxResourceTests()). Each answer checked here is Metal's on the M3 Max (macOS 27), for
 * the kind of object the wrapper stands for.
 *
 * Our own ObjC++; built on the Mac's native arch with the NSObject-based wrappers.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/message.h>

#include <cstdint>

#import "RgfxBuffer.h"
#import "RgfxHeap.h"
#import "RgfxMetalPrivate.h"
#import "RgfxUnsupported.h"

#include "RgfxObjectModel.hpp"
#include "RgfxRingWriter.hpp"
#include "RgfxSerializer.hpp"
#include "reims_graphics_wire.h"
#include "rgfx_test_ring.hpp" // CHECK, g_checks, TestRing, expectMessage

void rgfxResourceTests();
// shell_tests.mm's control ring helpers.
bool ringEmpty(TestRing &ring);
void drainRing(TestRing &ring);

namespace {

// A ring, an id space and a serialiser, as shell_tests.mm's fixture. The transport is never freed:
// an object a test lets go may dealloc after the test and post its destroy here.
struct Fixture {
	TestRing &ring;
	RgfxRingWriter &writer;
	RgfxIdSpace &ids;
	RgfxSerializer &ser;

	Fixture()
	    : ring(*new TestRing(0x8000)),
	      writer(*new RgfxRingWriter(ring.base(), ring.dataSize, RGFX_CONTROL_RING_ID,
	                                 TestRing::doorbell, nullptr)),
	      ids(*new RgfxIdSpace()), ser(*new RgfxSerializer(writer, ids))
	{
		ring.setConsumerIdle(1);
	}
};

MTLHeapDescriptor *heapDescriptor(MTLStorageMode storage, MTLHeapType type)
{
	MTLHeapDescriptor *d = [MTLHeapDescriptor new];
	d.size = 0x100000;
	d.storageMode = storage;
	d.type = type;
	return d;
}

// The heap's members past MTLHeap.
id<RgfxMetalHeapInternals> internals(RgfxHeap *heap)
{
	return (id<RgfxMetalHeapInternals>)heap;
}

// A new MTLBufferDescriptor of `length` bytes and `options` (Metal's own class, at run time).
id bufferDescriptor(NSUInteger length, MTLResourceOptions options)
{
	id d = [NSClassFromString(@"MTLBufferDescriptor") new];
	CHECK(d != nil);
	((void (*)(id, SEL, NSUInteger))objc_msgSend)(d, @selector(setLength:), length);
	((void (*)(id, SEL, NSUInteger))objc_msgSend)(d, @selector(setResourceOptions:), options);
	return d;
}

// A heap answers its base address, its options with a Default hazard tracking mode resolved to
// Untracked (unfiltered the same), and keeps the purgeable state it is told, as Metal's.
void test_heap_answers()
{
	Fixture fx;
	@autoreleasepool {
		RgfxHeap *priv = [RgfxHeap heapWithDescriptor:heapDescriptor(MTLStorageModePrivate,
		                                                             MTLHeapTypePlacement)
		                                       device:nil
		                                   serializer:&fx.ser];
		auto made = expectMessage<RgfxHeapCreate>(fx.ring, RGFX_CMD_HEAP_CREATE);
		CHECK(made.gpu_address != 0 && [internals(priv) gpuAddress] == made.gpu_address);
		// The wire carries the options as asked; the heap answers them resolved.
		CHECK(made.options == MTLResourceStorageModePrivate);
		CHECK(priv.resourceOptions == 0x120 && [internals(priv) unfilteredResourceOptions] == 0x120);
		CHECK(priv.hazardTrackingMode == MTLHazardTrackingModeUntracked);
		CHECK(priv.storageMode == MTLStorageModePrivate);
		// A buffer placed at 0 lies at the heap's base.
		id<MTLBuffer> at0 = [priv newBufferWithLength:0x100
		                                      options:MTLResourceStorageModePrivate
		                                       offset:0];
		CHECK(at0.gpuAddress == [internals(priv) gpuAddress]);

		MTLHeapDescriptor *sd = heapDescriptor(MTLStorageModeShared, MTLHeapTypeAutomatic);
		RgfxHeap *shared = [RgfxHeap heapWithDescriptor:sd device:nil serializer:&fx.ser];
		CHECK(shared.resourceOptions == 0x100 && [internals(shared) unfilteredResourceOptions] == 0x100);
		sd.hazardTrackingMode = MTLHazardTrackingModeTracked;
		RgfxHeap *tracked = [RgfxHeap heapWithDescriptor:sd device:nil serializer:&fx.ser];
		CHECK(tracked.resourceOptions == 0x200 && [internals(tracked) unfilteredResourceOptions] == 0x200);
		CHECK(tracked.hazardTrackingMode == MTLHazardTrackingModeTracked);

		// The state is kept and the previous one answered.
		CHECK([priv setPurgeableState:MTLPurgeableStateVolatile] == MTLPurgeableStateNonVolatile);
		CHECK([priv setPurgeableState:MTLPurgeableStateKeepCurrent] == MTLPurgeableStateVolatile);
		CHECK([priv setPurgeableState:MTLPurgeableStateEmpty] == MTLPurgeableStateVolatile);
		CHECK([priv setPurgeableState:MTLPurgeableStateNonVolatile] == MTLPurgeableStateEmpty);
		CHECK([priv setPurgeableState:MTLPurgeableStateKeepCurrent] ==
		      MTLPurgeableStateNonVolatile);

		// No acceleration structure: nil, logged, nothing posted.
		drainRing(fx.ring);
		CHECK([internals(priv) newAccelerationStructureWithSize:0x1000 resourceIndex:0] == nil);
		CHECK(RgfxUnsupportedWasLogged([RgfxHeap class],
		                               @selector(newAccelerationStructureWithSize:resourceIndex:)));
		CHECK(ringEmpty(fx.ring));
	}
	drainRing(fx.ring);
}

// An Automatic heap places a buffer of a descriptor's length as it places one of a length, at the
// descriptor's alignment where it is larger, with the heap's options whatever the descriptor asks;
// a Placement heap, a length of 0, no descriptor and a descriptor over memory of its own answer
// nil, posting nothing.
void test_heap_buffer_descriptors()
{
	Fixture fx;
	@autoreleasepool {
		RgfxHeap *heap = [RgfxHeap heapWithDescriptor:heapDescriptor(MTLStorageModePrivate,
		                                                             MTLHeapTypeAutomatic)
		                                       device:nil
		                                   serializer:&fx.ser];
		auto made = expectMessage<RgfxHeapCreate>(fx.ring, RGFX_CMD_HEAP_CREATE);
		auto placed = [&](uint64_t offset, uint64_t length) {
			auto m = expectMessage<RgfxBufferAtOffset>(fx.ring, RGFX_CMD_BUFFER_AT_OFFSET);
			CHECK(m.heap_obj_id == heap.rgfxObjId && m.offset == offset && m.length == length);
			// Placed with the heap's options, so the host's heap takes it.
			CHECK(m.options == MTLResourceStorageModePrivate);
		};
		id<MTLBuffer> a = [heap newBufferWithLength:100 options:MTLResourceStorageModePrivate];
		placed(0, 100);
		// Shared options on a Private heap still answer a buffer, placed as one of the heap's.
		id<MTLBuffer> b = [internals(heap) newBufferWithDescriptor:bufferDescriptor(100, 0)];
		placed(256, 100);
		CHECK(b != nil && b.heap == heap && b.length == 100 && b.heapOffset == 0);
		CHECK(b.gpuAddress == made.gpu_address + 256);
		CHECK(b.storageMode == MTLStorageModePrivate && b.contents == nullptr);
		id d = bufferDescriptor(100, MTLResourceStorageModePrivate);
		((void (*)(id, SEL, NSUInteger))objc_msgSend)(d, @selector(setAlignment:), 0x10000);
		id<MTLBuffer> c = [internals(heap) newBufferWithDescriptor:d];
		placed(0x10000, 100);
		CHECK(c.gpuAddress == made.gpu_address + 0x10000 && heap.usedSize == 300);

		CHECK([internals(heap) newBufferWithDescriptor:bufferDescriptor(0, 0)] == nil);
		CHECK([internals(heap) newBufferWithDescriptor:nil] == nil);
		CHECK(ringEmpty(fx.ring));
		id own = bufferDescriptor(100, 0);
		((void (*)(id, SEL, uint64_t))objc_msgSend)(own, @selector(setPinnedGPUAddress:), 0x1000);
		CHECK([internals(heap) newBufferWithDescriptor:own] == nil);
		CHECK(RgfxUnsupportedWasLogged([RgfxHeap class], @selector(newBufferWithDescriptor:)));
		CHECK(ringEmpty(fx.ring));

		RgfxHeap *placement = [RgfxHeap heapWithDescriptor:heapDescriptor(MTLStorageModePrivate,
		                                                                  MTLHeapTypePlacement)
		                                            device:nil
		                                        serializer:&fx.ser];
		drainRing(fx.ring);
		CHECK([internals(placement) newBufferWithDescriptor:bufferDescriptor(100, 0)] == nil);
		CHECK(ringEmpty(fx.ring));
		(void)a;
	}
	drainRing(fx.ring);
}

} // namespace

void rgfxResourceTests()
{
	test_heap_answers();
	test_heap_buffer_descriptors();
}
