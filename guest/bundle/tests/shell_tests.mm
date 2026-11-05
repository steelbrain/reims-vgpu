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
#import <objc/message.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <tuple>

#import "RgfxBuffer.h"
#import "RgfxResource.h"
#import "RgfxEvent.h"
#import "RgfxFence.h"
#import "RgfxSharedEvent.h"
#import "RgfxDepthStencilState.h"
#import "RgfxHeap.h"
#import "RgfxSampler.h"
#import "RgfxTexture.h"
#import "RgfxArgumentEncoder.h"
#import "RgfxBinaryArchive.h"
#import "RgfxBlockCall.h"
#import "RgfxComputePipelineState.h"
#import "RgfxFunction.h"
#import "RgfxLibrary.h"
#import "RgfxMetalError.h"
#import "RgfxRenderPipelineState.h"
#include <set>
#include <vector>
#import "RgfxCommandObjects.h"
#import "RgfxIOSurface.h"
#import "RgfxIndirectCommandBuffer.h"
#import "RgfxMetalPrivate.h"
#import "RgfxUnsupported.h"
#import "RgfxUploads.h"

#include "RgfxCommitOrder.hpp"
#include "RgfxHeapSizes.hpp"
#include "RgfxSlabs.hpp"
#include "../../kext/src/RgfxUserClientABI.hpp"

#include <CoreVideo/CoreVideo.h>
#include <IOSurface/IOSurface.h>

#include <crt_externs.h>
#include <mach/mach.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

#include "RgfxCompletionThread.hpp"
#include "RgfxEncoder.hpp"
#include "RgfxFunctionConstants.hpp"
#include "RgfxManagedLayout.hpp"
#include "RgfxMemoryShadow.hpp"
#include "RgfxObjectModel.hpp"
#include "RgfxRegion.hpp"
#include "RgfxReplyReader.hpp"
#include "RgfxRingWriter.hpp"
#include "RgfxSerializer.hpp"
#include "reims_graphics_wire.h"
#include "rgfx_test_ring.hpp" // CHECK, g_checks, TestRing, expectMessage


@interface TestHandle : RgfxResource <MTLBuffer, MTLTexture, MTLRenderPipelineState,
                                      MTLComputePipelineState, MTLDepthStencilState,
                                      MTLSamplerState, MTLFence, MTLHeap,
                                      MTLIndirectCommandBuffer>
@end

@implementation TestHandle
@end

// A drawable that records how it was presented: "now", "at T" or "after D", in order.
@interface TestDrawable : NSObject <MTLDrawable>
@property(nonatomic) NSMutableArray<NSString *> *presents;
@end

@implementation TestDrawable
- (instancetype)init
{
	if ((self = [super init]))
		_presents = [NSMutableArray array];
	return self;
}
- (void)present
{
	@synchronized(self) {
		[_presents addObject:@"now"];
	}
}
- (void)presentAtTime:(CFTimeInterval)presentationTime
{
	@synchronized(self) {
		[_presents addObject:[NSString stringWithFormat:@"at %g", presentationTime]];
	}
}
- (void)presentAfterMinimumDuration:(CFTimeInterval)duration
{
	@synchronized(self) {
		[_presents addObject:[NSString stringWithFormat:@"after %g", duration]];
	}
}
- (void)addPresentedHandler:(MTLDrawablePresentedHandler)block
{
	(void)block;
}
- (CFTimeInterval)presentedTime
{
	return 0;
}
- (NSUInteger)drawableID
{
	return 1;
}
- (NSArray<NSString *> *)seen
{
	@synchronized(self) {
		return [_presents copy];
	}
}
@end
// The control ring's helpers (below), which the tests in the namespace use too.
uint64_t nextDestroyed(TestRing &ring);
bool ringEmpty(TestRing &ring);
void drainRing(TestRing &ring);
void syncAsCommitted(std::initializer_list<id> textures);

namespace {

// A ring + id space + serialiser over a plain-buffer region, the fixture every
// wrapper routing test stands up. Mirrors how a device category would reach its
// serialiser, minus the device (nil here, as a wrapper allows).
// The transport is never freed: an object a test lets go may dealloc after the test (its
// autorelease pool drains later, or a retirement waits) and post its destroy here.
struct Fixture {
	TestRing &ring;
	RgfxRingWriter &writer;
	RgfxIdSpace &ids;
	RgfxSerializer &ser;

	// `ringBytes`: the control ring's, which a library's bytes must fit.
	explicit Fixture(uint32_t ringBytes = 0x8000)
	    : ring(*new TestRing(ringBytes)),
	      writer(*new RgfxRingWriter(ring.base(), ring.dataSize, RGFX_CONTROL_RING_ID,
	                                 TestRing::doorbell, nullptr)),
	      ids(*new RgfxIdSpace()), ser(*new RgfxSerializer(writer, ids))
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
	// Its options as Metal answers them: a Default hazard tracking mode is Tracked.
	CHECK([buf resourceOptions] == (options | MTLResourceHazardTrackingModeTracked));
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

// The bytes this process has resident or compressed, as its footprint counts them.
uint64_t physFootprint()
{
	task_vm_info_data_t info = {};
	mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
	if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) !=
	    KERN_SUCCESS)
		return 0;
	return info.phys_footprint;
}

// A Private buffer's bytes are the GPU's alone: its create routes as any other's, and no guest
// memory stands for them, whatever its length (`contents` answers NULL, as Metal documents for
// the mode). A Managed one made without a region still has zeroed contents of its own.
void test_private_buffer_has_no_contents()
{
	Fixture fx;
	// Drained while fx lives: a buffer's dealloc posts its destroy through fx.
	@autoreleasepool {
		const NSUInteger length = 0x4000000; // 64 MiB, which a zeroed backing would make resident
		const uint64_t before = physFootprint();
		RgfxBuffer *priv = [RgfxBuffer bufferWithLength:length
		                                        options:MTLResourceStorageModePrivate
		                                         device:nil
		                                     serializer:&fx.ser];
		CHECK(priv != nil);
		CHECK([priv contents] == nullptr);
		CHECK(physFootprint() < before + 0x800000);
		CHECK([priv length] == length && [priv storageMode] == MTLStorageModePrivate);
		auto msg = expectMessage<RgfxBufferCreate>(fx.ring, RGFX_CMD_BUFFER_CREATE);
		CHECK(msg.obj_id == priv.rgfxObjId && msg.gpu_address == [priv gpuAddress]);
		CHECK(msg.length == length);
		CHECK(msg.options == static_cast<uint32_t>(MTLResourceStorageModePrivate));
		// didModifyRange: on it posts nothing and touches nothing.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
		[priv didModifyRange:NSMakeRange(0, 0x1000)];
#pragma clang diagnostic pop

		RgfxBuffer *managed = [RgfxBuffer bufferWithLength:0x1000
		                                           options:MTLResourceStorageModeManaged
		                                            device:nil
		                                        serializer:&fx.ser];
		CHECK(managed != nil && [managed contents] != nullptr);
		CHECK(static_cast<uint8_t *>([managed contents])[0xfff] == 0);
		(void)expectMessage<RgfxBufferCreate>(fx.ring, RGFX_CMD_BUFFER_CREATE);
		RgfxMsgHeader extra = {};
		std::vector<uint8_t> extraBytes;
		CHECK(!fx.ring.next(&extra, &extraBytes)); // nothing more on the ring
	}
}

// The device's buffers answer as Metal's (measured on the M3 Max, macOS 27): a Shared one lies in
// a slab of the region with the bytes given (or zeros), a Managed one's CPU copy too, and a
// Private one has no contents. None of 0 bytes or past maxBufferLength, in any mode; none where
// the region has no room for a Shared or Managed one (no device buffer the host would never see);
// none for bytes with a Private one. A refusal posts nothing.
void test_device_buffers_answer_as_metal()
{
	Fixture fx;
	std::vector<uint8_t> mem(RGFX_ALLOCATIONS_OFFSET + 0x80000, 0xee);
	RgfxRegion region(mem.data(), mem.size());
	RgfxRegionSlabSource source(region, fx.ser);
	RgfxSlabs slabs(source);
	auto make = [&](NSUInteger length, MTLResourceOptions options, const void *bytes,
	                NSUInteger maxLength, RgfxSlabs *in) {
		return [RgfxBuffer bufferForDeviceWithLength:length
		                                     options:options
		                                       bytes:bytes
		                                   maxLength:maxLength
		                                       slabs:in
		                                      device:nil
		                                  serializer:&fx.ser];
	};
	auto nothingPosted = [&]() {
		RgfxMsgHeader h = {};
		std::vector<uint8_t> bytes;
		return !fx.ring.next(&h, &bytes);
	};
	@autoreleasepool {
		uint8_t pattern[0x100];
		for (size_t i = 0; i < sizeof(pattern); i++)
			pattern[i] = static_cast<uint8_t>(i * 3 + 1);
		RgfxBuffer *shared = make(sizeof(pattern), MTLResourceStorageModeShared, pattern, 0x2000,
		                          &slabs);
		CHECK(shared != nil && shared.length == sizeof(pattern));
		uint8_t *at = static_cast<uint8_t *>([shared contents]);
		CHECK(at >= mem.data() + RGFX_ALLOCATIONS_OFFSET && at < mem.data() + mem.size());
		CHECK(std::memcmp(at, pattern, sizeof(pattern)) == 0);
		// Its page past the bytes reads zero, whatever the region held there.
		CHECK(at[sizeof(pattern)] == 0 && at[0xfff] == 0);
		auto heap = expectMessage<RgfxHeapOverHostMemory>(fx.ring, RGFX_CMD_HEAP_OVER_HOST_MEMORY);
		auto placed = expectMessage<RgfxBufferAtOffset>(fx.ring, RGFX_CMD_BUFFER_AT_OFFSET);
		CHECK(placed.obj_id == shared.rgfxObjId && placed.length == sizeof(pattern) &&
		      placed.heap_obj_id == heap.obj_id);
		CHECK(heap.region_offset + placed.offset == static_cast<uint64_t>(at - mem.data()));
		CHECK(shared.gpuAddress == heap.gpu_address + placed.offset);

		RgfxBuffer *managed = make(0x1000, MTLResourceStorageModeManaged, nullptr, 0x2000, &slabs);
		CHECK(managed != nil && managed.storageMode == MTLStorageModeManaged);
		CHECK(static_cast<uint8_t *>([managed contents])[0] == 0 &&
		      static_cast<uint8_t *>([managed contents])[0xfff] == 0);
		auto over = expectMessage<RgfxBufferOverHostMemory>(fx.ring, RGFX_CMD_BUFFER_OVER_HOST_MEMORY);
		CHECK(over.obj_id == managed.rgfxObjId &&
		      over.region_offset ==
		          static_cast<uint64_t>(static_cast<uint8_t *>([managed contents]) - mem.data()));

		RgfxBuffer *priv = make(0x1000, MTLResourceStorageModePrivate, nullptr, 0x2000, &slabs);
		CHECK(priv != nil && [priv contents] == nullptr);
		(void)expectMessage<RgfxBufferCreate>(fx.ring, RGFX_CMD_BUFFER_CREATE);
		// No limit known: none held back.
		RgfxBuffer *unlimited = make(0x3000, MTLResourceStorageModePrivate, nullptr, 0, &slabs);
		CHECK(unlimited != nil);
		(void)expectMessage<RgfxBufferCreate>(fx.ring, RGFX_CMD_BUFFER_CREATE);
		CHECK(nothingPosted());

		const MTLResourceOptions modes[] = { MTLResourceStorageModeShared,
			                                 MTLResourceStorageModeManaged,
			                                 MTLResourceStorageModePrivate };
		for (MTLResourceOptions mode : modes) {
			CHECK(make(0, mode, nullptr, 0x2000, &slabs) == nil);
			CHECK(make(0x2001, mode, nullptr, 0x2000, &slabs) == nil);
		}
		// Bytes for a Private buffer; a Shared or Managed one with no slabs, or too large for
		// what the region has left.
		CHECK(make(0x100, MTLResourceStorageModePrivate, pattern, 0x2000, &slabs) == nil);
		CHECK(make(0x100, MTLResourceStorageModeShared, nullptr, 0x2000, nullptr) == nil);
		CHECK(make(0x80000, MTLResourceStorageModeShared, nullptr, 0, &slabs) == nil);
		CHECK(make(0x200000, MTLResourceStorageModeManaged, pattern, 0, &slabs) == nil);
		CHECK(nothingPosted());
	} // the buffers go while the region lives
	std::vector<uint8_t> drain;
	RgfxMsgHeader h;
	while (fx.ring.next(&h, &drain)) {
	}
}

// What a device buffer in a slab answers, as both x86-64 Macs' GPUs agree (measured, AMD 5300M and
// Intel 630): whole 4 KiB pages, its contents and GPU address a page apart from the next, its
// heap none. A buffer made over a freed one's pages reads zeros, and the freed one's destroy comes
// before the new one's create; its pages go back to its slab, not to the region.
void test_device_buffers_take_pages_of_slabs()
{
	Fixture fx;
	std::vector<uint8_t> mem(RGFX_ALLOCATIONS_OFFSET + 0x400000, 0xee);
	RgfxRegion region(mem.data(), mem.size());
	RgfxRegionSlabSource source(region, fx.ser);
	RgfxSlabs slabs(source);
	auto make = [&](NSUInteger length, MTLResourceOptions options) {
		return [RgfxBuffer bufferForDeviceWithLength:length
		                                     options:options
		                                       bytes:nullptr
		                                   maxLength:0
		                                       slabs:&slabs
		                                      device:nil
		                                  serializer:&fx.ser];
	};
	uint64_t firstId = 0;
	@autoreleasepool {
		NSMutableArray<RgfxBuffer *> *held = [NSMutableArray array];
		for (int i = 0; i < 64; i++)
			[held addObject:make(256, MTLResourceStorageModeShared)];
		for (NSUInteger i = 0; i < held.count; i++) {
			RgfxBuffer *b = held[i];
			CHECK(b.allocatedSize == 0x1000 && b.length == 256 && b.heap == nil &&
			      b.heapOffset == 0);
			CHECK(reinterpret_cast<uintptr_t>([b contents]) % 0x1000 == 0 && b.gpuAddress % 0x1000 == 0);
			if (i > 0 && i < 32) {
				CHECK(b.gpuAddress == held[i - 1].gpuAddress + 0x1000);
				CHECK(static_cast<uint8_t *>([b contents]) ==
				      static_cast<uint8_t *>([held[i - 1] contents]) + 0x1000);
			}
			// Metal's default hazard tracking for a buffer: Tracked.
			CHECK(b.resourceOptions == MTLResourceHazardTrackingModeTracked);
		}
		CHECK(make(0x10000, MTLResourceStorageModeShared).allocatedSize == 0x10000);
		CHECK(make(0x10001, MTLResourceStorageModeShared).allocatedSize == 0x11000);
		RgfxBuffer *mib = make(0x100001, MTLResourceStorageModeManaged);
		CHECK(mib.allocatedSize == 0x101000 && mib.length == 0x100001);
		// A Private one, the GPU's alone and in no slab, answers its length in whole pages too,
		// as the x86 Macs do (oracle_bundle buffers).
		CHECK(make(1, MTLResourceStorageModePrivate).allocatedSize == 0x1000);
		CHECK(make(0x1001, MTLResourceStorageModePrivate).allocatedSize == 0x2000);
		CHECK(make(0x10000, MTLResourceStorageModePrivate).allocatedSize == 0x10000);
		uint8_t *first = static_cast<uint8_t *>([held[0] contents]);
		firstId = held[0].rgfxObjId;
		std::memset(first, 0xa5, 256);
		drainRing(fx.ring);
		[held removeObjectAtIndex:0];
	}
	@autoreleasepool {
		// The first buffer's destroy, then a buffer over its page, zeroed.
		RgfxMsgHeader h = {};
		std::vector<uint8_t> bytes;
		bool destroyed = false;
		while (fx.ring.next(&h, &bytes)) {
			RgfxObjectDestroy d = {};
			if (h.type == RGFX_CMD_OBJECT_DESTROY && bytes.size() == sizeof(d)) {
				std::memcpy(&d, bytes.data(), sizeof(d));
				destroyed = destroyed || d.obj_id == firstId;
			}
		}
		CHECK(destroyed);
		CHECK(slabs.stats().usedBytes == 0 && slabs.stats().slabs >= 2);
		RgfxBuffer *again = make(0x1000, MTLResourceStorageModeShared);
		CHECK(again != nil && static_cast<uint8_t *>([again contents])[0] == 0 &&
		      static_cast<uint8_t *>([again contents])[255] == 0);
		auto placed = expectMessage<RgfxBufferAtOffset>(fx.ring, RGFX_CMD_BUFFER_AT_OFFSET);
		CHECK(placed.offset == 0);
	}
	drainRing(fx.ring);
}

// Where the device took RGFX_FEATURE_MANAGED_AT_OFFSET, a Managed buffer's slab is a Shared heap
// over its block, of the buffers' cache mode and hazard tracking, and each buffer a
// BUFFER_MANAGED_AT_OFFSET at an address of its own, its CPU copy its pages of the heap: the host
// imports no memory per buffer. Its ranges post as before; its free destroys it and gives its
// address back, and the pages go back to the slab.
void test_managed_buffers_place_their_cpu_copies_in_heaps()
{
	Fixture fx;
	std::vector<uint8_t> mem(RGFX_ALLOCATIONS_OFFSET + 0x400000, 0xee);
	RgfxRegion region(mem.data(), mem.size());
	RgfxRegionSlabSource source(region, fx.ser, true);
	RgfxSlabs slabs(source);
	auto make = [&](NSUInteger length, MTLResourceOptions options) {
		return [RgfxBuffer bufferForDeviceWithLength:length
		                                     options:options
		                                       bytes:nullptr
		                                   maxLength:0
		                                       slabs:&slabs
		                                      device:nil
		                                  serializer:&fx.ser];
	};
	uint64_t firstId = 0;
	@autoreleasepool {
		const MTLResourceOptions managed = MTLResourceStorageModeManaged;
		RgfxBuffer *a = make(256, managed);
		RgfxBuffer *b = make(0x1800, managed | MTLResourceCPUCacheModeWriteCombined);
		RgfxBuffer *c = make(0x1000, managed);
		CHECK(a != nil && b != nil && c != nil);
		CHECK(a.storageMode == MTLStorageModeManaged && a.allocatedSize == 0x1000);
		auto heap = expectMessage<RgfxHeapOverHostMemory>(fx.ring, RGFX_CMD_HEAP_OVER_HOST_MEMORY);
		CHECK(heap.options == static_cast<uint32_t>(MTLResourceHazardTrackingModeTracked) &&
		      heap.heap_type == 1);
		auto ma = expectMessage<RgfxBufferManagedAtOffset>(fx.ring, RGFX_CMD_BUFFER_MANAGED_AT_OFFSET);
		CHECK(ma.obj_id == a.rgfxObjId && ma.heap_obj_id == heap.obj_id && ma.offset == 0 &&
		      ma.length == 256 && ma.options == static_cast<uint32_t>(managed));
		CHECK(ma.gpu_address == a.gpuAddress && ma.gpu_address != 0);
		CHECK(heap.region_offset + ma.offset ==
		      static_cast<uint64_t>(static_cast<uint8_t *>([a contents]) - mem.data()));
		// Another cache mode is another class, with a heap of its own.
		auto wcHeap =
		    expectMessage<RgfxHeapOverHostMemory>(fx.ring, RGFX_CMD_HEAP_OVER_HOST_MEMORY);
		CHECK(wcHeap.options == static_cast<uint32_t>(MTLResourceCPUCacheModeWriteCombined |
		                                              MTLResourceHazardTrackingModeTracked));
		auto mb = expectMessage<RgfxBufferManagedAtOffset>(fx.ring, RGFX_CMD_BUFFER_MANAGED_AT_OFFSET);
		CHECK(mb.heap_obj_id == wcHeap.obj_id && mb.offset == 0 && mb.length == 0x1800);
		auto mc = expectMessage<RgfxBufferManagedAtOffset>(fx.ring, RGFX_CMD_BUFFER_MANAGED_AT_OFFSET);
		CHECK(mc.heap_obj_id == heap.obj_id && mc.offset == 0x1000 && mc.length == 0x1000);
		CHECK(c.gpuAddress != a.gpuAddress && c.gpuAddress != heap.gpu_address + 0x1000);
		// A range the CPU wrote posts as for any Managed buffer.
		static_cast<uint8_t *>([c contents])[8] = 0x5a;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
		[c didModifyRange:NSMakeRange(8, 1)];
#pragma clang diagnostic pop
		auto range =
		    expectMessage<RgfxBufferDidModifyRange>(fx.ring, RGFX_CMD_BUFFER_DID_MODIFY_RANGE);
		CHECK(range.obj_id == c.rgfxObjId && range.location == 8 && range.length == 1);
		firstId = a.rgfxObjId;
		drainRing(fx.ring);
	}
	// All three went: their destroys, and their pages back in the slabs.
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	bool destroyed = false;
	while (fx.ring.next(&h, &bytes)) {
		RgfxObjectDestroy d = {};
		if (h.type == RGFX_CMD_OBJECT_DESTROY && bytes.size() == sizeof(d)) {
			std::memcpy(&d, bytes.data(), sizeof(d));
			destroyed = destroyed || d.obj_id == firstId;
		}
	}
	CHECK(destroyed && slabs.stats().usedBytes == 0);
	// Without the feature, a Managed buffer is a BUFFER_OVER_HOST_MEMORY over its range.
	RgfxRegionSlabSource plain(region, fx.ser);
	CHECK(!plain.heaped(MTLResourceStorageModeManaged) &&
	      plain.heaped(MTLResourceStorageModeShared) &&
	      source.heaped(MTLResourceStorageModeManaged) &&
	      !source.heaped(MTLResourceStorageModePrivate));
}

// A heap the device makes is the host's, in Private storage, of any type. None of 0 bytes (plain
// Metal's nil; its validation stops one) or past maxBufferLength, in any type. A refusal posts
// nothing. A Shared one ends the process
// (test_shared_heaps_and_cpu_depth_textures_end_the_process).
void test_device_heaps_answer_as_metal()
{
	Fixture fx;
	auto make = [&](uint64_t size, MTLStorageMode mode, MTLHeapType type, NSUInteger maxLength) {
		MTLHeapDescriptor *d = [MTLHeapDescriptor new];
		d.size = size;
		d.storageMode = mode;
		d.type = type;
		return [RgfxHeap heapForDeviceWithDescriptor:d
		                                   maxLength:maxLength
		                                      region:nullptr
		                                      device:nil
		                                  serializer:&fx.ser];
	};
	@autoreleasepool {
		RgfxHeap *priv = make(0x1000, MTLStorageModePrivate, MTLHeapTypeAutomatic, 0x10000);
		CHECK(priv != nil && priv.rgfxContents == nullptr);
		(void)expectMessage<RgfxHeapCreate>(fx.ring, RGFX_CMD_HEAP_CREATE);
		// No limit known: none held back.
		RgfxHeap *unlimited = make(0x20000, MTLStorageModePrivate, MTLHeapTypePlacement, 0);
		CHECK(unlimited != nil);
		(void)expectMessage<RgfxHeapCreate>(fx.ring, RGFX_CMD_HEAP_CREATE);
		CHECK(ringEmpty(fx.ring));

		const MTLHeapType types[] = { MTLHeapTypeAutomatic, MTLHeapTypePlacement };
		for (MTLHeapType type : types) {
			CHECK(make(0, MTLStorageModePrivate, type, 0x10000) == nil);
			CHECK(make(0x10001, MTLStorageModePrivate, type, 0x10000) == nil);
		}
		CHECK(ringEmpty(fx.ring));
	}
	drainRing(fx.ring);
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

	// The creators with options (SPI) answer as the x86 Macs' do: an event for any options, a
	// shared event for options 0 and 1 and nil, with nothing sent, for any other.
	// Each is kept until the end, so that no destroy comes between the creates.
	NSMutableArray *kept = [NSMutableArray array];
	for (NSInteger options : {0, 1, 2, 3, 16}) {
		RgfxEvent *withOptions = [RgfxEvent eventWithOptions:options device:nil serializer:&fx.ser];
		CHECK(withOptions != nil);
		if (withOptions != nil)
			[kept addObject:withOptions];
		expectMessage<RgfxObjectCreate>(fx.ring, RGFX_CMD_EVENT_CREATE);
	}
	for (NSInteger options : {0, 1}) {
		RgfxSharedEvent *shared = [RgfxSharedEvent sharedEventWithOptions:options
		                                                           device:nil
		                                                       serializer:&fx.ser];
		CHECK(shared != nil);
		if (shared != nil)
			[kept addObject:shared];
		expectMessage<RgfxSharedEventCreate>(fx.ring, RGFX_CMD_SHARED_EVENT_CREATE);
	}
	for (NSInteger options : {2, 3, 4, 8, 16})
		CHECK([RgfxSharedEvent sharedEventWithOptions:options device:nil serializer:&fx.ser] == nil);
	CHECK(!fx.ring.next(&extra, &extraBytes));
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
// the initial value, and its signaledValue starts there. A CPU set goes to the host
// (RGFX_CMD_SHARED_EVENT_SET_VALUE) and raises the value; one to a lower value goes too,
// and leaves it, as Metal's does.
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

	CHECK(ev.signaledValue == initial);
	ev.signaledValue = initial + 7;
	CHECK(ev.signaledValue == initial + 7);
	auto set = expectMessage<RgfxSharedEventSetValue>(fx.ring, RGFX_CMD_SHARED_EVENT_SET_VALUE);
	CHECK(set.obj_id == ev.rgfxObjId && set.value == initial + 7);
	ev.signaledValue = 1;
	CHECK(ev.signaledValue == initial + 7);
	auto lower = expectMessage<RgfxSharedEventSetValue>(fx.ring, RGFX_CMD_SHARED_EVENT_SET_VALUE);
	CHECK(lower.value == 1);

	RgfxMsgHeader extra = {};
	std::vector<uint8_t> extraBytes;
	CHECK(!fx.ring.next(&extra, &extraBytes));
	CHECK([ev newSharedEventHandle] == nil);

	CHECK([RgfxSharedEvent sharedEventWithInitialValue:0 device:nil serializer:nullptr] == nil);
}


// The CPU's copies of a texture: one over a buffer writes and reads the region's rows in the
// buffer's bytes, at the texture's rows; a Private texture's are refused with nothing posted,
// as is a device texture's copy with no device to blit on.
void test_texture_cpu_copies()
{
	Fixture fx;
	@autoreleasepool {
	// A 4 x 3 BGRA8 texture over a buffer, its rows 32 bytes apart from offset 16.
	std::vector<uint8_t> memory(16 + 32 * 3, 0xee);
	RgfxBuffer *buf = [RgfxBuffer bufferOverHostMemoryWithContents:memory.data()
	                                                  regionOffset:0x10000
	                                                        length:memory.size()
	                                                       options:MTLResourceStorageModeShared
	                                                   deallocator:nil
	                                                        device:nil
	                                                    serializer:&fx.ser];
	MTLTextureDescriptor *desc =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:4
	                                                      height:3
	                                                   mipmapped:NO];
	desc.storageMode = MTLStorageModeShared;
	RgfxTexture *tex = [RgfxTexture textureWithDescriptor:desc
	                                               buffer:buf
	                                               offset:16
	                                          bytesPerRow:32
	                                               device:nil
	                                           serializer:&fx.ser];
	CHECK(tex != nil);
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	while (fx.ring.next(&h, &bytes)) {
	}

	// Two pixels of each of two rows, from (1, 1), the caller's rows 12 bytes apart.
	uint8_t in[24];
	for (int i = 0; i < 24; i++)
		in[i] = static_cast<uint8_t>(i);
	[tex replaceRegion:MTLRegionMake2D(1, 1, 2, 2) mipmapLevel:0 withBytes:in bytesPerRow:12];
	CHECK(std::memcmp(&memory[16 + 32 * 1 + 4], &in[0], 8) == 0);
	CHECK(std::memcmp(&memory[16 + 32 * 2 + 4], &in[12], 8) == 0);
	CHECK(memory[16 + 32 * 1] == 0xee && memory[16 + 32 * 1 + 12] == 0xee); // around it
	CHECK(memory[16 + 32 * 0 + 4] == 0xee);
	uint8_t out[16] = {};
	[tex getBytes:out bytesPerRow:8 fromRegion:MTLRegionMake2D(1, 1, 2, 2) mipmapLevel:0];
	CHECK(std::memcmp(&out[0], &in[0], 8) == 0 && std::memcmp(&out[8], &in[12], 8) == 0);
	// Past the texture, or rows shorter than the region's: nothing is copied.
	[tex replaceRegion:MTLRegionMake2D(3, 0, 2, 1) mipmapLevel:0 withBytes:in bytesPerRow:8];
	[tex replaceRegion:MTLRegionMake2D(0, 0, 2, 1) mipmapLevel:0 withBytes:in bytesPerRow:4];
	CHECK(memory[16] == 0xee && memory[16 + 12] == 0xee);

	// A Private texture: refused, nothing posted. A Shared device texture with no device:
	// no queue to blit on, nothing posted.
	desc.storageMode = MTLStorageModePrivate;
	RgfxTexture *priv = [RgfxTexture textureWithDescriptor:desc
	                                                  heap:RGFX_OBJ_NONE
	                                            heapOffset:0
	                                                device:nil
	                                            serializer:&fx.ser];
	desc.storageMode = MTLStorageModeShared;
	RgfxTexture *shared = [RgfxTexture textureWithDescriptor:desc
	                                                    heap:RGFX_OBJ_NONE
	                                              heapOffset:0
	                                                  device:nil
	                                              serializer:&fx.ser];
	CHECK(priv != nil && shared != nil);
	while (fx.ring.next(&h, &bytes)) {
	}
	[priv replaceRegion:MTLRegionMake2D(0, 0, 2, 2) mipmapLevel:0 withBytes:in bytesPerRow:12];
	[priv getBytes:out bytesPerRow:8 fromRegion:MTLRegionMake2D(0, 0, 2, 2) mipmapLevel:0];
	[shared replaceRegion:MTLRegionMake2D(0, 0, 2, 2) mipmapLevel:0 withBytes:in bytesPerRow:12];
	CHECK(!fx.ring.next(&h, &bytes));

	// A resource's purgeable state is kept as told, the previous one answered, and nothing
	// is posted: the device purges nothing. A texture over a buffer stays NonVolatile, as
	// Metal's (resource_tests.mm).
	CHECK([buf setPurgeableState:MTLPurgeableStateKeepCurrent] == MTLPurgeableStateNonVolatile);
	CHECK([buf setPurgeableState:MTLPurgeableStateVolatile] == MTLPurgeableStateNonVolatile);
	CHECK([buf setPurgeableState:MTLPurgeableStateKeepCurrent] == MTLPurgeableStateVolatile);
	CHECK([tex setPurgeableState:MTLPurgeableStateEmpty] == MTLPurgeableStateNonVolatile);
	CHECK([tex setPurgeableState:MTLPurgeableStateNonVolatile] == MTLPurgeableStateNonVolatile);
	CHECK(!fx.ring.next(&h, &bytes));

	// A buffer's resource options decode as a texture's; a resource made in no heap answers
	// none, and cannot be made aliasable.
	CHECK(buf.storageMode == MTLStorageModeShared && tex.storageMode == MTLStorageModeShared);
	CHECK(buf.cpuCacheMode == MTLCPUCacheModeDefaultCache);
	CHECK(buf.hazardTrackingMode == MTLHazardTrackingModeTracked);
	CHECK(buf.heap == nil && buf.heapOffset == 0 && tex.heap == nil);
	[buf makeAliasable];
	CHECK(!buf.isAliasable);
	CHECK([buf setOwnerWithIdentity:0] == KERN_SUCCESS);
	// No resource is protected: the device protects no content.
	CHECK([buf protectionOptions] == 0 && [tex protectionOptions] == 0);
	}
}

// A tiled texture (newTiledTextureWithDescriptor:offset:bytesPerRow:, RgfxBuffer.mm), as
// WindowServer's MetalTiledBacking makes one: a tile of a Shared backing buffer at the backing's
// pitch, from a Managed descriptor. It is a texture over the buffer's bytes with the buffer's
// storage (TEXTURE_FROM_BUFFER), whose CPU copies read and write the tile's bytes in the
// buffer. What Metal's validation asserts at answers nil and posts nothing.
void test_tiled_texture_over_backing()
{
	Fixture fx;
	@autoreleasepool {
	// A backing of 8 x 6 BGRA8 pixels, 32 bytes a row, and a 3 x 2 tile of it at (2, 3).
	const NSUInteger pitch = 32, tileOffset = 3 * pitch + 2 * 4;
	std::vector<uint8_t> memory(pitch * 6, 0xee);
	RgfxBuffer *buf = [RgfxBuffer bufferOverHostMemoryWithContents:memory.data()
	                                                  regionOffset:0x10000
	                                                        length:memory.size()
	                                                       options:MTLResourceStorageModeShared
	                                                   deallocator:nil
	                                                        device:nil
	                                                    serializer:&fx.ser];
	CHECK(buf != nil);
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	while (fx.ring.next(&h, &bytes)) {
	}
	MTLTextureDescriptor *desc =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:3
	                                                      height:2
	                                                   mipmapped:NO];
	desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
	desc.storageMode = MTLStorageModeManaged;
#pragma clang diagnostic pop
	id<RgfxMetalBufferInternals> tiled = (id<RgfxMetalBufferInternals>)buf;
	id<MTLTexture> tex = [tiled newTiledTextureWithDescriptor:desc
	                                                    offset:tileOffset
	                                               bytesPerRow:pitch];
	CHECK(tex != nil && tex.buffer == buf && tex.bufferOffset == tileOffset &&
	      tex.bufferBytesPerRow == pitch);
	CHECK(tex.storageMode == MTLStorageModeShared && tex.width == 3 && tex.height == 2);
	auto made = expectMessage<RgfxTextureFromBuffer>(fx.ring, RGFX_CMD_TEXTURE_FROM_BUFFER);
	CHECK(made.buffer_obj_id == buf.rgfxObjId && made.offset == tileOffset &&
	      made.bytes_per_row == pitch);
	CHECK(made.desc.width == 3 && made.desc.height == 2 &&
	      made.desc.pixel_format == MTLPixelFormatBGRA8Unorm &&
	      (made.desc.options & MTLResourceStorageModeMask) == MTLResourceStorageModeShared);
	CHECK(ringEmpty(fx.ring));

	// The CPU's bytes in the backing are the tile's texels, and the tile's writes land there.
	for (NSUInteger i = 0; i < memory.size(); i++)
		memory[i] = static_cast<uint8_t>(i);
	uint8_t out[3 * 4 * 2] = {};
	[tex getBytes:out bytesPerRow:12 fromRegion:MTLRegionMake2D(0, 0, 3, 2) mipmapLevel:0];
	CHECK(std::memcmp(&out[0], &memory[tileOffset], 12) == 0);
	CHECK(std::memcmp(&out[12], &memory[tileOffset + pitch], 12) == 0);
	uint8_t in[8] = {0xa0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7};
	[tex replaceRegion:MTLRegionMake2D(1, 1, 2, 1) mipmapLevel:0 withBytes:in bytesPerRow:8];
	CHECK(std::memcmp(&memory[tileOffset + pitch + 4], in, 8) == 0);
	CHECK(memory[tileOffset + pitch] == static_cast<uint8_t>(tileOffset + pitch)); // beside it
	CHECK(memory[tileOffset + pitch + 12] == static_cast<uint8_t>(tileOffset + pitch + 12));

	// Metal's validation's rules: a descriptor other than Managed, and a tile past the
	// buffer's bytes (offset + pitch x (height - 1) + the row's bytes), answer nil.
	desc.storageMode = MTLStorageModeShared;
	CHECK([tiled newTiledTextureWithDescriptor:desc offset:tileOffset bytesPerRow:pitch] == nil);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
	desc.storageMode = MTLStorageModeManaged;
#pragma clang diagnostic pop
	const NSUInteger last = memory.size() - (pitch + 3 * 4); // the last offset that fits
	id<MTLTexture> atLast = [tiled newTiledTextureWithDescriptor:desc
	                                                       offset:last
	                                                  bytesPerRow:pitch];
	CHECK(atLast != nil);
	CHECK(expectMessage<RgfxTextureFromBuffer>(fx.ring, RGFX_CMD_TEXTURE_FROM_BUFFER).offset ==
	      last);
	CHECK([tiled newTiledTextureWithDescriptor:desc offset:last + 4 bytesPerRow:pitch] == nil);
	// Any offset and pitch inside the buffer: no alignment is asked.
	id<MTLTexture> odd = [tiled newTiledTextureWithDescriptor:desc offset:4 bytesPerRow:pitch + 4];
	CHECK(odd != nil);
	CHECK(expectMessage<RgfxTextureFromBuffer>(fx.ring, RGFX_CMD_TEXTURE_FROM_BUFFER)
	          .bytes_per_row == pitch + 4);
	while (fx.ring.next(&h, &bytes)) {
	}

	// More than one level answers nil, sending nothing: the host does not carry levels 1 and up.
	desc.mipmapLevelCount = 2;
	CHECK([tiled newTiledTextureWithDescriptor:desc offset:tileOffset bytesPerRow:pitch] == nil);
	CHECK(ringEmpty(fx.ring));
	desc.mipmapLevelCount = 1;
	// So does a type other than 2D, which the AMD and the Intel GPU make and the host does not
	// carry over a buffer.
	desc.textureType = MTLTextureType2DArray;
	desc.arrayLength = 2;
	CHECK([tiled newTiledTextureWithDescriptor:desc offset:0 bytesPerRow:pitch] == nil);
	desc.textureType = MTLTextureType1D;
	desc.height = 1;
	desc.arrayLength = 1;
	CHECK([tiled newTiledTextureWithDescriptor:desc offset:0 bytesPerRow:pitch] == nil);
	CHECK(ringEmpty(fx.ring));
	desc.textureType = MTLTextureType2D;
	desc.height = 2;

	// A buffer other than Shared answers nil.
	std::vector<uint8_t> other(pitch * 6, 0);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
	RgfxBuffer *managed = [RgfxBuffer bufferOverHostMemoryWithContents:other.data()
	                                                      regionOffset:0x20000
	                                                            length:other.size()
	                                                           options:MTLResourceStorageModeManaged
	                                                       deallocator:nil
	                                                            device:nil
	                                                        serializer:&fx.ser];
#pragma clang diagnostic pop
	CHECK(managed != nil);
	while (fx.ring.next(&h, &bytes)) {
	}
	CHECK([(id<RgfxMetalBufferInternals>)managed newTiledTextureWithDescriptor:desc
	                                                                    offset:0
	                                                               bytesPerRow:pitch] == nil);
	CHECK(ringEmpty(fx.ring));
	}
}

// This binary's path, which the death checks run again with --dies CASE (main).
static const char *g_self = nullptr;

// The exit status of a death case that raised SIGABRT: its handler exits with it, so the death
// writes no crash report.
static const int kAborted = 128 + SIGABRT;

// Runs this binary again with --dies `name`, the case of `dies_case` that ends the process, and
// checks that it raised SIGABRT with `words` on its stderr, as Metal's failed assertion does.
static void checkDies(const char *name, const char *words)
{
	int fds[2];
	CHECK(pipe(fds) == 0);
	posix_spawn_file_actions_t fa;
	posix_spawn_file_actions_init(&fa);
	posix_spawn_file_actions_adddup2(&fa, fds[1], 2);
	posix_spawn_file_actions_addclose(&fa, fds[0]);
	char *args[] = {const_cast<char *>(g_self), const_cast<char *>("--dies"),
	                const_cast<char *>(name), nullptr};
	pid_t pid = 0;
	CHECK(posix_spawn(&pid, g_self, &fa, nullptr, args, *_NSGetEnviron()) == 0);
	posix_spawn_file_actions_destroy(&fa);
	close(fds[1]);
	std::string err;
	char chunk[512];
	ssize_t n;
	while ((n = read(fds[0], chunk, sizeof chunk)) > 0)
		err.append(chunk, static_cast<size_t>(n));
	close(fds[0]);
	int status = 0;
	CHECK(waitpid(pid, &status, 0) == pid);
	const bool aborted = WIFEXITED(status) && WEXITSTATUS(status) == kAborted;
	if (!aborted || err.find(words) == std::string::npos)
		std::fprintf(stderr, "--dies %s ended with status %d:\n%s\n", name, status, err.c_str());
	CHECK(aborted);
	CHECK(err.find(words) != std::string::npos);
}

// A .metallib of one function, `rgfx_exported`, of the TYPE a dynamic library's exported function
// has (3), in metallibOf's layout.
static std::vector<uint8_t> exportedFunctionMetallib();

// Textures over a 64 x 64 'BGRA' surface of 256 bytes a row, without planes (as IOSurfaceCreate
// makes WindowServer's) or of one plane, or without planes of 512 bytes a row, each as
// oracle_surftex asked it of the AMD 5300M and the
// Intel 630 under macOS 26.7.1 (docs/graphics-kext/dgpu-oracle.md, Q20), with Metal's words where
// both ended the process there; null where both went on.
enum SurfaceKind { kFlat, kOnePlane, kWideRows };
struct SurfaceCase {
	const char *name;
	SurfaceKind surface;
	NSUInteger plane;
	MTLPixelFormat format;
	MTLTextureType type;
	NSUInteger width, height, depth, arrayLength, levels, samples;
	MTLStorageMode storage;
	const char *words;
};
#define RGFX_VALIDATE                                                                              \
	"-[MTLTextureDescriptorInternal validateWithDevice:]: failed assertion `Texture "               \
	"Descriptor Validation\n"
#define RGFX_STRIDE                                                                                \
	"_mtlValidateStrideTextureParameters: failed assertion `Texture Descriptor Validation\n"
#define RGFX_INIT                                                                                  \
	"-[MTLIOAccelTexture initWithDevice:descriptor:iosurface:plane:field:args:argsSize:]: failed "  \
	"assertion `"
static const MTLTextureType k2D = MTLTextureType2D;
static const MTLStorageMode kManaged = MTLStorageModeManaged;
static const SurfaceCase kSurfaceCases[] = {
    {"surface-width-0", kFlat, 0, MTLPixelFormatBGRA8Unorm, k2D, 0, 64, 1, 1, 1, 1, kManaged,
     RGFX_VALIDATE "MTLTextureDescriptor has width of zero.\n'"},
    {"surface-height-0", kFlat, 0, MTLPixelFormatBGRA8Unorm, k2D, 64, 0, 1, 1, 1, 1, kManaged,
     RGFX_VALIDATE "MTLTextureDescriptor has height of zero.\n'"},
    {"surface-format-invalid", kFlat, 0, MTLPixelFormatInvalid, k2D, 64, 64, 1, 1, 1, 1, kManaged,
     RGFX_VALIDATE "MTLTextureDescriptor has invalid pixelFormat (0).\n'"},
    {"surface-format-lacked", kOnePlane, 0, MTLPixelFormatB5G6R5Unorm, k2D, 64, 64, 1, 1, 1, 1,
     kManaged, RGFX_VALIDATE "MTLTextureDescriptor has invalid pixelFormat (40).\n'"},
    {"surface-format-xr10", kFlat, 0, MTLPixelFormatBGR10_XR, k2D, 64, 64, 1, 1, 1, 1, kManaged,
     RGFX_VALIDATE "MTLTextureDescriptor has invalid pixelFormat (554).\n'"},
    {"surface-depth-2", kFlat, 0, MTLPixelFormatBGRA8Unorm, k2D, 64, 32, 2, 1, 1, 1, kManaged,
     RGFX_VALIDATE "MTLTextureDescriptor has depth (2) greater than the maximum allowed size of "
                   "1.\nMTLTextureDescriptor requests a depth of 2 but the texture type "
                   "MTLTextureType2D requires that depth is 1\n'"},
    {"surface-array-2", kFlat, 0, MTLPixelFormatBGRA8Unorm, k2D, 64, 32, 1, 2, 1, 1, kManaged,
     RGFX_VALIDATE "MTLTextureDescriptor has arrayLength (2) greater than the maximum allowed size "
                   "of 1.\nMTLTextureDescriptor requests an arrayLength of 2, but MTLTextureType2D "
                   "requires that arrayLength is 1\n'"},
    {"surface-samples-4", kFlat, 0, MTLPixelFormatBGRA8Unorm, k2D, 64, 64, 1, 1, 1, 4, kManaged,
     RGFX_VALIDATE "MTLTextureDescriptor has sampleCount set but is using a type that does not "
                   "allow sampleCount.\n'"},
    {"surface-multisample-1", kFlat, 0, MTLPixelFormatBGRA8Unorm, MTLTextureType2DMultisample, 64,
     64, 1, 1, 1, 1, kManaged,
     RGFX_VALIDATE "MTLTextureDescriptor sampleCount must be > 1 for multisample textures.\n"
                   "MTLTextureDescriptor: Multisample textures cannot be allocated with "
                   "MTLStorageModeShared or MTLStorageModeManaged on this device.\n'"},
    {"surface-multisample-managed", kFlat, 0, MTLPixelFormatBGRA8Unorm,
     MTLTextureType2DMultisample, 64, 64, 1, 1, 1, 4, kManaged,
     RGFX_VALIDATE "MTLTextureDescriptor: Multisample textures cannot be allocated with "
                   "MTLStorageModeShared or MTLStorageModeManaged on this device.\n'"},
    {"surface-memoryless", kFlat, 0, MTLPixelFormatBGRA8Unorm, k2D, 64, 64, 1, 1, 1, 1,
     MTLStorageModeMemoryless,
     RGFX_VALIDATE "MTLTextureDescriptor: MTLResourceStorageModeMemoryless is not a supported "
                   "storage mode on this device.\n'"},
    {"surface-depth-managed", kFlat, 0, MTLPixelFormatDepth32Float, k2D, 64, 64, 1, 1, 1, 1,
     kManaged,
     RGFX_VALIDATE "MTLTextureDescriptor: Depth, Stencil, DepthStencil textures cannot be "
                   "allocated with MTLStorageModeShared or MTLStorageModeManaged on this "
                   "device.\n'"},
    {"surface-type-1d", kFlat, 0, MTLPixelFormatBGRA8Unorm, MTLTextureType1D, 64, 1, 1, 1, 1, 1,
     kManaged,
     RGFX_STRIDE "IOSurface texture: must be of type MTLTextureType2D, textureType "
                 "(MTLTextureType1D) disallowed\n'"},
    {"surface-type-cube", kFlat, 0, MTLPixelFormatBGRA8Unorm, MTLTextureTypeCube, 64, 64, 1, 1, 1,
     1, kManaged,
     RGFX_STRIDE "IOSurface texture: must be of type MTLTextureType2D, textureType "
                 "(MTLTextureTypeCube) disallowed\n'"},
    {"surface-type-3d-depth-2", kFlat, 0, MTLPixelFormatBGRA8Unorm, MTLTextureType3D, 64, 32, 2, 1,
     1, 1, kManaged,
     RGFX_STRIDE "IOSurface texture: must have depth == 1, depth (2) disallowed\nIOSurface "
                 "texture: must be of type MTLTextureType2D, textureType (MTLTextureType3D) "
                 "disallowed\n'"},
    {"surface-levels-7", kFlat, 0, MTLPixelFormatBGRA8Unorm, k2D, 64, 64, 1, 1, 7, 1, kManaged,
     RGFX_STRIDE "IOSurface texture: cannot be mipmapped, mipmapLevelCount (7) disallowed\n'"},
    {"surface-wider", kFlat, 0, MTLPixelFormatBGRA8Unorm, k2D, 65, 64, 1, 1, 1, 1, kManaged,
     RGFX_STRIDE "IOSurface texture: bytesPerRow (256) must be greater or equal to (260) bytes\n'"},
    {"surface-wider-format", kFlat, 0, MTLPixelFormatRGBA32Float, k2D, 64, 64, 1, 1, 1, 1,
     kManaged,
     RGFX_STRIDE "IOSurface texture: bytesPerRow (256) must be greater or equal to (1024) "
                 "bytes\n'"},
    {"surface-depth-private", kFlat, 0, MTLPixelFormatDepth32Float, k2D, 64, 64, 1, 1, 1, 1,
     MTLStorageModePrivate,
     RGFX_STRIDE "IOSurface texture: cannot create compressed, depth, or stencil textures, "
                 "pixelFormat (MTLPixelFormatDepth32Float) disallowed\n'"},
    {"surface-plane-1", kFlat, 1, MTLPixelFormatBGRA8Unorm, k2D, 64, 64, 1, 1, 1, 1, kManaged,
     RGFX_INIT "IOSurface is non-planar, but plane 1 specified'"},
    {"surface-taller", kFlat, 0, MTLPixelFormatBGRA8Unorm, k2D, 64, 65, 1, 1, 1, 1, kManaged,
     RGFX_INIT "Descriptor height*arrayLength > IOSurface height'"},
    {"surface-array-depth-2", kFlat, 0, MTLPixelFormatBGRA8Unorm, MTLTextureType2DArray, 64, 32,
     2, 1, 1, 1, kManaged,
     RGFX_VALIDATE "MTLTextureDescriptor has depth (2) greater than the maximum allowed size of "
                   "1.\nMTLTextureDescriptor requests a depth of 2 but the texture type "
                   "MTLTextureType2DArray requires that depth is 1\n'"},
    {"surface-array-samples-4", kFlat, 0, MTLPixelFormatBGRA8Unorm, MTLTextureType2DArray, 64, 64,
     1, 1, 1, 4, kManaged,
     RGFX_VALIDATE "MTLTextureDescriptor has sampleCount set but is using a type that does not "
                   "allow sampleCount.\n'"},
    {"surface-plane-depth-2", kOnePlane, 0, MTLPixelFormatBGRA8Unorm, k2D, 64, 32, 2, 1, 1, 1,
     kManaged,
     RGFX_VALIDATE "MTLTextureDescriptor has depth (2) greater than the maximum allowed size of "
                   "1.\nMTLTextureDescriptor requests a depth of 2 but the texture type "
                   "MTLTextureType2D requires that depth is 1\n'"},
    {"surface-422-odd", kFlat, 0, MTLPixelFormatGBGR422, k2D, 63, 64, 1, 1, 1, 1, kManaged,
     RGFX_VALIDATE "MTLTextureDescriptor: MTLPixelFormat422 requires even width.\n'"},
    {"surface-multisample-private", kFlat, 0, MTLPixelFormatBGRA8Unorm,
     MTLTextureType2DMultisample, 64, 64, 1, 1, 1, 4, MTLStorageModePrivate,
     RGFX_STRIDE "IOSurface texture: must be of type MTLTextureType2D, textureType "
                 "(MTLTextureType2DMultisample) disallowed\n'"},
    {"surface-buffer-wider", kFlat, 0, MTLPixelFormatBGRA8Unorm, MTLTextureTypeTextureBuffer, 65,
     1, 1, 1, 1, 1, kManaged,
     RGFX_STRIDE "IOSurface texture: bytesPerRow (256) must be greater or equal to (260) bytes\n'"},
    {"surface-wide-rows-wider", kWideRows, 0, MTLPixelFormatBGRA8Unorm, k2D, 129, 64, 1, 1, 1, 1,
     kManaged,
     RGFX_STRIDE "IOSurface texture: bytesPerRow (512) must be greater or equal to (516) bytes\n'"},
    {"surface-plane-2", kFlat, 2, MTLPixelFormatBGRA8Unorm, k2D, 64, 64, 1, 1, 1, 1, kManaged,
     RGFX_INIT "IOSurface is non-planar, but plane 2 specified'"},
    {"surface-compressed-wider", kFlat, 0, MTLPixelFormatBC1_RGBA, k2D, 132, 64, 1, 1, 1, 1,
     kManaged, RGFX_INIT "Descriptor rowBytes (264) > IOSurface rowBytes (256)'"},
    {"surface-compressed-wider-bc7", kFlat, 0, MTLPixelFormatBC7_RGBAUnorm, k2D, 68, 64, 1, 1, 1,
     1, kManaged, RGFX_INIT "Descriptor rowBytes (272) > IOSurface rowBytes (256)'"},
    {"surface-compressed-taller", kFlat, 0, MTLPixelFormatBC1_RGBA, k2D, 64, 68, 1, 1, 1, 1,
     kManaged, RGFX_INIT "Descriptor height*arrayLength > IOSurface height'"},
    {"surface-wide-rows-taller", kWideRows, 0, MTLPixelFormatBGRA8Unorm, k2D, 64, 65, 1, 1, 1, 1,
     kManaged, RGFX_INIT "Descriptor height*arrayLength > IOSurface height'"},
    // What both made, or answered nil for.
    {"surface-made", kFlat, 0, MTLPixelFormatBGRA8Unorm, k2D, 64, 64, 1, 1, 1, 1, kManaged,
     nullptr},
    {"surface-made-smaller", kFlat, 0, MTLPixelFormatBGRA8Unorm, k2D, 32, 32, 1, 1, 1, 1, kManaged,
     nullptr},
    {"surface-made-row", kFlat, 0, MTLPixelFormatBGRA8Unorm, k2D, 64, 1, 1, 1, 1, 1, kManaged,
     nullptr},
    {"surface-made-buffer", kFlat, 0, MTLPixelFormatBGRA8Unorm, MTLTextureTypeTextureBuffer, 64, 1,
     1, 1, 1, 1, kManaged, nullptr},
    {"surface-made-r8", kFlat, 0, MTLPixelFormatR8Unorm, k2D, 256, 64, 1, 1, 1, 1, kManaged,
     nullptr},
    {"surface-made-rgba16f", kFlat, 0, MTLPixelFormatRGBA16Float, k2D, 32, 64, 1, 1, 1, 1,
     kManaged, nullptr},
    {"surface-made-bc1", kFlat, 0, MTLPixelFormatBC1_RGBA, k2D, 64, 64, 1, 1, 1, 1, kManaged,
     nullptr},
    {"surface-made-422", kFlat, 0, MTLPixelFormatGBGR422, k2D, 128, 64, 1, 1, 1, 1, kManaged,
     nullptr},
    {"surface-made-private", kFlat, 0, MTLPixelFormatBGRA8Unorm, k2D, 64, 64, 1, 1, 1, 1,
     MTLStorageModePrivate, nullptr},
    {"surface-made-shared", kFlat, 0, MTLPixelFormatBGRA8Unorm, k2D, 64, 64, 1, 1, 1, 1,
     MTLStorageModeShared, nullptr},
    {"surface-plane-wider", kOnePlane, 0, MTLPixelFormatBGRA8Unorm, k2D, 65, 64, 1, 1, 1, 1,
     kManaged, nullptr},
    {"surface-plane-taller", kOnePlane, 0, MTLPixelFormatBGRA8Unorm, k2D, 64, 65, 1, 1, 1, 1,
     kManaged, nullptr},
    {"surface-plane-1-nil", kOnePlane, 1, MTLPixelFormatBGRA8Unorm, k2D, 64, 64, 1, 1, 1, 1,
     kManaged, nullptr},
    // A plane of a planar surface takes what a surface without planes refuses.
    {"surface-plane-1d", kOnePlane, 0, MTLPixelFormatBGRA8Unorm, MTLTextureType1D, 64, 1, 1, 1, 1,
     1, kManaged, nullptr},
    {"surface-plane-array", kOnePlane, 0, MTLPixelFormatBGRA8Unorm, MTLTextureType2DArray, 64, 32,
     1, 2, 1, 1, kManaged, nullptr},
    {"surface-plane-cube", kOnePlane, 0, MTLPixelFormatBGRA8Unorm, MTLTextureTypeCube, 64, 64, 1,
     1, 1, 1, kManaged, nullptr},
    {"surface-plane-3d", kOnePlane, 0, MTLPixelFormatBGRA8Unorm, MTLTextureType3D, 64, 32, 2, 1, 1,
     1, kManaged, nullptr},
    {"surface-plane-multisample", kOnePlane, 0, MTLPixelFormatBGRA8Unorm,
     MTLTextureType2DMultisample, 64, 64, 1, 1, 1, 4, MTLStorageModePrivate, nullptr},
    {"surface-plane-levels", kOnePlane, 0, MTLPixelFormatBGRA8Unorm, k2D, 64, 64, 1, 1, 2, 1,
     kManaged, nullptr},
    {"surface-plane-depth", kOnePlane, 0, MTLPixelFormatDepth32Float, k2D, 64, 64, 1, 1, 1, 1,
     MTLStorageModePrivate, nullptr},
    {"surface-plane-bc1", kOnePlane, 0, MTLPixelFormatBC1_RGBA, k2D, 64, 64, 1, 1, 1, 1, kManaged,
     nullptr},
    {"surface-made-bc1-wide", kFlat, 0, MTLPixelFormatBC1_RGBA, k2D, 128, 64, 1, 1, 1, 1, kManaged,
     nullptr},
    {"surface-made-bc1-62", kFlat, 0, MTLPixelFormatBC1_RGBA, k2D, 62, 64, 1, 1, 1, 1, kManaged,
     nullptr},
    {"surface-made-bc1-private", kFlat, 0, MTLPixelFormatBC1_RGBA, k2D, 64, 64, 1, 1, 1, 1,
     MTLStorageModePrivate, nullptr},
    {"surface-made-422-odd-height", kFlat, 0, MTLPixelFormatGBGR422, k2D, 64, 63, 1, 1, 1, 1,
     kManaged, nullptr},
    {"surface-wide-rows-made", kWideRows, 0, MTLPixelFormatBGRA8Unorm, k2D, 128, 64, 1, 1, 1, 1,
     kManaged, nullptr},
    // The two-plane Y'CbCr formats, which the lane makes as the M3 Max does (Q13).
    {"surface-made-ycbcr", kFlat, 0, static_cast<MTLPixelFormat>(500), k2D, 64, 64, 1, 1, 1, 1,
     kManaged, nullptr},
};
#undef RGFX_VALIDATE
#undef RGFX_STRIDE
#undef RGFX_INIT

// The surface of `c` (one the caller releases) and its descriptor.
static IOSurfaceRef surfaceOf(const SurfaceCase &c)
{
	const size_t row = 64 * 4;
	NSMutableDictionary *props = [@{
		(id)kIOSurfaceWidth : @64,
		(id)kIOSurfaceHeight : @64,
		(id)kIOSurfaceBytesPerElement : @4,
		(id)kIOSurfacePixelFormat : @(0x42475241u), // 'BGRA'
	} mutableCopy];
	if (c.surface == kWideRows)
		props[(id)kIOSurfaceBytesPerRow] = @(2 * row);
	if (c.surface == kOnePlane) {
		[props removeObjectForKey:(id)kIOSurfaceBytesPerElement];
		props[(id)kIOSurfaceAllocSize] = @(row * 64);
		props[(id)kIOSurfacePlaneInfo] = @[ @{
			(id)kIOSurfacePlaneWidth : @64,
			(id)kIOSurfacePlaneHeight : @64,
			(id)kIOSurfacePlaneBytesPerElement : @4,
			(id)kIOSurfacePlaneBytesPerRow : @(row),
			(id)kIOSurfacePlaneOffset : @0,
			(id)kIOSurfacePlaneSize : @(row * 64),
		} ];
	}
	return IOSurfaceCreate((__bridge CFDictionaryRef)props);
}

static MTLTextureDescriptor *descriptorOf(const SurfaceCase &c)
{
	MTLTextureDescriptor *td = [MTLTextureDescriptor new];
	td.textureType = c.type;
	td.pixelFormat = c.format;
	td.width = c.width;
	td.height = c.height;
	td.depth = c.depth;
	td.arrayLength = c.arrayLength;
	td.mipmapLevelCount = c.levels;
	td.sampleCount = c.samples;
	td.usage = MTLTextureUsageShaderRead |
	           (c.format == MTLPixelFormatDepth32Float ? MTLTextureUsageRenderTarget : 0);
	td.storageMode = c.storage;
	return td;
}

// A YCBCR8_420_2P texture `w` x `h` over plane 0 of a 64 x 64 '420v' CVPixelBuffer, a render
// target where `target` says, through rgfxEndAtInvalidSurfaceTexture.
static void ycbcrTarget(unsigned w, unsigned h, bool target)
{
	CVPixelBufferRef pb = nullptr;
	NSDictionary *attrs = @{(id)kCVPixelBufferIOSurfacePropertiesKey : @{}};
	CHECK(CVPixelBufferCreate(kCFAllocatorDefault, 64, 64, '420v', (__bridge CFDictionaryRef)attrs,
	                          &pb) == kCVReturnSuccess);
	IOSurfaceRef surface = CVPixelBufferGetIOSurface(pb);
	CHECK(surface != nullptr && IOSurfaceGetPlaneCount(surface) == 2);
	MTLTextureDescriptor *d =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:static_cast<MTLPixelFormat>(500)
	                                                       width:w
	                                                      height:h
	                                                   mipmapped:NO];
	d.usage = MTLTextureUsageShaderRead | (target ? MTLTextureUsageRenderTarget : 0);
	rgfxEndAtInvalidSurfaceTexture(d, surface, 0);
	CVPixelBufferRelease(pb);
}

// The cases checkDies runs, each in a process of its own: each ends the process where it should,
// else it answers 0, which its check fails.
static int dies_case(const char *name)
{
	signal(SIGABRT, [](int) { _exit(kAborted); });
	Fixture fx;
	@autoreleasepool {
	for (const SurfaceCase &c : kSurfaceCases) {
		if (std::strcmp(name, c.name) != 0)
			continue;
		IOSurfaceRef surface = surfaceOf(c);
		rgfxEndAtInvalidSurfaceTexture(descriptorOf(c), surface, c.plane);
		CFRelease(surface);
	}
	unsigned w = 0, h = 0;
	if (std::sscanf(name, "surface-ycbcr-target-%ux%u", &w, &h) == 2)
		ycbcrTarget(w, h, true);
	if (std::strcmp(name, "buffer-texture-levels") == 0) {
		std::vector<uint8_t> memory(4096, 0);
		RgfxBuffer *buf =
		    [RgfxBuffer bufferOverHostMemoryWithContents:memory.data()
		                                    regionOffset:0x10000
		                                          length:memory.size()
		                                         options:MTLResourceStorageModeShared
		                                     deallocator:nil
		                                          device:nil
		                                      serializer:&fx.ser];
		MTLTextureDescriptor *desc =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
		                                                       width:16
		                                                      height:16
		                                                   mipmapped:NO];
		desc.mipmapLevelCount = 2;
		(void)[buf newTextureWithDescriptor:desc offset:0 bytesPerRow:64];
	}
	// buffer-texture-type-N and linear-texture-type-N: a texture of MTLTextureType N over a
	// buffer, by the public form or the linear SPI.
	const bool linearForm = std::strncmp(name, "linear-texture-type-", 20) == 0;
	if (linearForm || std::strncmp(name, "buffer-texture-type-", 20) == 0) {
		const auto type = static_cast<MTLTextureType>(std::atoi(name + 20));
		const bool oneD = type == MTLTextureType1D || type == MTLTextureType1DArray;
		const bool cube = type == MTLTextureTypeCube || type == MTLTextureTypeCubeArray;
		std::vector<uint8_t> memory(4096, 0);
		RgfxBuffer *buf =
		    [RgfxBuffer bufferOverHostMemoryWithContents:memory.data()
		                                    regionOffset:0x10000
		                                          length:memory.size()
		                                         options:MTLResourceStorageModeShared
		                                     deallocator:nil
		                                          device:nil
		                                      serializer:&fx.ser];
		MTLTextureDescriptor *desc =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
		                                                       width:16
		                                                      height:oneD ? 1 : cube ? 16 : 4
		                                                   mipmapped:NO];
		desc.textureType = type;
		desc.depth = type == MTLTextureType3D ? 2 : 1;
		desc.arrayLength = type == MTLTextureType1DArray || type == MTLTextureType2DArray ||
		                           type == MTLTextureTypeCubeArray ||
		                           type == MTLTextureType2DMultisampleArray
		                       ? 2
		                       : 1;
		if (type == MTLTextureType2DMultisample || type == MTLTextureType2DMultisampleArray)
			desc.sampleCount = 4;
		if (linearForm) {
			(void)[(id<RgfxMetalBufferInternals>)buf newLinearTextureWithDescriptor:desc
			                                                                offset:0
			                                                           bytesPerRow:64
			                                                         bytesPerImage:256];
		} else {
			(void)[buf newTextureWithDescriptor:desc offset:0 bytesPerRow:64];
		}
	}
	// A texture of a placement heap, in the format and type the case names.
	auto heapTexture = [&](MTLPixelFormat format, NSUInteger samples) {
		MTLHeapDescriptor *hd = [[MTLHeapDescriptor alloc] init];
		hd.size = 0x100000;
		hd.storageMode = MTLStorageModePrivate;
		hd.type = MTLHeapTypePlacement;
		RgfxHeap *heap = [RgfxHeap heapWithDescriptor:hd device:nil serializer:&fx.ser];
		MTLTextureDescriptor *td =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
		                                                       width:64
		                                                      height:64
		                                                   mipmapped:NO];
		td.storageMode = MTLStorageModePrivate;
		if (samples > 0) {
			td.textureType = MTLTextureType2DMultisample;
			td.sampleCount = samples;
			td.usage = MTLTextureUsageRenderTarget;
		}
		(void)[heap newTextureWithDescriptor:td offset:0];
	};
	if (std::strcmp(name, "texture-three-channel") == 0)
		heapTexture(static_cast<MTLPixelFormat>(45), 0);
	if (std::strcmp(name, "texture-multisample-compressed") == 0)
		heapTexture(MTLPixelFormatBC1_RGBA, 4);
	if (std::strcmp(name, "texture-multisample-422") == 0)
		heapTexture(MTLPixelFormatGBGR422, 2);
	if (std::strcmp(name, "icb-memoryless") == 0) {
		MTLIndirectCommandBufferDescriptor *d = [MTLIndirectCommandBufferDescriptor new];
		d.commandTypes = MTLIndirectCommandTypeDraw;
		(void)[RgfxIndirectCommandBuffer icbWithDescriptor:d
		                                   maxCommandCount:4
		                                           options:MTLResourceStorageModeMemoryless
		                                            device:nil
		                                        serializer:&fx.ser];
	}
	if (std::strcmp(name, "linear-alignment-compressed") == 0)
		rgfxEndAtUnalignableLinearFormat(MTLPixelFormatBC1_RGBA);
	if (std::strcmp(name, "linear-alignment-422") == 0)
		rgfxEndAtUnalignableLinearFormat(MTLPixelFormatBGRG422);
	if (std::strcmp(name, "linear-alignment-depth") == 0)
		rgfxEndAtUnalignableLinearFormat(MTLPixelFormatDepth32Float);
	if (std::strcmp(name, "linear-alignment-lacked") == 0)
		rgfxEndAtUnalignableLinearFormat(MTLPixelFormatASTC_4x4_LDR);
	if (std::strcmp(name, "linear-alignment-d24s8") == 0)
		rgfxEndAtUnalignableLinearFormat(MTLPixelFormatDepth24Unorm_Stencil8);
	if (std::strcmp(name, "linear-alignment-invalid") == 0)
		rgfxEndAtUnalignableLinearFormat(MTLPixelFormatInvalid);
	if (std::strcmp(name, "linear-alignment-ycbcr") == 0)
		rgfxEndAtUnalignableLinearFormat(static_cast<MTLPixelFormat>(500));
	if (std::strcmp(name, "linear-alignment-unnamed") == 0)
		rgfxEndAtUnalignableLinearFormat(static_cast<MTLPixelFormat>(4096));
	// A view of test_texture_views' texture: a Private 2D array of four slices and seven levels,
	// of usage 0x17, in RGBA8Unorm.
	auto viewOf = [&](MTLPixelFormat format, MTLTextureType type, NSRange levels, NSRange slices,
	                  bool subrange) {
		MTLTextureDescriptor *desc =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
		                                                       width:64
		                                                      height:32
		                                                   mipmapped:YES];
		desc.textureType = MTLTextureType2DArray;
		desc.arrayLength = 4;
		desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite |
		             MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
		desc.storageMode = MTLStorageModePrivate;
		RgfxTexture *tex = [RgfxTexture textureWithDescriptor:desc
		                                                 heap:RGFX_OBJ_NONE
		                                           heapOffset:0
		                                               device:nil
		                                           serializer:&fx.ser];
		if (subrange)
			(void)[tex newTextureViewWithPixelFormat:format
			                             textureType:type
			                                  levels:levels
			                                  slices:slices];
		else
			(void)[tex newTextureViewWithPixelFormat:format];
	};
	if (std::strcmp(name, "dynamic-exported-function") == 0) {
		// A library whose one function is a dynamic library's exported one (TYPE 3).
		const std::vector<uint8_t> bytes = exportedFunctionMetallib();
		uint32_t status = 0;
		RgfxLibrary *lib = [RgfxLibrary libraryWithData:bytes.data()
		                                        length:bytes.size()
		                                           key:nullptr
		                                        device:nil
		                                    serializer:&fx.ser
		                                        status:&status];
		(void)[lib newFunctionWithName:@"rgfx_exported"];
	}
	if (std::strcmp(name, "heap-shared") == 0) {
		MTLHeapDescriptor *d = [MTLHeapDescriptor new];
		d.size = 0x1000;
		d.storageMode = MTLStorageModeShared;
		(void)[RgfxHeap heapForDeviceWithDescriptor:d
		                                  maxLength:0
		                                     region:nullptr
		                                     device:nil
		                                 serializer:&fx.ser];
	}
	if (std::strcmp(name, "buffer-texture-depth") == 0) {
		// oracle_bundle buftex.depth's: Depth32Float, 64x64, 256 bytes a row, over a Shared buffer.
		std::vector<uint8_t> memory(0x10000, 0);
		RgfxBuffer *buf =
		    [RgfxBuffer bufferOverHostMemoryWithContents:memory.data()
		                                    regionOffset:0x10000
		                                          length:memory.size()
		                                         options:MTLResourceStorageModeShared
		                                     deallocator:nil
		                                          device:nil
		                                      serializer:&fx.ser];
		MTLTextureDescriptor *desc =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
		                                                       width:64
		                                                      height:64
		                                                   mipmapped:NO];
		desc.storageMode = MTLStorageModeShared;
		(void)[buf newTextureWithDescriptor:desc offset:0 bytesPerRow:256];
	}
	if (std::strcmp(name, "texture-stencil-managed") == 0) {
		MTLTextureDescriptor *desc =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatStencil8
		                                                       width:64
		                                                      height:64
		                                                   mipmapped:NO];
		desc.storageMode = MTLStorageModeManaged;
		rgfxEndAtInvalidTextureDescriptor(desc);
	}
	const NSRange one = NSMakeRange(0, 1);
	if (std::strcmp(name, "view-no-format") == 0)
		viewOf(MTLPixelFormatInvalid, MTLTextureType2DArray, one, one, false);
	if (std::strcmp(name, "view-format-class") == 0)
		viewOf(MTLPixelFormatRG8Unorm, MTLTextureType2DArray, one, one, false);
	if (std::strcmp(name, "view-levels") == 0)
		viewOf(MTLPixelFormatRGBA8Unorm, MTLTextureType2D, NSMakeRange(6, 3), one, true);
	if (std::strcmp(name, "view-slices") == 0)
		viewOf(MTLPixelFormatRGBA8Unorm, MTLTextureType2DArray, one, NSMakeRange(3, 2), true);
	if (std::strcmp(name, "view-type") == 0)
		viewOf(MTLPixelFormatRGBA8Unorm, MTLTextureType3D, one, one, true);
	if (std::strcmp(name, "view-cube-count") == 0)
		viewOf(MTLPixelFormatRGBA8Unorm, MTLTextureTypeCube, one, NSMakeRange(0, 4), true);
	}
	return 0;
}

// A view ends the process with Metal's words, before the host is asked, where Metal ends it on the
// AMD 5300M and the Intel 630 (oracle_bundle view.*): in no format, in a format of one texel a
// block of other bytes, at levels or slices past the texture's, of a type of another family, and
// as a cube of slices not a multiple of six.
void test_metal_ended_views_end_the_process()
{
	const std::string kWhere = "_mtlValidateArgumentsForTextureViewOnDevice: failed assertion "
	                           "`Texture Creation\n";
	checkDies("view-no-format",
	          (kWhere + "texture view pixelFormat (MTLPixelFormatInvalid) not castable.\nsource "
	                    "texture pixelFormat (MTLPixelFormatRGBA8Unorm) not compatible with "
	                    "texture view pixelFormat (MTLPixelFormatInvalid).\n'")
	              .c_str());
	checkDies("view-format-class",
	          (kWhere + "source texture pixelFormat (MTLPixelFormatRGBA8Unorm) not compatible with "
	                    "texture view pixelFormat (MTLPixelFormatRG8Unorm).\n'")
	              .c_str());
	checkDies("view-levels", (kWhere + "(newLevelRange.location + newLevelRange.length)(9) must "
	                                   "be <= [texture mipmapLevelCount](7).\n'")
	                             .c_str());
	checkDies("view-slices",
	          (kWhere + "(newSliceRange.location + newSliceRange.length)(5) must be <= (4).\n'")
	              .c_str());
	checkDies("view-type",
	          (kWhere + "source texture textureType (MTLTextureType2DArray) not compatible with "
	                    "texture view textureType (MTLTextureType3D).\n'")
	              .c_str());
	checkDies("view-cube-count",
	          (kWhere + "newSliceRange.length(4) must be a multiple of (6).\n'").c_str());
}

// A Memoryless indirect command buffer ends the process, before the host is asked, as on the AMD
// 5300M, the Intel 630 (an assertion) and the M3 Max (a trap).
void test_memoryless_icb_ends_the_process()
{
	checkDies("icb-memoryless", "an indirect command buffer cannot be Memoryless (storageMode 3)");
}

// A texture of more than one level over a buffer's bytes (the public
// newTextureWithDescriptor:offset:bytesPerRow:) ends the process with Metal's words, as Metal
// does on the AMD, the Intel and the M3 GPU (docs/graphics-kext/dgpu-oracle.md, Q11).
void test_mipmapped_buffer_texture_ends_the_process()
{
	checkDies("buffer-texture-levels",
	          "_mtlValidateStrideTextureParameters: failed assertion `Texture Descriptor "
	          "Validation\nLinear texture: cannot be mipmapped, mipmapLevelCount (2) disallowed\n'");
}

// A texture over a buffer's bytes of a type the form does not make ends the process with
// Metal's words, before the host is sent it, whose platform would end the host's process there
// (the bundle used to send it): by the public form, any type but 2D; by the linear SPI, any but
// 2D, 2D array and texture buffer. The AMD, the Intel and the M3 GPU say these words at a 1D,
// 1D array, cube or cube array type, and at a 2D array by the public form; at a 3D or multisample
// type they are the M3's, the AMD's too at the 3D one (`oracle_storage DEVICE lintypes
// FORM:TYPE`, docs/graphics-kext/dgpu-oracle.md, Q11).
void test_buffer_texture_types_end_the_process()
{
	const struct {
		MTLTextureType type;
		const char *name;
		bool linear; // the linear SPI dies at it too
	} types[] = {{MTLTextureType1D, "MTLTextureType1D", true},
	             {MTLTextureType1DArray, "MTLTextureType1DArray", true},
	             {MTLTextureType2DArray, "MTLTextureType2DArray", false},
	             {MTLTextureType2DMultisample, "MTLTextureType2DMultisample", true},
	             {MTLTextureTypeCube, "MTLTextureTypeCube", true},
	             {MTLTextureTypeCubeArray, "MTLTextureTypeCubeArray", true},
	             {MTLTextureType3D, "MTLTextureType3D", true},
	             {MTLTextureType2DMultisampleArray, "MTLTextureType2DMultisampleArray", true}};
	for (const auto &t : types) {
		for (const bool linear : {false, true}) {
			if (linear && !t.linear)
				continue;
			const std::string dies = std::string(linear ? "linear" : "buffer") +
			                         "-texture-type-" + std::to_string(t.type);
			std::string words = "_mtlValidateStrideTextureParameters: failed assertion `Texture "
			                    "Descriptor Validation\n";
			if (t.type == MTLTextureType3D)
				words += "Linear texture: must have depth == 1, depth (2) disallowed\n";
			words += std::string("Linear texture: must be of type MTLTextureType2D") +
			         (linear ? " or linear MTLTextureType2DArray" : "") + ", textureType (" +
			         t.name + ") disallowed\n'";
			checkDies(dies.c_str(), words.c_str());
		}
	}
}

// A Shared heap the device is asked for, and a depth or stencil texture the CPU reaches (over a
// Shared buffer's bytes, or of a Managed descriptor), end the process with Metal's words before
// the host is sent anything, as the heap and texture descriptor validation of the AMD 5300M and
// the Intel 630 under macOS 26.7.1 does (oracle_bundle heapbuffers, buftex.depth and devtex.*).
// A depth texture in Private storage passes, as there.
void test_shared_heaps_and_cpu_depth_textures_end_the_process()
{
	checkDies("heap-shared", "-[MTLHeapDescriptorInternal validateWithDevice:]: failed assertion "
	                         "`Heap Descriptor Validation\nShared storage mode disallowed\n'");
	const char *const kDepth =
	    "-[MTLTextureDescriptorInternal validateWithDevice:]: failed assertion `Texture Descriptor "
	    "Validation\nMTLTextureDescriptor: Depth, Stencil, DepthStencil textures cannot be allocated "
	    "with MTLStorageModeShared or MTLStorageModeManaged on this device.\n'";
	checkDies("buffer-texture-depth", kDepth);
	checkDies("texture-stencil-managed", kDepth);
	MTLTextureDescriptor *priv =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
	                                                       width:64
	                                                      height:64
	                                                   mipmapped:NO];
	priv.storageMode = MTLStorageModePrivate;
	rgfxEndAtInvalidTextureDescriptor(priv);
	g_checks++;
}

// minimumLinearTextureAlignmentForPixelFormat: ends the process with Metal's words where both x86
// Macs' Metal does (oracle_bundle linalign, raw values 0 to 799 and four past them: compressed and
// 4:2:2, depth and stencil, and every other value the device lacks, named Invalid: Invalid
// itself, the Y'CbCr formats, values no format has), and answers the 64 formats both answer.
void test_unalignable_linear_formats_end_the_process()
{
	const char *const kWhere = "-[_MTLDevice minimumLinearTextureAlignmentForPixelFormat:]: failed "
	                           "assertion `";
	checkDies("linear-alignment-compressed",
	          (std::string(kWhere) + "Linear textures do not support compressed pixel formats'")
	              .c_str());
	checkDies("linear-alignment-422", "Linear textures do not support compressed pixel formats'");
	checkDies("linear-alignment-depth",
	          (std::string(kWhere) + "Linear textures do not support depth/stencil pixel formats'")
	              .c_str());
	checkDies("linear-alignment-lacked",
	          (std::string(kWhere) + "MTLPixelFormatInvalid is not supported on this device.'")
	              .c_str());
	checkDies("linear-alignment-d24s8", "MTLPixelFormatInvalid is not supported on this device.'");
	checkDies("linear-alignment-invalid", "MTLPixelFormatInvalid is not supported on this device.'");
	checkDies("linear-alignment-ycbcr", "MTLPixelFormatInvalid is not supported on this device.'");
	checkDies("linear-alignment-unnamed", "MTLPixelFormatInvalid is not supported on this device.'");
	const uint64_t answered[] = {1,  10, 12, 13, 14, 20, 22, 23, 24, 25, 30, 32, 33, 34, 43,  53,
	                             54, 55, 60, 62, 63, 64, 65, 70, 71, 72, 73, 74, 80, 81, 90,  91,
	                             92, 93, 94, 103, 104, 105, 110, 112, 113, 114, 115, 123, 124, 125,
	                             // of no name in the SDK
	                             84, 85, 300, 301, 302, 303, 304, 313, 314, 315, 323, 324, 325, 556,
	                             586, 587, 628, 629};
	for (uint64_t f : answered) {
		rgfxEndAtUnalignableLinearFormat(static_cast<MTLPixelFormat>(f));
		g_checks++;
	}
}

// A texture Metal's descriptor validation ends the process at on every Mac (a three-channel
// format; a multisample one of a compressed or 4:2:2 format) ends it with Metal's words, the rules
// it breaks a line each (docs/graphics-kext/dgpu-oracle.md, Q12); the check passes every other.
void test_invalid_texture_descriptors_end_the_process()
{
	checkDies("texture-three-channel",
	          "-[MTLTextureDescriptorInternal validateWithDevice:]: failed assertion `Texture "
	          "Descriptor Validation\nMTLTextureDescriptor has invalid pixelFormat (45).\n'");
	checkDies("texture-multisample-compressed",
	          "-[MTLTextureDescriptorInternal validateWithDevice:]: failed assertion `Texture "
	          "Descriptor Validation\n"
	          "MTLTextureDescriptor pixelFormat (MTLPixelFormatBC1_RGBA) does not support "
	          "multisample textures.\n"
	          "MTLTextureDescriptor has a compressed pixel format set with a sample count greater "
	          "than one.\n"
	          "MTLTextureDescriptor: compressed pixel formats require texture type of "
	          "MTLTextureType2D[Array], MTLTextureType3D or MTLTextureTypeCube[Array].\n'");
	checkDies("texture-multisample-422",
	          "MTLTextureDescriptor pixelFormat (MTLPixelFormatGBGR422) does not support "
	          "multisample textures.\n"
	          "MTLTextureDescriptor has a compressed pixel format set with a sample count greater "
	          "than one.\n"
	          "MTLTextureDescriptor: compressed pixel formats require texture type of "
	          "MTLTextureType2D[Array], MTLTextureType3D or MTLTextureTypeCube[Array].\n"
	          "MTLTextureDescriptor: MTLPixelFormat422 requires MTLTextureType2D.\n'");

	// What Metal makes passes: a compressed 2D texture, a multisample one of a plain format, and
	// the formats beside the three-channel ones.
	auto passes = [](MTLPixelFormat format, MTLTextureType type, NSUInteger samples) {
		MTLTextureDescriptor *td =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
		                                                       width:64
		                                                      height:64
		                                                   mipmapped:NO];
		td.textureType = type;
		td.sampleCount = samples;
		rgfxEndAtInvalidTextureDescriptor(td);
		g_checks++;
	};
	passes(MTLPixelFormatBC1_RGBA, MTLTextureType2D, 1);
	passes(MTLPixelFormatGBGR422, MTLTextureType2D, 1);
	passes(MTLPixelFormatRGBA8Unorm, MTLTextureType2DMultisample, 4);
	passes(MTLPixelFormatBGR5A1Unorm, MTLTextureType2D, 1); // 43
	passes(MTLPixelFormatRGBA16Unorm, MTLTextureType2D, 1);  // 110
	passes(MTLPixelFormatRG32Uint, MTLTextureType2D, 1);     // 103
}

// A texture over an IOSurface ends the process with Metal's words, before the host is sent
// anything, where newTextureWithDescriptor:iosurface:plane: ends it on the AMD 5300M and the Intel
// 630 (oracle_surftex, Q20): at the descriptor's validation, then, over a surface without planes,
// at its type, depth, levels, rows and depth formats, then at a plane past 0, a compressed
// format's rows in blocks and rows past the surface's. What both Macs made passes, a plane of a
// planar surface taking every type, level count and format, as do a plane past a surface's planes
// and the M3 Max's two-plane Y'CbCr formats, which the creator answers.
void test_invalid_surface_textures_end_the_process()
{
	for (const SurfaceCase &c : kSurfaceCases) {
		if (c.words != nullptr) {
			checkDies(c.name, c.words);
			continue;
		}
		IOSurfaceRef surface = surfaceOf(c);
		const bool planar = c.surface == kOnePlane;
		CHECK(surface != nullptr && IOSurfaceGetPlaneCount(surface) == (planar ? 1u : 0u) &&
		      (planar ? IOSurfaceGetBytesPerRowOfPlane(surface, 0)
		              : IOSurfaceGetBytesPerRow(surface)) == (c.surface == kWideRows ? 512 : 256));
		rgfxEndAtInvalidSurfaceTexture(descriptorOf(c), surface, c.plane);
		g_checks++;
		CFRelease(surface);
	}
	// A render target of a two-plane Y'CbCr format at an odd width or height, which the x86 Macs
	// end at as they end at every texture of the format, and the M3 Max and the host make none
	// of; the others the lane makes as the M3 does (Q13).
	const char *const kYCbCr = "-[MTLTextureDescriptorInternal validateWithDevice:]: failed "
	                           "assertion `Texture Descriptor Validation\nMTLTextureDescriptor has "
	                           "invalid pixelFormat (500).\n'";
	checkDies("surface-ycbcr-target-63x63", kYCbCr);
	checkDies("surface-ycbcr-target-63x64", kYCbCr);
	checkDies("surface-ycbcr-target-64x63", kYCbCr);
	ycbcrTarget(64, 64, true);
	ycbcrTarget(63, 63, false);
	g_checks++;
}

// A texture wrapper's create routes an RGFX_CMD_TEXTURE_CREATE onto the ring
// (device-placed: a GPU address and a resource id are picked, heap_obj_id 0), and
// its getters answer the mapped descriptor. The RgfxBuffer routing pattern, for
// the texture's shape.
void test_texture_routing()
{
	Fixture fx;
	// Drained while fx lives: a texture's dealloc posts its destroy through fx.
	@autoreleasepool {

	MTLTextureDescriptor *desc =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:1920
	                                                      height:1080
	                                                   mipmapped:NO];
	desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
	desc.storageMode = MTLStorageModePrivate;

	RgfxTexture *tex = [RgfxTexture textureWithDescriptor:desc
	                                                 heap:RGFX_OBJ_NONE
	                                           heapOffset:0
	                                               device:nil
	                                           serializer:&fx.ser];
	CHECK(tex != nil);

	// Getters answer the mapped descriptor.
	CHECK(tex.rgfxObjId == 1);
	CHECK([tex textureType] == MTLTextureType2D);
	CHECK([tex pixelFormat] == MTLPixelFormatBGRA8Unorm);
	CHECK([tex width] == 1920 && [tex height] == 1080 && [tex depth] == 1);
	CHECK([tex mipmapLevelCount] == 1 && [tex arrayLength] == 1 && [tex sampleCount] == 1);
	CHECK([tex usage] == (MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead));
	CHECK([tex storageMode] == MTLStorageModePrivate);
	// Its Default hazard tracking mode answered resolved, as Metal's (resource_tests.mm).
	CHECK([tex resourceOptions] == (desc.resourceOptions | MTLResourceHazardTrackingModeTracked));
	MTLTextureSwizzleChannels sw = [tex swizzle]; // the descriptor's default swizzle
	CHECK(sw.red == MTLTextureSwizzleRed && sw.green == MTLTextureSwizzleGreen);
	CHECK(sw.blue == MTLTextureSwizzleBlue && sw.alpha == MTLTextureSwizzleAlpha);
	CHECK([tex gpuResourceID]._impl != 0); // a resource id was picked
	CHECK([tex parentTexture] == nil && [tex buffer] == nil);

	// The create wrote exactly one TEXTURE_CREATE matching the wrapper.
	auto msg = expectMessage<RgfxTextureCreate>(fx.ring, RGFX_CMD_TEXTURE_CREATE);
	CHECK(msg.header.seqno == 1);
	CHECK(msg.obj_id == tex.rgfxObjId);
	CHECK(msg.resource_id == [tex gpuResourceID]._impl);
	CHECK(msg.gpu_address != 0 && msg.heap_obj_id == 0); // device-placed
	CHECK(msg.desc.pixel_format == MTLPixelFormatBGRA8Unorm);
	CHECK(msg.desc.texture_type == MTLTextureType2D);
	CHECK(msg.desc.width == 1920 && msg.desc.height == 1080);
	CHECK(msg.desc.usage == (uint32_t)(MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead));
	CHECK(msg.desc.options == (uint32_t)desc.resourceOptions);
	}
}

// One chunk of texture resource ids (ids 1 + 4 * 64 to 64 + 4 * 64), as the kext would hand
// it, and none after it.
bool oneChunk(void *context, uint64_t *first_id)
{
	int *calls = static_cast<int *>(context);
	if ((*calls)++ != 0)
		return false;
	*first_id = 1 + 4 * RGFX_RESOURCE_ID_CHUNK;
	return true;
}

// A texture's resource id comes from the process's chunks and goes back as the texture goes,
// for the next texture; with every id held and no chunk left, newTexture answers nil and posts
// nothing.
void test_texture_resource_ids()
{
	Fixture fx;
	// Drained while fx lives: a texture's dealloc posts its destroy through fx.
	@autoreleasepool {
	int calls = 0;
	fx.ids.setResourceIdSource(&oneChunk, &calls);
	MTLTextureDescriptor *desc =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:16
	                                                      height:16
	                                                   mipmapped:NO];
	const uint64_t first = 1 + 4 * RGFX_RESOURCE_ID_CHUNK;
	NSMutableArray<RgfxTexture *> *held = [NSMutableArray array];
	for (uint64_t i = 0; i < RGFX_RESOURCE_ID_CHUNK; i++) {
		// Each texture's autoreleased return drains here, so `held` alone keeps it.
		@autoreleasepool {
			RgfxTexture *tex = [RgfxTexture textureWithDescriptor:desc
			                                                 heap:RGFX_OBJ_NONE
			                                           heapOffset:0
			                                               device:nil
			                                           serializer:&fx.ser];
			CHECK(tex != nil && [tex gpuResourceID]._impl == first + i);
			auto msg = expectMessage<RgfxTextureCreate>(fx.ring, RGFX_CMD_TEXTURE_CREATE);
			CHECK(msg.resource_id == first + i);
			[held addObject:tex];
		}
	}
	CHECK(calls == 1);
	RgfxTexture *none = [RgfxTexture textureWithDescriptor:desc
	                                                  heap:RGFX_OBJ_NONE
	                                            heapOffset:0
	                                                device:nil
	                                            serializer:&fx.ser];
	CHECK(none == nil);
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(!fx.ring.next(&h, &bytes));
	// The texture holding id first + 10 goes: the host drops it (no command buffer is in
	// flight), then the next texture takes its id, after the destroy on the ring.
	uint64_t goneObjId = 0;
	@autoreleasepool {
		goneObjId = held[10].rgfxObjId;
		[held removeObjectAtIndex:10];
	}
	auto gone = expectMessage<RgfxObjectDestroy>(fx.ring, RGFX_CMD_OBJECT_DESTROY);
	CHECK(gone.obj_id == goneObjId);
	RgfxTexture *again = [RgfxTexture textureWithDescriptor:desc
	                                                   heap:RGFX_OBJ_NONE
	                                             heapOffset:0
	                                                 device:nil
	                                             serializer:&fx.ser];
	CHECK(again != nil && [again gpuResourceID]._impl == first + 10);
	auto msg = expectMessage<RgfxTextureCreate>(fx.ring, RGFX_CMD_TEXTURE_CREATE);
	CHECK(msg.resource_id == first + 10 && msg.obj_id == again.rgfxObjId);
	}
}

// A sampler wrapper's create routes an RGFX_CMD_SAMPLER_CREATE, mapping every
// descriptor field (the LOD clamps as f32 bits, the booleans packed into flags).
void test_sampler_routing()
{
	Fixture fx;

	MTLSamplerDescriptor *desc = [[MTLSamplerDescriptor alloc] init];
	desc.minFilter = MTLSamplerMinMagFilterLinear;
	desc.magFilter = MTLSamplerMinMagFilterLinear;
	desc.mipFilter = MTLSamplerMipFilterLinear;
	desc.sAddressMode = MTLSamplerAddressModeClampToEdge;
	desc.tAddressMode = MTLSamplerAddressModeRepeat;
	desc.rAddressMode = MTLSamplerAddressModeMirrorRepeat;
	desc.maxAnisotropy = 4;
	desc.compareFunction = MTLCompareFunctionLess;
	desc.lodMinClamp = 0.0f;
	desc.lodMaxClamp = 7.5f;
	desc.normalizedCoordinates = YES;
	desc.supportArgumentBuffers = YES;

	RgfxSampler *smp = [RgfxSampler samplerWithDescriptor:desc device:nil serializer:&fx.ser];
	CHECK(smp != nil && smp.rgfxObjId == 1);
	CHECK([smp conformsToProtocol:@protocol(MTLSamplerState)]);

	auto msg = expectMessage<RgfxSamplerCreate>(fx.ring, RGFX_CMD_SAMPLER_CREATE);
	CHECK(msg.obj_id == smp.rgfxObjId);
	CHECK(msg.min_filter == (uint32_t)MTLSamplerMinMagFilterLinear);
	CHECK(msg.mag_filter == (uint32_t)MTLSamplerMinMagFilterLinear);
	CHECK(msg.mip_filter == (uint32_t)MTLSamplerMipFilterLinear);
	CHECK(msg.s_address_mode == (uint32_t)MTLSamplerAddressModeClampToEdge);
	CHECK(msg.t_address_mode == (uint32_t)MTLSamplerAddressModeRepeat);
	CHECK(msg.r_address_mode == (uint32_t)MTLSamplerAddressModeMirrorRepeat);
	CHECK(msg.max_anisotropy == 4);
	CHECK(msg.compare_function == (uint32_t)MTLCompareFunctionLess);
	float lodMax;
	std::memcpy(&lodMax, &msg.lod_max_clamp_bits, 4);
	CHECK(lodMax == 7.5f);
	CHECK(msg.flags == (0x1u | 0x4u)); // normalized coords + argument-buffer usable
}

// A depth/stencil wrapper's create routes an RGFX_CMD_DEPTH_STENCIL_CREATE,
// mapping the depth test/write and both faces' stencil state.
void test_depth_stencil_routing()
{
	Fixture fx;

	MTLDepthStencilDescriptor *desc = [[MTLDepthStencilDescriptor alloc] init];
	desc.depthCompareFunction = MTLCompareFunctionLessEqual;
	desc.depthWriteEnabled = YES;
	MTLStencilDescriptor *front = [[MTLStencilDescriptor alloc] init];
	front.stencilCompareFunction = MTLCompareFunctionEqual;
	front.stencilFailureOperation = MTLStencilOperationKeep;
	front.depthFailureOperation = MTLStencilOperationIncrementClamp;
	front.depthStencilPassOperation = MTLStencilOperationReplace;
	front.readMask = 0xfe;
	front.writeMask = 0xcd;
	desc.frontFaceStencil = front;

	RgfxDepthStencilState *ds =
	    [RgfxDepthStencilState depthStencilStateWithDescriptor:desc device:nil serializer:&fx.ser];
	CHECK(ds != nil && ds.rgfxObjId == 1);
	CHECK([ds conformsToProtocol:@protocol(MTLDepthStencilState)]);

	auto msg = expectMessage<RgfxDepthStencilCreate>(fx.ring, RGFX_CMD_DEPTH_STENCIL_CREATE);
	CHECK(msg.obj_id == ds.rgfxObjId);
	CHECK(msg.depth_compare == (uint32_t)MTLCompareFunctionLessEqual);
	CHECK(msg.depth_write_enabled == 1);
	CHECK(msg.front.stencil_compare == (uint32_t)MTLCompareFunctionEqual);
	CHECK(msg.front.stencil_fail_op == (uint32_t)MTLStencilOperationKeep);
	CHECK(msg.front.depth_fail_op == (uint32_t)MTLStencilOperationIncrementClamp);
	CHECK(msg.front.pass_op == (uint32_t)MTLStencilOperationReplace);
	CHECK(msg.front.read_mask == 0xfe && msg.front.write_mask == 0xcd);
	// The untouched back face carries Metal's default MTLStencilDescriptor:
	// stencilCompareFunction Always, keep operations, all-ones masks.
	CHECK(msg.back.stencil_compare == (uint32_t)MTLCompareFunctionAlways);
	CHECK(msg.back.read_mask == 0xffffffffu && msg.back.write_mask == 0xffffffffu);
}

// A heap wrapper's create routes an RGFX_CMD_HEAP_CREATE; its getters answer the
// descriptor; and a Placement heap sub-allocates a buffer (BUFFER_AT_OFFSET) and a
// texture (TEXTURE_CREATE into the heap), both wrapped. The offset-less variants
// are documented nil stubs this slice.
void test_heap_routing_and_suballoc()
{
	Fixture fx;
	// Drained while fx lives: a texture's dealloc posts its destroy through fx.
	@autoreleasepool {

	MTLHeapDescriptor *hd = [[MTLHeapDescriptor alloc] init];
	hd.size = 0x100000;
	hd.storageMode = MTLStorageModePrivate;
	hd.type = MTLHeapTypePlacement;

	RgfxHeap *heap = [RgfxHeap heapWithDescriptor:hd device:nil serializer:&fx.ser];
	CHECK(heap != nil && heap.rgfxObjId == 1);
	CHECK([heap size] == 0x100000);
	CHECK([heap type] == MTLHeapTypePlacement);
	CHECK([heap storageMode] == MTLStorageModePrivate);
	// Its options as Metal answers them: a Default hazard tracking mode is Untracked for a heap.
	CHECK([heap resourceOptions] == (hd.resourceOptions | MTLResourceHazardTrackingModeUntracked));

	auto hm = expectMessage<RgfxHeapCreate>(fx.ring, RGFX_CMD_HEAP_CREATE);
	CHECK(hm.obj_id == heap.rgfxObjId && hm.gpu_address != 0);
	CHECK(hm.size_bytes == 0x100000 && hm.heap_type == (int32_t)MTLHeapTypePlacement);
	CHECK(hm.options == (uint32_t)hd.resourceOptions);

	// Placement buffer at an offset -> BUFFER_AT_OFFSET into this heap.
	id<MTLBuffer> hb = [heap newBufferWithLength:0x400
	                                    options:MTLResourceStorageModePrivate
	                                     offset:0x2000];
	CHECK(hb != nil);
	CHECK([(RgfxBuffer *)hb rgfxObjId] == heap.rgfxObjId + 1); // next objId
	CHECK([hb length] == 0x400);
	auto bm = expectMessage<RgfxBufferAtOffset>(fx.ring, RGFX_CMD_BUFFER_AT_OFFSET);
	CHECK(bm.obj_id == [(RgfxBuffer *)hb rgfxObjId]);
	CHECK(bm.heap_obj_id == heap.rgfxObjId);
	CHECK(bm.offset == 0x2000 && bm.length == 0x400);

	// Placement texture at an offset -> TEXTURE_CREATE into this heap.
	MTLTextureDescriptor *td =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                                                       width:256
	                                                      height:256
	                                                   mipmapped:NO];
	td.storageMode = MTLStorageModePrivate;
	id<MTLTexture> ht = [heap newTextureWithDescriptor:td offset:0x4000];
	CHECK(ht != nil);
	CHECK([ht width] == 256 && [ht height] == 256);
	auto tm = expectMessage<RgfxTextureCreate>(fx.ring, RGFX_CMD_TEXTURE_CREATE);
	CHECK(tm.obj_id == [(RgfxTexture *)ht rgfxObjId]);
	CHECK(tm.heap_obj_id == heap.rgfxObjId && tm.heap_offset == 0x4000);
	CHECK(tm.gpu_address == 0); // heap-placed: no device address
	CHECK(tm.desc.pixel_format == MTLPixelFormatRGBA8Unorm);
	// Each answers its heap and offset, and is aliasable, as a Placement heap's resources are
	// from their making.
	CHECK(hb.heap == heap && hb.heapOffset == 0x2000);
	CHECK(ht.heap == heap && ht.heapOffset == 0x4000);
	CHECK(hb.storageMode == MTLStorageModePrivate);
	CHECK([hb contents] == nullptr); // a Private heap's buffer: the GPU's bytes alone
	CHECK(ht.isAliasable && hb.isAliasable);
	[ht makeAliasable];
	CHECK(ht.isAliasable);

	// The offset-less (Automatic-placement) variants are nil stubs this slice.
	CHECK([heap newBufferWithLength:0x100 options:MTLResourceStorageModePrivate] == nil);
	CHECK([heap newTextureWithDescriptor:td] == nil);

	// Of an Automatic heap, a placement at an offset is nil, and nothing is posted.
	hd.type = MTLHeapTypeAutomatic;
	RgfxHeap *automatic = [RgfxHeap heapWithDescriptor:hd device:nil serializer:&fx.ser];
	(void)expectMessage<RgfxHeapCreate>(fx.ring, RGFX_CMD_HEAP_CREATE);
	CHECK([automatic newTextureWithDescriptor:td offset:0x4000] == nil);
	CHECK([automatic newBufferWithLength:0x400
	                             options:MTLResourceStorageModePrivate
	                              offset:0x2000] == nil);
	RgfxMsgHeader h = {};
	std::vector<uint8_t> posted;
	CHECK(!fx.ring.next(&h, &posted));
	}
}

// A wrapper with no serialiser route cannot be created (each resource must reach
// the host).
void test_resource_guards()
{
	MTLTextureDescriptor *td =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:16
	                                                      height:16
	                                                   mipmapped:NO];
	CHECK([RgfxTexture textureWithDescriptor:td
	                                    heap:RGFX_OBJ_NONE
	                              heapOffset:0
	                                  device:nil
	                              serializer:nullptr] == nil);
	CHECK([RgfxSampler samplerWithDescriptor:[[MTLSamplerDescriptor alloc] init]
	                                  device:nil
	                              serializer:nullptr] == nil);
	CHECK([RgfxDepthStencilState depthStencilStateWithDescriptor:[[MTLDepthStencilDescriptor alloc]
	                                                                 init]
	                                                     device:nil
	                                                 serializer:nullptr] == nil);
	CHECK([RgfxHeap heapWithDescriptor:[[MTLHeapDescriptor alloc] init]
	                            device:nil
	                        serializer:nullptr] == nil);
}


// Reads the next message off the ring into `bytes` and checks its type; returns the
// full message bytes (head + tail) for a variable-size wire struct the fixed-size
// expectMessage<T> cannot read. Aborts (via CHECK) if there is no message.
std::vector<uint8_t> nextMessage(TestRing &ring, uint32_t type)
{
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(ring.next(&h, &bytes));
	CHECK(h.type == type);
	return bytes;
}

// A library create forwards its bytes unchanged (RGFX_CMD_LIBRARY_CREATE), and the
// library vends a function by name as a guest-side RgfxFunction (no wire message).
// Libraries made from bytes of one content key share one function table, read once for the
// process (a framework library loaded again is not read again); bytes with no key read their own.
void test_libraries_of_one_key_share_their_function_table()
{
	Fixture fx;
	std::vector<uint8_t> metallib(0x30, 0);
	std::memcpy(metallib.data(), "MTLB", 4);
	metallib[0x18] = 0x30; // the function list's offset: one of no functions, at the end
	metallib.resize(0x34, 0);
	uint8_t key[32] = {};
	key[0] = 0x5a;
	key[31] = 0xa5;
	auto make = [&](const uint8_t *k) {
		uint32_t status = 0;
		RgfxLibrary *lib = [RgfxLibrary libraryWithData:metallib.data()
		                                        length:metallib.size()
		                                           key:k
		                                        device:nil
		                                    serializer:&fx.ser
		                                        status:&status];
		(void)nextMessage(fx.ring, RGFX_CMD_LIBRARY_CREATE);
		return lib;
	};
	RgfxLibrary *first = make(key);
	RgfxLibrary *again = make(key);
	RgfxLibrary *unkeyed = make(nullptr);
	CHECK(first != nil && again != nil && unkeyed != nil);
	CHECK([first rgfxFunctionTable] != nullptr);
	CHECK([again rgfxFunctionTable] == [first rgfxFunctionTable]);
	CHECK([unkeyed rgfxFunctionTable] != [first rgfxFunctionTable]);
	CHECK(first.functionNames.count == 0 && again.functionNames.count == 0);
}

void test_library_and_function_routing()
{
	Fixture fx;

	const uint8_t libBytes[] = {0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03};
	uint32_t libStatus = 0;
	RgfxLibrary *lib = [RgfxLibrary libraryWithData:libBytes
	                                        length:sizeof(libBytes)
	                                           key:nullptr
	                                        device:nil
	                                    serializer:&fx.ser
	                                        status:&libStatus];
	CHECK(lib != nil);
	CHECK(lib.rgfxObjId == 1); // first object on this device
	CHECK([lib conformsToProtocol:@protocol(MTLLibrary)]);

	auto bytes = nextMessage(fx.ring, RGFX_CMD_LIBRARY_CREATE);
	RgfxLibraryCreate head;
	std::memcpy(&head, bytes.data(), sizeof(head));
	CHECK(head.header.seqno == 1);
	CHECK(head.obj_id == lib.rgfxObjId);
	CHECK(head.byte_length == sizeof(libBytes));
	CHECK(std::memcmp(bytes.data() + RGFX_LIBRARY_BYTES_OFFSET, libBytes, sizeof(libBytes)) == 0);

	// A function is a guest-side reference: it holds the library objId + name, picks
	// no objId, and posts nothing.
	id<MTLFunction> fn = [lib newFunctionWithName:@"vmain"];
	CHECK(fn != nil);
	RgfxFunction *rf = (RgfxFunction *)fn;
	CHECK(rf.rgfxObjId == 0); // RGFX_OBJ_NONE: no host handle of its own
	CHECK(rf.rgfxLibraryObjId == lib.rgfxObjId);
	CHECK(std::strcmp([rf rgfxEntryPointUTF8], "vmain") == 0);
	CHECK([rf rgfxConstantCount] == 0);
	CHECK([[fn name] isEqualToString:@"vmain"]);

	RgfxMsgHeader extra = {};
	std::vector<uint8_t> extraBytes;
	CHECK(!fx.ring.next(&extra, &extraBytes)); // newFunctionWithName: posts nothing

	// No function table: a stand-in hash, the same for the same function, another for
	// another name.
	id<MTLFunction> other = [lib newFunctionWithName:@"fmain"];
	CHECK(std::memcmp([rf bitCodeHash]->bytes, [(RgfxFunction *)[lib newFunctionWithName:@"vmain"]
	                                               bitCodeHash]->bytes, 32) == 0);
	CHECK(std::memcmp([rf bitCodeHash]->bytes, [(RgfxFunction *)other bitCodeHash]->bytes, 32) != 0);

	// A .metallib's function list: its entry's stage from TYPE and its hash from HASH.
	std::vector<uint8_t> metallib(0x30, 0);
	std::memcpy(metallib.data(), "MTLB", 4);
	metallib[0x18] = 0x30; // the function list's offset
	auto put32 = [&](uint32_t v) {
		for (int i = 0; i < 4; i++)
			metallib.push_back(static_cast<uint8_t>(v >> (8 * i)));
	};
	auto tag = [&](const char *name, const uint8_t *data, uint16_t size) {
		metallib.insert(metallib.end(), name, name + 4);
		metallib.push_back(static_cast<uint8_t>(size));
		metallib.push_back(static_cast<uint8_t>(size >> 8));
		metallib.insert(metallib.end(), data, data + size);
	};
	uint8_t hash[32];
	for (int i = 0; i < 32; i++)
		hash[i] = static_cast<uint8_t>(0xa0 + i);
	const uint8_t stage = 2; // a kernel
	put32(1);
	put32(4 + (6 + 6) + (6 + 1) + (6 + 32) + 4);
	tag("NAME", reinterpret_cast<const uint8_t *>("kmain"), 6);
	tag("TYPE", &stage, 1);
	tag("HASH", hash, 32);
	metallib.insert(metallib.end(), {'E', 'N', 'D', 'T'});
	RgfxLibrary *tabled = [RgfxLibrary libraryWithData:metallib.data()
	                                            length:metallib.size()
	                                               key:nullptr
	                                            device:nil
	                                        serializer:&fx.ser
	                                            status:&libStatus];
	CHECK(tabled != nil);
	RgfxFunction *k = (RgfxFunction *)[tabled newFunctionWithName:@"kmain"];
	CHECK(k != nil && k.functionType == MTLFunctionTypeKernel);
	CHECK(std::memcmp([k bitCodeHash]->bytes, hash, 32) == 0);
	CHECK([tabled newFunctionWithName:@"vmain"] == nil);
	while (fx.ring.next(&extra, &extraBytes)) {
	}

	// A library with no route cannot be created.
	RgfxLibrary *none = [RgfxLibrary libraryWithData:libBytes
	                                         length:sizeof(libBytes)
	                                            key:nullptr
	                                         device:nil
	                                     serializer:nullptr
	                                         status:&libStatus];
	CHECK(none == nil);
}

// A render pipeline create maps the MTLRenderPipelineDescriptor into the wire: the
// functions, label, colour attachment, flags and vertex descriptor; the wrapper
// holds objId + seqno and answers reflection getters with documented placeholders.
void test_render_pipeline_routing()
{
	Fixture fx;

	const uint8_t libBytes[] = {1, 2, 3, 4};
	uint32_t libStatus = 0;
	RgfxLibrary *lib = [RgfxLibrary libraryWithData:libBytes
	                                        length:sizeof(libBytes)
	                                           key:nullptr
	                                        device:nil
	                                    serializer:&fx.ser
	                                        status:&libStatus];
	(void)nextMessage(fx.ring, RGFX_CMD_LIBRARY_CREATE); // drain the library create

	id<MTLFunction> vfn = [lib newFunctionWithName:@"vmain"];
	id<MTLFunction> ffn = [lib newFunctionWithName:@"fmain"];

	MTLRenderPipelineDescriptor *desc = [[MTLRenderPipelineDescriptor alloc] init];
	desc.vertexFunction = vfn;
	desc.fragmentFunction = ffn;
	desc.label = @"pipe";
	desc.rasterSampleCount = 4;
	desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
	desc.colorAttachments[0].blendingEnabled = YES;
	desc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
	desc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
	desc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
	desc.alphaToCoverageEnabled = YES;
	desc.supportIndirectCommandBuffers = YES;
	desc.vertexLinkedFunctions.functions = @[ [lib newFunctionWithName:@"vlink"] ];
	desc.fragmentLinkedFunctions.privateFunctions = @[ [lib newFunctionWithName:@"flink"] ];

	MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
	vd.layouts[0].stride = 32;
	vd.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;
	vd.layouts[0].stepRate = 1;
	vd.attributes[0].format = MTLVertexFormatFloat3;
	vd.attributes[0].offset = 0;
	vd.attributes[0].bufferIndex = 0;
	desc.vertexDescriptor = vd;

	NSError *err = nil;
	RgfxRenderPipelineState *ps =
	    [RgfxRenderPipelineState renderPipelineStateWithDescriptor:desc
	                                                       device:nil
	                                                   serializer:&fx.ser
	                                                        error:&err];
	CHECK(ps != nil);
	CHECK(err == nil);
	CHECK(ps.rgfxObjId == 2); // library was objId 1, the pipeline the next
	CHECK(ps.rgfxSeqno == 2); // library post was seqno 1, the pipeline the next
	CHECK([ps conformsToProtocol:@protocol(MTLRenderPipelineState)]);

	auto bytes = nextMessage(fx.ring, RGFX_CMD_RENDER_PIPELINE_CREATE);
	RgfxRenderPipelineCreate head;
	std::memcpy(&head, bytes.data(), sizeof(head));
	CHECK(head.obj_id == ps.rgfxObjId);
	CHECK(head.vertex_library_obj_id == lib.rgfxObjId);
	CHECK(head.fragment_library_obj_id == lib.rgfxObjId);
	CHECK(head.raster_sample_count == 4);
	CHECK(head.color_attachment_count == 1);
	CHECK(head.colors[0].pixel_format == static_cast<uint32_t>(MTLPixelFormatBGRA8Unorm));
	CHECK(head.colors[0].blending_enabled == 1);
	CHECK(head.colors[0].source_rgb_blend_factor == static_cast<uint32_t>(MTLBlendFactorSourceAlpha));
	CHECK(head.colors[0].destination_rgb_blend_factor ==
	      static_cast<uint32_t>(MTLBlendFactorOneMinusSourceAlpha));
	CHECK(head.depth_pixel_format == static_cast<uint32_t>(MTLPixelFormatDepth32Float));
	CHECK((head.flags & RGFX_RENDER_PIPELINE_ALPHA_TO_COVERAGE) != 0);
	CHECK((head.flags & RGFX_RENDER_PIPELINE_RASTERIZATION_ENABLED) != 0); // default on
	CHECK((head.flags & RGFX_RENDER_PIPELINE_SUPPORT_ICB) != 0);
	// Each stage's linked functions, in their lists.
	CHECK(head.vertex_linked_count == 1 && head.fragment_linked_count == 1);
	RgfxLinkedFunction vlink, flink;
	std::memcpy(&vlink, bytes.data() + head.vertex_linked_offset, sizeof(vlink));
	std::memcpy(&flink, bytes.data() + head.fragment_linked_offset, sizeof(flink));
	CHECK(vlink.list == RGFX_LINKED_FUNCTIONS && vlink.name_len == 5);
	CHECK(std::memcmp(bytes.data() + vlink.name_offset, "vlink", 5) == 0);
	CHECK(flink.list == RGFX_LINKED_PRIVATE_FUNCTIONS && flink.name_len == 5);
	CHECK(std::memcmp(bytes.data() + flink.name_offset, "flink", 5) == 0);
	CHECK(vlink.library_obj_id == lib.rgfxObjId && flink.library_obj_id == lib.rgfxObjId);

	// The functions, label and vertex descriptor are in the tail at their offsets.
	CHECK(head.vertex_name_len == 5);
	CHECK(std::memcmp(bytes.data() + head.vertex_name_offset, "vmain", 5) == 0);
	CHECK(head.fragment_name_len == 5);
	CHECK(std::memcmp(bytes.data() + head.fragment_name_offset, "fmain", 5) == 0);
	CHECK(head.label_len == 4);
	CHECK(std::memcmp(bytes.data() + head.label_offset, "pipe", 4) == 0);

	CHECK(head.vertex_layout_count == 1);
	RgfxVertexBufferLayout vl;
	std::memcpy(&vl, bytes.data() + head.vertex_layouts_offset, sizeof(vl));
	CHECK(vl.buffer_index == 0);
	CHECK(vl.stride == 32);
	CHECK(vl.step_function == static_cast<uint32_t>(MTLVertexStepFunctionPerVertex));

	CHECK(head.vertex_attribute_count == 1);
	RgfxVertexAttribute va;
	std::memcpy(&va, bytes.data() + head.vertex_attributes_offset, sizeof(va));
	CHECK(va.attribute_index == 0);
	CHECK(va.format == static_cast<uint32_t>(MTLVertexFormatFloat3));
	CHECK(va.buffer_index == 0);

	// Reflection getters are documented placeholders until the reply reader lands.
	CHECK([ps maxTotalThreadsPerThreadgroup] == 1024);
	CHECK(ps.gpuResourceID._impl == 0);

	// No serialiser route is a refusal with an error.
	NSError *err2 = nil;
	RgfxRenderPipelineState *noRoute =
	    [RgfxRenderPipelineState renderPipelineStateWithDescriptor:desc
	                                                       device:nil
	                                                   serializer:nullptr
	                                                        error:&err2];
	CHECK(noRoute == nil);
	CHECK(err2 != nil);
}

// A compute pipeline create maps the function (and the descriptor knobs) into the
// wire; the wrapper holds objId + seqno.
void test_compute_pipeline_routing()
{
	Fixture fx;

	const uint8_t libBytes[] = {9, 9, 9};
	uint32_t libStatus = 0;
	RgfxLibrary *lib = [RgfxLibrary libraryWithData:libBytes
	                                        length:sizeof(libBytes)
	                                           key:nullptr
	                                        device:nil
	                                    serializer:&fx.ser
	                                        status:&libStatus];
	(void)nextMessage(fx.ring, RGFX_CMD_LIBRARY_CREATE);

	id<MTLFunction> kfn = [lib newFunctionWithName:@"kmain"];

	NSError *err = nil;
	RgfxComputePipelineState *cs =
	    [RgfxComputePipelineState computePipelineStateWithFunction:kfn
	                                                       device:nil
	                                                   serializer:&fx.ser
	                                                        error:&err];
	CHECK(cs != nil);
	CHECK(err == nil);
	CHECK(cs.rgfxObjId == 2);
	CHECK(cs.rgfxSeqno == 2);

	auto bytes = nextMessage(fx.ring, RGFX_CMD_COMPUTE_PIPELINE_CREATE);
	RgfxComputePipelineCreate head;
	std::memcpy(&head, bytes.data(), sizeof(head));
	CHECK(head.obj_id == cs.rgfxObjId);
	CHECK(head.function_library_obj_id == lib.rgfxObjId);
	CHECK(head.name_len == 5);
	CHECK(std::memcmp(bytes.data() + head.name_offset, "kmain", 5) == 0);
	CHECK(head.flags == 0);
	CHECK(head.max_call_stack_depth == 1); // Metal's default
	CHECK(head.stage_input_layout_count == 0 && head.stage_input_attribute_count == 0);
	CHECK(head.linked_count == 0);

	CHECK([cs threadExecutionWidth] == 32);           // documented placeholder
	CHECK([cs maxTotalThreadsPerThreadgroup] == 1024); // documented placeholder

	// The descriptor variant carries the ICB/exec-width flags and the thread cap.
	MTLComputePipelineDescriptor *desc = [[MTLComputePipelineDescriptor alloc] init];
	desc.computeFunction = kfn;
	desc.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
	desc.maxTotalThreadsPerThreadgroup = 256;
	NSError *err2 = nil;
	RgfxComputePipelineState *cs2 =
	    [RgfxComputePipelineState computePipelineStateWithDescriptor:desc
	                                                         device:nil
	                                                     serializer:&fx.ser
	                                                          error:&err2];
	CHECK(cs2 != nil);
	CHECK(err2 == nil);
	auto bytes2 = nextMessage(fx.ring, RGFX_CMD_COMPUTE_PIPELINE_CREATE);
	RgfxComputePipelineCreate head2;
	std::memcpy(&head2, bytes2.data(), sizeof(head2));
	CHECK((head2.flags & RGFX_COMPUTE_PIPELINE_TG_MULTIPLE_OF_EXEC_WIDTH) != 0);
	CHECK(head2.max_total_threads_per_threadgroup == 256);

	// The stage-input descriptor's layouts and attributes in use, its index type and buffer,
	// the call stack depth, and the linked functions in their lists, a function made by
	// descriptor with its options, and one with a specialized name, which it answers, under
	// its entry point.
	MTLFunctionDescriptor *fd = [MTLFunctionDescriptor functionDescriptor];
	fd.name = @"bin";
	fd.options = MTLFunctionOptionCompileToBinary;
	id<MTLFunction> binary = [lib newFunctionWithDescriptor:fd error:nil];
	CHECK(binary.options == MTLFunctionOptionCompileToBinary);
	CHECK([binary.name isEqualToString:@"bin"]);
	MTLFunctionDescriptor *pd = [MTLFunctionDescriptor functionDescriptor];
	pd.name = @"fp";
	pd.specializedName = @"custom_fn";
	id<MTLFunction> specialized = [lib newFunctionWithDescriptor:pd error:nil];
	CHECK([specialized.name isEqualToString:@"custom_fn"]);
	MTLComputePipelineDescriptor *whole = [[MTLComputePipelineDescriptor alloc] init];
	whole.computeFunction = kfn;
	whole.maxCallStackDepth = 3;
	MTLStageInputOutputDescriptor *sd = [MTLStageInputOutputDescriptor stageInputOutputDescriptor];
	sd.layouts[2].stride = 8;
	sd.layouts[2].stepFunction = MTLStepFunctionThreadPositionInGridX;
	sd.attributes[1].format = MTLAttributeFormatFloat2;
	sd.attributes[1].bufferIndex = 2;
	sd.indexType = MTLIndexTypeUInt32;
	sd.indexBufferIndex = 5;
	whole.stageInputDescriptor = sd;
	MTLLinkedFunctions *linked = [MTLLinkedFunctions linkedFunctions];
	linked.functions = @[ [lib newFunctionWithName:@"fa"] ];
	linked.binaryFunctions = @[ binary ];
	linked.groups = @{
		@"zeta" : @[ [lib newFunctionWithName:@"gz"] ],
		@"alpha" : @[ [lib newFunctionWithName:@"ga"], [lib newFunctionWithName:@"gb"] ],
	};
	linked.privateFunctions = @[ specialized ];
	whole.linkedFunctions = linked;
	RgfxComputePipelineState *cs4 =
	    [RgfxComputePipelineState computePipelineStateWithDescriptor:whole
	                                                          device:nil
	                                                      serializer:&fx.ser
	                                                           error:nil];
	CHECK(cs4 != nil);
	auto bytes4 = nextMessage(fx.ring, RGFX_CMD_COMPUTE_PIPELINE_CREATE);
	RgfxComputePipelineCreate head4;
	std::memcpy(&head4, bytes4.data(), sizeof(head4));
	CHECK(head4.max_call_stack_depth == 3);
	CHECK(head4.stage_input_layout_count == 1);
	RgfxVertexBufferLayout sl;
	std::memcpy(&sl, bytes4.data() + head4.stage_input_layouts_offset, sizeof(sl));
	CHECK(sl.buffer_index == 2 && sl.stride == 8);
	CHECK(sl.step_function == static_cast<uint32_t>(MTLStepFunctionThreadPositionInGridX));
	CHECK(head4.stage_input_attribute_count == 1);
	RgfxVertexAttribute sa;
	std::memcpy(&sa, bytes4.data() + head4.stage_input_attributes_offset, sizeof(sa));
	CHECK(sa.attribute_index == 1 && sa.buffer_index == 2);
	CHECK(sa.format == static_cast<uint32_t>(MTLAttributeFormatFloat2));
	CHECK(head4.stage_input_index_type == static_cast<uint32_t>(MTLIndexTypeUInt32));
	CHECK(head4.stage_input_index_buffer_index == 5);
	CHECK(head4.linked_count == 6);
	struct Want {
		const char *name;
		uint32_t list;
		uint32_t options;
		const char *group;
		const char *specialized;
	};
	const Want want[6] = {
		{"fa", RGFX_LINKED_FUNCTIONS, 0, "", ""},
		{"bin", RGFX_LINKED_BINARY_FUNCTIONS, MTLFunctionOptionCompileToBinary, "", ""},
		{"ga", RGFX_LINKED_GROUP, 0, "alpha", ""},
		{"gb", RGFX_LINKED_GROUP, 0, "alpha", ""},
		{"gz", RGFX_LINKED_GROUP, 0, "zeta", ""},
		{"fp", RGFX_LINKED_PRIVATE_FUNCTIONS, 0, "", "custom_fn"},
	};
	for (uint32_t i = 0; i < 6; i++) {
		RgfxLinkedFunction r;
		std::memcpy(&r, bytes4.data() + head4.linked_offset + i * sizeof(r), sizeof(r));
		CHECK(r.library_obj_id == lib.rgfxObjId);
		CHECK(r.name_len == std::strlen(want[i].name));
		CHECK(std::memcmp(bytes4.data() + r.name_offset, want[i].name, r.name_len) == 0);
		CHECK(r.list == want[i].list);
		CHECK(r.options == want[i].options);
		CHECK(r.group_len == std::strlen(want[i].group));
		CHECK(std::memcmp(bytes4.data() + r.group_offset, want[i].group, r.group_len) == 0);
		CHECK(r.specialized_name_len == std::strlen(want[i].specialized));
		CHECK((r.specialized_name_offset == 0) == (r.specialized_name_len == 0));
		CHECK(std::memcmp(bytes4.data() + r.specialized_name_offset, want[i].specialized,
		                  r.specialized_name_len) == 0);
	}

	// A linked function another device's library made is refused before anything is sent:
	// the next create is the one the ring holds next.
	id<MTLDevice> metal = MTLCreateSystemDefaultDevice();
	id<MTLLibrary> real =
	    [metal newLibraryWithSource:@"[[visible]] int f(int x) { return x + 1; }" options:nil error:nil];
	id<MTLFunction> foreign = [real newFunctionWithName:@"f"];
	CHECK(foreign != nil);
	MTLLinkedFunctions *other = [MTLLinkedFunctions linkedFunctions];
	other.privateFunctions = @[ foreign ];
	whole.linkedFunctions = other;
	NSError *foreignErr = nil;
	CHECK([RgfxComputePipelineState computePipelineStateWithDescriptor:whole
	                                                            device:nil
	                                                        serializer:&fx.ser
	                                                             error:&foreignErr] == nil);
	CHECK(foreignErr.code == RgfxMetalErrorInvalidDescriptor);
	whole.linkedFunctions = nil;
	RgfxComputePipelineState *after =
	    [RgfxComputePipelineState computePipelineStateWithDescriptor:whole
	                                                          device:nil
	                                                      serializer:&fx.ser
	                                                           error:nil];
	auto bytes5 = nextMessage(fx.ring, RGFX_CMD_COMPUTE_PIPELINE_CREATE);
	RgfxComputePipelineCreate head5;
	std::memcpy(&head5, bytes5.data(), sizeof(head5));
	CHECK(head5.obj_id == after.rgfxObjId);
	CHECK(head5.linked_count == 0);

	// A compute create with no function is a refusal with an error. (The nil goes
	// through a variable so the nonnull parameter's literal-nil diagnostic does not
	// fire; a guest can still reach this path with a nil id at runtime.)
	NSError *err3 = nil;
	id<MTLFunction> nilFn = nil;
	RgfxComputePipelineState *noFn =
	    [RgfxComputePipelineState computePipelineStateWithFunction:nilFn
	                                                       device:nil
	                                                   serializer:&fx.ser
	                                                        error:&err3];
	CHECK(noFn == nil);
	CHECK(err3 != nil);
}

// One function of a test .metallib: its entry point, its stage (the TYPE tag's byte: 0
// vertex, 1 fragment, 2 kernel) and the constants its public metadata declares.
struct TestFunction {
	const char *name;
	uint8_t stage;
	std::vector<RgfxConstantDecl> constants;
};

void put16(std::vector<uint8_t> &v, uint16_t x) { v.insert(v.end(), (uint8_t *)&x, (uint8_t *)&x + 2); }
void put32(std::vector<uint8_t> &v, uint32_t x) { v.insert(v.end(), (uint8_t *)&x, (uint8_t *)&x + 4); }
void put64(std::vector<uint8_t> &v, uint64_t x) { v.insert(v.end(), (uint8_t *)&x, (uint8_t *)&x + 8); }

// A tag: its four-character name, a u16 size and the data.
void putTag(std::vector<uint8_t> &v, const char *name, const std::vector<uint8_t> &data)
{
	v.insert(v.end(), name, name + 4);
	put16(v, static_cast<uint16_t>(data.size()));
	v.insert(v.end(), data.begin(), data.end());
}

// A .metallib of `functions` in the layout Apple's compiler writes (measured: macOS 27's
// `metal`): the header's function list offset and size at 0x18, its public metadata's at 0x28;
// per function a group of tags (NAME, TYPE, OFFT, whose first field places the function's
// entry in the public metadata), and per function a public entry (its size, then a CNST tag
// where it declares constants: a u16 count, then each name, MTLDataType byte, u16 index and
// required byte), whose size counts itself where `sizeCountsItself` (macOS 27's) and not
// otherwise (macOS 26's). No bitcode: the shell never reads it.
std::vector<uint8_t> metallibOf(const std::vector<TestFunction> &functions,
                                bool sizeCountsItself = true)
{
	std::vector<uint8_t> list, pub;
	put32(list, static_cast<uint32_t>(functions.size()));
	for (const TestFunction &f : functions) {
		std::vector<uint8_t> entry;
		put32(entry, 0);
		if (!f.constants.empty()) {
			std::vector<uint8_t> cnst;
			put16(cnst, static_cast<uint16_t>(f.constants.size()));
			for (const RgfxConstantDecl &d : f.constants) {
				cnst.insert(cnst.end(), d.name.begin(), d.name.end());
				cnst.push_back(0);
				cnst.push_back(static_cast<uint8_t>(d.type));
				put16(cnst, static_cast<uint16_t>(d.index));
				cnst.push_back(d.required ? 1 : 0);
			}
			putTag(entry, "CNST", cnst);
		}
		entry.insert(entry.end(), { 'E', 'N', 'D', 'T' });
		uint32_t entrySize = static_cast<uint32_t>(entry.size() - (sizeCountsItself ? 0 : 4));
		std::memcpy(entry.data(), &entrySize, 4);
		std::vector<uint8_t> group, offt;
		put32(group, 0);
		std::vector<uint8_t> name(f.name, f.name + std::strlen(f.name) + 1);
		putTag(group, "NAME", name);
		putTag(group, "TYPE", { f.stage });
		put64(offt, pub.size());
		put64(offt, 0);
		put64(offt, 0);
		putTag(group, "OFFT", offt);
		group.insert(group.end(), { 'E', 'N', 'D', 'T' });
		uint32_t groupSize = static_cast<uint32_t>(group.size());
		std::memcpy(group.data(), &groupSize, 4);
		list.insert(list.end(), group.begin(), group.end());
		pub.insert(pub.end(), entry.begin(), entry.end());
	}
	std::vector<uint8_t> out(0x58, 0);
	std::memcpy(out.data(), "MTLB", 4);
	uint64_t fields[4] = { 0x58, list.size(), 0x58 + list.size(), pub.size() };
	std::memcpy(out.data() + 0x18, fields, sizeof(fields));
	out.insert(out.end(), list.begin(), list.end());
	out.insert(out.end(), pub.begin(), pub.end());
	return out;
}

// The constant `function` carries at `index`, or null.
const RgfxFunctionConstant *constantAt(id<MTLFunction> function, uint32_t index)
{
	RgfxFunction *rf = (RgfxFunction *)function;
	for (uint32_t i = 0; i < [rf rgfxConstantCount]; i++) {
		if ([rf rgfxConstants][i].index == index)
			return &[rf rgfxConstants][i];
	}
	return nullptr;
}

// A stitched library is Metal's own, stitched over the shadows of our functions and of no binary
// archive, and made ours from the bytes Metal answers for it: its functions are the graphs'.
void test_stitched_library_over_shadows()
{
	Fixture fx;
	const char *path = getenv("RGFX_STITCHABLE");
	CHECK(path != nullptr);
	if (path == nullptr)
		return;
	NSData *bytes = [NSData dataWithContentsOfFile:@(path)];
	id<MTLDevice> metal = MTLCreateSystemDefaultDevice();
	auto make = [&](const void *p, size_t n) {
		uint32_t status = 0;
		return [RgfxLibrary libraryWithData:p
		                             length:n
		                                key:nullptr
		                             device:nil
		                         serializer:&fx.ser
		                             status:&status];
	};
	RgfxLibrary *lib = make(bytes.bytes, bytes.length);
	[lib rgfxSetSource:dispatch_data_create(bytes.bytes, bytes.length, nil, ^{
		     (void)bytes;
	     })
	       shadowMaker:^id<MTLLibrary>(dispatch_data_t d) {
		       return [metal newLibraryWithData:d error:nil];
	       }];
	// A function's reflection is the shadow's, Metal's own answer (for this library, none), with
	// no unsupported use; a library with no shadow answers none, as unsupported.
	SEL reflect = @selector(reflectionForFunctionWithName:);
	MTLFunctionReflection *reflected = [lib reflectionForFunctionWithName:@"stitch_caller"];
	MTLFunctionReflection *metals =
	    [[lib rgfxShadow] reflectionForFunctionWithName:@"stitch_caller"];
	CHECK((reflected == nil) == (metals == nil) && reflected.bindings.count == metals.bindings.count);
	CHECK(!RgfxUnsupportedWasLogged([RgfxLibrary class], reflect));
	CHECK([make(bytes.bytes, bytes.length) reflectionForFunctionWithName:@"stitch_caller"] == nil);
	CHECK(RgfxUnsupportedWasLogged([RgfxLibrary class], reflect));
	id<MTLFunction> add = [lib newFunctionWithName:@"stitch_add"];
	id<MTLFunction> mul = [lib newFunctionWithName:@"stitch_mul"];
	CHECK([(id)add isKindOfClass:[RgfxFunction class]] && [(RgfxFunction *)add rgfxShadow] == nil);

	MTLFunctionStitchingInputNode *a = [[MTLFunctionStitchingInputNode alloc] initWithArgumentIndex:0];
	MTLFunctionStitchingInputNode *b = [[MTLFunctionStitchingInputNode alloc] initWithArgumentIndex:1];
	MTLFunctionStitchingInputNode *c = [[MTLFunctionStitchingInputNode alloc] initWithArgumentIndex:2];
	MTLFunctionStitchingFunctionNode *sum =
	    [[MTLFunctionStitchingFunctionNode alloc] initWithName:@"stitch_add"
	                                                 arguments:@[ a, b ]
	                                       controlDependencies:@[]];
	MTLFunctionStitchingFunctionNode *product =
	    [[MTLFunctionStitchingFunctionNode alloc] initWithName:@"stitch_mul"
	                                                 arguments:@[ sum, c ]
	                                       controlDependencies:@[]];
	MTLStitchedLibraryDescriptor *d = [[MTLStitchedLibraryDescriptor alloc] init];
	d.functionGraphs = @[ [[MTLFunctionStitchingGraph alloc] initWithFunctionName:@"fused"
	                                                                        nodes:@[ sum ]
	                                                                   outputNode:product
	                                                                   attributes:@[]] ];
	d.functions = @[ add, mul ];
	d.binaryArchives = @[ [metal newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new] error:nil] ];

	__block MTLStitchedLibraryDescriptor *given = nil;
	__block id<MTLLibrary> stitched = nil;
	NSError *err = nil;
	id<MTLLibrary> ours = RgfxStitchOverShadows(
	    d, nil,
	    ^id<MTLLibrary>(MTLStitchedLibraryDescriptor *shadowed, NSError **e) {
		    given = shadowed;
		    stitched = [metal newLibraryWithStitchedDescriptor:shadowed error:e];
		    return stitched;
	    },
	    ^id<MTLLibrary>(dispatch_data_t data, NSError **e) {
		    const void *p = nullptr;
		    size_t n = 0;
		    dispatch_data_t map = dispatch_data_create_map(data, &p, &n);
		    (void)map;
		    return make(p, n);
	    },
	    &err);
	CHECK(err == nil && stitched != nil);
	CHECK(given != d && given.binaryArchives.count == 0 && d.binaryArchives.count == 1);
	CHECK(given.functions.count == 2 && given.functions[0] == [(RgfxFunction *)add rgfxShadow] &&
	      given.functions[1] == [(RgfxFunction *)mul rgfxShadow]);
	CHECK(![(id)given.functions[0] isKindOfClass:[RgfxFunction class]]);
	CHECK(d.functions[0] == add); // the caller's descriptor is unchanged
	CHECK([(id)ours isKindOfClass:[RgfxLibrary class]]);
	CHECK([[ours functionNames] isEqualToArray:@[ @"fused" ]]);
	CHECK([(RgfxLibrary *)ours rgfxShadow] == stitched);
	CHECK([ours newFunctionWithName:@"fused"].functionType == MTLFunctionTypeVisible);
	std::vector<uint8_t> drain;
	RgfxMsgHeader h;
	while (fx.ring.next(&h, &drain)) {
	}
}

// A stitched library of a DAG (newLibraryWithDAG:functions:error:, SkyLight's and MPS's) is
// Metal's own over the functions' shadows too: its text is Metal's to read, and its answer, here
// Metal's parser's error, is the guest's, with nothing sent to the host.
void test_stitched_dag_over_shadows()
{
	Fixture fx;
	const char *path = getenv("RGFX_STITCHABLE");
	CHECK(path != nullptr);
	if (path == nullptr)
		return;
	NSData *bytes = [NSData dataWithContentsOfFile:@(path)];
	id<MTLDevice> metal = MTLCreateSystemDefaultDevice();
	uint32_t status = 0;
	RgfxLibrary *lib = [RgfxLibrary libraryWithData:bytes.bytes
	                                         length:bytes.length
	                                            key:nullptr
	                                         device:nil
	                                     serializer:&fx.ser
	                                         status:&status];
	[lib rgfxSetSource:dispatch_data_create(bytes.bytes, bytes.length, nil, ^{
		     (void)bytes;
	     })
	       shadowMaker:^id<MTLLibrary>(dispatch_data_t d) {
		       return [metal newLibraryWithData:d error:nil];
	       }];
	id<MTLFunction> add = [lib newFunctionWithName:@"stitch_add"];
	id<MTLFunction> mul = [lib newFunctionWithName:@"stitch_mul"];
	std::vector<uint8_t> drain;
	RgfxMsgHeader h;
	while (fx.ring.next(&h, &drain)) {
	}
	__block NSArray<id<MTLFunction>> *given = nil;
	__block NSError *metals = nil;
	NSError *err = nil;
	id<MTLLibrary> ours = RgfxLinkOverShadows(
	    @[ add, mul ],
	    ^id<MTLLibrary>(NSArray<id<MTLFunction>> *shadows, NSError **e) {
		    given = shadows;
		    id<MTLLibrary> made = [(IOGPUMetalDevice *)metal newLibraryWithDAG:@"fused" functions:shadows error:e];
		    metals = *e;
		    return made;
	    },
	    ^id<MTLLibrary>(dispatch_data_t, NSError **) {
		    CHECK(false); // nothing to make of no library
		    return nil;
	    },
	    &err);
	CHECK(ours == nil && metals != nil && err == metals);
	CHECK([err.domain isEqualToString:MTLLibraryErrorDomain] && err.code == MTLLibraryErrorCompileFailure);
	CHECK([err.localizedDescription hasPrefix:@"error while parsing script: "]);
	CHECK(given.count == 2 && given[0] == [(RgfxFunction *)add rgfxShadow] &&
	      given[1] == [(RgfxFunction *)mul rgfxShadow] && given[0] != nil);
	CHECK(!fx.ring.next(&h, &drain));
}

// A runtime source compile (RgfxCompileOverMetal) over Metal's own frontend, as the device's
// newLibraryWithSource: runs it: _MTLDevice's compiler-task form, on this Mac's device. The
// library is ours, made over the bytes Metal's public compile of the same source makes, its
// shadow Metal's library, with no error; a source Metal refuses answers nil and Metal's own error,
// with the same text as the public compile's, and reaches neither the maker nor the ring.
void test_source_compiles_through_metals_frontend()
{
	Fixture fx;
	RgfxSerializer *ser = &fx.ser;
	id<MTLDevice> metal = MTLCreateSystemDefaultDevice();
	SEL sel = @selector(newLibraryWithSource:options:compilerTask:error:);
	Class base = objc_getClass("_MTLDevice");
	CHECK(class_getInstanceMethod(base, sel) != nullptr);
	if (class_getInstanceMethod(base, sel) == nullptr)
		return;
	IMP frontend = class_getMethodImplementation(base, sel);
	MTLCompileOptions *options = [MTLCompileOptions new];
	__block id<MTLLibrary> metals = nil;
	__block NSError *given = nil;
	RgfxCompileBlock (^compileOf)(NSString *) = ^RgfxCompileBlock(NSString *source) {
		return ^id<MTLLibrary>(NSError **e) {
			metals = ((id(*)(id, SEL, NSString *, MTLCompileOptions *, id, NSError **))frontend)(
			    metal, sel, source, options, nil, e);
			given = *e;
			return metals;
		};
	};
	__block NSData *made = nil;
	RgfxLibraryMaker makeLibrary = ^id<MTLLibrary>(dispatch_data_t data, NSError **) {
		const void *bytes = nullptr;
		size_t size = 0;
		dispatch_data_t map = dispatch_data_create_map(data, &bytes, &size);
		made = [NSData dataWithBytes:bytes length:size];
		uint32_t status = 0;
		RgfxLibrary *lib = [RgfxLibrary libraryWithData:bytes
		                                         length:size
		                                            key:nullptr
		                                         device:nil
		                                     serializer:ser
		                                         status:&status];
		(void)map;
		return lib;
	};

	NSString *source = @"#include <metal_stdlib>\n"
	                   @"using namespace metal;\n"
	                   @"kernel void doubled(device float *o [[buffer(0)]],\n"
	                   @"                    uint i [[thread_position_in_grid]]) { o[i] = float(i) * 2.0f; }\n";
	NSError *err = nil;
	id<MTLLibrary> ours = RgfxCompileOverMetal(compileOf(source), makeLibrary, &err);
	CHECK([(id)ours isKindOfClass:[RgfxLibrary class]] && err == nil && given == nil);
	CHECK(metals != nil && [(RgfxLibrary *)ours rgfxShadow] == metals);
	NSError *publicError = nil;
	id<MTLLibrary> pub = [metal newLibraryWithSource:source options:options error:&publicError];
	NSData *publicBytes = [(id<RgfxMetalLibraryInternals>)pub libraryDataContents];
	CHECK(pub != nil && publicError == nil);
	CHECK(made.length > 0 && [made isEqualToData:publicBytes]);
	auto bytes = nextMessage(fx.ring, RGFX_CMD_LIBRARY_CREATE);
	RgfxLibraryCreate head;
	std::memcpy(&head, bytes.data(), sizeof(head));
	CHECK(head.obj_id == [(RgfxLibrary *)ours rgfxObjId] && head.byte_length == made.length);
	CHECK(std::memcmp(bytes.data() + RGFX_LIBRARY_BYTES_OFFSET, made.bytes, made.length) == 0);

	NSString *bad = @"#include <metal_stdlib>\n"
	                @"kernel void broken(device float *o [[buffer(0)]]) { o[0] = undeclared; }\n";
	err = nil;
	id<MTLLibrary> refused = RgfxCompileOverMetal(
	    compileOf(bad),
	    ^id<MTLLibrary>(dispatch_data_t, NSError **) {
		    CHECK(false); // nothing to make of no library
		    return nil;
	    },
	    &err);
	CHECK(refused == nil && metals == nil && given != nil && err == given);
	CHECK([err.domain isEqualToString:MTLLibraryErrorDomain] && err.code == MTLLibraryErrorCompileFailure);
	publicError = nil;
	CHECK([metal newLibraryWithSource:bad options:options error:&publicError] == nil);
	CHECK([err.localizedDescription isEqualToString:publicError.localizedDescription]);
	RgfxMsgHeader h = {};
	std::vector<uint8_t> drain;
	CHECK(!fx.ring.next(&h, &drain));
	CHECK(ours != nil); // kept until here: its release would post
}

} // namespace

// A device that stitches as ours does (RgfxStitchOverShadows), on Metal's own stitch, making its
// libraries with `make`: what an archive adds a stitched library through.
@interface StitchingDevice : NSObject <RgfxStitchingDevice>
@property(nonatomic, strong) id<MTLDevice> metal;
@property(nonatomic, copy) RgfxLibraryMaker make;
@property(nonatomic) int stitches;
@end

@implementation StitchingDevice
- (id<MTLLibrary>)newLibraryWithStitchedDescriptor:(MTLStitchedLibraryDescriptor *)descriptor
                          destinationBinaryArchive:(id<MTLBinaryArchive>)archive
                                             error:(NSError **)error
{
	return RgfxStitchOverShadows(
	    descriptor, archive,
	    ^id<MTLLibrary>(MTLStitchedLibraryDescriptor *shadowed, NSError **e) {
		    self.stitches++;
		    return [self.metal newLibraryWithStitchedDescriptor:shadowed error:e];
	    },
	    self.make, error);
}
@end

namespace {

// A library keeps each entry point's arguments as its AIR module declares them, read once from the
// bytes it was made from (RgfxAirArguments.hpp, over the .metallib built from
// tests/air_arguments.metal), and its functions and their pipelines carry them; a library that
// kept no bytes has none to give, and a render pipeline's stage with no function binds nothing.
void test_library_air_arguments()
{
	Fixture fx(0x40000); // the library's bytes fit
	const char *path = getenv("RGFX_AIR_ARGUMENTS");
	CHECK(path != nullptr);
	if (path == nullptr)
		return;
	NSData *bytes = [NSData dataWithContentsOfFile:@(path)];
	CHECK(bytes.length > 0);
	uint32_t status = 0;
	RgfxLibrary *lib = [RgfxLibrary libraryWithData:bytes.bytes
	                                         length:bytes.length
	                                            key:nullptr
	                                         device:nil
	                                     serializer:&fx.ser
	                                         status:&status];
	CHECK(lib != nil);
	[lib rgfxSetSource:dispatch_data_create(bytes.bytes, bytes.length, nil, ^{
		     (void)bytes;
	     })
	       shadowMaker:^id<MTLLibrary>(dispatch_data_t) {
		       return nil;
	       }];
	RgfxFunction *kernel = (RgfxFunction *)[lib newFunctionWithName:@"every_access"];
	std::shared_ptr<const RgfxAirFunction> air = [kernel rgfxAirFunction];
	CHECK(air != nullptr && air->name == "every_access" && air->stage == RgfxAirStage::Kernel &&
	      air->arguments.size() == 11);
	CHECK(air != nullptr && rgfxAirWritesTexture(*air, 3) && !rgfxAirWritesTexture(*air, 1));
	RgfxFunction *again = (RgfxFunction *)[lib newFunctionWithName:@"every_access"];
	CHECK([again rgfxAirFunction] == air); // read once
	RgfxFunction *vertex = (RgfxFunction *)[lib newFunctionWithName:@"reads_in_vertex"];
	RgfxFunction *fragment = (RgfxFunction *)[lib newFunctionWithName:@"writes_in_fragment"];
	CHECK([vertex rgfxAirFunction] != nullptr &&
	      [vertex rgfxAirFunction]->stage == RgfxAirStage::Vertex);
	CHECK([fragment rgfxAirFunction] != nullptr &&
	      rgfxAirWritesTexture(*[fragment rgfxAirFunction], 5));

	// every_access declares a constant: its pipeline is of the function specialised by a value.
	MTLFunctionConstantValues *values = [[MTLFunctionConstantValues alloc] init];
	const bool scale = true;
	[values setConstantValue:&scale type:MTLDataTypeBool atIndex:0];
	NSError *err = nil;
	RgfxFunction *specialised = (RgfxFunction *)[lib newFunctionWithName:@"every_access"
	                                                      constantValues:values
	                                                               error:&err];
	CHECK(specialised != nil && [specialised rgfxAirFunction] == air);
	RgfxComputePipelineState *cs =
	    [RgfxComputePipelineState computePipelineStateWithFunction:specialised
	                                                       device:nil
	                                                   serializer:&fx.ser
	                                                        error:&err];
	CHECK(cs != nil && [cs rgfxAirFunction] == air);
	MTLRenderPipelineDescriptor *desc = [[MTLRenderPipelineDescriptor alloc] init];
	desc.vertexFunction = vertex;
	desc.fragmentFunction = fragment;
	desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
	RgfxRenderPipelineState *rs = [RgfxRenderPipelineState renderPipelineStateWithDescriptor:desc
	                                                                                 device:nil
	                                                                             serializer:&fx.ser
	                                                                                  error:&err];
	CHECK(rs != nil && [rs rgfxVertexAirFunction] == [vertex rgfxAirFunction] &&
	      [rs rgfxFragmentAirFunction] == [fragment rgfxAirFunction]);
	desc.fragmentFunction = nil;
	RgfxRenderPipelineState *vertexOnly =
	    [RgfxRenderPipelineState renderPipelineStateWithDescriptor:desc
	                                                       device:nil
	                                                   serializer:&fx.ser
	                                                        error:&err];
	CHECK(vertexOnly != nil && [vertexOnly rgfxFragmentAirFunction] != nullptr &&
	      [vertexOnly rgfxFragmentAirFunction]->arguments.empty());

	// Made from a universal .metallib's AIR slice, the whole file kept: read from the slice.
	std::vector<uint8_t> fat(8 + 20 + 16, 0);
	const uint32_t fatHeader[7] = {0xCAFEBABE, 1, 0x01000017, 0, static_cast<uint32_t>(fat.size()),
	                               static_cast<uint32_t>(bytes.length), 4};
	for (int i = 0; i < 7; i++)
		for (int k = 0; k < 4; k++)
			fat[4 * i + k] = static_cast<uint8_t>(fatHeader[i] >> (24 - 8 * k));
	fat.insert(fat.end(), static_cast<const uint8_t *>(bytes.bytes),
	           static_cast<const uint8_t *>(bytes.bytes) + bytes.length);
	NSData *fatBytes = [NSData dataWithBytes:fat.data() length:fat.size()];
	drainRing(fx.ring);
	RgfxLibrary *sliced = [RgfxLibrary libraryWithData:bytes.bytes
	                                            length:bytes.length
	                                               key:nullptr
	                                            device:nil
	                                        serializer:&fx.ser
	                                            status:&status];
	[sliced rgfxSetSource:dispatch_data_create(fatBytes.bytes, fatBytes.length, nil, ^{
		        (void)fatBytes;
	        })
	          shadowMaker:^id<MTLLibrary>(dispatch_data_t) {
		          return nil;
	          }];
	CHECK(sliced != nil);
	std::shared_ptr<const RgfxAirFunction> fromSlice =
	    [(RgfxFunction *)[sliced newFunctionWithName:@"writes_in_fragment"] rgfxAirFunction];
	CHECK(fromSlice != nullptr && rgfxAirWritesTexture(*fromSlice, 5));

	// No bytes kept: the arguments are unknown.
	drainRing(fx.ring);
	RgfxLibrary *bare = [RgfxLibrary libraryWithData:bytes.bytes
	                                          length:bytes.length
	                                             key:nullptr
	                                          device:nil
	                                      serializer:&fx.ser
	                                          status:&status];
	CHECK(bare != nil);
	CHECK([(RgfxFunction *)[bare newFunctionWithName:@"every_access"] rgfxAirFunction] == nullptr);
	drainRing(fx.ring);

	// A function whose module does not read (its function table's kind spoilt), in two libraries
	// of the same bytes: unknown in each, its line logged once a process.
	std::vector<uint8_t> spoilt(static_cast<const uint8_t *>(bytes.bytes),
	                            static_cast<const uint8_t *>(bytes.bytes) + bytes.length);
	const char kind[] = "air.visible_function_table";
	auto it = std::search(spoilt.begin(), spoilt.end(), kind, kind + sizeof(kind) - 1);
	CHECK(it != spoilt.end());
	if (it == spoilt.end())
		return;
	*(it + sizeof(kind) - 2) = '_';
	NSData *spoiltBytes = [NSData dataWithBytes:spoilt.data() length:spoilt.size()];
	for (int i = 0; i < 2; i++) {
		RgfxLibrary *again = [RgfxLibrary libraryWithData:spoiltBytes.bytes
		                                           length:spoiltBytes.length
		                                              key:nullptr
		                                           device:nil
		                                       serializer:&fx.ser
		                                           status:&status];
		[again rgfxSetSource:dispatch_data_create(spoiltBytes.bytes, spoiltBytes.length, nil, ^{
			       (void)spoiltBytes;
		       })
		         shadowMaker:^id<MTLLibrary>(dispatch_data_t) {
			         return nil;
		         }];
		CHECK(again != nil);
		CHECK([(RgfxFunction *)[again newFunctionWithName:@"calls_through_table"] rgfxAirFunction] ==
		      nullptr);
		CHECK([(RgfxFunction *)[again newFunctionWithName:@"conditional_planes"] rgfxAirFunction] !=
		      nullptr);
		drainRing(fx.ring);
	}
	CHECK(!rgfxAirUnreadFirstLog("calls_through_table"));
	CHECK(rgfxAirUnreadFirstLog("no_function_of_this_name") &&
	      !rgfxAirUnreadFirstLog("no_function_of_this_name"));
}

// A stitched library kept in a binary archive is found as Metal's archive finds one (measured:
// macOS 27, M3 Max): a create that fails on a miss is made where an archive of the descriptor
// holds the same graphs over the same functions, in any order and of any library over the same
// bytes, and otherwise answers Metal's error with nothing stitched; another graph name, swapped
// arguments or another attribute miss. The archive holds it in no key and writes it to no file. An
// archive adds one by stitching it on its device into itself, answering Metal's error for a graph
// Metal cannot stitch, and for a descriptor that fails on a miss and that no archive holds.
void test_stitched_libraries_in_archives()
{
	Fixture fx;
	const char *path = getenv("RGFX_STITCHABLE");
	CHECK(path != nullptr);
	if (path == nullptr)
		return;
	NSData *bytes = [NSData dataWithContentsOfFile:@(path)];
	id<MTLDevice> metal = MTLCreateSystemDefaultDevice();
	auto make = [&](const void *p, size_t n) {
		uint32_t status = 0;
		return [RgfxLibrary libraryWithData:p
		                             length:n
		                                key:nullptr
		                             device:nil
		                         serializer:&fx.ser
		                             status:&status];
	};
	auto drain = [&] {
		std::vector<uint8_t> b;
		RgfxMsgHeader h;
		while (fx.ring.next(&h, &b)) {
		}
	};
	auto source = [&](RgfxLibrary *lib) {
		[lib rgfxSetSource:dispatch_data_create(bytes.bytes, bytes.length, nil, ^{
			     (void)bytes;
		     })
		       shadowMaker:^id<MTLLibrary>(dispatch_data_t d) {
			       return [metal newLibraryWithData:d error:nil];
		       }];
		return lib;
	};
	RgfxLibrary *lib = source(make(bytes.bytes, bytes.length));
	RgfxLibrary *again = source(make(bytes.bytes, bytes.length));
	drain();
	auto graph = [](NSString *name, bool swapped, NSArray *attributes) {
		MTLFunctionStitchingInputNode *a = [[MTLFunctionStitchingInputNode alloc] initWithArgumentIndex:0];
		MTLFunctionStitchingInputNode *b = [[MTLFunctionStitchingInputNode alloc] initWithArgumentIndex:1];
		MTLFunctionStitchingInputNode *c = [[MTLFunctionStitchingInputNode alloc] initWithArgumentIndex:2];
		MTLFunctionStitchingFunctionNode *sum =
		    [[MTLFunctionStitchingFunctionNode alloc] initWithName:@"stitch_add"
		                                                 arguments:@[ a, b ]
		                                       controlDependencies:@[]];
		NSArray *last = swapped ? @[ c, sum ] : @[ sum, c ];
		MTLFunctionStitchingFunctionNode *product =
		    [[MTLFunctionStitchingFunctionNode alloc] initWithName:@"stitch_mul"
		                                                 arguments:last
		                                       controlDependencies:@[]];
		return [[MTLFunctionStitchingGraph alloc] initWithFunctionName:name
		                                                         nodes:@[ sum ]
		                                                    outputNode:product
		                                                    attributes:attributes];
	};
	auto descriptor = [&](RgfxLibrary *from, MTLFunctionStitchingGraph *g, NSArray *archives, bool reversed) {
		MTLStitchedLibraryDescriptor *d = [[MTLStitchedLibraryDescriptor alloc] init];
		id<MTLFunction> add = [from newFunctionWithName:@"stitch_add"];
		id<MTLFunction> mul = [from newFunctionWithName:@"stitch_mul"];
		d.functions = reversed ? @[ mul, add ] : @[ add, mul ];
		d.functionGraphs = @[ g ];
		d.binaryArchives = archives;
		d.options = MTLStitchedLibraryOptionFailOnBinaryArchiveMiss;
		return d;
	};
	StitchingDevice *device = [StitchingDevice new];
	device.metal = metal;
	device.make = ^id<MTLLibrary>(dispatch_data_t data, NSError **e) {
		const void *p = nullptr;
		size_t n = 0;
		dispatch_data_t map = dispatch_data_create_map(data, &p, &n);
		(void)map;
		return make(p, n);
	};
	NSError *err = nil;
	RgfxBinaryArchive *archive = [RgfxBinaryArchive archiveWithDescriptor:[MTLBinaryArchiveDescriptor new]
	                                                                device:(id<MTLDevice>)device
	                                                          architecture:@"arch_a"
	                                                                 error:&err];
	CHECK(archive != nil);
	auto create = [&](MTLStitchedLibraryDescriptor *d, NSError **e) {
		return [device newLibraryWithStitchedDescriptor:d destinationBinaryArchive:nil error:e];
	};
	auto missed = [&](MTLStitchedLibraryDescriptor *d) {
		NSError *e = nil;
		const int before = device.stitches;
		id<MTLLibrary> made = create(d, &e);
		return made == nil && device.stitches == before && [e.domain isEqualToString:MTLLibraryErrorDomain] &&
		       e.code == MTLLibraryErrorCompileFailure &&
		       [e.localizedDescription isEqualToString:@"Failed to find library in binary archive."];
	};
	MTLFunctionStitchingGraph *fused = graph(@"fused", false, @[]);
	CHECK(missed(descriptor(lib, fused, @[], false)));
	CHECK(missed(descriptor(lib, fused, nil, false)));
	CHECK(missed(descriptor(lib, fused, @[ archive ], false)));

	// An add that fails on a miss misses too, and keeps nothing.
	err = nil;
	CHECK(![archive addLibraryWithDescriptor:descriptor(lib, fused, @[ archive ], false) error:&err]);
	CHECK([err.localizedDescription isEqualToString:@"Failed to find library in binary archive."]);
	CHECK(device.stitches == 0 && missed(descriptor(lib, fused, @[ archive ], false)));

	// Added: stitched on the device into the archive, then found.
	MTLStitchedLibraryDescriptor *plain = descriptor(lib, fused, @[], false);
	plain.options = 0;
	CHECK([archive addLibraryWithDescriptor:plain error:&err] && err == nil && device.stitches == 1);
	drain();
	id<MTLLibrary> found = create(descriptor(lib, fused, @[ archive ], false), &err);
	CHECK([(id)found isKindOfClass:[RgfxLibrary class]] && [[found functionNames] isEqualToArray:@[ @"fused" ]]);
	CHECK(device.stitches == 2);
	drain();
	CHECK(create(descriptor(lib, fused, @[ archive ], true), &err) != nil);
	CHECK(create(descriptor(again, fused, @[ archive ], false), &err) != nil);
	drain();
	CHECK(missed(descriptor(lib, graph(@"other", false, @[]), @[ archive ], false)));
	CHECK(missed(descriptor(lib, graph(@"fused", true, @[]), @[ archive ], false)));
	CHECK(missed(descriptor(lib, graph(@"fused", false, @[ [MTLFunctionStitchingAttributeAlwaysInline new] ]),
	                        @[ archive ], false)));
	// Kept in no key and in no file.
	CHECK(archive.keys.count == 0);
	NSURL *url = [NSURL fileURLWithPath:[NSTemporaryDirectory()
	                                        stringByAppendingPathComponent:@"rgfx-stitched-archive.bin"]];
	CHECK(![archive serializeToURL:url error:&err]);
	CHECK([err.localizedDescription isEqualToString:@"Nothing to serialize"]);

	// A graph Metal cannot stitch is not added, with Metal's error.
	MTLFunctionStitchingFunctionNode *missing =
	    [[MTLFunctionStitchingFunctionNode alloc] initWithName:@"nope" arguments:@[] controlDependencies:@[]];
	MTLStitchedLibraryDescriptor *bad = descriptor(lib, fused, @[], false);
	bad.options = 0;
	bad.functionGraphs = @[ [[MTLFunctionStitchingGraph alloc] initWithFunctionName:@"bad"
	                                                                          nodes:@[]
	                                                                     outputNode:missing
	                                                                     attributes:@[]] ];
	err = nil;
	CHECK(![archive addLibraryWithDescriptor:bad error:&err]);
	CHECK([err.domain isEqualToString:MTLLibraryErrorDomain] && err.code == MTLLibraryErrorCompileFailure &&
	      [err.localizedDescription containsString:@"missing function 'nope' required by DAG 'bad'"]);
	// An archive with no device to stitch on adds none.
	RgfxBinaryArchive *alone = [RgfxBinaryArchive archiveWithDescriptor:[MTLBinaryArchiveDescriptor new]
	                                                              device:nil
	                                                        architecture:@"arch_a"
	                                                               error:&err];
	CHECK(![alone addLibraryWithDescriptor:plain error:&err] && err != nil);
	drain();
}

// A visible function answers its reflection from its signature in its AIR module
// (RgfxReflection.h), as Metal's own function on a device with a compiler does, and asks no shadow
// for it: over tests/air_arguments.metal's stitchable functions, each return type and argument
// is equal to Metal's own (its -isEqual:, both ways), of the same classes, and -bindings is
// -arguments' object, made once. A visible function that is not stitchable answers nil and no
// arguments. Where the signature has a type the reader does not know, or an acceleration
// structure, the shadow answers, and the library's other functions still answer their own. tools/reflection_check.mm compares them
// getter by getter on a device.
void test_stitchable_reflection_from_air()
{
	Fixture fx(0x80000); // the library's bytes fit, twice
	const char *path = getenv("RGFX_AIR_ARGUMENTS");
	CHECK(path != nullptr);
	if (path == nullptr)
		return;
	NSData *bytes = [NSData dataWithContentsOfFile:@(path)];
	id<MTLDevice> metal = MTLCreateSystemDefaultDevice();
	auto library = [&](NSData *data) {
		uint32_t status = 0;
		RgfxLibrary *lib = [RgfxLibrary libraryWithData:data.bytes
		                                         length:data.length
		                                            key:nullptr
		                                         device:nil
		                                     serializer:&fx.ser
		                                         status:&status];
		[lib rgfxSetSource:dispatch_data_create(data.bytes, data.length, nil, ^{
			     (void)data;
		     })
		       shadowMaker:^id<MTLLibrary>(dispatch_data_t) {
			       return nil;
		       }];
		return lib;
	};
	RgfxLibrary *lib = library(bytes);
	id<MTLLibrary> genuine = [metal newLibraryWithData:dispatch_data_create(bytes.bytes, bytes.length, nil, ^{
		                                (void)bytes;
	                                }) error:nil];
	CHECK(lib != nil && genuine != nil);
	auto ask = [](id object, NSString *member) {
		return ((id (*)(id, SEL))objc_msgSend)(object, NSSelectorFromString(member));
	};
	auto number = [](id object, NSString *member) {
		return ((NSUInteger(*)(id, SEL))objc_msgSend)(object, NSSelectorFromString(member));
	};
	auto same = [](id ours, id theirs) {
		return ours == nil ? theirs == nil
		                   : theirs != nil && [ours class] == [theirs class] && [ours isEqual:theirs] &&
		                         [theirs isEqual:ours];
	};
	unsigned compared = 0;
	for (NSString *name in genuine.functionNames) {
		id<MTLFunction> theirs = [genuine newFunctionWithName:name];
		if (theirs.functionType != MTLFunctionTypeVisible)
			continue;
		RgfxFunction *fn = (RgfxFunction *)[lib newFunctionWithName:name];
		if ([name isEqualToString:@"sig_acceleration"]) {
			// Metal's own reflection of an acceleration structure ends the process, so it is not
			// asked. Ours is refused, and its shadow answers (none here).
			CHECK(ask(fn, @"arguments") == nil && ask(fn, @"returnType") == nil);
			continue;
		}
		NSArray *arguments = ask(fn, @"arguments"), *metals = ask(theirs, @"arguments");
		CHECK(arguments != nil && metals != nil && arguments.count == metals.count);
		for (NSUInteger i = 0; i < arguments.count && i < metals.count; i++)
			CHECK(same(arguments[i], metals[i]));
		CHECK(same(ask(fn, @"returnType"), ask(theirs, @"returnType")));
		CHECK(ask(fn, @"bindings") == arguments && ask(fn, @"arguments") == arguments);
		CHECK(ask(fn, @"returnType") == ask(fn, @"returnType"));
		CHECK([fn rgfxShadow] == nil); // no shadow made to answer them
		compared++;
	}
	CHECK(compared == 18);

	// Field by field, what Core Image reads among them: the return's data type, and each
	// argument's name, binding type, and type's data type and name.
	RgfxFunction *sigStruct = (RgfxFunction *)[lib newFunctionWithName:@"sig_struct"];
	MTLStructType *pair = ask(sigStruct, @"returnType");
	CHECK([pair isKindOfClass:[MTLStructType class]] && pair.dataType == MTLDataTypeStruct &&
	      [ask(pair, @"typeName") isEqualToString:@"SigPair"] && pair.members.count == 3);
	if (pair.members.count == 3) {
		CHECK([pair.members[0].name isEqualToString:@"xy"] && pair.members[0].offset == 0 &&
		      pair.members[0].dataType == MTLDataTypeFloat2);
		CHECK([pair.members[2].name isEqualToString:@"n"] && pair.members[2].offset == 12 &&
		      pair.members[2].dataType == MTLDataTypeInt);
	}
	NSArray *structArguments = ask(sigStruct, @"arguments");
	CHECK(structArguments.count == 5);
	if (structArguments.count == 5) {
		id o = structArguments[0];
		MTLStructType *outer = ask(o, @"dataTypeDescription");
		CHECK([ask(o, @"name") isEqualToString:@"o"] && number(o, @"type") == 29 &&
		      [ask(outer, @"typeName") isEqualToString:@"SigOuter"]);
		// A struct field's struct type is nameless (measured, as Metal's).
		CHECK(outer.members.count == 3 && outer.members[0].structType != nil &&
		      [ask(outer.members[0].structType, @"typeName") isEqualToString:@""] &&
		      outer.members[0].structType.members.count == 3);
		id i = structArguments[1];
		CHECK([ask(i, @"name") isEqualToString:@"i"] && number(i, @"type") == 18 &&
		      number(ask(i, @"dataTypeDescription"), @"dataType") == MTLDataTypeInt);
	}
	CHECK(number(ask([lib newFunctionWithName:@"sig_void"], @"returnType"), @"dataType") == 61);
	RgfxFunction *doubled = (RgfxFunction *)[lib newFunctionWithName:@"doubled"];
	NSArray *none = ask(doubled, @"arguments");
	CHECK(ask(doubled, @"returnType") == nil && none != nil && none.count == 0 && ask(doubled, @"bindings") == none);

	// A signature with a type the reader does not know: the shadow answers (none here: nil).
	NSMutableData *spoilt = [bytes mutableCopy];
	const char kind[] = "air.sampler_type";
	uint8_t *b = static_cast<uint8_t *>(spoilt.mutableBytes);
	for (uint8_t *at = b; (at = std::search(at, b + spoilt.length, kind, kind + sizeof(kind) - 1)) != b + spoilt.length;
	     at++)
		at[sizeof(kind) - 2] = '_';
	RgfxLibrary *other = library(spoilt);
	RgfxFunction *unknown = (RgfxFunction *)[other newFunctionWithName:@"sig_fields"];
	CHECK(other != nil && unknown != nil);
	CHECK(ask(unknown, @"arguments") == nil && ask(unknown, @"returnType") == nil);
	CHECK(RgfxUnsupportedWasLogged([RgfxFunction class], NSSelectorFromString(@"arguments")));
	CHECK(same(ask([other newFunctionWithName:@"sig_struct"], @"returnType"), ask(sigStruct, @"returnType")));
}

// A function answers Metal's private members from its shadow, Metal's own function of the same
// entry point, made of its library's on first use: a stitchable function's bindings and return
// type, a kernel's nils, and each one's AIR. One made with constant values has no shadow: it
// answers nil for its constants, as Metal's, and nil, logged, for the rest. Its options are what
// it was last given, and its plugin data nil.
void test_function_members_from_shadow()
{
	Fixture fx;
	const char *path = getenv("RGFX_STITCHABLE");
	CHECK(path != nullptr);
	if (path == nullptr)
		return;
	NSData *bytes = [NSData dataWithContentsOfFile:@(path)];
	id<MTLDevice> metal = MTLCreateSystemDefaultDevice();
	uint32_t status = 0;
	RgfxLibrary *lib = [RgfxLibrary libraryWithData:bytes.bytes
	                                        length:bytes.length
	                                           key:nullptr
	                                        device:nil
	                                    serializer:&fx.ser
	                                        status:&status];
	dispatch_data_t source = dispatch_data_create(bytes.bytes, bytes.length, nil, ^{
		(void)bytes;
	});
	[lib rgfxSetSource:source
	       shadowMaker:^id<MTLLibrary>(dispatch_data_t d) {
		       return [metal newLibraryWithData:d error:nil];
	       }];
	id<MTLLibrary> genuine = [metal newLibraryWithData:source error:nil];
	auto ask = [](id object, NSString *member) {
		return ((id (*)(id, SEL))objc_msgSend)(object, NSSelectorFromString(member));
	};
	for (NSString *name in @[ @"stitch_add", @"stitch_caller" ]) {
		RgfxFunction *fn = (RgfxFunction *)[lib newFunctionWithName:name];
		id<MTLFunction> theirs = [genuine newFunctionWithName:name];
		CHECK(fn != nil && theirs != nil && [fn rgfxShadow] == nil);
		NSArray *arguments = ask(fn, @"arguments"), *metals = ask(theirs, @"arguments");
		CHECK((arguments == nil) == (metals == nil) && arguments.count == metals.count);
		for (NSUInteger i = 0; i < arguments.count && i < metals.count; i++)
			CHECK([[arguments[i] name] isEqualToString:[metals[i] name]]);
		id type = ask(fn, @"returnType"), metalType = ask(theirs, @"returnType");
		CHECK((type == nil) == (metalType == nil));
		CHECK(type == nil || [type dataType] == [metalType dataType]);
		CHECK(ask(fn, @"importedLibraries") == nil && ask(theirs, @"importedLibraries") == nil);
		CHECK(ask(fn, @"functionConstants") == nil && ask(theirs, @"functionConstants") == nil);
		id air = ask(fn, @"bitcodeData"), metalAir = ask(theirs, @"bitcodeData");
		CHECK(air != nil && dispatch_data_get_size((dispatch_data_t)air) ==
		                        dispatch_data_get_size((dispatch_data_t)metalAir));
		CHECK([fn rgfxShadow] != nil && ![(id)[fn rgfxShadow] isKindOfClass:[RgfxFunction class]]);
		CHECK(ask(fn, @"pluginData") == nil);
	}
	CHECK(((NSArray *)ask([lib newFunctionWithName:@"stitch_add"], @"arguments")).count == 2);
	CHECK(ask([lib newFunctionWithName:@"stitch_add"], @"returnType") != nil);
	CHECK(ask([lib newFunctionWithName:@"stitch_caller"], @"returnType") == nil);

	RgfxFunctionConstant c = {};
	c.data_type = static_cast<uint32_t>(MTLDataTypeFloat);
	RgfxFunction *valued = [RgfxFunction functionWithLibrary:lib
	                                                    name:@"stitch_caller"
	                                               constants:&c
	                                                   count:1
	                                                  device:nil
	                                              serializer:&fx.ser];
	CHECK(ask(valued, @"functionConstants") == nil && ask(valued, @"bitcodeData") == nil);
	CHECK([valued rgfxShadowOrMake] == nil);
	CHECK(RgfxUnsupportedWasLogged([RgfxFunction class], NSSelectorFromString(@"bitcodeData")));

	SEL setOptions = NSSelectorFromString(@"setOptions:");
	CHECK(valued.options == MTLFunctionOptionNone && [valued respondsToSelector:setOptions]);
	((void (*)(id, SEL, MTLFunctionOptions))objc_msgSend)(valued, setOptions,
	                                                      MTLFunctionOptionCompileToBinary);
	CHECK(valued.options == MTLFunctionOptionCompileToBinary);
	std::vector<uint8_t> drain;
	RgfxMsgHeader h;
	while (fx.ring.next(&h, &drain)) {
	}
}

// A library whose function list does not read (bytes this reader does not follow, which the host
// took) answers from its shadow, Metal's own library over the same bytes: its function names, nil
// for a name the shadow lacks, each function's type and patch, its declared constants, and values
// set by name resolved against them, a value of another type failing as Metal's does. With no
// shadow, its functions still answer the placeholder stage.
void test_unread_function_list_answers_from_shadow()
{
	Fixture fx(0x80000); // the library's bytes fit
	const char *path = getenv("RGFX_AIR_ARGUMENTS");
	CHECK(path != nullptr);
	if (path == nullptr)
		return;
	NSData *bytes = [NSData dataWithContentsOfFile:@(path)];
	id<MTLDevice> metal = MTLCreateSystemDefaultDevice();
	id<MTLLibrary> genuine =
	    [metal newLibraryWithData:dispatch_data_create(bytes.bytes, bytes.length, nil, ^{
		           (void)bytes;
	           })
	                        error:nil];
	CHECK(genuine != nil);
	// Bytes whose function list does not read: the .metallib with its list's offset past its end.
	NSMutableData *unread = [bytes mutableCopy];
	const uint64_t past = unread.length + 1;
	std::memcpy(static_cast<uint8_t *>(unread.mutableBytes) + 0x18, &past, sizeof(past));
	auto make = [&]() {
		uint32_t status = 0;
		return [RgfxLibrary libraryWithData:unread.bytes
		                             length:unread.length
		                                key:nullptr
		                             device:nil
		                         serializer:&fx.ser
		                             status:&status];
	};
	RgfxLibrary *lib = make();
	CHECK(lib != nil);
	[lib rgfxAdoptShadow:genuine];
	NSArray *names = [[lib functionNames] sortedArrayUsingSelector:@selector(compare:)];
	CHECK([names isEqualToArray:[genuine.functionNames sortedArrayUsingSelector:@selector(compare:)]]);
	CHECK([names containsObject:@"no_texture"]);
	CHECK([lib newFunctionWithName:@"no_texture"].functionType == MTLFunctionTypeKernel);
	CHECK([lib newFunctionWithName:@"reads_in_vertex"].functionType == MTLFunctionTypeVertex);
	CHECK([lib newFunctionWithName:@"writes_in_fragment"].functionType == MTLFunctionTypeFragment);
	CHECK([lib newFunctionWithName:@"no_such_function"] == nil);
	NSError *err = nil;
	CHECK([lib newFunctionWithName:@"no_such_function" constantValues:[MTLFunctionConstantValues new]
	                          error:&err] == nil &&
	      err.code == MTLLibraryErrorFunctionNotFound);
	CHECK([lib newFunctionWithName:@"no_texture"].patchControlPointCount == -1);

	// every_access declares kScale (index 0, bool), as Metal's own function does.
	id<MTLFunction> unspecialised = [lib newFunctionWithName:@"every_access"];
	MTLFunctionConstant *declared = unspecialised.functionConstantsDictionary[@"kScale"];
	MTLFunctionConstant *metals =
	    [genuine newFunctionWithName:@"every_access"].functionConstantsDictionary[@"kScale"];
	CHECK(unspecialised.functionConstantsDictionary.count == 1 && declared.index == 0 &&
	      declared.type == MTLDataTypeBool && metals != nil && declared.required == metals.required);
	MTLFunctionConstantValues *byName = [MTLFunctionConstantValues new];
	bool yes = true;
	[byName setConstantValue:&yes type:MTLDataTypeBool withName:@"kScale"];
	err = nil;
	RgfxFunction *fn = (RgfxFunction *)[lib newFunctionWithName:@"every_access"
	                                             constantValues:byName
	                                                      error:&err];
	CHECK(fn != nil && err == nil && [fn rgfxConstantCount] == 1);
	if ([fn rgfxConstantCount] == 1) {
		const RgfxFunctionConstant &c = [fn rgfxConstants][0];
		CHECK(c.index == 0 && c.data_type == MTLDataTypeBool && c.value[0] == 1);
	}
	float wrong = 1.0f;
	MTLFunctionConstantValues *mistyped = [MTLFunctionConstantValues new];
	[mistyped setConstantValue:&wrong type:MTLDataTypeFloat withName:@"kScale"];
	err = nil;
	CHECK([lib newFunctionWithName:@"every_access" constantValues:mistyped error:&err] == nil &&
	      err.code == MTLLibraryErrorCompileFailure);

	// With no shadow there is nothing to ask: a function of any name, of the placeholder stage.
	RgfxLibrary *alone = make();
	CHECK([alone functionNames].count == 0);
	CHECK([alone newFunctionWithName:@"no_texture"].functionType == MTLFunctionTypeVertex);
	std::vector<uint8_t> drain;
	RgfxMsgHeader h;
	while (fx.ring.next(&h, &drain)) {
	}
}

// A library whose function list reads but whose function's metadata names no constants (no CNST
// tag: here every_access's, renamed) takes them from its AIR module, as it takes a CNST tag's: the
// function declares kScale as Metal's own over the tagged bytes does, values set by name and by
// index reach it, a value of another type fails as Metal's does, and a function of no constants
// still declares none.
void test_constants_without_cnst_resolve_from_air()
{
	Fixture fx(0x80000); // the library's bytes fit
	const char *path = getenv("RGFX_AIR_ARGUMENTS");
	CHECK(path != nullptr);
	if (path == nullptr)
		return;
	NSData *bytes = [NSData dataWithContentsOfFile:@(path)];
	NSMutableData *untagged = [bytes mutableCopy];
	// every_access's CNST tag, the one that lists kScale first: a 4-byte name, a u16 size, a u16
	// count, then the first constant's name.
	uint8_t *p = static_cast<uint8_t *>(untagged.mutableBytes);
	size_t renamed = 0;
	for (size_t i = 0; i + 15 <= untagged.length; i++) {
		if (std::memcmp(p + i, "CNST", 4) == 0 && std::memcmp(p + i + 8, "kScale", 7) == 0) {
			std::memcpy(p + i, "XNST", 4);
			renamed++;
		}
	}
	CHECK(renamed == 1);
	uint32_t status = 0;
	RgfxLibrary *lib = [RgfxLibrary libraryWithData:untagged.bytes
	                                         length:untagged.length
	                                            key:nullptr
	                                         device:nil
	                                     serializer:&fx.ser
	                                         status:&status];
	CHECK(lib != nil);
	[lib rgfxSetSource:dispatch_data_create(untagged.bytes, untagged.length, nil, ^{
		     (void)untagged;
	     })
	       shadowMaker:^id<MTLLibrary>(dispatch_data_t) {
		       return nil;
	       }];
	id<MTLDevice> metal = MTLCreateSystemDefaultDevice();
	id<MTLLibrary> genuine =
	    [metal newLibraryWithData:dispatch_data_create(bytes.bytes, bytes.length, nil, ^{
		           (void)bytes;
	           })
	                        error:nil];
	MTLFunctionConstant *metals =
	    [genuine newFunctionWithName:@"every_access"].functionConstantsDictionary[@"kScale"];
	CHECK(metals != nil);

	id<MTLFunction> unspecialised = [lib newFunctionWithName:@"every_access"];
	MTLFunctionConstant *declared = unspecialised.functionConstantsDictionary[@"kScale"];
	CHECK(unspecialised.functionConstantsDictionary.count == 1 && declared.index == 0 &&
	      declared.type == MTLDataTypeBool && declared.required == metals.required);
	MTLFunctionConstantValues *byName = [MTLFunctionConstantValues new];
	bool yes = true;
	[byName setConstantValue:&yes type:MTLDataTypeBool withName:@"kScale"];
	NSError *err = nil;
	id<MTLFunction> named = [lib newFunctionWithName:@"every_access"
	                                  constantValues:byName
	                                           error:&err];
	CHECK(named != nil && err == nil && [(RgfxFunction *)named rgfxConstantCount] == 1);
	const RgfxFunctionConstant *k = constantAt(named, 0);
	CHECK(k != nullptr && k->data_type == MTLDataTypeBool && k->value[0] == 1);
	MTLFunctionConstantValues *byIndex = [MTLFunctionConstantValues new];
	[byIndex setConstantValue:&yes type:MTLDataTypeBool atIndex:0];
	id<MTLFunction> indexed = [lib newFunctionWithName:@"every_access"
	                                    constantValues:byIndex
	                                             error:&err];
	CHECK(indexed != nil && err == nil && [(RgfxFunction *)indexed rgfxConstantCount] == 1);
	k = constantAt(indexed, 0);
	CHECK(k != nullptr && k->data_type == MTLDataTypeBool && k->value[0] == 1);
	float wrong = 1.0f;
	MTLFunctionConstantValues *mistyped = [MTLFunctionConstantValues new];
	[mistyped setConstantValue:&wrong type:MTLDataTypeFloat withName:@"kScale"];
	err = nil;
	CHECK([lib newFunctionWithName:@"every_access" constantValues:mistyped error:&err] == nil &&
	      err.code == MTLLibraryErrorCompileFailure);
	CHECK([lib newFunctionWithName:@"no_texture"].functionConstantsDictionary.count == 0);
	std::vector<uint8_t> drain;
	RgfxMsgHeader h;
	while (fx.ring.next(&h, &drain)) {
	}
}

// A library made from bytes answers nil for its bitcode, as Metal's, and is written by its shadow
// as the bytes it was made from, with Foundation's error for a URL it cannot write; one made from
// a URL is refused with Metal's error, and one with no shadow is refused, logged. (Metal's answers
// measured on macOS 27, M3 Max.)
void test_library_members_as_metal()
{
	Fixture fx;
	const char *path = getenv("RGFX_STITCHABLE");
	CHECK(path != nullptr);
	if (path == nullptr)
		return;
	NSData *bytes = [NSData dataWithContentsOfFile:@(path)];
	id<MTLDevice> metal = MTLCreateSystemDefaultDevice();
	auto make = [&]() {
		uint32_t status = 0;
		RgfxLibrary *lib = [RgfxLibrary libraryWithData:bytes.bytes
		                                        length:bytes.length
		                                           key:nullptr
		                                        device:nil
		                                    serializer:&fx.ser
		                                        status:&status];
		[lib rgfxSetSource:dispatch_data_create(bytes.bytes, bytes.length, nil, ^{
			     (void)bytes;
		     })
		       shadowMaker:^id<MTLLibrary>(dispatch_data_t d) {
			       return [metal newLibraryWithData:d error:nil];
		       }];
		return lib;
	};
	NSString *dir = [NSTemporaryDirectory()
	    stringByAppendingPathComponent:[NSString stringWithFormat:@"rgfx-library-tests-%d", getpid()]];
	[[NSFileManager defaultManager] createDirectoryAtPath:dir
	                          withIntermediateDirectories:YES
	                                           attributes:nil
	                                                error:nil];
	NSURL *out = [NSURL fileURLWithPath:[dir stringByAppendingPathComponent:@"out.metallib"]];

	RgfxLibrary *lib = make();
	CHECK([lib bitcodeData] == nil);
	NSError *err = nil;
	CHECK([lib serializeToURL:out error:&err] && err == nil);
	CHECK([[NSData dataWithContentsOfURL:out] isEqualToData:bytes]);
	CHECK([lib bitcodeData] == nil); // its shadow's, now made, is nil too
	CHECK([lib serializeToURL:out error:&err]); // over the file it wrote
	NSURL *missing = [NSURL fileURLWithPath:[dir stringByAppendingPathComponent:@"no/out.metallib"]];
	CHECK(![lib serializeToURL:missing error:&err]);
	CHECK([err.domain isEqualToString:NSCocoaErrorDomain] && err.code == NSFileNoSuchFileError);
	err = nil;
	CHECK(![lib serializeToURL:[NSURL URLWithString:@"https://example.invalid/out.metallib"]
	                     error:&err]);
	CHECK([err.domain isEqualToString:NSCocoaErrorDomain] &&
	      err.code == NSFileWriteUnsupportedSchemeError);

	RgfxLibrary *fromURL = make();
	[fromURL rgfxSetMadeFromURL:out];
	err = nil;
	CHECK(![fromURL serializeToURL:out error:&err]);
	CHECK([err.domain isEqualToString:MTLLibraryErrorDomain] && err.code == MTLLibraryErrorUnsupported &&
	      [err.localizedDescription isEqualToString:@"MTLLibrary is not formatted as a MetalLib file."]);
	CHECK([fromURL bitcodeData] == nil);

	std::vector<uint8_t> drain;
	RgfxMsgHeader h;
	while (fx.ring.next(&h, &drain)) {
	}
	uint32_t status = 0;
	RgfxLibrary *bare = [RgfxLibrary libraryWithData:bytes.bytes
	                                         length:bytes.length
	                                            key:nullptr
	                                         device:nil
	                                     serializer:&fx.ser
	                                         status:&status];
	CHECK(bare != nil);
	err = nil;
	CHECK(![bare serializeToURL:out error:&err] && [err.domain isEqualToString:MTLLibraryErrorDomain]);
	CHECK(RgfxUnsupportedWasLogged([RgfxLibrary class], @selector(serializeToURL:error:)));
	[[NSFileManager defaultManager] removeItemAtPath:dir error:nil];
	while (fx.ring.next(&h, &drain)) {
	}
}

// A library answers its type and install name as Metal does on the x86 Macs (oracle_bundle libtype
// and functypes): an executable library's bytes Executable and, as its install name, the last part
// of the URL it was made from, or default.metallib; a dynamic library's bytes, whose header's file
// type is 2, Dynamic and the install name they hold, which its shadow answers, nil with none.
void test_library_type_and_install_name()
{
	Fixture fx(0x40000); // the five libraries' bytes fit
	const char *path = getenv("RGFX_STITCHABLE");
	CHECK(path != nullptr);
	if (path == nullptr)
		return;
	id<MTLDevice> metal = MTLCreateSystemDefaultDevice();
	auto make = [&](NSData *bytes, bool shadowed) {
		uint32_t status = 0;
		RgfxLibrary *lib = [RgfxLibrary libraryWithData:bytes.bytes
		                                        length:bytes.length
		                                           key:nullptr
		                                        device:nil
		                                    serializer:&fx.ser
		                                        status:&status];
		if (shadowed)
			[lib rgfxSetSource:dispatch_data_create(bytes.bytes, bytes.length, nil, ^{
				     (void)bytes;
			     })
			       shadowMaker:^id<MTLLibrary>(dispatch_data_t d) {
				       return [metal newLibraryWithData:d error:nil];
			       }];
		return lib;
	};
	NSData *executable = [NSData dataWithContentsOfFile:@(path)];
	RgfxLibrary *fromData = make(executable, true);
	CHECK(fromData.type == MTLLibraryTypeExecutable);
	CHECK([fromData.installName isEqualToString:@"default.metallib"]);
	RgfxLibrary *fromURL = make(executable, false);
	[fromURL rgfxSetMadeFromURL:[NSURL fileURLWithPath:@"/Library/Shaders/stitchable.metallib"]];
	CHECK(fromURL.type == MTLLibraryTypeExecutable);
	CHECK([fromURL.installName isEqualToString:@"stitchable.metallib"]);

	// A dynamic library's bytes, as Metal's compiler writes them.
	MTLCompileOptions *o = [MTLCompileOptions new];
	o.libraryType = MTLLibraryTypeDynamic;
	o.installName = @"@executable_path/librgfx_dyn.metallib";
	NSError *err = nil;
	id<MTLLibrary> compiled =
	    [metal newLibraryWithSource:@"float rgfx_dyn(float x) { return x * 3.0f; }"
	                        options:o
	                          error:&err];
	CHECK(compiled != nil);
	SEL contents = sel_registerName("libraryDataContents");
	CHECK([(id)compiled respondsToSelector:contents]);
	NSData *dynamic = compiled != nil && [(id)compiled respondsToSelector:contents]
	                      ? ((id (*)(id, SEL))objc_msgSend)(compiled, contents)
	                      : nil;
	CHECK(dynamic.length > 0x0a && static_cast<const uint8_t *>(dynamic.bytes)[0x0a] == 2);
	if (dynamic.length > 0x0a) {
		RgfxLibrary *dyn = make(dynamic, true);
		CHECK(dyn.type == MTLLibraryTypeDynamic);
		CHECK([dyn.installName isEqualToString:@"@executable_path/librgfx_dyn.metallib"]);
		RgfxLibrary *dynURL = make(dynamic, true);
		[dynURL rgfxSetMadeFromURL:[NSURL fileURLWithPath:@"/tmp/librgfx_dyn.metallib"]];
		CHECK([dynURL.installName isEqualToString:@"@executable_path/librgfx_dyn.metallib"]);
		RgfxLibrary *bare = make(dynamic, false);
		CHECK(bare.type == MTLLibraryTypeDynamic && bare.installName == nil);
	}
	// A function of its exported function ends the process, with Metal's words, as on both x86
	// Macs (oracle_bundle functypes.dyn).
	checkDies("dynamic-exported-function", "validateMTLFunctionType: failed assertion `type is "
	                                       "not a valid MTLFunctionType.'");
	std::vector<uint8_t> drain;
	RgfxMsgHeader h;
	while (fx.ring.next(&h, &drain)) {
	}
}

static std::vector<uint8_t> exportedFunctionMetallib()
{
	return metallibOf({ { "rgfx_exported", 3, {} } });
}

// A post-tessellation vertex function answers its [[patch]]'s type and control point count, from
// the library's function list, and every other function MTLPatchTypeNone and -1, as Metal's own
// functions answer for the same bytes (RgfxLibrary.mm, rgfxPatchOf).
void test_function_patch_answers()
{
	Fixture fx(0x20000); // the library's bytes fit
	const char *path = getenv("RGFX_PATCH");
	CHECK(path != nullptr);
	if (path == nullptr)
		return;
	NSData *bytes = [NSData dataWithContentsOfFile:@(path)];
	CHECK(bytes.length > 0);
	uint32_t status = 0;
	RgfxLibrary *ours = [RgfxLibrary libraryWithData:bytes.bytes
	                                          length:bytes.length
	                                             key:nullptr
	                                          device:nil
	                                      serializer:&fx.ser
	                                          status:&status];
	CHECK(ours != nil);
	dispatch_data_t data = dispatch_data_create(bytes.bytes, bytes.length, nil, ^{
		(void)bytes;
	});
	id<MTLLibrary> theirs = [MTLCreateSystemDefaultDevice() newLibraryWithData:data error:nil];
	CHECK(theirs != nil);
	const struct {
		const char *name;
		MTLPatchType type;
		NSInteger count;
	} want[] = {
		{"patch_quad4", MTLPatchTypeQuad, 4},
		{"patch_quad_any", MTLPatchTypeQuad, -1},
		{"patch_quad32", MTLPatchTypeQuad, 32},
		{"patch_triangle_any", MTLPatchTypeTriangle, -1},
		{"patch_triangle3", MTLPatchTypeTriangle, 3},
		{"patch_quad_no_points", MTLPatchTypeQuad, -1},
		{"plain_vertex", MTLPatchTypeNone, -1},
		{"plain_fragment", MTLPatchTypeNone, -1},
		{"plain_kernel", MTLPatchTypeNone, -1},
	};
	for (const auto &w : want) {
		NSString *name = @(w.name);
		id<MTLFunction> f = [ours newFunctionWithName:name];
		CHECK(f != nil);
		CHECK(f.patchType == w.type);
		CHECK(f.patchControlPointCount == w.count);
		id<MTLFunction> g = [theirs newFunctionWithName:name];
		CHECK(g != nil && g.patchType == f.patchType &&
		      g.patchControlPointCount == f.patchControlPointCount);
	}
	// Each function's stage is Metal's own, those of the mesh, object and intersection stages
	// among them (the .metallib's TYPE tag bytes 7, 8 and 6).
	CHECK([[ours.functionNames sortedArrayUsingSelector:@selector(compare:)]
	    isEqualToArray:[theirs.functionNames sortedArrayUsingSelector:@selector(compare:)]]);
	for (NSString *name in theirs.functionNames) {
		const MTLFunctionType type = [ours newFunctionWithName:name].functionType;
		if (type != [theirs newFunctionWithName:name].functionType)
			NSLog(@"FAIL detail: %@ answers functionType %lu", name, (unsigned long)type);
		CHECK(type == [theirs newFunctionWithName:name].functionType);
	}
	CHECK([ours newFunctionWithName:@"plain_mesh"].functionType == MTLFunctionTypeMesh);
	CHECK([ours newFunctionWithName:@"plain_object"].functionType == MTLFunctionTypeObject);
	CHECK([ours newFunctionWithName:@"plain_intersection"].functionType ==
	      MTLFunctionTypeIntersection);
	// A function made apart from the library's table states no patch.
	RgfxFunction *bare = [RgfxFunction functionWithLibrary:ours
	                                                  name:@"patch_quad4"
	                                             constants:nullptr
	                                                 count:0
	                                                device:nil
	                                            serializer:&fx.ser];
	CHECK(bare.patchType == MTLPatchTypeNone && bare.patchControlPointCount == -1);
	// A kernel made to fail on an archive miss, from an archive that does not hold it: nil, with
	// Metal's error (RgfxBinaryArchive's rgfxMissOfFunction); made once the archive holds it.
	RgfxBinaryArchive *archive =
	    [RgfxBinaryArchive archiveWithDescriptor:[MTLBinaryArchiveDescriptor new]
	                                      device:nil
	                                architecture:@"arch_a"
	                                       error:nil];
	MTLFunctionDescriptor *fd = [MTLFunctionDescriptor functionDescriptor];
	fd.name = @"plain_kernel";
	fd.options = MTLFunctionOptionFailOnBinaryArchiveMiss;
	fd.binaryArchives = @[ archive ];
	NSError *err = nil;
	CHECK([ours newFunctionWithDescriptor:fd error:&err] == nil &&
	      err.code == MTLLibraryErrorFunctionNotFound);
	MTLFunctionDescriptor *added = [MTLFunctionDescriptor functionDescriptor];
	added.name = @"plain_kernel";
	CHECK([archive addFunctionWithDescriptor:added library:ours error:&err]);
	err = nil;
	CHECK([ours newFunctionWithDescriptor:fd error:&err] != nil && err == nil);
	std::vector<uint8_t> drain;
	RgfxMsgHeader h;
	while (fx.ring.next(&h, &drain)) {
	}
}

// A Core Image kernel library's extern functions come from its shadow, Metal's own library over
// the same bytes: their names, each a function of functionType 4 that holds Metal's own, and nil
// for a name the shadow does not hold; functionNames leaves them out, as Metal's does. Core
// Image's link hands Metal each function's shadow, and makes the library Metal's link answers
// ours from the bytes Metal answers for it, keeping Metal's as its shadow.
void test_core_image_extern_functions()
{
	Fixture fx;
	const char *path = getenv("RGFX_CI_KERNEL");
	CHECK(path != nullptr);
	if (path == nullptr)
		return;
	NSData *bytes = [NSData dataWithContentsOfFile:@(path)];
	CHECK(bytes.length > 0);
	id<MTLDevice> metal = MTLCreateSystemDefaultDevice();
	auto make = [&](NSData *from) {
		uint32_t status = 0;
		return [RgfxLibrary libraryWithData:from.bytes
		                             length:from.length
		                                key:nullptr
		                             device:nil
		                         serializer:&fx.ser
		                             status:&status];
	};
	RgfxLibrary *lib = make(bytes);
	CHECK(lib != nil);

	// With no source kept, no shadow and no extern functions.
	CHECK([lib rgfxShadow] == nil);
	CHECK([lib externFunctionNames].count == 0);
	CHECK([lib newExternFunctionWithName:@"swap_red_blue"] == nil);

	// The shadow is made once, on first use, from the source.
	dispatch_data_t source = dispatch_data_create(bytes.bytes, bytes.length, nil, ^{
		(void)bytes;
	});
	__block int made = 0;
	[lib rgfxSetSource:source
	       shadowMaker:^id<MTLLibrary>(dispatch_data_t d) {
		       made++;
		       return [metal newLibraryWithData:d error:nil];
	       }];
	id<MTLLibrary> shadow = [lib rgfxShadow];
	CHECK(shadow != nil && [lib rgfxShadow] == shadow && made == 1);
	id<RgfxMetalLibraryInternals> internals = (id<RgfxMetalLibraryInternals>)shadow;
	NSArray<NSString *> *names = [internals externFunctionNames];
	CHECK(names.count == 2);
	CHECK([[lib externFunctionNames] isEqualToArray:names]);
	CHECK([shadow functionNames].count == 0 && [lib functionNames].count == 0);

	RgfxFunction *swap = (RgfxFunction *)[lib newExternFunctionWithName:@"swap_red_blue"];
	CHECK([swap isKindOfClass:[RgfxFunction class]]);
	CHECK(swap.functionType == kRgfxFunctionTypeExtern);
	CHECK([swap.name isEqualToString:@"swap_red_blue"]);
	CHECK([swap rgfxShadow].functionType == kRgfxFunctionTypeExtern);
	CHECK([[swap rgfxShadow].name isEqualToString:@"swap_red_blue"]);
	CHECK([lib newExternFunctionWithName:@"nope"] == nil);
	// Its shadow answers what Core Image reflects of the kernel (its return type); a function by
	// name answers it from a shadow made of its library's on first use.
	SEL returnType = NSSelectorFromString(@"returnType");
	CHECK([[swap rgfxShadow] respondsToSelector:returnType] && [swap respondsToSelector:returnType]);
	id answered = ((id (*)(id, SEL))objc_msgSend)(swap, returnType);
	id metals = ((id (*)(id, SEL))objc_msgSend)([swap rgfxShadow], returnType);
	CHECK(answered != nil && [[answered description] isEqualToString:[metals description]]);
	RgfxFunction *swapByName = (RgfxFunction *)[lib newFunctionWithName:@"swap_red_blue"];
	CHECK([swapByName rgfxShadow] == nil);
	id swapType = ((id (*)(id, SEL))objc_msgSend)(swapByName, returnType);
	CHECK([swapByName rgfxShadow] != nil &&
	      [[swapType description] isEqualToString:[metals description]]);
	// A kernel by name is of functionType 4 too, as Metal's.
	id<MTLFunction> byName = [lib newFunctionWithName:@"scale_red_green"];
	CHECK(byName.functionType == kRgfxFunctionTypeExtern);
	CHECK([shadow newFunctionWithName:@"scale_red_green"].functionType == kRgfxFunctionTypeExtern);

	// The link's functions: ours as their shadows (one made by name given one of its library's
	// shadow), Metal's own as they are.
	id<MTLFunction> genuine = [internals newExternFunctionWithName:@"scale_red_green"];
	NSError *err = nil;
	NSArray<id<MTLFunction>> *shadows = RgfxShadowFunctions(@[ swap, byName, genuine ], &err);
	CHECK(shadows.count == 3 && err == nil);
	CHECK(shadows[0] == [swap rgfxShadow] && shadows[2] == genuine);
	CHECK(shadows[1] == [(RgfxFunction *)byName rgfxShadow] &&
	      ![(id)shadows[1] isKindOfClass:[RgfxFunction class]]);
	// A function whose library keeps no source has no shadow, and the link is refused.
	RgfxLibrary *bare = make(bytes);
	CHECK(RgfxShadowFunctions(@[ [bare newFunctionWithName:@"swap_red_blue"] ], &err) == nil);
	CHECK(err.code == RgfxMetalErrorInvalidDescriptor);
	// Nor any answer of Metal's own: nil, logged.
	CHECK(((id (*)(id, SEL))objc_msgSend)([bare newFunctionWithName:@"swap_red_blue"], returnType) ==
	      nil);
	CHECK(RgfxUnsupportedWasLogged([RgfxFunction class], returnType));

	// Metal's link answers its own library: made ours from its bytes, Metal's kept as the shadow.
	__block NSArray<id<MTLFunction>> *linkedFrom = nil;
	__block dispatch_data_t madeFrom = nil;
	RgfxLibraryMaker maker = ^id<MTLLibrary>(dispatch_data_t data, NSError **e) {
		madeFrom = data;
		const void *p = nullptr;
		size_t n = 0;
		dispatch_data_t map = dispatch_data_create_map(data, &p, &n);
		(void)map;
		return make([NSData dataWithBytes:p length:n]);
	};
	err = nil;
	id<MTLLibrary> ours = RgfxLinkOverShadows(
	    @[ swap ],
	    ^id<MTLLibrary>(NSArray<id<MTLFunction>> *given, NSError **e) {
		    linkedFrom = given;
		    return shadow;
	    },
	    maker, &err);
	CHECK(linkedFrom.count == 1 && linkedFrom[0] == [swap rgfxShadow]);
	CHECK([(id)ours isKindOfClass:[RgfxLibrary class]] && ours != lib);
	CHECK([(RgfxLibrary *)ours rgfxShadow] == shadow);
	NSData *contents = [internals libraryDataContents];
	CHECK(contents.length > 0 && madeFrom != nil &&
	      dispatch_data_get_size(madeFrom) == contents.length);
	CHECK([[(RgfxLibrary *)ours externFunctionNames] isEqualToArray:names]);
	id ourBitcode = [(RgfxLibrary *)ours bitcodeData];
	id metalBitcode = ((id (*)(id, SEL))objc_msgSend)(shadow, @selector(bitcodeData));
	CHECK((ourBitcode == nil) == (metalBitcode == nil));
	// A library the link answers that is ours already is answered as it is, and a refused link
	// answers nil with its error.
	madeFrom = nil;
	CHECK(RgfxLinkOverShadows(
	          @[ swap ], ^id<MTLLibrary>(NSArray<id<MTLFunction>> *given, NSError **e) { return lib; },
	          maker, &err) == lib);
	CHECK(madeFrom == nil);
	err = nil;
	CHECK(RgfxLinkOverShadows(
	          @[ swap ],
	          ^id<MTLLibrary>(NSArray<id<MTLFunction>> *given, NSError **e) {
		          *e = RgfxMakeError(RgfxMetalErrorRefused, @"refused");
		          return nil;
	          },
	          maker, &err) == nil);
	CHECK(err.code == RgfxMetalErrorRefused);
	std::vector<uint8_t> drain;
	RgfxMsgHeader h;
	while (fx.ring.next(&h, &drain)) {
	}
}

// A pipeline links a function of another library made with constant values under a specialized
// name, as RenderBox's custom effects link theirs to the `custom_fn` their fragment calls (Metal
// refuses one linked under its own name, "unresolved visible function reference: custom_fn";
// measured: macOS 27): the create carries the fragment's constants, and for the linked function
// its library, list, specialized name and constant values, all the host makes it of.
void test_linked_function_specialised_on_the_wire()
{
	Fixture fx;
	auto make = [&](const std::vector<uint8_t> &bytes) {
		uint32_t status = 0;
		RgfxLibrary *lib = [RgfxLibrary libraryWithData:bytes.data()
		                                        length:bytes.size()
		                                           key:nullptr
		                                        device:nil
		                                    serializer:&fx.ser
		                                        status:&status];
		CHECK(lib != nil);
		(void)nextMessage(fx.ring, RGFX_CMD_LIBRARY_CREATE);
		return lib;
	};
	RgfxLibrary *effects = make(metallibOf({
	    { "effect_vertex", 0, {} },
	    { "effect_fragment", 1, { { "state", 36, 0, true } } },
	}));
	// A visible function's entry (TYPE 4, as Apple's compiler writes one).
	RgfxLibrary *gradients = make(metallibOf({ { "gradient_v1", 4, { { "variant", 33, 1, true } } } }));
	static const uint32_t state[4] = { 0x465, 0x82, 0x1145e, 0xc };
	static const uint32_t variant = 7;
	MTLFunctionConstantValues *fragmentValues = [MTLFunctionConstantValues new];
	[fragmentValues setConstantValue:state type:MTLDataTypeUInt4 atIndex:0];
	NSError *err = nil;
	id<MTLFunction> fragment = [effects newFunctionWithName:@"effect_fragment"
	                                         constantValues:fragmentValues
	                                                  error:&err];
	MTLFunctionDescriptor *fd = [MTLFunctionDescriptor functionDescriptor];
	fd.name = @"gradient_v1";
	fd.specializedName = @"custom_fn";
	fd.constantValues = [MTLFunctionConstantValues new];
	[fd.constantValues setConstantValue:&variant type:MTLDataTypeUInt withName:@"variant"];
	id<MTLFunction> linked = [gradients newFunctionWithDescriptor:fd error:&err];
	CHECK(fragment != nil && linked != nil && err == nil);
	CHECK([linked.name isEqualToString:@"custom_fn"]);

	MTLRenderPipelineDescriptor *desc = [MTLRenderPipelineDescriptor new];
	desc.vertexFunction = [effects newFunctionWithName:@"effect_vertex"];
	desc.fragmentFunction = fragment;
	desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
	desc.fragmentLinkedFunctions.functions = @[ linked ];
	RgfxRenderPipelineState *ps = [RgfxRenderPipelineState renderPipelineStateWithDescriptor:desc
	                                                                                 device:nil
	                                                                             serializer:&fx.ser
	                                                                                  error:&err];
	CHECK(ps != nil && err == nil);
	auto bytes = nextMessage(fx.ring, RGFX_CMD_RENDER_PIPELINE_CREATE);
	RgfxRenderPipelineCreate head;
	std::memcpy(&head, bytes.data(), sizeof(head));
	CHECK(head.fragment_library_obj_id == effects.rgfxObjId && head.fragment_constant_count == 1);
	RgfxFunctionConstant fc;
	std::memcpy(&fc, bytes.data() + head.fragment_constants_offset, sizeof(fc));
	CHECK(fc.index == 0 && fc.data_type == static_cast<uint32_t>(MTLDataTypeUInt4));
	CHECK(std::memcmp(fc.value, state, sizeof(state)) == 0);
	CHECK(head.vertex_linked_count == 0 && head.fragment_linked_count == 1);
	RgfxLinkedFunction r;
	std::memcpy(&r, bytes.data() + head.fragment_linked_offset, sizeof(r));
	CHECK(r.library_obj_id == gradients.rgfxObjId && r.library_obj_id != effects.rgfxObjId);
	CHECK(r.list == RGFX_LINKED_FUNCTIONS && r.options == 0 && r.group_len == 0);
	CHECK(r.name_len == 11 && std::memcmp(bytes.data() + r.name_offset, "gradient_v1", 11) == 0);
	CHECK(r.specialized_name_len == 9 &&
	      std::memcmp(bytes.data() + r.specialized_name_offset, "custom_fn", 9) == 0);
	CHECK(r.constant_count == 1);
	RgfxFunctionConstant lc;
	std::memcpy(&lc, bytes.data() + r.constants_offset, sizeof(lc));
	CHECK(lc.index == 1 && lc.data_type == static_cast<uint32_t>(MTLDataTypeUInt));
	CHECK(std::memcmp(lc.value, &variant, sizeof(variant)) == 0);
}

// An asynchronous creator returns before its handler is done, and runs the handler on the global
// concurrent queue of default QoS, never on the caller's thread, as Metal's do on the M3 Max
// (macOS 27.0.1, `oracle_cmdbuf Apple timing`: `com.apple.root.default-qos`; the 5300M and the UHD
// 630 run theirs on a serial queue of the device's, docs/graphics-kext/dgpu-oracle.md, Q7). So a
// handler that waits for the creator to return goes on, and so does one that runs a block
// synchronously on the caller's own serial queue, which would deadlock (or end the process, in
// libdispatch's check) were it called on that queue. Every creation ends, its block released,
// before the library goes.
void test_async_creators_return_before_their_handlers()
{
	Fixture fx;
	const std::vector<uint8_t> bytes = metallibOf({ { "frg", 1, { { "useRed", 53, 0, true } } } });
	uint32_t status = 0;
	RgfxLibrary *lib = [RgfxLibrary libraryWithData:bytes.data()
	                                        length:bytes.size()
	                                           key:nullptr
	                                        device:nil
	                                    serializer:&fx.ser
	                                        status:&status];
	CHECK(lib != nil);
	(void)nextMessage(fx.ring, RGFX_CMD_LIBRARY_CREATE);
	// A value of another type than its constant's: the create fails with no message posted.
	static const float five = 5;
	MTLFunctionConstantValues *wrong = [MTLFunctionConstantValues new];
	[wrong setConstantValue:&five type:MTLDataTypeFloat atIndex:0];

	static char kCaller;
	dispatch_queue_t caller = dispatch_queue_create("rgfx.test.caller", DISPATCH_QUEUE_SERIAL);
	dispatch_queue_set_specific(caller, &kCaller, &kCaller, nullptr);
	dispatch_semaphore_t returned = dispatch_semaphore_create(0);
	dispatch_semaphore_t handled = dispatch_semaphore_create(0);
	dispatch_time_t (^after)(int64_t) = ^(int64_t seconds) {
		return dispatch_time(DISPATCH_TIME_NOW, seconds * static_cast<int64_t>(NSEC_PER_SEC));
	};
	__block BOOL afterReturn = NO, onCaller = YES, synced = NO;
	__block NSString *queue = nil;
	__block NSError *handed = nil;
	dispatch_async(caller, ^{
		[lib newFunctionWithName:@"frg"
		          constantValues:wrong
		       completionHandler:^(id<MTLFunction> f, NSError *e) {
			       (void)f;
			       handed = e;
			       afterReturn = dispatch_semaphore_wait(returned, after(5)) == 0;
			       onCaller = dispatch_get_specific(&kCaller) != nullptr;
			       queue = @(dispatch_queue_get_label(DISPATCH_CURRENT_QUEUE_LABEL));
			       if (!onCaller)
				       dispatch_sync(caller, ^{
					       synced = YES;
				       });
			       dispatch_semaphore_signal(handled);
		       }];
		dispatch_semaphore_signal(returned);
	});
	CHECK(dispatch_semaphore_wait(handled, after(10)) == 0);
	CHECK(afterReturn && !onCaller && synced);
	CHECK([queue isEqualToString:@"com.apple.root.default-qos"]);
	CHECK(handed.code == MTLLibraryErrorCompileFailure);
	CHECK(RgfxWaitForCreations(after(10)));
	RgfxMsgHeader h = {};
	std::vector<uint8_t> posted;
	CHECK(!fx.ring.next(&h, &posted));
}

// A pending asynchronous create keeps what it uses: a library released right after the call
// lives until its create ends, then goes, and a descriptor changed after the call is not what
// the create reads, as Metal copies it at the call. The handler waits for the release, so a
// create that kept no reference finds the library gone; the change races the create's start,
// which comes later in practice (about 0.15 ms on the global queue), so a create that copied
// nothing reads the changed name and fails as a function not found.
void test_async_creates_keep_what_they_use()
{
	Fixture fx;
	const std::vector<uint8_t> bytes = metallibOf({ { "frg", 1, { { "useRed", 53, 0, true } } } });
	dispatch_time_t (^after)(int64_t) = ^(int64_t seconds) {
		return dispatch_time(DISPATCH_TIME_NOW, seconds * static_cast<int64_t>(NSEC_PER_SEC));
	};
	dispatch_semaphore_t released = dispatch_semaphore_create(0);
	__block BOOL alive = NO;
	__block NSError *handed = nil;
	__weak RgfxLibrary *weak = nil;
	@autoreleasepool {
		uint32_t status = 0;
		RgfxLibrary *lib = [RgfxLibrary libraryWithData:bytes.data()
		                                        length:bytes.size()
		                                           key:nullptr
		                                        device:nil
		                                    serializer:&fx.ser
		                                        status:&status];
		CHECK(lib != nil);
		(void)nextMessage(fx.ring, RGFX_CMD_LIBRARY_CREATE);
		weak = lib;
		// A value of another type than its constant's: the create fails, with no message posted.
		static const float five = 5;
		MTLFunctionConstantValues *wrong = [MTLFunctionConstantValues new];
		[wrong setConstantValue:&five type:MTLDataTypeFloat atIndex:0];
		MTLFunctionDescriptor *fd = [MTLFunctionDescriptor functionDescriptor];
		fd.name = @"frg";
		fd.constantValues = wrong;
		[lib newFunctionWithDescriptor:fd
		             completionHandler:^(id<MTLFunction> f, NSError *e) {
			             (void)f;
			             dispatch_semaphore_wait(released, after(5));
			             alive = weak != nil;
			             handed = e;
		             }];
		// Changed after the call: a function the library does not hold would fail otherwise.
		fd.name = @"nosuch";
		fd.constantValues = nil;
		lib = nil;
	}
	dispatch_semaphore_signal(released);
	CHECK(RgfxWaitForCreations(after(10)));
	CHECK(alive);
	CHECK(handed.code == MTLLibraryErrorCompileFailure);
	CHECK(weak == nil);
}

// A library's functions are specialised as Metal specialises them: a value set by name or by
// range reaches its constant's index, an indexed value wins over a named one, values for
// constants the function does not declare are ignored, a required constant left unset reads 0
// and an optional one is not defined, and a value of another type than its constant's fails the
// function with Metal's error. A function made with no values answers its constants and builds
// no pipeline. (Metal's answers measured on macOS 27.0.1, M3 Max.)
void test_function_constants_resolve_as_metal()
{
	Fixture fx;
	const std::vector<uint8_t> bytes = metallibOf({
	    { "frg",
	      1,
	      { { "useRed", 53, 0, true }, { "mode", 29, 7, true }, { "opt", 53, 9, false },
	        { "tint", 6, 11, true } } },
	    { "knl", 2, { { "scale", 3, 3, true }, { "mode", 29, 7, true } } },
	    { "plain", 1, {} },
	});
	uint32_t status = 0;
	RgfxLibrary *lib = [RgfxLibrary libraryWithData:bytes.data()
	                                        length:bytes.size()
	                                           key:nullptr
	                                        device:nil
	                                    serializer:&fx.ser
	                                        status:&status];
	CHECK(lib != nil);
	(void)nextMessage(fx.ring, RGFX_CMD_LIBRARY_CREATE);
	static const bool yes = true;
	static const int32_t one = 1, two = 2, three = 3;
	static const float four[4] = { 1, 2, 3, 4 }, five = 5, nine = 9;

	// By name: each value reaches its constant's index, with its bytes; `opt` is not defined.
	MTLFunctionConstantValues *named = [MTLFunctionConstantValues new];
	[named setConstantValue:&yes type:MTLDataTypeBool withName:@"useRed"];
	[named setConstantValue:&two type:MTLDataTypeInt withName:@"mode"];
	[named setConstantValue:four type:MTLDataTypeFloat4 withName:@"tint"];
	NSError *err = nil;
	id<MTLFunction> frg = [lib newFunctionWithName:@"frg" constantValues:named error:&err];
	CHECK(frg != nil && err == nil);
	CHECK([(RgfxFunction *)frg rgfxConstantCount] == 3);
	const RgfxFunctionConstant *k = constantAt(frg, 0);
	CHECK(k != nullptr && k->data_type == 53 && k->value[0] == 1);
	k = constantAt(frg, 7);
	CHECK(k != nullptr && k->data_type == 29 && std::memcmp(k->value, &two, 4) == 0);
	k = constantAt(frg, 11);
	CHECK(k != nullptr && k->data_type == 6 && std::memcmp(k->value, four, 16) == 0);
	CHECK(constantAt(frg, 9) == nullptr);
	CHECK(frg.functionConstantsDictionary.count == 0); // specialised: none, as Metal answers
	CHECK(![(RgfxFunction *)frg rgfxUnspecialised]);

	// A required constant left unset reads 0; the optional one stays undefined.
	MTLFunctionConstantValues *some = [MTLFunctionConstantValues new];
	[some setConstantValue:&two type:MTLDataTypeInt atIndex:7];
	id<MTLFunction> partial = [lib newFunctionWithName:@"frg" constantValues:some error:&err];
	CHECK(partial != nil && err == nil);
	CHECK([(RgfxFunction *)partial rgfxConstantCount] == 3);
	static const uint8_t zeros[RGFX_FUNCTION_CONSTANT_VALUE_BYTES] = {};
	k = constantAt(partial, 0);
	CHECK(k != nullptr && k->data_type == 53 && std::memcmp(k->value, zeros, sizeof(zeros)) == 0);
	k = constantAt(partial, 11);
	CHECK(k != nullptr && k->data_type == 6 && std::memcmp(k->value, zeros, sizeof(zeros)) == 0);
	CHECK(constantAt(partial, 9) == nullptr);
	// nil values specialise with none set.
	MTLFunctionConstantValues *nilValues = nil; // through a variable: the parameter is nonnull
	id<MTLFunction> none = [lib newFunctionWithName:@"frg" constantValues:nilValues error:&err];
	CHECK(none != nil && err == nil && [(RgfxFunction *)none rgfxConstantCount] == 3);

	// By range, an index over a name (whose type is then not checked), and values for
	// constants the function does not declare, ignored.
	MTLFunctionConstantValues *mixed = [MTLFunctionConstantValues new];
	[mixed setConstantValues:&five type:MTLDataTypeFloat withRange:NSMakeRange(3, 1)];
	[mixed setConstantValue:&nine type:MTLDataTypeFloat withName:@"mode"];
	[mixed setConstantValue:&one type:MTLDataTypeInt atIndex:7];
	[mixed setConstantValue:&three type:MTLDataTypeInt withName:@"nosuch"];
	[mixed setConstantValue:four type:MTLDataTypeBool4 atIndex:30];
	id<MTLFunction> knl = [lib newFunctionWithName:@"knl" constantValues:mixed error:&err];
	CHECK(knl != nil && err == nil);
	CHECK([(RgfxFunction *)knl rgfxConstantCount] == 2);
	k = constantAt(knl, 3);
	CHECK(k != nullptr && k->data_type == 3 && std::memcmp(k->value, &five, 4) == 0);
	k = constantAt(knl, 7);
	CHECK(k != nullptr && k->data_type == 29 && std::memcmp(k->value, &one, 4) == 0);

	// A value of another type than its constant's: nil, and Metal's error.
	MTLFunctionConstantValues *wrong = [MTLFunctionConstantValues new];
	[wrong setConstantValue:&five type:MTLDataTypeFloat atIndex:0];
	CHECK([lib newFunctionWithName:@"frg" constantValues:wrong error:&err] == nil);
	CHECK(err != nil && [err.domain isEqualToString:MTLLibraryErrorDomain] &&
	      err.code == MTLLibraryErrorCompileFailure);
	CHECK([err.localizedDescription
	    isEqualToString:@"Function frg: constant useRed (0) is of type MTLDataTypeBool but value "
	                    @"found has type MTLDataTypeFloat"]);
	MTLFunctionConstantValues *wrongName = [MTLFunctionConstantValues new];
	[wrongName setConstantValue:&two type:MTLDataTypeUInt withName:@"mode"];
	err = nil;
	CHECK([lib newFunctionWithName:@"knl" constantValues:wrongName error:&err] == nil);
	CHECK([err.localizedDescription
	    isEqualToString:@"Function knl: constant mode (7) is of type MTLDataTypeInt but value "
	                    @"found has type MTLDataTypeUInt"]);
	static const float span[5] = { 1, 2, 3, 4, 5 };
	MTLFunctionConstantValues *wideRange = [MTLFunctionConstantValues new];
	[wideRange setConstantValues:span type:MTLDataTypeFloat withRange:NSMakeRange(3, 5)];
	err = nil;
	CHECK([lib newFunctionWithName:@"knl" constantValues:wideRange error:&err] == nil);
	CHECK(err.code == MTLLibraryErrorCompileFailure);
	// The completion-handler and descriptor forms answer alike.
	__block NSError *handed = nil;
	__block id<MTLFunction> handedFn = nil;
	[lib newFunctionWithName:@"frg"
	          constantValues:wrong
	       completionHandler:^(id<MTLFunction> f, NSError *e) {
		       handedFn = f;
		       handed = e;
	       }];
	CHECK(RgfxWaitForCreations(dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC)));
	CHECK(handedFn == nil && handed.code == MTLLibraryErrorCompileFailure);
	MTLFunctionDescriptor *fd = [MTLFunctionDescriptor functionDescriptor];
	fd.name = @"frg";
	fd.constantValues = named;
	id<MTLFunction> described = [lib newFunctionWithDescriptor:fd error:&err];
	CHECK(described != nil && [(RgfxFunction *)described rgfxConstantCount] == 3);
	CHECK(constantAt(described, 7) != nullptr);
	fd.constantValues = nil;
	described = [lib newFunctionWithDescriptor:fd error:&err];
	CHECK(described != nil && ![(RgfxFunction *)described rgfxUnspecialised]);

	// A name the library does not hold: Metal's error.
	err = nil;
	CHECK([lib newFunctionWithName:@"nosuch" constantValues:named error:&err] == nil);
	CHECK([err.domain isEqualToString:MTLLibraryErrorDomain] &&
	      err.code == MTLLibraryErrorFunctionNotFound);
	CHECK([err.localizedDescription isEqualToString:@"Function nosuch was not found in the library"]);

	// Made with no values: the function answers its constants, and builds no pipeline.
	id<MTLFunction> unspecialised = [lib newFunctionWithName:@"frg"];
	CHECK([(RgfxFunction *)unspecialised rgfxUnspecialised]);
	NSDictionary<NSString *, MTLFunctionConstant *> *table = unspecialised.functionConstantsDictionary;
	CHECK(table.count == 4);
	CHECK(table[@"opt"].index == 9 && table[@"opt"].type == MTLDataTypeBool && !table[@"opt"].required);
	CHECK(table[@"tint"].index == 11 && table[@"tint"].type == MTLDataTypeFloat4 && table[@"tint"].required);
	id<MTLFunction> plain = [lib newFunctionWithName:@"plain"];
	CHECK(![(RgfxFunction *)plain rgfxUnspecialised] && plain.functionConstantsDictionary.count == 0);
	MTLRenderPipelineDescriptor *rd = [[MTLRenderPipelineDescriptor alloc] init];
	rd.vertexFunction = knl;
	rd.fragmentFunction = unspecialised;
	rd.colorAttachments[0].pixelFormat = MTLPixelFormatR16Float;
	err = nil;
	CHECK([RgfxRenderPipelineState renderPipelineStateWithDescriptor:rd
	                                                          device:nil
	                                                      serializer:&fx.ser
	                                                           error:&err] == nil);
	CHECK([err.localizedDescription hasPrefix:@"fragmentFunction frg cannot be used to build a pipeline state."]);
	err = nil;
	CHECK([RgfxComputePipelineState computePipelineStateWithFunction:[lib newFunctionWithName:@"knl"]
	                                                          device:nil
	                                                      serializer:&fx.ser
	                                                           error:&err] == nil);
	CHECK([err.localizedDescription hasPrefix:@"function knl cannot be used to build a pipeline state."]);
	RgfxMsgHeader h = {};
	std::vector<uint8_t> posted;
	CHECK(!fx.ring.next(&h, &posted)); // neither refusal posted a create

	// A specialised pipeline carries its functions' records in its create's tail.
	rd.fragmentFunction = frg;
	RgfxRenderPipelineState *ps = [RgfxRenderPipelineState renderPipelineStateWithDescriptor:rd
	                                                                                  device:nil
	                                                                              serializer:&fx.ser
	                                                                                   error:&err];
	CHECK(ps != nil);
	auto create = nextMessage(fx.ring, RGFX_CMD_RENDER_PIPELINE_CREATE);
	RgfxRenderPipelineCreate head;
	std::memcpy(&head, create.data(), sizeof(head));
	CHECK(head.vertex_constant_count == 2 && head.fragment_constant_count == 3);
	RgfxFunctionConstant tint;
	std::memcpy(&tint, create.data() + head.fragment_constants_offset + 2 * sizeof(tint), sizeof(tint));
	CHECK(tint.index == 11 && tint.data_type == 6 && std::memcmp(tint.value, four, 16) == 0);

	// A metallib built for macOS 26, whose entries' sizes do not count themselves: its constants
	// are read alike, by name and against their types.
	const std::vector<uint8_t> older = metallibOf(
	    {
	        { "knl", 2, { { "scale", 3, 3, true }, { "mode", 29, 7, true } } },
	        { "plain", 1, {} },
	    },
	    false);
	RgfxLibrary *lib26 = [RgfxLibrary libraryWithData:older.data()
	                                           length:older.size()
	                                              key:nullptr
	                                           device:nil
	                                       serializer:&fx.ser
	                                           status:&status];
	CHECK(lib26 != nil);
	(void)nextMessage(fx.ring, RGFX_CMD_LIBRARY_CREATE);
	MTLFunctionConstantValues *byName = [MTLFunctionConstantValues new];
	[byName setConstantValue:&five type:MTLDataTypeFloat withName:@"scale"];
	err = nil;
	id<MTLFunction> knl26 = [lib26 newFunctionWithName:@"knl" constantValues:byName error:&err];
	CHECK(knl26 != nil && [(RgfxFunction *)knl26 rgfxConstantCount] == 2);
	CHECK(constantAt(knl26, 3) != nullptr && std::memcmp(constantAt(knl26, 3)->value, &five, 4) == 0);
	CHECK(constantAt(knl26, 7) != nullptr && constantAt(knl26, 7)->value[0] == 0);
	[byName setConstantValue:&yes type:MTLDataTypeBool withName:@"mode"];
	err = nil;
	CHECK([lib26 newFunctionWithName:@"knl" constantValues:byName error:&err] == nil);
	CHECK(err.code == MTLLibraryErrorCompileFailure);
	CHECK([lib26 newFunctionWithName:@"plain"].functionConstantsDictionary.count == 0);
}

static TestHandle *handle(uint64_t objId)
{
	return [[TestHandle alloc] initWithObjId:objId device:nil serializer:nullptr];
}

// The host-side op-stream walker, the same minimal consumer as the serialiser check
// (serializer_tests.cpp) and crates/.../decode/cmdbuf.rs.
struct OpWalk {
	const uint8_t *ops;
	size_t size;
	size_t pos = 0;
	OpWalk(const uint8_t *p, size_t n) : ops(p), size(n) {}
	bool done() const { return pos >= size; }
	bool next(uint16_t *tag, uint32_t *len, const uint8_t **body, size_t *body_len)
	{
		if (size - pos < sizeof(RgfxOpHeader))
			return false;
		RgfxOpHeader h;
		std::memcpy(&h, ops + pos, sizeof(h));
		if (h.len < RGFX_OP_BODY_OFFSET || (h.len % 8) != 0 || h.len > size - pos)
			return false;
		*tag = h.tag;
		*len = h.len;
		*body = ops + pos + RGFX_OP_BODY_OFFSET;
		*body_len = h.len - RGFX_OP_BODY_OFFSET;
		pos += h.len;
		return true;
	}
};

static uint16_t opGroup(uint16_t tag) { return static_cast<uint16_t>(tag >> 12); }

// The queue/command-buffer/encoder object graph routes the whole Metal call stream
// through RgfxEncoder onto the queue's submission ring. This drives a command buffer
// through each encoder type and asserts the exact op stream the host decoder expects —
// the routing-assertion pattern of this slice.
void test_command_graph()
{
	// Control ring for the serialiser (QUEUE_CREATE), and a separate submission ring
	// the queue's command buffers post on (COMMAND_BUFFER).
	// Never freed, as Fixture's: the queue and its buffers may go after the test.
	TestRing &control = *new TestRing(0x8000);
	control.setConsumerIdle(1);
	RgfxRingWriter &controlWriter = *new RgfxRingWriter(
	    control.base(), control.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell, nullptr);
	RgfxIdSpace &ids = *new RgfxIdSpace();
	RgfxSerializer &ser = *new RgfxSerializer(controlWriter, ids);

	TestRing submit(0x8000);
	submit.setConsumerIdle(1);
	RgfxRingWriter submitWriter(submit.base(), submit.dataSize, /*ring_id*/ 1, TestRing::doorbell,
	                            nullptr);

	// The queue: its create routes a QUEUE_CREATE on the control ring and picks objId 1.
	RgfxCommandQueue *queue = [RgfxCommandQueue queueWithSubmitRing:&submitWriter
	                                                  submitOffset:0
	                                                completeOffset:0
	                                              completeDataSize:0
	                                                           ids:&ids
	                                                      maxCount:0
	                                                        device:nil
	                                                    serializer:&ser];
	CHECK(queue != nil);
	CHECK(queue.rgfxObjId == 1);
	CHECK([queue conformsToProtocol:@protocol(MTLCommandQueue)]);
	auto qc = expectQueueCreate(control);
	CHECK(qc.obj_id == 1);
	CHECK(qc.submit_data_size == submit.dataSize);

	// A command buffer: the next objId after the queue's eight barrier fences (10), bound to
	// the queue, not yet enqueued.
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	CHECK(cb != nil);
	CHECK([cb conformsToProtocol:@protocol(MTLCommandBuffer)]);
	CHECK(((RgfxResource *)cb).rgfxObjId == 10);
	CHECK(cb.commandQueue == queue);
	CHECK(cb.retainedReferences == YES);
	CHECK(cb.status == MTLCommandBufferStatusNotEnqueued);

	// --- render encoder ---
	MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
	rp.colorAttachments[0].texture = handle(42);
	rp.colorAttachments[0].loadAction = MTLLoadActionClear;
	rp.colorAttachments[0].storeAction = MTLStoreActionStore;
	rp.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
	rp.renderTargetWidth = 1920;
	rp.renderTargetHeight = 1080;
	id<MTLRenderCommandEncoder> renc = [cb renderCommandEncoderWithDescriptor:rp];
	CHECK(renc != nil);
	CHECK(renc.device == nil);
	[renc setRenderPipelineState:handle(7)];
	[renc setVertexBuffer:handle(10) offset:16 atIndex:0];
	[renc setViewport:(MTLViewport){ 0.0, 0.0, 1920.0, 1080.0, 0.0, 1.0 }];
	{
		id<MTLBuffer> bufs[2] = { handle(10), handle(11) };
		NSUInteger offs[2] = { 0, 256 };
		[renc setVertexBuffers:bufs offsets:offs withRange:NSMakeRange(0, 2)];
	}
	[renc setFragmentTexture:handle(42) atIndex:3];
	[renc setScissorRect:(MTLScissorRect){ 1, 2, 3, 4 }];
	[renc setDepthStencilState:handle(8)];
	[renc setCullMode:MTLCullModeBack];
	[renc useResource:handle(9) usage:MTLResourceUsageRead];
	[renc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
	[renc drawIndexedPrimitives:MTLPrimitiveTypeTriangle
	                 indexCount:6
	                  indexType:MTLIndexTypeUInt16
	                indexBuffer:handle(12)
	          indexBufferOffset:0
	              instanceCount:2];
	[renc endEncoding];

	// --- compute encoder ---
	id<MTLComputeCommandEncoder> cenc = [cb computeCommandEncoder];
	CHECK(cenc != nil);
	[cenc setComputePipelineState:handle(20)];
	[cenc setBuffer:handle(21) offset:64 atIndex:0];
	[cenc setTexture:handle(22) atIndex:1];
	[cenc dispatchThreadgroups:MTLSizeMake(16, 8, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
	[cenc memoryBarrierWithScope:MTLBarrierScopeBuffers];
	[cenc updateFence:handle(30)];
	[cenc endEncoding];

	// --- blit encoder ---
	id<MTLBlitCommandEncoder> benc = [cb blitCommandEncoder];
	CHECK(benc != nil);
	[benc copyFromBuffer:handle(40)
	        sourceOffset:16
	            toBuffer:handle(41)
	   destinationOffset:32
	                size:4096];
	[benc fillBuffer:handle(40) range:NSMakeRange(0, 256) value:0xab];
	[benc generateMipmapsForTexture:handle(50)];
	[benc synchronizeTexture:handle(51) slice:0 level:1];
	[benc waitForFence:handle(60)];
	[benc endEncoding];

	// Holding a completed handler must not fault; commit posts the command buffer.
	[cb addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull done) {
		(void)done;
	}];
	[cb commit];
	CHECK(cb.status == MTLCommandBufferStatusCommitted);

	// Read the command buffer off the submission ring and check its header.
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(submit.next(&h, &bytes));
	CHECK(h.type == RGFX_CMD_COMMAND_BUFFER);
	CHECK(h.size == bytes.size());
	RgfxCommandBuffer cbh = {};
	std::memcpy(&cbh, bytes.data(), sizeof(cbh));
	CHECK(cbh.queue_obj_id == 1);
	CHECK(cbh.cb_obj_id == 10);
	CHECK(cbh.op_bytes == bytes.size() - RGFX_COMMAND_BUFFER_OPS_OFFSET);

	// The exact op sequence the three encoders produced.
	const uint16_t want[] = {
		RGFX_OP_ENCODER_BEGIN_RENDER,
		RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE,
		RGFX_OP_RENDER_SET_VERTEX_BUFFER,
		RGFX_OP_RENDER_SET_VIEWPORT,
		RGFX_OP_RENDER_SET_VERTEX_BUFFERS,
		RGFX_OP_RENDER_SET_FRAGMENT_TEXTURE,
		RGFX_OP_RENDER_SET_SCISSOR_RECT,
		RGFX_OP_RENDER_SET_DEPTH_STENCIL_STATE,
		RGFX_OP_RENDER_SET_CULL_MODE,
		RGFX_OP_RENDER_USE_RESOURCES,
		RGFX_OP_RENDER_DRAW_PRIMITIVES,
		RGFX_OP_RENDER_DRAW_INDEXED_PRIMITIVES_INSTANCES,
		RGFX_OP_RENDER_UPDATE_FENCE, // the queue barrier fence (QUEUE BARRIERS)
		RGFX_OP_ENCODER_END,
		RGFX_OP_ENCODER_BEGIN_COMPUTE,
		RGFX_OP_COMPUTE_SET_COMPUTE_PIPELINE_STATE,
		RGFX_OP_COMPUTE_SET_BUFFERS,
		RGFX_OP_COMPUTE_SET_TEXTURE,
		RGFX_OP_COMPUTE_DISPATCH_THREADGROUPS,
		RGFX_OP_COMPUTE_MEMORY_BARRIER_SCOPE,
		RGFX_OP_COMPUTE_UPDATE_FENCE,
		RGFX_OP_COMPUTE_UPDATE_FENCE, // the queue barrier fence
		RGFX_OP_ENCODER_END,
		RGFX_OP_ENCODER_BEGIN_BLIT,
		RGFX_OP_BLIT_COPY_BUFFER_TO_BUFFER,
		RGFX_OP_BLIT_FILL_BUFFER,
		RGFX_OP_BLIT_GENERATE_MIPMAPS,
		RGFX_OP_BLIT_SYNCHRONIZE_TEXTURE,
		RGFX_OP_BLIT_WAIT_FOR_FENCE,
		RGFX_OP_BLIT_UPDATE_FENCE, // the queue barrier fence
		RGFX_OP_ENCODER_END,
	};
	const size_t want_n = sizeof(want) / sizeof(want[0]);

	OpWalk walk(bytes.data() + RGFX_COMMAND_BUFFER_OPS_OFFSET, cbh.op_bytes);
	int open = 0, begins = 0, ends = 0;
	size_t i = 0;
	while (!walk.done()) {
		uint16_t tag = 0;
		uint32_t len = 0;
		const uint8_t *body = nullptr;
		size_t body_len = 0;
		CHECK(walk.next(&tag, &len, &body, &body_len));
		CHECK(i < want_n);
		CHECK(tag == want[i]);

		if (opGroup(tag) == RGFX_OP_GROUP_FRAMING) {
			if (tag == RGFX_OP_ENCODER_BEGIN_RENDER) {
				CHECK(open == 0);
				open = 1;
				begins++;
			} else if (tag == RGFX_OP_ENCODER_BEGIN_COMPUTE) {
				CHECK(open == 0);
				open = 2;
				begins++;
			} else if (tag == RGFX_OP_ENCODER_BEGIN_BLIT) {
				CHECK(open == 0);
				open = 3;
				begins++;
			} else if (tag == RGFX_OP_ENCODER_END) {
				CHECK(open != 0);
				open = 0;
				ends++;
			}
		} else {
			CHECK(static_cast<int>(opGroup(tag)) == open);
		}

		// Representative bodies across the three encoders, with the extracted handles.
		if (tag == RGFX_OP_ENCODER_BEGIN_RENDER) {
			CHECK(len == 8 + sizeof(RgfxRenderPassDesc) + sizeof(RgfxRenderPassColorAttachment));
			RgfxRenderPassDesc pass;
			std::memcpy(&pass, body, sizeof(pass));
			CHECK(pass.color_count == 1);
			CHECK(pass.flags == RGFX_PASS_FLAG_RT_SIZE);
			CHECK(pass.render_target_width == 1920 && pass.render_target_height == 1080);
			RgfxRenderPassColorAttachment col;
			std::memcpy(&col, body + RGFX_RENDER_PASS_COLOR_OFFSET, sizeof(col));
			CHECK(col.texture_obj_id == 42);
			CHECK(col.load_action == MTLLoadActionClear);
			CHECK(col.store_action == MTLStoreActionStore);
			double a;
			std::memcpy(&a, &col.clear_color_bits[3], 8);
			CHECK(a == 1.0);
		} else if (tag == RGFX_OP_RENDER_SET_VERTEX_BUFFER) {
			RgfxOpSetVertexBuffer b;
			std::memcpy(&b, body, sizeof(b));
			CHECK(b.buffer_obj_id == 10 && b.offset == 16 && b.index == 0);
		} else if (tag == RGFX_OP_RENDER_SET_VERTEX_BUFFERS) {
			CHECK(len == 8 + sizeof(RgfxOpSetVertexBuffers) + 2 * sizeof(RgfxBufferBinding));
			RgfxOpSetVertexBuffers head;
			std::memcpy(&head, body, sizeof(head));
			CHECK(head.range_location == 0 && head.range_length == 2);
			RgfxBufferBinding e0, e1;
			std::memcpy(&e0, body + sizeof(head), sizeof(e0));
			std::memcpy(&e1, body + sizeof(head) + sizeof(e0), sizeof(e1));
			CHECK(e0.buffer_obj_id == 10 && e0.offset == 0);
			CHECK(e1.buffer_obj_id == 11 && e1.offset == 256);
		} else if (tag == RGFX_OP_RENDER_SET_FRAGMENT_TEXTURE) {
			RgfxOpSetFragmentTexture b;
			std::memcpy(&b, body, sizeof(b));
			CHECK(b.texture_obj_id == 42 && b.index == 3);
		} else if (tag == RGFX_OP_RENDER_USE_RESOURCES) {
			RgfxOpUseResources head;
			std::memcpy(&head, body, sizeof(head));
			CHECK(head.count == 1 && head.usage == MTLResourceUsageRead);
			RgfxResourceBinding e;
			std::memcpy(&e, body + sizeof(head), sizeof(e));
			CHECK(e.obj_id == 9 && e.kind == RGFX_RESOURCE_KIND_BUFFER);
		} else if (tag == RGFX_OP_RENDER_DRAW_PRIMITIVES) {
			RgfxOpDrawPrimitives b;
			std::memcpy(&b, body, sizeof(b));
			CHECK(b.primitive_type == MTLPrimitiveTypeTriangle);
			CHECK(b.vertex_start == 0 && b.vertex_count == 3);
		} else if (tag == RGFX_OP_RENDER_DRAW_INDEXED_PRIMITIVES_INSTANCES) {
			RgfxOpDrawIndexedPrimitivesInstances b;
			std::memcpy(&b, body, sizeof(b));
			CHECK(b.index_count == 6 && b.index_buffer_obj_id == 12 && b.instance_count == 2);
		} else if (tag == RGFX_OP_COMPUTE_SET_BUFFERS) {
			RgfxComputeOpSetBuffers head;
			std::memcpy(&head, body, sizeof(head));
			CHECK(head.location == 0 && head.length == 1);
			RgfxComputeBufferBinding e;
			std::memcpy(&e, body + sizeof(head), sizeof(e));
			CHECK(e.buffer_obj_id == 21 && e.offset == 64);
		} else if (tag == RGFX_OP_COMPUTE_SET_TEXTURE) {
			RgfxComputeOpSetTexture b;
			std::memcpy(&b, body, sizeof(b));
			CHECK(b.texture_obj_id == 22 && b.index == 1);
		} else if (tag == RGFX_OP_COMPUTE_DISPATCH_THREADGROUPS) {
			RgfxComputeOpDispatchThreadgroups b;
			std::memcpy(&b, body, sizeof(b));
			CHECK(b.threadgroups_width == 16 && b.threadgroups_height == 8);
			CHECK(b.threads_width == 32);
		} else if (tag == RGFX_OP_COMPUTE_UPDATE_FENCE) {
			// The app's fence, then the queue's barrier fence at the encoder's end.
			RgfxComputeOpUpdateFence b;
			std::memcpy(&b, body, sizeof(b));
			CHECK(b.fence_obj_id == 30 || (i + 1 < want_n && want[i + 1] == RGFX_OP_ENCODER_END));
		} else if (tag == RGFX_OP_BLIT_COPY_BUFFER_TO_BUFFER) {
			RgfxBlitOpCopyBufferToBuffer b;
			std::memcpy(&b, body, sizeof(b));
			CHECK(b.source_buffer_obj_id == 40 && b.destination_buffer_obj_id == 41);
			CHECK(b.source_offset == 16 && b.destination_offset == 32 && b.size == 4096);
		} else if (tag == RGFX_OP_BLIT_FILL_BUFFER) {
			RgfxBlitOpFillBuffer b;
			std::memcpy(&b, body, sizeof(b));
			CHECK(b.buffer_obj_id == 40 && b.range_length == 256 && b.value == 0xab);
		} else if (tag == RGFX_OP_BLIT_SYNCHRONIZE_TEXTURE) {
			RgfxBlitOpSynchronizeTexture b;
			std::memcpy(&b, body, sizeof(b));
			CHECK(b.texture_obj_id == 51 && b.slice == 0 && b.level == 1);
		} else if (tag == RGFX_OP_BLIT_WAIT_FOR_FENCE) {
			RgfxBlitOpWaitForFence b;
			std::memcpy(&b, body, sizeof(b));
			CHECK(b.fence_obj_id == 60);
		}
		i++;
	}
	CHECK(i == want_n);
	CHECK(open == 0);
	CHECK(begins == 3 && ends == 3);
	CHECK(walk.pos == cbh.op_bytes && cbh.op_bytes % 8 == 0);

	// The commit posted the one message; a second commit ends the process, as Metal's does
	// (tests/queue_tests.mm).
	RgfxMsgHeader extra = {};
	std::vector<uint8_t> extraBytes;
	CHECK(!submit.next(&extra, &extraBytes));
}

// A parallel render encoder owns one render pass: its sub-encoders forward ops to the
// shared RgfxEncoder without their own framing, and the parallel encoder writes the
// single ENCODER_END.
void test_parallel_render_encoder()
{
	// Never freed, as Fixture's: the queue and its buffers may go after the test.
	TestRing &control = *new TestRing(0x8000);
	control.setConsumerIdle(1);
	RgfxRingWriter &controlWriter = *new RgfxRingWriter(
	    control.base(), control.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell, nullptr);
	RgfxIdSpace &ids = *new RgfxIdSpace();
	RgfxSerializer &ser = *new RgfxSerializer(controlWriter, ids);
	TestRing submit(0x8000);
	submit.setConsumerIdle(1);
	RgfxRingWriter submitWriter(submit.base(), submit.dataSize, 1, TestRing::doorbell, nullptr);

	RgfxCommandQueue *queue = [RgfxCommandQueue queueWithSubmitRing:&submitWriter
	                                                  submitOffset:0
	                                                completeOffset:0
	                                              completeDataSize:0
	                                                           ids:&ids
	                                                      maxCount:0
	                                                        device:nil
	                                                    serializer:&ser];
	(void)expectQueueCreate(control);
	id<MTLCommandBuffer> cb = [queue commandBuffer];

	MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
	rp.colorAttachments[0].texture = handle(42);
	id<MTLParallelRenderCommandEncoder> penc =
	    [cb parallelRenderCommandEncoderWithDescriptor:rp];
	CHECK(penc != nil);
	id<MTLRenderCommandEncoder> sub1 = [penc renderCommandEncoder];
	[sub1 setRenderPipelineState:handle(7)];
	[sub1 drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
	[sub1 endEncoding]; // sub-encoder writes NO framing
	id<MTLRenderCommandEncoder> sub2 = [penc renderCommandEncoder];
	[sub2 drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:3 vertexCount:3];
	[sub2 endEncoding];
	[penc endEncoding]; // the one ENCODER_END for the pass
	[cb commit];

	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(submit.next(&h, &bytes));
	RgfxCommandBuffer cbh = {};
	std::memcpy(&cbh, bytes.data(), sizeof(cbh));
	const uint16_t want[] = {
		RGFX_OP_ENCODER_BEGIN_RENDER,     RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE,
		RGFX_OP_RENDER_DRAW_PRIMITIVES,   RGFX_OP_RENDER_DRAW_PRIMITIVES,
		RGFX_OP_RENDER_UPDATE_FENCE, // the pass's queue barrier fence (QUEUE BARRIERS)
		RGFX_OP_ENCODER_END,
	};
	const size_t want_n = sizeof(want) / sizeof(want[0]);
	OpWalk walk(bytes.data() + RGFX_COMMAND_BUFFER_OPS_OFFSET, cbh.op_bytes);
	size_t i = 0;
	int begins = 0, ends = 0;
	while (!walk.done()) {
		uint16_t tag = 0;
		uint32_t len = 0;
		const uint8_t *body = nullptr;
		size_t body_len = 0;
		CHECK(walk.next(&tag, &len, &body, &body_len));
		CHECK(i < want_n && tag == want[i]);
		if (tag == RGFX_OP_ENCODER_BEGIN_RENDER)
			begins++;
		if (tag == RGFX_OP_ENCODER_END)
			ends++;
		i++;
	}
	CHECK(i == want_n);
	CHECK(begins == 1 && ends == 1); // one pass, one begin/end though two sub-encoders
}

// Metal runs a parallel pass's sub-encoders in the order they were made, whatever order their
// commands were encoded in, and each starts from the default state (measured on the M3 Max, macOS
// 27: a sub-encoder made second and encoded first draws last; a 1x1 scissor set in the first does
// not clip the second's draw). Here the second sub-encoder is encoded first: its draw (vertex
// start 3) must follow the first's (vertex start 0) in the stream, and the first's scissor must
// not reach the second's draw.
void test_parallel_sub_encoders_run_in_creation_order()
{
	TestRing &control = *new TestRing(0x8000);
	control.setConsumerIdle(1);
	RgfxRingWriter &controlWriter = *new RgfxRingWriter(
	    control.base(), control.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell, nullptr);
	RgfxIdSpace &ids = *new RgfxIdSpace();
	RgfxSerializer &ser = *new RgfxSerializer(controlWriter, ids);
	TestRing submit(0x8000);
	submit.setConsumerIdle(1);
	RgfxRingWriter submitWriter(submit.base(), submit.dataSize, 1, TestRing::doorbell, nullptr);
	RgfxCommandQueue *queue = [RgfxCommandQueue queueWithSubmitRing:&submitWriter
	                                                  submitOffset:0
	                                                completeOffset:0
	                                              completeDataSize:0
	                                                           ids:&ids
	                                                      maxCount:0
	                                                        device:nil
	                                                    serializer:&ser];
	(void)expectQueueCreate(control);
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
	rp.colorAttachments[0].texture = handle(42);
	id<MTLParallelRenderCommandEncoder> penc = [cb parallelRenderCommandEncoderWithDescriptor:rp];
	id<MTLRenderCommandEncoder> first = [penc renderCommandEncoder];
	id<MTLRenderCommandEncoder> second = [penc renderCommandEncoder];
	[second setRenderPipelineState:handle(7)];
	[second drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:3 vertexCount:3];
	[second endEncoding];
	[first setRenderPipelineState:handle(7)];
	[first setScissorRect:(MTLScissorRect){0, 0, 1, 1}];
	[first drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
	[first endEncoding];
	[penc endEncoding];
	[cb commit];

	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(submit.next(&h, &bytes));
	RgfxCommandBuffer cbh = {};
	std::memcpy(&cbh, bytes.data(), sizeof(cbh));
	OpWalk walk(bytes.data() + RGFX_COMMAND_BUFFER_OPS_OFFSET, cbh.op_bytes);
	std::vector<uint64_t> starts;
	bool scissored = false, scissoredBeforeSecond = false;
	while (!walk.done()) {
		uint16_t tag = 0;
		uint32_t len = 0;
		const uint8_t *body = nullptr;
		size_t body_len = 0;
		CHECK(walk.next(&tag, &len, &body, &body_len));
		if (tag == RGFX_OP_RENDER_SET_SCISSOR_RECT) {
			RgfxOpSetScissorRect r = {};
			std::memcpy(&r, body, sizeof(r));
			scissored = r.width == 1 && r.height == 1;
		}
		if (tag == RGFX_OP_RENDER_DRAW_PRIMITIVES) {
			RgfxOpDrawPrimitives d = {};
			std::memcpy(&d, body, sizeof(d));
			starts.push_back(d.vertex_start);
			if (d.vertex_start == 3)
				scissoredBeforeSecond = scissored;
		}
	}
	CHECK(starts.size() == 2);
	CHECK(starts.size() == 2 && starts[0] == 0 && starts[1] == 3); // creation order
	CHECK(!scissoredBeforeSecond); // the first's scissor does not clip the second's draw
}

// A queue must have a route, an id space and a ring.
void test_queue_guards()
{
	// Never freed, as Fixture's: the queue and its buffers may go after the test.
	TestRing &control = *new TestRing(0x8000);
	control.setConsumerIdle(1);
	RgfxRingWriter &controlWriter = *new RgfxRingWriter(
	    control.base(), control.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell, nullptr);
	RgfxIdSpace &ids = *new RgfxIdSpace();
	RgfxSerializer &ser = *new RgfxSerializer(controlWriter, ids);
	TestRing submit(0x8000);
	RgfxRingWriter submitWriter(submit.base(), submit.dataSize, 1, TestRing::doorbell, nullptr);

	CHECK([RgfxCommandQueue queueWithSubmitRing:&submitWriter
	                               submitOffset:0
	                             completeOffset:0
	                           completeDataSize:0
	                                        ids:&ids
	                                   maxCount:0
	                                     device:nil
	                                 serializer:nullptr] == nil);
	CHECK([RgfxCommandQueue queueWithSubmitRing:&submitWriter
	                               submitOffset:0
	                             completeOffset:0
	                           completeDataSize:0
	                                        ids:nullptr
	                                   maxCount:0
	                                     device:nil
	                                 serializer:&ser] == nil);
	CHECK([RgfxCommandQueue queueWithSubmitRing:nullptr
	                               submitOffset:0
	                             completeOffset:0
	                           completeDataSize:0
	                                        ids:&ids
	                                   maxCount:0
	                                     device:nil
	                                 serializer:&ser] == nil);
}

// A queue whose create the kext refuses (its 257th, kMaxQueues in RgfxAcceleratorUserClient.hpp)
// is no queue: the device never opened its rings, so a command buffer committed to it would
// never complete. newCommandQueue answers nil, as Metal does when it cannot make one.
void test_refused_queue_create_is_nil()
{
	TestRing &control = *new TestRing(0x8000);
	control.setConsumerIdle(1);
	RgfxRingWriter &controlWriter = *new RgfxRingWriter(
	    control.base(), control.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell, nullptr);
	RgfxIdSpace &ids = *new RgfxIdSpace();
	RgfxSerializer &ser = *new RgfxSerializer(controlWriter, ids);
	ser.setQueueOpener([](void *, const RgfxQueueCreate &) { return false; }, nullptr);
	TestRing submit(0x8000);
	RgfxRingWriter submitWriter(submit.base(), submit.dataSize, 1, TestRing::doorbell, nullptr);

	CHECK([RgfxCommandQueue queueWithSubmitRing:&submitWriter
	                               submitOffset:0
	                             completeOffset:0
	                           completeDataSize:0
	                                        ids:&ids
	                                   maxCount:0
	                                     device:nil
	                                 serializer:&ser] == nil);
}

// =============================================================================
// The transport layer: the region allocator and the reply reader
// =============================================================================

// The region allocator hands out aligned, non-overlapping blocks past the floor
// (RGFX_ALLOCATIONS_OFFSET), never touching the host-reserved control rings / caps
// page below it, and places ring pairs with power-of-two byte rings.
void test_region_allocator()
{
	std::vector<uint8_t> mem(0x800000, 0xff); // 8 MiB, pre-dirtied like BAR memory
	RgfxRegion region(mem.data(), mem.size());
	CHECK(region.floor() == RGFX_ALLOCATIONS_OFFSET);
	CHECK(region.size() == mem.size());

	RgfxRegion::Block a = region.allocate(0x1000);
	CHECK(a.ok);
	CHECK(a.offset >= RGFX_ALLOCATIONS_OFFSET); // never below the floor
	CHECK(a.offset % 0x1000 == 0);              // aligned (default page)
	CHECK(a.ptr == region.base() + a.offset);
	CHECK(a.size == 0x1000);

	RgfxRegion::Block b = region.allocate(0x800, 256);
	CHECK(b.ok);
	CHECK(b.offset % 256 == 0);
	CHECK(a.offset + a.size <= b.offset); // non-overlap with a

	RgfxRegion::Block c = region.allocate(0x1234, 64);
	CHECK(c.ok);
	CHECK(b.offset + b.size <= c.offset); // non-overlap with b
	CHECK(region.used() >= (c.offset - region.floor()) + c.size);

	// A non-power-of-two alignment, and a request past the region's end, both fail.
	CHECK(!region.allocate(16, 3).ok);
	CHECK(!region.allocate(mem.size(), 0x1000).ok);

	// A submission + completion ring pair: both offsets past the floor, non-overlapping,
	// each a ring header plus a power-of-two byte ring.
	RgfxRegion::RingPair rings = region.allocateRingPair(0x8000, 0x8000);
	CHECK(rings.ok);
	CHECK(rings.submit_offset >= RGFX_ALLOCATIONS_OFFSET);
	CHECK(rings.submit_data_size == 0x8000);
	CHECK(rings.complete_data_size == 0x8000);
	CHECK(rings.complete_offset >= rings.submit_offset + RGFX_RING_HEADER_SIZE + 0x8000);
	// A completion ring of 0 places none (the single-ring model).
	RgfxRegion::RingPair submitOnly = region.allocateRingPair(0x1000, 0);
	CHECK(submitOnly.ok);
	CHECK(submitOnly.complete_offset == 0);
	// A non-power-of-two byte ring is refused.
	CHECK(!region.allocateRingPair(0x7000, 0).ok);

	// contains / offsetOf: inside the allocable region, below the floor is out.
	CHECK(region.contains(mem.data() + RGFX_ALLOCATIONS_OFFSET, 16));
	CHECK(!region.contains(mem.data(), 16)); // below the floor
	CHECK(!region.contains(mem.data() + mem.size() - 8, 16)); // runs off the end
	CHECK(region.offsetOf(mem.data() + a.offset) == a.offset);
}

// newCommandQueue's path: a queue built over a ring pair the region allocator placed
// carries those offsets in its RGFX_CMD_QUEUE_CREATE (the device category does the same
// with its region; here the factory is driven directly with a fixture region).
void test_queue_carries_allocated_offsets()
{
	// Never freed, as Fixture's: the queue and its buffers may go after the test.
	TestRing &control = *new TestRing(0x8000);
	control.setConsumerIdle(1);
	RgfxRingWriter &controlWriter = *new RgfxRingWriter(
	    control.base(), control.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell, nullptr);
	RgfxIdSpace &ids = *new RgfxIdSpace();
	RgfxSerializer &ser = *new RgfxSerializer(controlWriter, ids);

	std::vector<uint8_t> mem(0x800000, 0);
	RgfxRegion region(mem.data(), mem.size());
	RgfxRegion::RingPair rings = region.allocateRingPair(0x8000, 0x8000);
	CHECK(rings.ok);

	// The submission ring's writer over the region at the allocated submit offset.
	RgfxRingWriter submitWriter(region.base() + rings.submit_offset, rings.submit_data_size, 1,
	                            TestRing::doorbell, nullptr);

	RgfxCommandQueue *queue = [RgfxCommandQueue queueWithSubmitRing:&submitWriter
	                                                  submitOffset:rings.submit_offset
	                                                completeOffset:rings.complete_offset
	                                              completeDataSize:rings.complete_data_size
	                                                           ids:&ids
	                                                      maxCount:0
	                                                        device:nil
	                                                    serializer:&ser];
	CHECK(queue != nil);
	RgfxQueueCreate qc = expectQueueCreate(control);
	CHECK(qc.submit_offset == rings.submit_offset);
	CHECK(qc.complete_offset == rings.complete_offset);
	CHECK(qc.submit_data_size == rings.submit_data_size);
	CHECK(qc.complete_data_size == rings.complete_data_size);
	CHECK(qc.submit_offset >= RGFX_ALLOCATIONS_OFFSET);
}

// A host command-buffer completion the reader dispatches: the CB's completed handler
// fires with the buffer, its status flips, and an error completion carries the error.
void test_reply_reader_completion()
{
	// Never freed, as Fixture's: the queue and its buffers may go after the test.
	TestRing &control = *new TestRing(0x8000);
	control.setConsumerIdle(1);
	RgfxRingWriter &controlWriter = *new RgfxRingWriter(
	    control.base(), control.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell, nullptr);
	RgfxIdSpace &ids = *new RgfxIdSpace();
	RgfxSerializer &ser = *new RgfxSerializer(controlWriter, ids);
	TestRing submit(0x8000);
	submit.setConsumerIdle(1);
	RgfxRingWriter submitWriter(submit.base(), submit.dataSize, 1, TestRing::doorbell, nullptr);

	TestRing completion(0x8000);
	RgfxReplyReader reader(completion.base(), completion.dataSize);

	RgfxCommandQueue *queue = [RgfxCommandQueue queueWithSubmitRing:&submitWriter
	                                                  submitOffset:0
	                                                completeOffset:0
	                                              completeDataSize:0
	                                                           ids:&ids
	                                                      maxCount:0
	                                                        device:nil
	                                                    serializer:&ser];
	queue.replyReader = &reader;
	(void)expectQueueCreate(control);

	// --- a clean completion ---
	__block int fired = 0;
	__block id<MTLCommandBuffer> handlerGot = nil;
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	uint64_t cbObjId = ((RgfxResource *)cb).rgfxObjId;
	[cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
		fired++;
		handlerGot = done;
	}];
	// No time is known before the commit, and the GPU's none before the completion.
	CHECK(cb.kernelStartTime == 0 && cb.kernelEndTime == 0);
	CHECK(cb.GPUStartTime == 0 && cb.GPUEndTime == 0);
	[cb commit];
	CHECK(cb.status == MTLCommandBufferStatusCommitted);
	CHECK(cb.kernelStartTime > 0 && cb.kernelEndTime >= cb.kernelStartTime);
	CHECK(cb.GPUStartTime == 0 && cb.GPUEndTime == 0);

	// Read the submission seqno the host's completion must carry.
	RgfxMsgHeader submitHdr = {};
	std::vector<uint8_t> submitBytes;
	CHECK(submit.next(&submitHdr, &submitBytes));
	CHECK(submitHdr.type == RGFX_CMD_COMMAND_BUFFER);
	uint64_t seqno = submitHdr.seqno;

	// Nothing has arrived: no dispatch, handler unfired.
	CHECK(reader.pump() == 0);
	CHECK(fired == 0);
	CHECK(reader.awaitingCompletion(seqno));

	RgfxCbComplete done = {};
	done.header.type = RGFX_EVT_CB_COMPLETE;
	done.header.size = sizeof(done);
	done.header.seqno = seqno;
	done.cb_obj_id = cbObjId;
	done.status = RGFX_CB_OK;
	// A 2 ms run on the device's clock.
	done.gpu_start_ns = 5'000'000'000;
	done.gpu_end_ns = 5'002'000'000;
	CHECK(completion.postFromHost(&done, sizeof(done)));
	CHECK(reader.pump() == 1);
	CHECK(fired == 1);
	CHECK(handlerGot == cb);
	CHECK(cb.status == MTLCommandBufferStatusCompleted);
	CHECK(cb.error == nil);
	// The run ends at the delivery, in the guest's time, and lasts what the device measured,
	// starting no earlier than the CPU posted the buffer.
	CHECK(cb.GPUEndTime >= cb.GPUStartTime && cb.GPUStartTime >= cb.kernelEndTime);
	CHECK(cb.GPUEndTime - cb.GPUStartTime <= 0.002 + 1e-9);
	CHECK(!reader.awaitingCompletion(seqno)); // the one-shot was removed

	// --- an error completion on a second buffer, waited on ---
	id<MTLCommandBuffer> cb2 = [queue commandBuffer];
	uint64_t cb2ObjId = ((RgfxResource *)cb2).rgfxObjId;
	[cb2 commit];
	RgfxMsgHeader submitHdr2 = {};
	std::vector<uint8_t> submitBytes2;
	CHECK(submit.next(&submitHdr2, &submitBytes2));
	uint64_t seqno2 = submitHdr2.seqno;

	RgfxCbComplete failed = {};
	failed.header.type = RGFX_EVT_CB_COMPLETE;
	failed.header.size = sizeof(failed);
	failed.header.seqno = seqno2;
	failed.cb_obj_id = cb2ObjId;
	failed.status = RGFX_CB_TIMEOUT;
	failed.error_code = 2; // MTLCommandBufferErrorTimeout
	CHECK(completion.postFromHost(&failed, sizeof(failed)));
	[cb2 waitUntilCompleted]; // pumps until the completion lands
	CHECK(cb2.status == MTLCommandBufferStatusError);
	CHECK(cb2.error != nil);
	CHECK(cb2.error.code == 2);
	// Metal's description of a hang, with nothing under it.
	CHECK([cb2.error.localizedDescription
	    isEqualToString:@"Caused GPU Hang Error (00000003:kIOGPUCommandBufferCallbackErrorHang)"]);
	CHECK(cb2.error.userInfo[NSUnderlyingErrorKey] == nil);
	// No run length known: the run starts as it ends.
	CHECK(cb2.GPUEndTime > 0 && cb2.GPUStartTime == cb2.GPUEndTime);

	// An invalid resource carries Metal's description and the IOGPU error under it, as Metal's
	// does (measured on the M3 Max); a code Metal was not measured failing with carries the
	// code alone.
	auto failWith = [&](uint32_t status, uint32_t code) -> id<MTLCommandBuffer> {
		id<MTLCommandBuffer> b = [queue commandBuffer];
		uint64_t objId = ((RgfxResource *)b).rgfxObjId;
		[b commit];
		RgfxMsgHeader hdr = {};
		std::vector<uint8_t> bytes;
		CHECK(submit.next(&hdr, &bytes));
		RgfxCbComplete msg = {};
		msg.header.type = RGFX_EVT_CB_COMPLETE;
		msg.header.size = sizeof(msg);
		msg.header.seqno = hdr.seqno;
		msg.cb_obj_id = objId;
		msg.status = status;
		msg.error_code = code;
		CHECK(completion.postFromHost(&msg, sizeof(msg)));
		[b waitUntilCompleted];
		CHECK(b.status == MTLCommandBufferStatusError);
		return b;
	};
	id<MTLCommandBuffer> invalid = failWith(RGFX_CB_INVALID_RESOURCE, 9);
	CHECK([invalid.error.domain isEqualToString:MTLCommandBufferErrorDomain]);
	CHECK(invalid.error.code == MTLCommandBufferErrorInvalidResource);
	CHECK([invalid.error.localizedDescription
	    isEqualToString:@"Invalid Resource (00000009:kIOGPUCommandBufferCallbackErrorInvalidResource)"]);
	NSError *under = invalid.error.userInfo[NSUnderlyingErrorKey];
	CHECK(under != nil && [under.domain isEqualToString:@"IOGPUCommandQueueErrorDomain"] &&
	      under.code == 9 && under.userInfo.count == 0);
	CHECK(invalid.error.userInfo.count == 2);
	id<MTLCommandBuffer> unknown = failWith(RGFX_CB_INTERNAL, 0);
	CHECK(unknown.error.code == 0 && unknown.error.userInfo.count == 0);

	// A descriptor's error options are kept for the buffer to answer, and no buffer has logs.
	MTLCommandBufferDescriptor *desc = [[MTLCommandBufferDescriptor alloc] init];
	desc.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
	id<MTLCommandBuffer> cb3 = [queue commandBufferWithDescriptor:desc];
	CHECK(cb3.errorOptions == MTLCommandBufferErrorOptionEncoderExecutionStatus);
	// The content protection asked of a buffer is answered as set.
	id<MTLCommandBufferSPI> spi = (id<MTLCommandBufferSPI>)cb3;
	CHECK([spi protectionOptions] == 0);
	[spi setProtectionOptions:0x5];
	CHECK([spi protectionOptions] == 0x5);
	CHECK(cb.errorOptions == MTLCommandBufferErrorOptionNone);
	NSUInteger logs = 0;
	for (id log in cb3.logs) {
		(void)log;
		logs++;
	}
	CHECK(logs == 0);
}

// A host pipeline reflection the reader dispatches fills the pipeline-state wrapper, so
// its reflection getters report the host's values instead of placeholders.
void test_reply_reader_reflection()
{
	Fixture fx; // control ring + serialiser

	const uint8_t libBytes[] = { 1, 2, 3, 4 };
	uint32_t libStatus = 0;
	RgfxLibrary *lib = [RgfxLibrary libraryWithData:libBytes
	                                        length:sizeof(libBytes)
	                                           key:nullptr
	                                        device:nil
	                                    serializer:&fx.ser
	                                        status:&libStatus];
	(void)nextMessage(fx.ring, RGFX_CMD_LIBRARY_CREATE);
	id<MTLFunction> cfn = [lib newFunctionWithName:@"kmain"];

	NSError *err = nil;
	RgfxComputePipelineState *ps =
	    [RgfxComputePipelineState computePipelineStateWithFunction:cfn
	                                                        device:nil
	                                                    serializer:&fx.ser
	                                                         error:&err];
	CHECK(ps != nil);
	(void)nextMessage(fx.ring, RGFX_CMD_COMPUTE_PIPELINE_CREATE);

	TestRing completion(0x8000);
	RgfxReplyReader reader(completion.base(), completion.dataSize);
	[ps rgfxRegisterWithReplyReader:&reader];
	CHECK(reader.awaitingReflection(ps.rgfxSeqno));

	// Before the reply the getters answer placeholders.
	CHECK([ps maxTotalThreadsPerThreadgroup] == 1024);
	CHECK(ps.gpuResourceID._impl == 0);

	RgfxPipelineReflection refl = {};
	refl.header.type = RGFX_EVT_PIPELINE_REFLECTION;
	refl.header.size = sizeof(refl);
	refl.header.seqno = ps.rgfxSeqno;
	refl.obj_id = ps.rgfxObjId;
	refl.gpu_resource_id = 0xABCDEF;
	refl.max_total_threads_per_threadgroup = 512;
	refl.static_threadgroup_memory_length = 2048;
	refl.thread_execution_width = 64;
	refl.status = RGFX_PIPELINE_OK;
	refl.kind = RGFX_PIPELINE_KIND_COMPUTE;
	CHECK(completion.postFromHost(&refl, sizeof(refl)));
	CHECK(reader.pump() == 1);

	CHECK([ps maxTotalThreadsPerThreadgroup] == 512);
	CHECK([ps threadExecutionWidth] == 64);
	CHECK([ps staticThreadgroupMemoryLength] == 2048);
	CHECK(ps.gpuResourceID._impl == 0xABCDEF);
	CHECK(!reader.awaitingReflection(ps.rgfxSeqno)); // one-shot removed
	CHECK(![ps rgfxRefused]);

	// A refused create: the state reports the refusal, which the device's creators answer as
	// nil and an NSError.
	RgfxComputePipelineState *refusedPs =
	    [RgfxComputePipelineState computePipelineStateWithFunction:cfn
	                                                        device:nil
	                                                    serializer:&fx.ser
	                                                         error:&err];
	CHECK(refusedPs != nil);
	(void)nextMessage(fx.ring, RGFX_CMD_COMPUTE_PIPELINE_CREATE);
	[refusedPs rgfxRegisterWithReplyReader:&reader];
	CHECK(![refusedPs rgfxRefused]);
	RgfxPipelineReflection no = {};
	no.header.type = RGFX_EVT_PIPELINE_REFLECTION;
	no.header.size = sizeof(no);
	no.header.seqno = refusedPs.rgfxSeqno;
	no.obj_id = refusedPs.rgfxObjId;
	no.status = RGFX_PIPELINE_REFUSED;
	no.kind = RGFX_PIPELINE_KIND_COMPUTE;
	CHECK(completion.postFromHost(&no, sizeof(no)));
	CHECK(reader.pump() == 1);
	CHECK([refusedPs rgfxRefused]);
}

// A host event-signalled message raises the shared event's value and fires the
// notifyListener blocks it reached, on their listener's queue, with the value each was
// registered at, as Metal's do (measured on the M3 Max). The host posts it on the queue's
// completion ring, whose reader hands it to the device's, where the event listens. A lower
// value leaves the value; a block registered at a value reached fires at once, on its queue.
void test_reply_reader_event()
{
	// The readers are declared before the inner pool so they outlive the autoreleased
	// event, whose dealloc deregisters its listener from the device's reader (as the
	// device's reader outlives its events in the guest).
	RgfxReplyReader device;
	TestRing completion(0x8000);
	RgfxReplyReader queue(completion.base(), completion.dataSize);
	queue.setEventsTo(&device);
	@autoreleasepool {
	Fixture fx;

	const uint64_t initial = 5;
	RgfxSharedEvent *ev = [RgfxSharedEvent sharedEventWithInitialValue:initial
	                                                           device:nil
	                                                       serializer:&fx.ser];
	CHECK(ev != nil);
	(void)expectMessage<RgfxSharedEventCreate>(fx.ring, RGFX_CMD_SHARED_EVENT_CREATE);

	[ev rgfxRegisterWithReplyReader:&device];
	CHECK(ev.signaledValue == initial);

	// A listener on a queue of its own, which fires once the value reaches 45.
	dispatch_queue_t lq = dispatch_queue_create("rgfx.test.listener", DISPATCH_QUEUE_SERIAL);
	static char onListener;
	dispatch_queue_set_specific(lq, &onListener, &onListener, nullptr);
	MTLSharedEventListener *listener = [[MTLSharedEventListener alloc] initWithDispatchQueue:lq];
	dispatch_semaphore_t fired = dispatch_semaphore_create(0);
	__block uint64_t notified = 0;
	__block bool onItsQueue = false;
	[ev notifyListener:listener
	           atValue:45
	             block:^(id<MTLSharedEvent> e, uint64_t v) {
		             (void)e;
		             notified = v;
		             onItsQueue = dispatch_get_specific(&onListener) == &onListener;
		             dispatch_semaphore_signal(fired);
	             }];

	auto signal = [&](uint64_t value) {
		RgfxEventSignalled sig = {};
		sig.header.type = RGFX_EVT_EVENT_SIGNALLED;
		sig.header.size = sizeof(sig);
		sig.obj_id = ev.rgfxObjId;
		sig.value = value;
		CHECK(completion.postFromHost(&sig, sizeof(sig)));
		CHECK(queue.pump() == 1);
	};
	// The host raises the event to 42: the value rises, the 45-listener holds.
	signal(42);
	CHECK(ev.signaledValue == 42);
	dispatch_sync(lq, ^{
	});
	CHECK(notified == 0);

	// The host raises it to 50: the value rises and the listener fires on its queue, with 45.
	signal(50);
	CHECK(ev.signaledValue == 50);
	CHECK(dispatch_semaphore_wait(fired, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC)) == 0);
	CHECK(notified == 45);
	CHECK(onItsQueue);

	// A lower value leaves it.
	signal(30);
	CHECK(ev.signaledValue == 50);

	// One registered at a value reached fires at once, on its queue, with its value.
	onItsQueue = false;
	[ev notifyListener:listener
	           atValue:10
	             block:^(id<MTLSharedEvent> e, uint64_t v) {
		             (void)e;
		             notified = v;
		             onItsQueue = dispatch_get_specific(&onListener) == &onListener;
		             dispatch_semaphore_signal(fired);
	             }];
	CHECK(dispatch_semaphore_wait(fired, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC)) == 0);
	CHECK(notified == 10);
	CHECK(onItsQueue);

	// A CPU set reaches a listener too.
	[ev notifyListener:listener
	           atValue:60
	             block:^(id<MTLSharedEvent> e, uint64_t v) {
		             (void)e;
		             notified = v;
		             dispatch_semaphore_signal(fired);
	             }];
	ev.signaledValue = 61;
	CHECK(dispatch_semaphore_wait(fired, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC)) == 0);
	CHECK(notified == 60);
	(void)expectMessage<RgfxSharedEventSetValue>(fx.ring, RGFX_CMD_SHARED_EVENT_SET_VALUE);
	dispatch_sync(lq, ^{
	});
	} // @autoreleasepool: ev drains here, while the readers still live
}

// --- the completion thread ---------------------------------------------------

// The kext's side of the completion thread, in user space: the wake count the thread
// sleeps on (kRgfxAccelMethodWaitCompletions), and the host's post, which raises the
// vector where the ring is armed and so wakes the thread (queueInterrupt).
struct FakeKext {
	std::mutex lock;
	std::condition_variable changed;
	uint64_t count = 0;
	bool closed = false;
	int waits = 0;

	void wake()
	{
		std::lock_guard<std::mutex> hold(lock);
		count++;
		changed.notify_all();
	}

	void close()
	{
		std::lock_guard<std::mutex> hold(lock);
		closed = true;
		changed.notify_all();
	}

	int waitCount()
	{
		std::lock_guard<std::mutex> hold(lock);
		return waits;
	}

	static RgfxCompletionThread::WaitResult wait(void *context, uint64_t last, uint64_t *now)
	{
		FakeKext *k = static_cast<FakeKext *>(context);
		std::unique_lock<std::mutex> hold(k->lock);
		k->waits++;
		k->changed.wait(hold, [&] { return k->count != last || k->closed; });
		*now = k->count;
		return k->closed ? RgfxCompletionThread::kStop : RgfxCompletionThread::kWoke;
	}
};

uint32_t ringArmed(TestRing &ring)
{
	return __atomic_load_n(reinterpret_cast<uint32_t *>(ring.mem.data() +
	                                                    __builtin_offsetof(RgfxRingHeader, armed)),
	                       __ATOMIC_SEQ_CST);
}

// The host's post (ring.rs Producer::post): the record published, a full fence, then
// `armed` read and cleared, the vector raised where it was set.
void hostPost(TestRing &ring, FakeKext &kext, const void *msg, uint32_t size)
{
	CHECK(ring.postFromHost(msg, size));
	__atomic_thread_fence(__ATOMIC_SEQ_CST);
	uint32_t *armed = reinterpret_cast<uint32_t *>(ring.mem.data() +
	                                               __builtin_offsetof(RgfxRingHeader, armed));
	if (__atomic_exchange_n(armed, 0u, __ATOMIC_SEQ_CST) == 1)
		kext.wake();
}

RgfxCbComplete cbComplete(uint64_t seqno, uint64_t cbObjId)
{
	RgfxCbComplete done = {};
	done.header.type = RGFX_EVT_CB_COMPLETE;
	done.header.size = sizeof(done);
	done.header.seqno = seqno;
	done.cb_obj_id = cbObjId;
	done.status = RGFX_CB_OK;
	return done;
}

RgfxCbSubmitted cbSubmitted(uint64_t seqno, uint64_t cbObjId)
{
	RgfxCbSubmitted submitted = {};
	submitted.header.type = RGFX_EVT_CB_SUBMITTED;
	submitted.header.size = sizeof(submitted);
	submitted.header.seqno = seqno;
	submitted.cb_obj_id = cbObjId;
	return submitted;
}

// How many RGFX_OP_SUBMITTED ops the command buffer message `bytes` holds.
int submittedOps(const std::vector<uint8_t> &bytes)
{
	int n = 0;
	size_t at = RGFX_COMMAND_BUFFER_OPS_OFFSET;
	while (at + sizeof(RgfxOpHeader) <= bytes.size()) {
		RgfxOpHeader op = {};
		std::memcpy(&op, bytes.data() + at, sizeof(op));
		if (op.len < RGFX_OP_BODY_OFFSET || at + op.len > bytes.size())
			return -1;
		if (op.tag == RGFX_OP_SUBMITTED)
			n += op.len == RGFX_OP_BODY_OFFSET ? 1 : 100;
		at += op.len;
	}
	return n;
}

RgfxCbScheduled cbScheduled(uint64_t seqno, uint64_t cbObjId)
{
	RgfxCbScheduled scheduled = {};
	scheduled.header.type = RGFX_EVT_CB_SCHEDULED;
	scheduled.header.size = sizeof(scheduled);
	scheduled.header.seqno = seqno;
	scheduled.cb_obj_id = cbObjId;
	return scheduled;
}

// Waits up to 5 s for `pred`, polling.
template <typename F> bool eventually(F pred)
{
	for (int i = 0; i < 5000; i++) {
		if (pred())
			return true;
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	return pred();
}

// A queue over its own submission and completion rings, as the device makes one, with an
// owned reader the completion thread can drive.
// The rings and the serialiser are never freed, as Fixture's: the queue lives on while a
// command buffer the test let go holds it.
struct QueueFixture {
	TestRing &control = *new TestRing(0x8000);
	RgfxRingWriter &controlWriter = *new RgfxRingWriter(
	    control.base(), control.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell, nullptr);
	RgfxIdSpace &ids = *new RgfxIdSpace();
	RgfxSerializer &ser = *new RgfxSerializer(controlWriter, ids);
	TestRing &submit;
	RgfxRingWriter &submitWriter;
	TestRing &completion = *new TestRing(0x8000);
	RgfxCommandQueue *queue = nil;

	// A queue whose submission ring has `submitData` bytes, made with `maxCount`, its doorbell
	// `doorbell` (with `doorbellContext`).
	explicit QueueFixture(uint32_t submitData = 0x8000, NSUInteger maxCount = 0,
	                      RgfxDoorbell doorbell = TestRing::doorbell, void *doorbellContext = nullptr)
	    : submit(*new TestRing(submitData)),
	      submitWriter(*new RgfxRingWriter(submit.base(), submit.dataSize, 1, doorbell,
	                                       doorbellContext))
	{
		control.setConsumerIdle(1);
		submit.setConsumerIdle(1);
		queue = [RgfxCommandQueue queueWithSubmitRing:&submitWriter
		                                 submitOffset:0
		                               completeOffset:0
		                             completeDataSize:0
		                                          ids:&ids
		                                     maxCount:maxCount
		                                       device:nil
		                                   serializer:&ser];
		[queue adoptReplyReader:new RgfxReplyReader(completion.base(), completion.dataSize)];
	}

	// The submission seqno of the next command buffer committed on the queue.
	uint64_t nextSeqno()
	{
		RgfxMsgHeader h = {};
		std::vector<uint8_t> bytes;
		CHECK(submit.next(&h, &bytes));
		CHECK(h.type == RGFX_CMD_COMMAND_BUFFER);
		return h.seqno;
	}
};

// A command buffer's event ops go in its stream between its encoders, in order, each naming
// its event and value, an event and a shared event alike; one naming an event no device of
// ours made is not carried.
void test_command_buffer_event_ops()
{
	QueueFixture f;
	@autoreleasepool {
		RgfxEvent *ev = [RgfxEvent eventWithDevice:nil serializer:&f.ser];
		RgfxSharedEvent *sev = [RgfxSharedEvent sharedEventWithInitialValue:0
		                                                             device:nil
		                                                         serializer:&f.ser];
		CHECK(ev != nil && sev != nil);
		id<MTLCommandBuffer> cb = [f.queue commandBuffer];
		[cb encodeWaitForEvent:sev value:3];
		id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
		[blit endEncoding];
		[cb encodeSignalEvent:ev value:4];
		[cb encodeSignalEvent:(id<MTLEvent>)[NSObject new] value:9];
		[cb encodeSignalEvent:sev value:5];
		[cb commit];

		RgfxMsgHeader h = {};
		std::vector<uint8_t> bytes;
		CHECK(f.submit.next(&h, &bytes));
		CHECK(h.type == RGFX_CMD_COMMAND_BUFFER);
		RgfxCommandBuffer cbh = {};
		std::memcpy(&cbh, bytes.data(), sizeof(cbh));
		struct Want {
			uint16_t tag;
			uint64_t event;
			uint64_t value;
		};
		const Want want[] = {
			{ RGFX_OP_WAIT_EVENT, sev.rgfxObjId, 3 },
			{ RGFX_OP_ENCODER_BEGIN_BLIT, 0, 0 },
			{ RGFX_OP_ENCODER_END, 0, 0 },
			{ RGFX_OP_SIGNAL_EVENT, ev.rgfxObjId, 4 },
			{ RGFX_OP_SIGNAL_EVENT, sev.rgfxObjId, 5 },
		};
		size_t i = 0;
		OpWalk walk(bytes.data() + RGFX_COMMAND_BUFFER_OPS_OFFSET, cbh.op_bytes);
		while (!walk.done()) {
			uint16_t tag = 0;
			uint32_t len = 0;
			const uint8_t *body = nullptr;
			size_t bodyLen = 0;
			CHECK(walk.next(&tag, &len, &body, &bodyLen));
			if (opGroup(tag) != RGFX_OP_GROUP_FRAMING)
				continue; // the queue's barrier fence
			CHECK(i < sizeof(want) / sizeof(want[0]));
			CHECK(tag == want[i].tag);
			if (tag == RGFX_OP_SIGNAL_EVENT || tag == RGFX_OP_WAIT_EVENT) {
				RgfxOpEvent op = {};
				CHECK(bodyLen == sizeof(op));
				std::memcpy(&op, body, sizeof(op));
				CHECK(op.event_obj_id == want[i].event && op.value == want[i].value);
			}
			i++;
		}
		CHECK(i == sizeof(want) / sizeof(want[0]));
		RgfxCbComplete done = cbComplete(h.seqno, ((RgfxResource *)cb).rgfxObjId);
		CHECK(f.completion.postFromHost(&done, sizeof(done)));
		[cb waitUntilCompleted];
	}
}

// RgfxResourceId reads the id of each of our objects, whatever its class, and answers 0 for nil
// and for an object no device of ours made.
void test_resource_ids()
{
	QueueFixture f;
	@autoreleasepool {
		RgfxEvent *ev = [RgfxEvent eventWithDevice:nil serializer:&f.ser];
		id<MTLCommandBuffer> cb = [f.queue commandBuffer];
		CHECK(ev.rgfxObjId != 0 && RgfxResourceId(ev) == ev.rgfxObjId);
		CHECK(f.queue.rgfxObjId != 0 && RgfxResourceId(f.queue) == f.queue.rgfxObjId);
		CHECK(((RgfxResource *)cb).rgfxObjId != 0 &&
		      RgfxResourceId(cb) == ((RgfxResource *)cb).rgfxObjId);
		CHECK(RgfxResourceId(nil) == 0);
		CHECK(RgfxResourceId([NSObject new]) == 0);
		id<MTLDevice> metal = MTLCreateSystemDefaultDevice();
		id<MTLBuffer> theirs = [metal newBufferWithLength:16 options:MTLResourceStorageModeShared];
		CHECK(theirs != nil && RgfxResourceId(theirs) == 0);
	}
}

// A queue's command buffers start their messages with room for the largest of its recent ones:
// a commit's size raises the queue's reserve, each smaller one takes an eighth off it, and one
// too large for the ring raises it too, for the block the next such message moves to. A message
// made with a reserve has that room before its first op, up to what its ring takes.
void test_message_reserve()
{
	QueueFixture f;
	@autoreleasepool {
		CHECK(f.queue.rgfxMessageReserve == 0);
		RgfxEvent *ev = [RgfxEvent eventWithDevice:nil serializer:&f.ser];
		// Commits a buffer of `signals` event signals, and completes it where it reached the
		// ring: its message's size there, 0 where it did not.
		auto commit = [&](int signals) -> uint32_t {
			id<MTLCommandBuffer> cb = [f.queue commandBuffer];
			for (int i = 0; i < signals; i++)
				[cb encodeSignalEvent:ev value:static_cast<uint64_t>(i)];
			[cb commit];
			RgfxMsgHeader h = {};
			std::vector<uint8_t> bytes;
			if (!f.submit.next(&h, &bytes)) {
				[cb waitUntilCompleted];
				return 0;
			}
			CHECK(h.type == RGFX_CMD_COMMAND_BUFFER);
			RgfxCbComplete done = cbComplete(h.seqno, ((RgfxResource *)cb).rgfxObjId);
			CHECK(f.completion.postFromHost(&done, sizeof(done)));
			[cb waitUntilCompleted];
			return h.size;
		};
		const uint32_t large = commit(200);
		CHECK(large > 0x1000 && f.queue.rgfxMessageReserve == large);
		const uint32_t small = commit(1);
		CHECK(small > 0 && small < large - large / 8);
		const uint32_t once = large - large / 8;
		CHECK(f.queue.rgfxMessageReserve == once);
		(void)commit(1);
		CHECK(f.queue.rgfxMessageReserve == once - once / 8);
		CHECK(commit(250) > large && f.queue.rgfxMessageReserve > large);
		// One too large for the ring, here with no region to go to, sizes the next at its own
		// size.
		CHECK(commit(3000) == 0);
		CHECK(f.queue.rgfxMessageReserve > f.submitWriter.dataSize());

		RgfxEncoder reserved(f.submitWriter, 1, 2, 0x1800);
		CHECK(reserved.capacity() >= 0x1800 && reserved.messageSize() == RGFX_COMMAND_BUFFER_OPS_OFFSET);
		RgfxEncoder unreserved(f.submitWriter, 1, 3);
		CHECK(unreserved.capacity() >= RGFX_COMMAND_BUFFER_OPS_OFFSET);
	}
}

// The ops of the next command buffer posted on `submit`, inline: each one's tag and body.
static std::vector<std::pair<uint16_t, std::vector<uint8_t>>> postedOps(TestRing &submit)
{
	std::vector<std::pair<uint16_t, std::vector<uint8_t>>> ops;
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(submit.next(&h, &bytes) && h.type == RGFX_CMD_COMMAND_BUFFER);
	RgfxCommandBuffer cbh = {};
	std::memcpy(&cbh, bytes.data(), sizeof(cbh));
	OpWalk walk(bytes.data() + RGFX_COMMAND_BUFFER_OPS_OFFSET, cbh.op_bytes);
	while (!walk.done()) {
		uint16_t tag = 0;
		uint32_t len = 0;
		const uint8_t *body = nullptr;
		size_t body_len = 0;
		CHECK(walk.next(&tag, &len, &body, &body_len));
		ops.push_back({tag, std::vector<uint8_t>(body, body + body_len)});
	}
	return ops;
}

// Metal starts each sub-encoder of a parallel pass from the default state (MTLRenderCommandEncoder.h;
// measured on the M3 Max, an AMD 5300M and an Intel 630 for the cull mode and the scissor): each
// kind of state a sub-encoder set is set back to Metal's default, with the op that sets it, before
// the next one's ops, the viewport and scissor to the pass's target (level 1 of a 64 x 32
// attachment: 32 x 16), and none that no sub-encoder before set.
void test_parallel_sub_encoders_start_from_defaults()
{
	QueueFixture f;
	@autoreleasepool {
		int calls = 0;
		f.ids.setResourceIdSource(&oneChunk, &calls);
		MTLTextureDescriptor *td =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
		                                                       width:64
		                                                      height:32
		                                                   mipmapped:YES];
		td.usage = MTLTextureUsageRenderTarget;
		RgfxTexture *target = [RgfxTexture textureWithDescriptor:td
		                                                    heap:RGFX_OBJ_NONE
		                                              heapOffset:0
		                                                  device:nil
		                                              serializer:&f.ser];
		CHECK(target != nil);
		MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
		rp.colorAttachments[0].texture = target;
		rp.colorAttachments[0].level = 1;
		id<MTLCommandBuffer> cb = [f.queue commandBuffer];
		id<MTLParallelRenderCommandEncoder> pass = [cb parallelRenderCommandEncoderWithDescriptor:rp];
		id<MTLRenderCommandEncoder> a = [pass renderCommandEncoder];
		id<MTLRenderCommandEncoder> b = [pass renderCommandEncoder];
		id<MTLRenderCommandEncoder> c = [pass renderCommandEncoder];
		[a setRenderPipelineState:handle(7)];
		[a setCullMode:MTLCullModeBack];
		[a setFrontFacingWinding:MTLWindingCounterClockwise];
		[a setTriangleFillMode:MTLTriangleFillModeLines];
		[a setDepthClipMode:MTLDepthClipModeClamp];
		[a setDepthBias:1 slopeScale:2 clamp:3];
		[a setBlendColorRed:1 green:1 blue:1 alpha:1];
		[a setStencilReferenceValue:5];
		[a setViewport:(MTLViewport){1, 1, 4, 4, 0.25, 0.5}];
		[a setScissorRect:(MTLScissorRect){1, 1, 2, 2}];
		[a setDepthStencilState:handle(8)];
		[a setVisibilityResultMode:MTLVisibilityResultModeBoolean offset:64];
		[a setTessellationFactorScale:2];
		[a drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
		[a endEncoding];
		[b setRenderPipelineState:handle(7)];
		[b drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:3 vertexCount:3];
		[b endEncoding];
		[c drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:6 vertexCount:3];
		[c endEncoding];
		[pass endEncoding];
		[cb commit];
		drainRing(f.control);
		std::vector<std::pair<uint16_t, std::vector<uint8_t>>> ops = postedOps(f.submit);
		// From the first draw's end to the second's pipeline: the defaults, in this order.
		size_t at = 0;
		while (at < ops.size() && ops[at].first != RGFX_OP_RENDER_DRAW_PRIMITIVES)
			at++;
		CHECK(at < ops.size());
		auto body = [&](size_t i) { return ops[i].second; };
		auto bytesOf = [](const auto &v) {
			return std::vector<uint8_t>(reinterpret_cast<const uint8_t *>(&v),
			                            reinterpret_cast<const uint8_t *>(&v) + sizeof(v));
		};
		RgfxOpSetCullMode cull = {};
		RgfxOpSetFrontFacingWinding winding = {};
		RgfxOpSetTriangleFillMode fill = {};
		RgfxOpSetDepthClipModeSpi clip = {};
		RgfxOpSetDepthBias bias = {};
		RgfxOpSetBlendColorRed blend = {};
		RgfxOpSetStencilFrontReferenceValue stencil = {};
		RgfxOpSetViewport viewport = {};
		const double w = 32, h = 16, one = 1;
		std::memcpy(&viewport.width_bits, &w, 8);
		std::memcpy(&viewport.height_bits, &h, 8);
		std::memcpy(&viewport.zfar_bits, &one, 8);
		RgfxOpSetScissorRect scissor = {};
		scissor.width = 32;
		scissor.height = 16;
		RgfxOpSetDepthStencilState depthStencil = {};
		RgfxOpSetVisibilityResultMode visibility = {};
		RgfxOpSetTessellationFactorScale scale = {};
		const float fone = 1;
		std::memcpy(&scale, &fone, 4);
		const std::vector<std::pair<uint16_t, std::vector<uint8_t>>> defaults = {
		    {RGFX_OP_RENDER_SET_CULL_MODE, bytesOf(cull)},
		    {RGFX_OP_RENDER_SET_FRONT_FACING_WINDING, bytesOf(winding)},
		    {RGFX_OP_RENDER_SET_TRIANGLE_FILL_MODE, bytesOf(fill)},
		    {RGFX_OP_RENDER_SET_DEPTH_CLIP_MODE_SPI, bytesOf(clip)},
		    {RGFX_OP_RENDER_SET_DEPTH_BIAS, bytesOf(bias)},
		    {RGFX_OP_RENDER_SET_BLEND_COLOR_RED, bytesOf(blend)},
		    {RGFX_OP_RENDER_SET_STENCIL_FRONT_REFERENCE_VALUE, bytesOf(stencil)},
		    {RGFX_OP_RENDER_SET_VIEWPORT, bytesOf(viewport)},
		    {RGFX_OP_RENDER_SET_SCISSOR_RECT, bytesOf(scissor)},
		    {RGFX_OP_RENDER_SET_DEPTH_STENCIL_STATE, bytesOf(depthStencil)},
		    {RGFX_OP_RENDER_SET_VISIBILITY_RESULT_MODE, bytesOf(visibility)},
		    {RGFX_OP_RENDER_SET_TESSELLATION_FACTOR_SCALE, bytesOf(scale)},
		};
		CHECK(ops.size() >= at + 1 + defaults.size() + 4);
		bool same = true;
		for (size_t i = 0; i < defaults.size(); i++)
			same = same && ops[at + 1 + i] == defaults[i];
		CHECK(same);
		at += 1 + defaults.size();
		// The second sub-encoder set only its pipeline, which has no default to set back: the
		// third's draw follows the second's at once.
		CHECK(ops[at].first == RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE);
		CHECK(ops[at + 1].first == RGFX_OP_RENDER_DRAW_PRIMITIVES && body(at + 1)[8] == 3);
		CHECK(ops[at + 2].first == RGFX_OP_RENDER_DRAW_PRIMITIVES && body(at + 2)[8] == 6);
		// Then the pass's end (after its queue barrier fence's update).
		at += 3;
		while (at < ops.size() && ops[at].first == RGFX_OP_RENDER_UPDATE_FENCE)
			at++;
		CHECK(at + 1 == ops.size() && ops[at].first == RGFX_OP_ENCODER_END);
		CHECK(f.queue != nil);
	}
}

// Threads encode their own sub-encoders of one pass at once, as Metal allows: every draw of each
// lands, each sub-encoder's whole and in the order they were made, whichever ends first.
void test_parallel_sub_encoders_on_threads()
{
	QueueFixture f(0x100000); // half of it takes a round's message
	@autoreleasepool {
		TestHandle *buffers[2] = {handle(0x100), handle(0x101)};
		for (int round = 0; round < 20; round++) {
			id<MTLCommandBuffer> cb = [f.queue commandBuffer];
			MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
			rp.colorAttachments[0].texture = handle(42);
			id<MTLParallelRenderCommandEncoder> pass = [cb parallelRenderCommandEncoderWithDescriptor:rp];
			id<MTLRenderCommandEncoder> subs[2] = {[pass renderCommandEncoder], [pass renderCommandEncoder]};
			std::atomic<int> go{0};
			std::vector<std::thread> threads;
			for (int t = 0; t < 2; t++) {
				threads.emplace_back([&, t] {
					go.fetch_add(1);
					while (go.load() < 2) {
					}
					for (int i = 0; i < 2000; i++) {
						[subs[t] setVertexBuffer:buffers[t] offset:static_cast<NSUInteger>(i) atIndex:0];
						[subs[t] drawPrimitives:MTLPrimitiveTypeTriangle
						            vertexStart:static_cast<NSUInteger>(t * 100000 + i)
						            vertexCount:3];
					}
					[subs[t] endEncoding];
				});
			}
			for (std::thread &th : threads)
				th.join();
			[pass endEncoding];
			[cb commit];
			std::vector<uint64_t> starts;
			for (const auto &op : postedOps(f.submit)) {
				if (op.first != RGFX_OP_RENDER_DRAW_PRIMITIVES)
					continue;
				RgfxOpDrawPrimitives d = {};
				std::memcpy(&d, op.second.data(), sizeof(d));
				starts.push_back(d.vertex_start);
			}
			bool ordered = starts.size() == 4000;
			for (size_t i = 0; ordered && i < 4000; i++)
				ordered = starts[i] == (i < 2000 ? i : 100000 + (i - 2000));
			CHECK(ordered);
		}
	}
}

// A command buffer keeps each object its encoders name once, however often they name it and in
// whichever of its encoders, a parallel pass's sub-encoders among them: binds alternating between
// objects keep each one once, from the first (its retain count rises by one), until it completes.
void test_command_buffer_keeps_once()
{
	QueueFixture f;
	TestHandle *a = handle(0x51), *b = handle(0x52);
	const long a0 = CFGetRetainCount((__bridge CFTypeRef)a);
	const long b0 = CFGetRetainCount((__bridge CFTypeRef)b);
	uint64_t seqno = 0;
	id<MTLCommandBuffer> cb = nil;
	@autoreleasepool {
		cb = [f.queue commandBuffer];
		MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
		rp.colorAttachments[0].texture = handle(42);
		id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
		for (int i = 0; i < 100; i++) { // within half the ring, with no region to go to
			[e setVertexBuffer:(i & 1) ? a : b offset:0 atIndex:0];
			[e setFragmentBuffer:(i & 1) ? b : a offset:0 atIndex:0];
			[e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
		}
		[e endEncoding];
		id<MTLParallelRenderCommandEncoder> pass = [cb parallelRenderCommandEncoderWithDescriptor:rp];
		id<MTLRenderCommandEncoder> s1 = [pass renderCommandEncoder];
		id<MTLRenderCommandEncoder> s2 = [pass renderCommandEncoder];
		[s1 setVertexBuffer:a offset:0 atIndex:0];
		[s2 setVertexBuffer:a offset:0 atIndex:0];
		[s1 endEncoding];
		[s2 endEncoding];
		[pass endEncoding];
		[cb commit];
		seqno = f.nextSeqno();
	}
	CHECK(CFGetRetainCount((__bridge CFTypeRef)a) == a0 + 1);
	CHECK(CFGetRetainCount((__bridge CFTypeRef)b) == b0 + 1);
	RgfxCbComplete done = cbComplete(seqno, ((RgfxResource *)cb).rgfxObjId);
	CHECK(f.completion.postFromHost(&done, sizeof(done)));
	[cb waitUntilCompleted];
	CHECK(CFGetRetainCount((__bridge CFTypeRef)a) == a0);
	CHECK(CFGetRetainCount((__bridge CFTypeRef)b) == b0);
}

// armAndCheck sets the ring's `armed` and says whether records wait; a ringless reader
// has nothing to arm.
void test_reply_reader_arm()
{
	RgfxReplyReader ringless;
	CHECK(!ringless.armAndCheck());
	TestRing ring(0x1000);
	RgfxReplyReader reader(ring.base(), ring.dataSize);
	CHECK(!reader.driven());
	CHECK(!reader.armAndCheck());
	CHECK(ringArmed(ring) == 1);
	RgfxCbComplete done = cbComplete(7, 1);
	CHECK(ring.postFromHost(&done, sizeof(done)));
	CHECK(reader.armAndCheck()); // a record waits: pump before sleeping
	reader.pump();
	CHECK(!reader.armAndCheck());
	reader.setDriven(true);
	CHECK(reader.driven());
}

// With nobody waiting, the completion thread takes each command buffer's records as the host
// posts them, and the buffer's scheduled then completed handlers run in the ring's order on the
// queue's own completion queue (com.Metal.CompletionQueueDispatch, as Metal's), not on the
// thread; the ring is left drained and armed.
void test_completion_thread_runs_handlers()
{
	@autoreleasepool {
	QueueFixture f;
	FakeKext *kext = new FakeKext(); // outlives the thread, which may sleep in it
	RgfxCompletionThread thread(&FakeKext::wait, kext);
	thread.add([f.queue sharedReplyReader]);
	CHECK(![f.queue sharedReplyReader]->driven()); // not started: the waiters pump
	CHECK(thread.start());
	CHECK(thread.running());
	CHECK([f.queue sharedReplyReader]->driven());
	CHECK(ringArmed(f.completion) == 1);
	CHECK(eventually([&] { return kext->waitCount() >= 1; })); // asleep in the kext

	std::mutex lock;
	std::vector<std::string> order;
	__block std::mutex *orderLock = &lock;
	__block std::vector<std::string> *orderLog = &order;
	__block int offThread = 0;
	dispatch_semaphore_t last = dispatch_semaphore_create(0);
	NSMutableArray<id<MTLCommandBuffer>> *cbs = [NSMutableArray array];
	std::vector<uint64_t> seqnos;
	for (int i = 0; i < 3; i++) {
		id<MTLCommandBuffer> cb = [f.queue commandBuffer];
		std::string n = std::to_string(i);
		[cb addScheduledHandler:^(id<MTLCommandBuffer> done) {
			(void)done;
			std::lock_guard<std::mutex> hold(*orderLock);
			orderLog->push_back("s" + n);
			if (RgfxCompletionThread::onCurrentThread() ||
			    std::strcmp(dispatch_queue_get_label(DISPATCH_CURRENT_QUEUE_LABEL),
			                "com.Metal.CompletionQueueDispatch") != 0)
				offThread++;
		}];
		[cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
			std::lock_guard<std::mutex> hold(*orderLock);
			orderLog->push_back("c" + n + (done.status == MTLCommandBufferStatusCompleted ? "" : "!"));
			if (RgfxCompletionThread::onCurrentThread() ||
			    std::strcmp(dispatch_queue_get_label(DISPATCH_CURRENT_QUEUE_LABEL),
			                "com.Metal.CompletionQueueDispatch") != 0)
				offThread++;
			if (n == "2")
				dispatch_semaphore_signal(last);
		}];
		[cb commit];
		[cbs addObject:cb];
		seqnos.push_back(f.nextSeqno());
	}
	// The first completion wakes the thread; the next two land while it may be busy, and
	// are taken by the same pass or the next.
	for (int i = 0; i < 3; i++) {
		RgfxCbComplete done = cbComplete(seqnos[i], ((RgfxResource *)cbs[i]).rgfxObjId);
		hostPost(f.completion, *kext, &done, sizeof(done));
	}
	CHECK(dispatch_semaphore_wait(last, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC)) == 0);
	{
		std::lock_guard<std::mutex> hold(lock);
		std::vector<std::string> want = {"s0", "c0", "s1", "c1", "s2", "c2"};
		CHECK(order == want);
		CHECK(offThread == 0);
	}
	for (id<MTLCommandBuffer> cb in cbs)
		CHECK(cb.status == MTLCommandBufferStatusCompleted);
	CHECK(eventually([&] { return f.completion.consumerCount() == f.completion.produced; }));
	CHECK(eventually([&] { return ringArmed(f.completion) == 1; })); // armed for the next

	// A waiter after the fact returns at once; the handlers ran once.
	[cbs[0] waitUntilCompleted];
	{
		std::lock_guard<std::mutex> hold(lock);
		CHECK(order.size() == 6);
	}
	kext->close();
	CHECK(eventually([&] { return !thread.running(); }));
	}
}

// waitUntilCompleted on a driven reader blocks until the thread delivers, and the
// handlers run once, before the wait returns.
void test_wait_with_the_completion_thread()
{
	@autoreleasepool {
	QueueFixture f;
	FakeKext *kext = new FakeKext();
	RgfxCompletionThread thread(&FakeKext::wait, kext);
	CHECK(thread.start());
	thread.add([f.queue sharedReplyReader]);
	CHECK([f.queue sharedReplyReader]->driven());

	std::atomic<int> fired{0};
	std::atomic<int> *firedp = &fired;
	id<MTLCommandBuffer> cb = [f.queue commandBuffer];
	[cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
		(void)done;
		std::this_thread::sleep_for(std::chrono::milliseconds(20)); // the waiter must outwait it
		(*firedp)++;
	}];
	[cb commit];
	uint64_t seqno = f.nextSeqno();
	uint64_t cbObjId = ((RgfxResource *)cb).rgfxObjId;
	TestRing *ring = &f.completion;
	std::thread host([ring, kext, seqno, cbObjId] {
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		RgfxCbComplete done = cbComplete(seqno, cbObjId);
		hostPost(*ring, *kext, &done, sizeof(done));
	});
	[cb waitUntilCompleted];
	CHECK(fired.load() == 1); // the handler ran before the wait returned
	CHECK(cb.status == MTLCommandBufferStatusCompleted);
	host.join();
	[cb waitUntilCompleted];
	CHECK(fired.load() == 1); // and only once

	// A handler waiting for a later buffer of its own queue waits for good, as Metal's does on
	// the M3 Max, an AMD 5300M and an Intel 630; one waiting for another queue's buffer is
	// released (tests/queue_tests.mm).
	kext->close();
	CHECK(eventually([&] { return !thread.running(); }));
	}
}

// A buffer's CB_SCHEDULED, ahead of its completion: commitAndWaitUntilSubmitted returns, the
// buffer is Scheduled, its scheduled handlers run and waitUntilScheduled returns, all before it
// completes; its CB_COMPLETE then runs its completed handlers alone. A completion with no
// CB_SCHEDULED before it (a buffer the host could not carry out) runs the scheduled handlers
// first and lets the submitted and scheduled waiters go, and drops the reader's registration.
void test_scheduled_before_completion()
{
	@autoreleasepool {
	QueueFixture f;
	FakeKext *kext = new FakeKext();
	RgfxCompletionThread thread(&FakeKext::wait, kext);
	CHECK(thread.start());
	thread.add([f.queue sharedReplyReader]);

	std::mutex lock;
	std::vector<std::string> order;
	__block std::mutex *orderLock = &lock;
	__block std::vector<std::string> *orderLog = &order;
	id<MTLCommandBuffer> cb = [f.queue commandBuffer];
	[cb addScheduledHandler:^(id<MTLCommandBuffer> done) {
		std::lock_guard<std::mutex> hold(*orderLock);
		orderLog->push_back(done.status == MTLCommandBufferStatusScheduled ? "s" : "s!");
	}];
	[cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
		std::lock_guard<std::mutex> hold(*orderLock);
		orderLog->push_back(done.status == MTLCommandBufferStatusCompleted ? "c" : "c!");
	}];
	uint64_t cbObjId = ((RgfxResource *)cb).rgfxObjId;
	std::atomic<uint64_t> taken{0};
	std::atomic<uint64_t> *takenp = &taken;
	QueueFixture *fp = &f;
	// The host takes the buffer, then posts its CB_SCHEDULED a while later: the call returns
	// only then.
	std::thread host([fp, kext, cbObjId, takenp] {
		RgfxMsgHeader h = {};
		std::vector<uint8_t> bytes;
		for (int i = 0; i < 5000 && !fp->submit.next(&h, &bytes); i++)
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		takenp->store(h.seqno);
		std::this_thread::sleep_for(std::chrono::milliseconds(30));
		RgfxCbScheduled scheduled = cbScheduled(h.seqno, cbObjId);
		hostPost(fp->completion, *kext, &scheduled, sizeof(scheduled));
	});
	auto start = std::chrono::steady_clock::now();
	CHECK([(id<MTLCommandBufferSPI>)cb commitAndWaitUntilSubmitted]);
	CHECK(std::chrono::steady_clock::now() - start >= std::chrono::milliseconds(25));
	host.join();
	uint64_t seqno = taken.load();
	CHECK(seqno != 0);
	[cb waitUntilScheduled];
	CHECK(cb.status == MTLCommandBufferStatusScheduled);
	{
		std::lock_guard<std::mutex> hold(lock);
		CHECK((order == std::vector<std::string>{"s"}));
	}
	CHECK(![f.queue sharedReplyReader]->awaitingScheduled(seqno));
	CHECK([f.queue sharedReplyReader]->awaitingCompletion(seqno));
	RgfxCbComplete done = cbComplete(seqno, cbObjId);
	hostPost(f.completion, *kext, &done, sizeof(done));
	[cb waitUntilCompleted];
	CHECK(cb.status == MTLCommandBufferStatusCompleted);
	{
		std::lock_guard<std::mutex> hold(lock);
		CHECK((order == std::vector<std::string>{"s", "c"}));
	}

	// No CB_SCHEDULED: the completion stands for it.
	id<MTLCommandBuffer> failed = [f.queue commandBuffer];
	__block int scheduledRuns = 0;
	[failed addScheduledHandler:^(id<MTLCommandBuffer> b) {
		(void)b;
		scheduledRuns++;
	}];
	[failed commit];
	uint64_t s2 = f.nextSeqno();
	CHECK([f.queue sharedReplyReader]->awaitingScheduled(s2));
	CHECK(failed.status == MTLCommandBufferStatusCommitted);
	RgfxCbComplete d2 = cbComplete(s2, ((RgfxResource *)failed).rgfxObjId);
	d2.status = RGFX_CB_INTERNAL;
	hostPost(f.completion, *kext, &d2, sizeof(d2));
	[failed waitUntilScheduled];
	CHECK(scheduledRuns == 1);
	[failed waitUntilCompleted];
	CHECK(failed.status == MTLCommandBufferStatusError);
	CHECK(![f.queue sharedReplyReader]->awaitingScheduled(s2));
	CHECK(scheduledRuns == 1);
	kext->close();
	CHECK(eventually([&] { return !thread.running(); }));
	}
}

// A drawable's present rides its command buffer's scheduling, as MTLCommandBuffer.h describes:
// presentDrawable: and its variants present nothing at the call or the commit, and present the
// drawable once, by the variant's own call, when the buffer is scheduled (its CB_SCHEDULED). A
// nil drawable adds nothing.
void test_present_drawable()
{
	@autoreleasepool {
	QueueFixture f;
	FakeKext *kext = new FakeKext();
	RgfxCompletionThread thread(&FakeKext::wait, kext);
	CHECK(thread.start());
	thread.add([f.queue sharedReplyReader]);

	TestDrawable *now = [TestDrawable new], *at = [TestDrawable new], *after = [TestDrawable new];
	id<MTLCommandBuffer> cb = [f.queue commandBuffer];
	[cb presentDrawable:now];
	[cb presentDrawable:at atTime:12.5];
	[cb presentDrawable:after afterMinimumDuration:0.25];
	id<MTLDrawable> none = nil;
	[cb presentDrawable:none];
	CHECK(now.seen.count == 0 && at.seen.count == 0 && after.seen.count == 0);
	[cb commit];
	uint64_t seqno = f.nextSeqno();
	CHECK(seqno != 0);
	CHECK(now.seen.count == 0 && at.seen.count == 0 && after.seen.count == 0); // not yet scheduled
	RgfxCbScheduled scheduled = cbScheduled(seqno, ((RgfxResource *)cb).rgfxObjId);
	hostPost(f.completion, *kext, &scheduled, sizeof(scheduled));
	[cb waitUntilScheduled];
	CHECK([now.seen isEqualToArray:@[ @"now" ]]);
	CHECK([at.seen isEqualToArray:@[ @"at 12.5" ]]);
	CHECK([after.seen isEqualToArray:@[ @"after 0.25" ]]);
	RgfxCbComplete done = cbComplete(seqno, ((RgfxResource *)cb).rgfxObjId);
	hostPost(f.completion, *kext, &done, sizeof(done));
	[cb waitUntilCompleted];
	CHECK(cb.status == MTLCommandBufferStatusCompleted);
	CHECK(now.seen.count == 1 && at.seen.count == 1 && after.seen.count == 1); // once each
	kext->close();
	CHECK(eventually([&] { return !thread.running(); }));
	}
}

// The host's side of a commitAndWaitUntilSubmitted: takes the buffer's message off `ring`,
// counts its RGFX_OP_SUBMITTED ops into `asked`, and after 30 ms posts `type`'s record for it
// (CB_SUBMITTED or CB_SCHEDULED) on `completion`. `seqno` gets the buffer's seqno.
std::thread hostAnswers(TestRing *ring, TestRing *completion, FakeKext *kext, uint64_t cbObjId,
                        uint32_t type, std::atomic<uint64_t> *seqno, std::atomic<int> *asked)
{
	return std::thread([=] {
		RgfxMsgHeader h = {};
		std::vector<uint8_t> bytes;
		for (int i = 0; i < 5000 && !ring->next(&h, &bytes); i++)
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		seqno->store(h.seqno);
		asked->store(submittedOps(bytes));
		std::this_thread::sleep_for(std::chrono::milliseconds(30));
		if (type == RGFX_EVT_CB_SUBMITTED) {
			RgfxCbSubmitted submitted = cbSubmitted(h.seqno, cbObjId);
			hostPost(*completion, *kext, &submitted, sizeof(submitted));
		} else {
			RgfxCbScheduled scheduled = cbScheduled(h.seqno, cbObjId);
			hostPost(*completion, *kext, &scheduled, sizeof(scheduled));
		}
	});
}

// A queue whose device answers CB_SUBMITTED: a buffer commitAndWaitUntilSubmitted commits asks
// for its submission once, and the call returns at its CB_SUBMITTED, the buffer still Committed
// and its scheduled handlers not run, which its CB_SCHEDULED then runs. A buffer committed
// otherwise asks nothing. One the host posts no CB_SUBMITTED for is submitted at its
// CB_SCHEDULED, which drops the reader's registration. A queue whose device does not answer it
// asks nothing.
void test_submitted_before_scheduled()
{
	@autoreleasepool {
	QueueFixture f;
	f.queue.asksSubmitted = YES;
	FakeKext *kext = new FakeKext();
	RgfxCompletionThread thread(&FakeKext::wait, kext);
	CHECK(thread.start());
	thread.add([f.queue sharedReplyReader]);

	std::atomic<int> scheduledRuns{0};
	std::atomic<int> *runs = &scheduledRuns;
	id<MTLCommandBuffer> cb = [f.queue commandBuffer];
	[cb addScheduledHandler:^(id<MTLCommandBuffer> b) {
		(void)b;
		runs->fetch_add(1);
	}];
	uint64_t cbObjId = ((RgfxResource *)cb).rgfxObjId;
	std::atomic<uint64_t> taken{0};
	std::atomic<int> asked{-1};
	std::thread host =
	    hostAnswers(&f.submit, &f.completion, kext, cbObjId, RGFX_EVT_CB_SUBMITTED, &taken, &asked);
	auto start = std::chrono::steady_clock::now();
	CHECK([(id<MTLCommandBufferSPI>)cb commitAndWaitUntilSubmitted]);
	CHECK(std::chrono::steady_clock::now() - start >= std::chrono::milliseconds(25));
	host.join();
	uint64_t seqno = taken.load();
	CHECK(seqno != 0);
	CHECK(asked.load() == 1);
	CHECK(cb.status == MTLCommandBufferStatusCommitted);
	CHECK(scheduledRuns.load() == 0);
	CHECK(![f.queue sharedReplyReader]->awaitingSubmitted(seqno));
	CHECK([f.queue sharedReplyReader]->awaitingScheduled(seqno));
	RgfxCbScheduled scheduled = cbScheduled(seqno, cbObjId);
	hostPost(f.completion, *kext, &scheduled, sizeof(scheduled));
	[cb waitUntilScheduled];
	CHECK(cb.status == MTLCommandBufferStatusScheduled);
	CHECK(scheduledRuns.load() == 1);
	RgfxCbComplete done = cbComplete(seqno, cbObjId);
	hostPost(f.completion, *kext, &done, sizeof(done));
	[cb waitUntilCompleted];
	CHECK(cb.status == MTLCommandBufferStatusCompleted);

	// A plain commit asks nothing, though the device answers it.
	id<MTLCommandBuffer> plain = [f.queue commandBuffer];
	[plain commit];
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(f.submit.next(&h, &bytes) && h.type == RGFX_CMD_COMMAND_BUFFER);
	CHECK(submittedOps(bytes) == 0);
	CHECK(![f.queue sharedReplyReader]->awaitingSubmitted(h.seqno));
	RgfxCbComplete plainDone = cbComplete(h.seqno, ((RgfxResource *)plain).rgfxObjId);
	hostPost(f.completion, *kext, &plainDone, sizeof(plainDone));
	[plain waitUntilCompleted];

	// No CB_SUBMITTED: the scheduling stands for it.
	id<MTLCommandBuffer> second = [f.queue commandBuffer];
	uint64_t secondId = ((RgfxResource *)second).rgfxObjId;
	asked.store(-1);
	host = hostAnswers(&f.submit, &f.completion, kext, secondId, RGFX_EVT_CB_SCHEDULED, &taken,
	                   &asked);
	CHECK([(id<MTLCommandBufferSPI>)second commitAndWaitUntilSubmitted]);
	host.join();
	uint64_t s2 = taken.load();
	CHECK(asked.load() == 1);
	// Submitted as the completion thread takes the CB_SCHEDULED; Scheduled once its record has
	// run on the queue's completion queue, which may be after the call returns, as Metal's may.
	CHECK(second.status == MTLCommandBufferStatusCommitted ||
	      second.status == MTLCommandBufferStatusScheduled);
	[second waitUntilScheduled];
	CHECK(second.status == MTLCommandBufferStatusScheduled);
	CHECK(![f.queue sharedReplyReader]->awaitingSubmitted(s2));
	RgfxCbComplete d2 = cbComplete(s2, secondId);
	hostPost(f.completion, *kext, &d2, sizeof(d2));
	[second waitUntilCompleted];
	kext->close();
	CHECK(eventually([&] { return !thread.running(); }));
	}

	// A queue whose device does not answer it: the call returns at the CB_SCHEDULED, the ring
	// taken by the waiting thread itself.
	@autoreleasepool {
	QueueFixture f;
	FakeKext kext;
	id<MTLCommandBuffer> cb = [f.queue commandBuffer];
	uint64_t cbObjId = ((RgfxResource *)cb).rgfxObjId;
	std::atomic<uint64_t> taken{0};
	std::atomic<int> asked{-1};
	std::thread host =
	    hostAnswers(&f.submit, &f.completion, &kext, cbObjId, RGFX_EVT_CB_SCHEDULED, &taken, &asked);
	CHECK([(id<MTLCommandBufferSPI>)cb commitAndWaitUntilSubmitted]);
	host.join();
	CHECK(asked.load() == 0);
	CHECK(cb.status == MTLCommandBufferStatusScheduled);
	CHECK(![f.queue sharedReplyReader]->awaitingSubmitted(taken.load()));
	RgfxCbComplete done = cbComplete(taken.load(), cbObjId);
	hostPost(f.completion, kext, &done, sizeof(done));
	[cb waitUntilCompleted];
	}
}

// A host that answers on the doorbell's thread: the doorbell write takes the buffer off the
// submission ring and posts its CB_SUBMITTED before it returns, as the host's post does (the
// record published, `armed` read and cleared), but the vector's wake of the completion thread
// comes only `lateMs` later, from a thread of its own, as an interrupt still on its way.
struct DoorbellAnswer {
	TestRing *submit = nullptr;
	TestRing *completion = nullptr;
	FakeKext *kext = nullptr;
	uint64_t cbObjId = 0;
	int lateMs = 0;
	std::atomic<uint64_t> seqno{0};
	std::atomic<int> answered{0};
	std::thread late;

	static void ring(void *context, uint32_t)
	{
		DoorbellAnswer *d = static_cast<DoorbellAnswer *>(context);
		if (d->submit == nullptr)
			return;
		RgfxMsgHeader h = {};
		std::vector<uint8_t> bytes;
		if (!d->submit->next(&h, &bytes) || h.type != RGFX_CMD_COMMAND_BUFFER ||
		    submittedOps(bytes) != 1)
			return;
		d->seqno.store(h.seqno);
		RgfxCbSubmitted submitted = cbSubmitted(h.seqno, d->cbObjId);
		CHECK(d->completion->postFromHost(&submitted, sizeof(submitted)));
		__atomic_thread_fence(__ATOMIC_SEQ_CST);
		uint32_t *armed = reinterpret_cast<uint32_t *>(
		    d->completion->mem.data() + __builtin_offsetof(RgfxRingHeader, armed));
		const bool raise = __atomic_exchange_n(armed, 0u, __ATOMIC_SEQ_CST) == 1;
		d->answered.fetch_add(1);
		FakeKext *kext = d->kext;
		const int ms = d->lateMs;
		d->late = std::thread([kext, ms, raise] {
			std::this_thread::sleep_for(std::chrono::milliseconds(ms));
			if (raise)
				kext->wake();
		});
	}
};

// A buffer the host answered on its doorbell's thread: commitAndWaitUntilSubmitted finds its
// CB_SUBMITTED on the ring after its post and returns, the buffer submitted and still Committed,
// without waiting for the completion thread's wake, which comes 400 ms later; the record is
// still on the ring at the return. The completion thread then takes it, and the buffer's
// scheduling and completion run as before. A buffer the doorbell does not answer waits for the
// thread as before.
void test_submitted_at_the_doorbell_wakes_no_thread()
{
	@autoreleasepool {
	DoorbellAnswer *door = new DoorbellAnswer();
	QueueFixture f(0x8000, 0, &DoorbellAnswer::ring, door);
	f.queue.asksSubmitted = YES;
	FakeKext *kext = new FakeKext();
	door->completion = &f.completion;
	door->kext = kext;
	door->lateMs = 400;
	RgfxCompletionThread thread(&FakeKext::wait, kext);
	CHECK(thread.start());
	std::shared_ptr<RgfxReplyReader> reader = [f.queue sharedReplyReader];
	thread.add(reader);
	CHECK(eventually([&] { return ringArmed(f.completion) == 1 && kext->waitCount() >= 1; }));

	id<MTLCommandBuffer> cb = [f.queue commandBuffer];
	door->cbObjId = ((RgfxResource *)cb).rgfxObjId;
	door->submit = &f.submit;
	auto start = std::chrono::steady_clock::now();
	CHECK([(id<MTLCommandBufferSPI>)cb commitAndWaitUntilSubmitted]);
	auto took = std::chrono::steady_clock::now() - start;
	const uint64_t seqno = door->seqno.load();
	CHECK(door->answered.load() == 1);
	CHECK(seqno != 0);
	CHECK(took < std::chrono::milliseconds(200));
	// Returned with the record still on the ring, untaken: the completion thread is asleep.
	CHECK(reader->awaitingSubmitted(seqno));
	CHECK(cb.status == MTLCommandBufferStatusCommitted);
	// The late wake: the thread takes the record, which finds the buffer submitted.
	door->late.join();
	CHECK(eventually([&] { return !reader->awaitingSubmitted(seqno); }));
	RgfxCbScheduled scheduled = cbScheduled(seqno, door->cbObjId);
	hostPost(f.completion, *kext, &scheduled, sizeof(scheduled));
	[cb waitUntilScheduled];
	CHECK(cb.status == MTLCommandBufferStatusScheduled);
	RgfxCbComplete done = cbComplete(seqno, door->cbObjId);
	hostPost(f.completion, *kext, &done, sizeof(done));
	[cb waitUntilCompleted];
	CHECK(cb.status == MTLCommandBufferStatusCompleted);

	// Not answered at its doorbell: the call returns at the CB_SUBMITTED the completion thread
	// takes.
	door->submit = nullptr;
	id<MTLCommandBuffer> second = [f.queue commandBuffer];
	uint64_t secondId = ((RgfxResource *)second).rgfxObjId;
	std::atomic<uint64_t> taken{0};
	std::atomic<int> asked{-1};
	std::thread host = hostAnswers(&f.submit, &f.completion, kext, secondId, RGFX_EVT_CB_SUBMITTED,
	                               &taken, &asked);
	start = std::chrono::steady_clock::now();
	CHECK([(id<MTLCommandBufferSPI>)second commitAndWaitUntilSubmitted]);
	CHECK(std::chrono::steady_clock::now() - start >= std::chrono::milliseconds(25));
	host.join();
	CHECK(asked.load() == 1);
	CHECK(!reader->awaitingSubmitted(taken.load()));
	RgfxCbComplete d2 = cbComplete(taken.load(), secondId);
	hostPost(f.completion, *kext, &d2, sizeof(d2));
	[second waitUntilCompleted];
	kext->close();
	CHECK(eventually([&] { return !thread.running(); }));
	}
}

// The reader's look for a buffer's submission reads the records not yet taken, takes none, and
// reads a CB_SCHEDULED or CB_COMPLETE as one: only its own seqno's, past the pads, and none of a
// closed ring.
void test_reply_reader_holds_submitted()
{
	TestRing ring(0x200);
	RgfxReplyReader reader(ring.base(), ring.dataSize);
	CHECK(!reader.holdsSubmitted(1));
	RgfxCbComplete other = cbComplete(1, 50);
	CHECK(ring.postFromHost(&other, sizeof(other)));
	CHECK(reader.holdsSubmitted(1));
	CHECK(!reader.holdsSubmitted(2));
	RgfxCbSubmitted submitted = cbSubmitted(2, 51);
	CHECK(ring.postFromHost(&submitted, sizeof(submitted)));
	CHECK(reader.holdsSubmitted(2));
	CHECK(!reader.holdsSubmitted(3));
	int fired = 0;
	reader.expectSubmitted(2, [&fired](const RgfxCbSubmitted &) { fired++; });
	CHECK(reader.awaitingSubmitted(2));
	// Taken by the pump, in order, the handler run there and not at the look.
	CHECK(fired == 0);
	CHECK(reader.pump() == 1);
	CHECK(fired == 1);
	CHECK(!reader.holdsSubmitted(2));
	// Past the ring's end: records taken until fewer bytes than a record's are left before it,
	// so the next record follows the host's pad there.
	for (;;) {
		const uint32_t left =
		    ring.dataSize - (static_cast<uint32_t>(ring.produced) & (ring.dataSize - 1));
		if (left < sizeof(RgfxCbScheduled))
			break;
		RgfxCbScheduled filler = cbScheduled(9, 59);
		CHECK(ring.postFromHost(&filler, sizeof(filler)));
		CHECK(reader.pump() == 0);
	}
	RgfxCbScheduled wrapped = cbScheduled(3, 52);
	CHECK(ring.postFromHost(&wrapped, sizeof(wrapped)));
	CHECK((ring.produced & (ring.dataSize - 1)) == sizeof(wrapped));
	CHECK(reader.holdsSubmitted(3));
	CHECK(!reader.holdsSubmitted(4));
	reader.close();
	CHECK(!reader.holdsSubmitted(3));
}

// Without a completion thread, waitUntilScheduled takes the ring itself and returns at the
// buffer's CB_SCHEDULED, before its completion.
void test_wait_until_scheduled_pumps()
{
	@autoreleasepool {
	QueueFixture f;
	id<MTLCommandBuffer> cb = [f.queue commandBuffer];
	[cb commit];
	uint64_t seqno = f.nextSeqno();
	FakeKext kext;
	RgfxCbScheduled scheduled = cbScheduled(seqno, ((RgfxResource *)cb).rgfxObjId);
	hostPost(f.completion, kext, &scheduled, sizeof(scheduled));
	[cb waitUntilScheduled];
	CHECK(cb.status == MTLCommandBufferStatusScheduled);
	CHECK([f.queue sharedReplyReader]->awaitingCompletion(seqno));
	RgfxCbComplete done = cbComplete(seqno, ((RgfxResource *)cb).rgfxObjId);
	hostPost(f.completion, kext, &done, sizeof(done));
	[cb waitUntilCompleted];
	CHECK(cb.status == MTLCommandBufferStatusCompleted);
	}
}

// Once the kext ends the wait (the process's context is gone) or the thread is
// destroyed, the readers go undriven, and waitUntilCompleted takes the ring itself.
void test_completion_thread_stops()
{
	@autoreleasepool {
	QueueFixture f;
	FakeKext *kext = new FakeKext();
	RgfxCompletionThread thread(&FakeKext::wait, kext);
	CHECK(thread.start());
	thread.add([f.queue sharedReplyReader]);
	CHECK([f.queue sharedReplyReader]->driven());
	kext->close();
	CHECK(eventually([&] { return !thread.running(); }));
	CHECK(![f.queue sharedReplyReader]->driven());

	__block int fired = 0;
	id<MTLCommandBuffer> cb = [f.queue commandBuffer];
	[cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
		(void)done;
		fired++;
	}];
	[cb commit];
	RgfxCbComplete done = cbComplete(f.nextSeqno(), ((RgfxResource *)cb).rgfxObjId);
	CHECK(f.completion.postFromHost(&done, sizeof(done)));
	[cb waitUntilCompleted]; // pumps: nothing else takes the ring
	CHECK(fired == 1);
	CHECK(cb.status == MTLCommandBufferStatusCompleted);

	// A thread destroyed while it sleeps lets its readers go at once.
	QueueFixture g;
	FakeKext *kext2 = new FakeKext();
	{
		RgfxCompletionThread doomed(&FakeKext::wait, kext2);
		CHECK(doomed.start());
		doomed.add([g.queue sharedReplyReader]);
		CHECK([g.queue sharedReplyReader]->driven());
		CHECK(eventually([&] { return kext2->waitCount() >= 1; }));
	}
	CHECK(![g.queue sharedReplyReader]->driven());
	kext2->wake(); // the detached thread wakes, sees it is stopping, and ends
	}
}

// A command buffer too large for half its queue's ring puts its whole COMMAND_BUFFER message
// in a region block, which the ring names with a COMMAND_BUFFER_IN_REGION whose seqno the
// completion carries; the block goes back as the buffer completes.
struct BlockLog {
	std::vector<uint64_t> allocated;
	std::vector<uint64_t> released;
	uint64_t next = RGFX_ALLOCATIONS_OFFSET + 0x10000;
	static bool alloc(void *c, uint64_t size, uint64_t align, uint64_t *offset)
	{
		BlockLog *log = static_cast<BlockLog *>(c);
		uint64_t at = (log->next + align - 1) & ~(align - 1);
		log->next = at + size;
		log->allocated.push_back(at);
		*offset = at;
		return true;
	}
	static void release(void *c, uint64_t offset)
	{
		static_cast<BlockLog *>(c)->released.push_back(offset);
	}
};

// A queue's pool of blocks for out-of-line command buffers keeps every block given back: a
// second stream as deep as the first makes none and gives none back to the region, a take reuses
// the smallest free block that holds its message, and the pool gives back all it holds as it goes.
void test_body_pool_keeps_every_block()
{
	std::vector<uint8_t> mem(RGFX_ALLOCATIONS_OFFSET + 0x4000000, 0);
	RgfxRegion region(mem.data(), mem.size());
	BlockLog blocks;
	region.setBlockSource(&BlockLog::alloc, &BlockLog::release, &blocks);
	constexpr size_t kDeep = 40; // far past the few a pool once kept
	{
		RgfxBodyPool pool(&region);
		for (int pass = 0; pass < 2; pass++) {
			std::vector<RgfxBodyBlock> out(kDeep);
			for (RgfxBodyBlock &b : out)
				CHECK(pool.take(RgfxBodyPool::kGranule, &b) && b.size == RgfxBodyPool::kGranule);
			for (const RgfxBodyBlock &b : out)
				pool.giveBack(b);
			CHECK(pool.made() == kDeep);
			CHECK(blocks.allocated.size() == kDeep && blocks.released.empty());
		}
		// A larger block, then a take of the granule: the smallest that holds it is reused.
		RgfxBodyBlock big = {};
		CHECK(pool.take(2 * RgfxBodyPool::kGranule, &big) && big.size == 2 * RgfxBodyPool::kGranule);
		CHECK(pool.made() == kDeep + 1);
		pool.giveBack(big);
		RgfxBodyBlock small = {};
		CHECK(pool.take(RgfxBodyPool::kGranule, &small) && small.size == RgfxBodyPool::kGranule);
		pool.giveBack(small);
		// With only the large one free of its size, a large take finds it.
		RgfxBodyBlock again = {};
		CHECK(pool.take(2 * RgfxBodyPool::kGranule, &again) && again.offset == big.offset);
		pool.giveBack(again);
		CHECK(pool.made() == kDeep + 1 && blocks.released.empty());
	}
	CHECK(blocks.released.size() == kDeep + 1);
}

void test_command_buffer_out_of_line()
{
	// Outside the pool: the queue's pool of blocks goes back into the region as the queue goes,
	// which the pool's drain may do.
	std::vector<uint8_t> mem(RGFX_ALLOCATIONS_OFFSET + 0x100000, 0);
	RgfxRegion region(mem.data(), mem.size());
	BlockLog blocks;
	region.setBlockSource(&BlockLog::alloc, &BlockLog::release, &blocks);
	@autoreleasepool {
	QueueFixture f(0x1000); // up to 2 KiB on the ring
	f.queue.region = &region;

	// A small buffer still rides the ring.
	id<MTLCommandBuffer> small = [f.queue commandBuffer];
	[small commit];
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(f.submit.next(&h, &bytes));
	CHECK(h.type == RGFX_CMD_COMMAND_BUFFER);
	CHECK(blocks.allocated.empty());
	RgfxCbComplete smallDone = cbComplete(h.seqno, ((RgfxResource *)small).rgfxObjId);
	CHECK(f.completion.postFromHost(&smallDone, sizeof(smallDone)));
	[small waitUntilCompleted];

	// 200 fills: far past half the ring.
	id<MTLCommandBuffer> cb = [f.queue commandBuffer];
	id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
	for (int i = 0; i < 200; i++)
		[blit fillBuffer:handle(40) range:NSMakeRange(4 * i, 4) value:(uint8_t)i];
	[blit endEncoding];
	__block int fired = 0;
	[cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
		(void)done;
		fired++;
	}];
	[cb commit];
	CHECK(f.submit.next(&h, &bytes));
	CHECK(h.type == RGFX_CMD_COMMAND_BUFFER_IN_REGION);
	CHECK(h.size == sizeof(RgfxCommandBufferInRegion));
	RgfxCommandBufferInRegion ref = {};
	std::memcpy(&ref, bytes.data(), sizeof(ref));
	// Built in the block once it outgrew the ring, after the room a commit's surface uses take.
	CHECK(blocks.allocated.size() == 1 && ref.region_offset > blocks.allocated[0] &&
	      ref.region_offset + ref.byte_length <= blocks.allocated[0] + RgfxBodyPool::kGranule);
	CHECK(ref.byte_length > 2048 && ref.byte_length % 8 == 0 && ref.region_offset % 8 == 0);

	// The block holds the whole message: a COMMAND_BUFFER of that length, seqno 0, its ops.
	RgfxCommandBuffer body = {};
	std::memcpy(&body, mem.data() + ref.region_offset, sizeof(body));
	CHECK(body.header.type == RGFX_CMD_COMMAND_BUFFER);
	CHECK(body.header.size == ref.byte_length);
	CHECK(body.header.seqno == 0);
	CHECK(body.cb_obj_id == ((RgfxResource *)cb).rgfxObjId);
	CHECK(body.op_bytes == ref.byte_length - RGFX_COMMAND_BUFFER_OPS_OFFSET);
	RgfxOpHeader first = {};
	std::memcpy(&first, mem.data() + ref.region_offset + RGFX_COMMAND_BUFFER_OPS_OFFSET,
	            sizeof(first));
	CHECK(first.tag == RGFX_OP_ENCODER_BEGIN_BLIT);

	// Its completion carries the reference's seqno; the block goes back to the queue's pool
	// then, still the process's, and the next buffer too large for the ring takes it again.
	CHECK(blocks.released.empty());
	RgfxCbComplete done = cbComplete(h.seqno, ((RgfxResource *)cb).rgfxObjId);
	CHECK(f.completion.postFromHost(&done, sizeof(done)));
	[cb waitUntilCompleted];
	CHECK(fired == 1);
	CHECK(cb.status == MTLCommandBufferStatusCompleted);
	CHECK(blocks.released.empty());
	id<MTLCommandBuffer> next = [f.queue commandBuffer];
	id<MTLBlitCommandEncoder> nextBlit = [next blitCommandEncoder];
	for (int i = 0; i < 200; i++)
		[nextBlit fillBuffer:handle(40) range:NSMakeRange(4 * i, 4) value:(uint8_t)i];
	[nextBlit endEncoding];
	[next commit];
	CHECK(f.submit.next(&h, &bytes));
	CHECK(h.type == RGFX_CMD_COMMAND_BUFFER_IN_REGION);
	RgfxCommandBufferInRegion nextRef = {};
	std::memcpy(&nextRef, bytes.data(), sizeof(nextRef));
	CHECK(nextRef.region_offset == ref.region_offset && blocks.allocated.size() == 1);
	RgfxCbComplete nextDone = cbComplete(h.seqno, ((RgfxResource *)next).rgfxObjId);
	CHECK(f.completion.postFromHost(&nextDone, sizeof(nextDone)));
	[next waitUntilCompleted];
	CHECK(next.status == MTLCommandBufferStatusCompleted && blocks.released.empty());

	// With no region, a buffer too large for the ring fails as out of memory, off the
	// committing thread, and never reaches the ring.
	QueueFixture g(0x1000);
	id<MTLCommandBuffer> lost = [g.queue commandBuffer];
	id<MTLBlitCommandEncoder> b2 = [lost blitCommandEncoder];
	for (int i = 0; i < 200; i++)
		[b2 fillBuffer:handle(40) range:NSMakeRange(4 * i, 4) value:1];
	[b2 endEncoding];
	dispatch_semaphore_t failed = dispatch_semaphore_create(0);
	__block int failedOnMain = -1;
	[lost addCompletedHandler:^(id<MTLCommandBuffer> d) {
		failedOnMain = pthread_main_np();
		(void)d;
		dispatch_semaphore_signal(failed);
	}];
	[lost commit];
	CHECK(dispatch_semaphore_wait(failed, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC)) == 0);
	CHECK(lost.status == MTLCommandBufferStatusError);
	CHECK(lost.error.code == MTLCommandBufferErrorOutOfMemory);
	CHECK(failedOnMain == 0);
	CHECK(!g.submit.next(&h, &bytes));
	f.queue.region = nullptr;
	}
}

// A buffer's completion is expected before the host can see the buffer: at the doorbell that
// publishing it rings, the reader already awaits its seqno.
struct ExpectAtDoorbell {
	TestRing *submit = nullptr;
	RgfxReplyReader *reader = nullptr;
	int rung = 0;
	bool awaited = false;
	uint64_t seqno = 0;
	static void ring(void *c, uint32_t)
	{
		ExpectAtDoorbell *e = static_cast<ExpectAtDoorbell *>(c);
		RgfxMsgHeader h = {};
		std::vector<uint8_t> bytes;
		if (e->submit->next(&h, &bytes)) {
			e->rung++;
			e->seqno = h.seqno;
			e->awaited = e->reader->awaitingCompletion(h.seqno);
		}
	}
};

void test_completion_expected_before_publish()
{
	@autoreleasepool {
	ExpectAtDoorbell e;
	QueueFixture f(0x8000, 0, &ExpectAtDoorbell::ring, &e);
	e.submit = &f.submit;
	e.reader = [f.queue sharedReplyReader].get();
	id<MTLCommandBuffer> cb = [f.queue commandBuffer];
	[cb commit];
	CHECK(e.rung == 1);
	CHECK(e.awaited);
	CHECK(cb.status == MTLCommandBufferStatusCommitted);
	RgfxCbComplete done = cbComplete(e.seqno, ((RgfxResource *)cb).rgfxObjId);
	CHECK(f.completion.postFromHost(&done, sizeof(done)));
	[cb waitUntilCompleted];
	}
}

// -commandBuffer blocks while maxCommandBufferCount buffers of the queue are alive and not
// completed, as Metal's does; a buffer that completes, or one released uncommitted, gives its
// slot back; and a completed handler that makes the next buffer does not wait for its own.
void test_command_buffer_back_pressure()
{
	@autoreleasepool {
	QueueFixture f(0x8000, 2);
	CHECK(f.queue.maxCommandBufferCount == 2);
	QueueFixture d;
	CHECK(d.queue.maxCommandBufferCount == 64);

	FakeKext *kext = new FakeKext();
	RgfxCompletionThread thread(&FakeKext::wait, kext);
	CHECK(thread.start());
	thread.add([f.queue sharedReplyReader]);

	id<MTLCommandBuffer> a = [f.queue commandBuffer];
	id<MTLCommandBuffer> b = [f.queue commandBuffer];
	[a commit];
	uint64_t sa = f.nextSeqno();
	[b commit];
	uint64_t sb = f.nextSeqno();

	// A third waits for a's completion, posted 50 ms from now.
	RgfxCommandQueue *queue = f.queue;
	std::atomic<bool> got{false};
	std::atomic<bool> *gotp = &got;
	id<MTLCommandBuffer> third = nil;
	id<MTLCommandBuffer> __strong *thirdp = &third;
	dispatch_semaphore_t made = dispatch_semaphore_create(0);
	std::thread maker([queue, gotp, made, thirdp] {
		@autoreleasepool {
			*thirdp = [queue commandBuffer];
			gotp->store(true);
			dispatch_semaphore_signal(made);
		}
	});
	std::this_thread::sleep_for(std::chrono::milliseconds(50));
	CHECK(!got.load()); // still blocked: two in flight
	RgfxCbComplete done = cbComplete(sa, ((RgfxResource *)a).rgfxObjId);
	hostPost(f.completion, *kext, &done, sizeof(done));
	CHECK(dispatch_semaphore_wait(made, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC)) == 0);
	maker.join();
	CHECK(got.load() && third != nil);

	// b and the third hold both slots. Released uncommitted, the third gives its slot back.
	third = nil;
	id<MTLCommandBuffer> fourth = nil;
	@autoreleasepool {
		fourth = [f.queue commandBuffer]; // did not block
	}
	CHECK(fourth != nil);
	RgfxCbComplete bDone = cbComplete(sb, ((RgfxResource *)b).rgfxObjId);
	hostPost(f.completion, *kext, &bDone, sizeof(bDone));
	[b waitUntilCompleted];
	kext->close();
	CHECK(eventually([&] { return !thread.running(); }));

	// With one slot, a completed handler that makes the next buffer has the slot back.
	QueueFixture g(0x8000, 1);
	id<MTLCommandBuffer> first = [g.queue commandBuffer];
	__block id<MTLCommandBuffer> next = nil;
	RgfxCommandQueue *gq = g.queue;
	[first addCompletedHandler:^(id<MTLCommandBuffer> x) {
		(void)x;
		next = [gq commandBuffer];
	}];
	[first commit];
	RgfxCbComplete d1 = cbComplete(g.nextSeqno(), ((RgfxResource *)first).rgfxObjId);
	CHECK(g.completion.postFromHost(&d1, sizeof(d1)));
	[first waitUntilCompleted]; // pumps: the handler runs here, and gets its buffer
	CHECK(next != nil);
	}
}

// A texture a command buffer names stays on the host while the buffer is in flight, though the
// app let it go, as Metal's buffer keeps what it encodes: its OBJECT_DESTROY comes after the
// buffer's handlers and before its waiter wakes, and its resource id goes to the next texture only
// after that. A texture the buffer never names goes at once meanwhile. A buffer made with
// unretained references keeps what it names too, though it answers retainedReferences NO.
void test_texture_destroy_waits_for_buffers()
{
	@autoreleasepool {
	QueueFixture f;
	(void)expectQueueCreate(f.control);
	int calls = 0;
	f.ids.setResourceIdSource(&oneChunk, &calls);
	MTLTextureDescriptor *desc =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:8
	                                                      height:8
	                                                   mipmapped:NO];
	auto texture = [&] {
		RgfxTexture *t = [RgfxTexture textureWithDescriptor:desc
		                                               heap:RGFX_OBJ_NONE
		                                         heapOffset:0
		                                             device:nil
		                                         serializer:&f.ser];
		CHECK(t != nil);
		(void)expectMessage<RgfxTextureCreate>(f.control, RGFX_CMD_TEXTURE_CREATE);
		return t;
	};
	uint64_t texObjId = 0, texResourceId = 0;
	id<MTLCommandBuffer> cb = [f.queue commandBuffer];
	CHECK(cb.retainedReferences);
	// Its handler, added before the commit as Metal requires, looks for the destroy at completion.
	__block bool destroyedBeforeHandler = true;
	TestRing *control = &f.control;
	[cb addCompletedHandler:^(id<MTLCommandBuffer> d) {
		(void)d;
		RgfxMsgHeader hh = {};
		std::vector<uint8_t> bb;
		destroyedBeforeHandler = control->next(&hh, &bb);
	}];
	@autoreleasepool {
		RgfxTexture *tex = texture();
		texObjId = tex.rgfxObjId;
		texResourceId = [tex gpuResourceID]._impl;
		id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
		[blit generateMipmapsForTexture:tex];
		[blit endEncoding];
		[cb commit];
	} // the app lets the texture go while the buffer is in flight
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(!f.control.next(&h, &bytes)); // no destroy yet
	// A texture made meanwhile gets another id: the first's is still the host's. It goes at once.
	uint64_t otherObjId = 0;
	@autoreleasepool {
		RgfxTexture *other = texture();
		otherObjId = other.rgfxObjId;
		CHECK([other gpuResourceID]._impl != texResourceId);
	}
	CHECK(expectMessage<RgfxObjectDestroy>(f.control, RGFX_CMD_OBJECT_DESTROY).obj_id ==
	      otherObjId);

	RgfxCbComplete done = cbComplete(f.nextSeqno(), ((RgfxResource *)cb).rgfxObjId);
	CHECK(f.completion.postFromHost(&done, sizeof(done)));
	[cb waitUntilCompleted];
	CHECK(!destroyedBeforeHandler);
	CHECK(expectMessage<RgfxObjectDestroy>(f.control, RGFX_CMD_OBJECT_DESTROY).obj_id == texObjId);
	CHECK(ringEmpty(f.control));

	// Unretained references: kept all the same.
	id<MTLCommandBuffer> unretained = [f.queue commandBufferWithUnretainedReferences];
	CHECK(!unretained.retainedReferences);
	@autoreleasepool {
		RgfxTexture *tex = texture();
		texObjId = tex.rgfxObjId;
		id<MTLBlitCommandEncoder> blit = [unretained blitCommandEncoder];
		[blit generateMipmapsForTexture:tex];
		[blit endEncoding];
		[unretained commit];
	}
	CHECK(ringEmpty(f.control));
	done = cbComplete(f.nextSeqno(), ((RgfxResource *)unretained).rgfxObjId);
	CHECK(f.completion.postFromHost(&done, sizeof(done)));
	[unretained waitUntilCompleted];
	CHECK(expectMessage<RgfxObjectDestroy>(f.control, RGFX_CMD_OBJECT_DESTROY).obj_id == texObjId);
	}
}

// A buffer whose completion never comes holds back only what it names: a buffer it fills stays
// on the host until the command buffer goes, with its queue's reader, while one it never names
// goes at once. Its completion group is left at its dealloc.
void test_uncompleted_buffer_releases()
{
	@autoreleasepool {
		QueueFixture f;
		(void)expectQueueCreate(f.control);
		uint64_t namedId = 0, otherId = 0;
		{
			RgfxReplyReader reader(f.completion.base(), f.completion.dataSize);
			f.queue.replyReader = &reader;
			@autoreleasepool {
				RgfxBuffer *named = [RgfxBuffer bufferWithLength:64
				                                         options:MTLResourceStorageModePrivate
				                                          device:nil
				                                      serializer:&f.ser];
				RgfxBuffer *other = [RgfxBuffer bufferWithLength:64
				                                         options:MTLResourceStorageModePrivate
				                                          device:nil
				                                      serializer:&f.ser];
				namedId = named.rgfxObjId;
				otherId = other.rgfxObjId;
				drainRing(f.control);
				id<MTLCommandBuffer> cb = [f.queue commandBuffer];
				id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
				[blit fillBuffer:named range:NSMakeRange(0, 64) value:1];
				[blit endEncoding];
				[cb commit];
				(void)f.nextSeqno();
				CHECK(cb.status == MTLCommandBufferStatusCommitted);
			}
			CHECK(nextDestroyed(f.control) == otherId && ringEmpty(f.control));
			f.queue.replyReader = nullptr;
		} // the reader goes, and with it its handler, the buffer's last hold
		CHECK(nextDestroyed(f.control) == namedId && ringEmpty(f.control));
	}
}

} // namespace

// The layout of a Managed texture's CPU copy, for the cases crates/reims-graphics/src/managed.rs
// holds too: a 2D RGBA8 texture with three levels, a BC1 one whose sides are no multiple of its
// blocks, a cube array, and a 3D one; and the blocks of a few formats.
void test_managed_layout()
{
	auto desc = [](uint32_t type, uint32_t w, uint32_t h, uint32_t d, uint32_t levels,
	               uint32_t array) {
		RgfxTextureDescriptor t = {};
		t.pixel_format = 70;
		t.texture_type = type;
		t.width = w;
		t.height = h;
		t.depth = d;
		t.mipmap_level_count = levels;
		t.array_length = array;
		t.sample_count = 1;
		return t;
	};
	auto level = [](const RgfxManagedLayout &l, uint32_t i, uint64_t offset, uint64_t row,
	                uint64_t image, uint64_t slice) {
		const RgfxManagedLevel &lv = l.level(i);
		return lv.offset == offset && lv.bytesPerRow == row && lv.bytesPerImage == image &&
		       lv.bytesPerSlice == slice;
	};
	RgfxFormatBlock rgba = {}, bc1 = {}, astc = {}, none = {};
	CHECK(rgfxPixelFormatBlock(MTLPixelFormatRGBA8Unorm, &rgba) && rgba.bytes == 4 &&
	      rgba.width == 1 && rgba.linear);
	CHECK(rgfxPixelFormatBlock(MTLPixelFormatBC1_RGBA, &bc1) && bc1.bytes == 8 &&
	      bc1.width == 4 && bc1.height == 4 && !bc1.linear);
	CHECK(rgfxPixelFormatBlock(MTLPixelFormatASTC_10x8_LDR, &astc) && astc.bytes == 16 &&
	      astc.width == 10 && astc.height == 8);
	CHECK(!rgfxPixelFormatBlock(MTLPixelFormatDepth32Float, &none));
	CHECK(!rgfxPixelFormatBlock(191, &none));

	RgfxManagedLayout l;
	CHECK(l.init(desc(2, 100, 60, 1, 3, 1), rgba));
	CHECK(level(l, 0, 0, 512, 30720, 30720) && level(l, 1, 30720, 256, 7680, 7680) &&
	      level(l, 2, 38400, 256, 3840, 3840));
	CHECK(l.size() == 42240 && l.slices() == 1);
	CHECK(l.init(desc(2, 30, 30, 1, 2, 1), bc1));
	CHECK(level(l, 0, 0, 256, 2048, 2048) && level(l, 1, 2048, 256, 1024, 1024));
	uint64_t at = 0;
	const uint64_t bcOrigin[3] = {4, 8, 0}, bcSize[3] = {26, 22, 1}, offBlock[3] = {2, 0, 0};
	CHECK(l.region(0, 0, bcOrigin, bcSize, &at) && at == 512 + 8);
	CHECK(!l.region(0, 0, offBlock, bcSize, &at));
	CHECK(l.init(desc(6, 16, 16, 1, 2, 2), rgba));
	CHECK(l.slices() == 12 && level(l, 0, 0, 256, 4096, 4096) &&
	      level(l, 1, 49152, 256, 2048, 2048) && l.size() == 49152 + 12 * 2048);
	const uint64_t o[3] = {2, 1, 0}, sz[3] = {4, 3, 1};
	CHECK(l.region(1, 3, o, sz, &at) && at == 49152 + 3 * 2048 + 256 + 8);
	CHECK(!l.region(1, 12, o, sz, &at) && !l.region(2, 0, o, sz, &at));
	CHECK(l.init(desc(7, 8, 8, 4, 2, 1), rgba));
	CHECK(level(l, 0, 0, 256, 2048, 8192) && level(l, 1, 8192, 256, 1024, 2048));
	CHECK(l.init(desc(2, 8, 8, 4, 1, 1), rgba) && l.size() == 2048);
	CHECK(!l.init(desc(2, 8, 8, 1, 5, 1), rgba)); // more levels than 8 halves to
	CHECK(!l.init(desc(2, 0, 8, 1, 1, 1), rgba));
}

// A Managed buffer's contents are a region block: didModifyRange: tells the host the range
// (BUFFER_DID_MODIFY_RANGE) and a Shared buffer's posts nothing.
void test_managed_buffer_range()
{
	Fixture fx;
	std::vector<uint8_t> mem(0x800000, 0);
	RgfxRegion region(mem.data(), mem.size());
	RgfxRegion::Block block = region.allocate(0x1000);
	CHECK(block.ok);
	const MTLResourceOptions managed = MTLResourceStorageModeManaged;
	RgfxBuffer *buf = [RgfxBuffer
	    bufferOverHostMemoryWithContents:const_cast<uint8_t *>(block.ptr)
	                        regionOffset:block.offset
	                              length:0x1000
	                             options:managed
	                         deallocator:nil
	                              device:nil
	                          serializer:&fx.ser];
	CHECK(buf != nil && [buf contents] == block.ptr);
	CHECK([buf storageMode] == MTLStorageModeManaged &&
	      [buf cpuCacheMode] == MTLCPUCacheModeDefaultCache &&
	      [buf hazardTrackingMode] == MTLHazardTrackingModeTracked);
	auto create =
	    expectMessage<RgfxBufferOverHostMemory>(fx.ring, RGFX_CMD_BUFFER_OVER_HOST_MEMORY);
	CHECK(create.options == static_cast<uint32_t>(managed));
	CHECK(create.region_offset == block.offset);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
	[buf didModifyRange:NSMakeRange(0x40, 0x20)];
	[buf didModifyRange:NSMakeRange(0x40, 0)];
#pragma clang diagnostic pop
	auto range =
	    expectMessage<RgfxBufferDidModifyRange>(fx.ring, RGFX_CMD_BUFFER_DID_MODIFY_RANGE);
	CHECK(range.obj_id == buf.rgfxObjId && range.location == 0x40 && range.length == 0x20);
	RgfxMsgHeader extra = {};
	std::vector<uint8_t> extraBytes;
	CHECK(!fx.ring.next(&extra, &extraBytes)); // an empty range posts nothing

	RgfxBuffer *shared = [RgfxBuffer
	    bufferOverHostMemoryWithContents:const_cast<uint8_t *>(block.ptr)
	                        regionOffset:block.offset
	                              length:0x1000
	                             options:MTLResourceStorageModeShared
	                         deallocator:nil
	                              device:nil
	                          serializer:&fx.ser];
	expectMessage<RgfxBufferOverHostMemory>(fx.ring, RGFX_CMD_BUFFER_OVER_HOST_MEMORY);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
	[shared didModifyRange:NSMakeRange(0, 0x10)];
#pragma clang diagnostic pop
	CHECK(!fx.ring.next(&extra, &extraBytes));
}

// replaceRegion: of a texture over a Managed buffer, the plain form's and the linear form's,
// writes the rows in the buffer's bytes and tells the host their span (BUFFER_DID_MODIFY_RANGE,
// from the first row's first byte written to the last row's last), as Metal's reaches the GPU
// with no didModifyRange: of the app's (the AMD 5300M's two copies, `oracle_storage AMD lintex
// managed`); a no-copy buffer's block takes the app's bytes first. getBytes: posts nothing, and
// nor does either over a Shared buffer.
void test_buffer_texture_replace_region_tells_the_host()
{
	Fixture fx;
	std::vector<uint8_t> mem(0x800000, 0);
	RgfxRegion region(mem.data(), mem.size());
	// A no-copy buffer's GPU copy is a range of a slab of the region's.
	RgfxRegionSlabSource slabSource(region, fx.ser);
	RgfxSlabs slabs(slabSource);
	const NSUInteger length = 0x2000, offset = 0x100, pitch = 0x100;
	MTLTextureDescriptor *desc =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:8
	                                                      height:4
	                                                   mipmapped:NO];
	desc.usage = MTLTextureUsageShaderRead;
	uint8_t in[3 * 4 * 2];
	for (size_t i = 0; i < sizeof in; i++)
		in[i] = static_cast<uint8_t>(0xa0 + i);
	// 3 x 2 texels at (2, 1), rows 12 bytes apart in `in`.
	const MTLRegion written = MTLRegionMake2D(2, 1, 3, 2);
	const uint64_t first = offset + 1 * pitch + 2 * 4, span = pitch + 3 * 4;
	auto rowsLanded = [&](const uint8_t *bytes) {
		return std::memcmp(bytes + first, in, 12) == 0 &&
		       std::memcmp(bytes + first + pitch, in + 12, 12) == 0 &&
		       bytes[first - 1] == 0 && bytes[first + 12] == 0;
	};
	auto overBuffer = [&](RgfxBuffer *buf, bool linear) -> id<MTLTexture> {
		desc.storageMode = buf.storageMode;
		id<RgfxMetalBufferInternals> spi = (id<RgfxMetalBufferInternals>)buf;
		id<MTLTexture> tex = linear ? [spi newLinearTextureWithDescriptor:desc
		                                                           offset:offset
		                                                      bytesPerRow:pitch
		                                                    bytesPerImage:pitch * 4]
		                            : [buf newTextureWithDescriptor:desc
		                                                     offset:offset
		                                                bytesPerRow:pitch];
		CHECK(tex != nil && tex.buffer == buf && tex.storageMode == buf.storageMode);
		drainRing(fx.ring);
		return tex;
	};
	for (const bool linear : {false, true}) {
		@autoreleasepool {
			RgfxRegion::Block block = region.allocate(length);
			CHECK(block.ok);
			std::memset(const_cast<uint8_t *>(block.ptr), 0, length);
			RgfxBuffer *managed = [RgfxBuffer
			    bufferOverHostMemoryWithContents:const_cast<uint8_t *>(block.ptr)
			                        regionOffset:block.offset
			                              length:length
			                             options:MTLResourceStorageModeManaged
			                         deallocator:nil
			                              device:nil
			                          serializer:&fx.ser];
			CHECK(managed != nil);
			drainRing(fx.ring);
			id<MTLTexture> tex = overBuffer(managed, linear);
			[tex replaceRegion:written mipmapLevel:0 withBytes:in bytesPerRow:12];
			CHECK(rowsLanded(const_cast<const uint8_t *>(block.ptr)));
			auto range =
			    expectMessage<RgfxBufferDidModifyRange>(fx.ring, RGFX_CMD_BUFFER_DID_MODIFY_RANGE);
			CHECK(range.obj_id == managed.rgfxObjId && range.location == first &&
			      range.length == span);
			CHECK(ringEmpty(fx.ring));
			uint8_t out[sizeof in] = {};
			[tex getBytes:out bytesPerRow:12 fromRegion:written mipmapLevel:0];
			CHECK(std::memcmp(out, in, sizeof in) == 0 && ringEmpty(fx.ring));

			RgfxBuffer *shared = [RgfxBuffer
			    bufferOverHostMemoryWithContents:const_cast<uint8_t *>(block.ptr)
			                        regionOffset:block.offset
			                              length:length
			                             options:MTLResourceStorageModeShared
			                         deallocator:nil
			                              device:nil
			                          serializer:&fx.ser];
			drainRing(fx.ring);
			std::memset(const_cast<uint8_t *>(block.ptr), 0, length);
			id<MTLTexture> overShared = overBuffer(shared, linear);
			[overShared replaceRegion:written mipmapLevel:0 withBytes:in bytesPerRow:12];
			CHECK(rowsLanded(const_cast<const uint8_t *>(block.ptr)) && ringEmpty(fx.ring));
		}
		drainRing(fx.ring);
	}

	// A no-copy Managed buffer: its contents are the app's memory, which its block takes before
	// the host is told.
	@autoreleasepool {
		std::vector<uint8_t> app(length + 0x20, 0);
		uint8_t *p = app.data() + 0x13;
		RgfxBuffer *noCopy = [RgfxBuffer bufferWithBytesNoCopy:p
		                                                length:length
		                                               options:MTLResourceStorageModeManaged
		                                           deallocator:nil
		                                                region:&region
		                                                 slabs:&slabs
		                                                device:nil
		                                            serializer:&fx.ser];
		CHECK(noCopy != nil && [noCopy contents] == p && [noCopy rgfxShadow] != nullptr);
		auto made =
		    expectMessage<RgfxBufferOverHostMemory>(fx.ring, RGFX_CMD_BUFFER_OVER_HOST_MEMORY);
		drainRing(fx.ring);
		id<MTLTexture> tex = overBuffer(noCopy, false);
		[tex replaceRegion:written mipmapLevel:0 withBytes:in bytesPerRow:12];
		CHECK(rowsLanded(p) && rowsLanded(&mem[made.region_offset]));
		auto range =
		    expectMessage<RgfxBufferDidModifyRange>(fx.ring, RGFX_CMD_BUFFER_DID_MODIFY_RANGE);
		CHECK(range.obj_id == noCopy.rgfxObjId && range.location == first && range.length == span);
		CHECK(ringEmpty(fx.ring));
	}
	drainRing(fx.ring);
}

// replaceRegion: and getBytes: of each slice of a linear 2D array over a buffer reach that slice's
// rows in the buffer, each slice its rows' bytes rounded up to 128 after the one before, whatever
// the bytesPerImage the texture was made with, as the host lays the array out; through a view of
// one of its slices too, where a view of more than one is nil, sending nothing (Metal asserts at
// it). Over a Managed buffer, the host is told the slice's span. A slice past the array copies
// nothing.
void test_buffer_texture_slices()
{
	Fixture fx;
	std::vector<uint8_t> mem(0x800000, 0);
	RgfxRegion region(mem.data(), mem.size());
	// 4 x 3 BGRA8 slices, rows 32 bytes apart from offset 128: each slice's 96 bytes of rows take
	// 128.
	const NSUInteger length = 0x1000, offset = 128, pitch = 32, slicePitch = 128;
	MTLTextureDescriptor *desc =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:4
	                                                      height:3
	                                                   mipmapped:NO];
	desc.textureType = MTLTextureType2DArray;
	desc.arrayLength = 3;
	desc.usage = MTLTextureUsageShaderRead;
	RgfxRegion::Block block = region.allocate(length);
	CHECK(block.ok);
	uint8_t *bytes = const_cast<uint8_t *>(block.ptr);
	std::memset(bytes, 0, length);
	RgfxBuffer *managed = [RgfxBuffer bufferOverHostMemoryWithContents:bytes
	                                                      regionOffset:block.offset
	                                                            length:length
	                                                           options:MTLResourceStorageModeManaged
	                                                       deallocator:nil
	                                                            device:nil
	                                                        serializer:&fx.ser];
	CHECK(managed != nil);
	desc.storageMode = MTLStorageModeManaged;
	id<MTLTexture> tex = [(id<RgfxMetalBufferInternals>)managed newLinearTextureWithDescriptor:desc
	                                                                                  offset:offset
	                                                                             bytesPerRow:pitch
	                                                                           bytesPerImage:0x400];
	CHECK(tex != nil && tex.arrayLength == 3);
	drainRing(fx.ring);

	// 2 x 2 texels at (1, 1) of each slice, rows 8 bytes apart in `in`: slice s holds 0x10 * s.
	const MTLRegion written = MTLRegionMake2D(1, 1, 2, 2);
	uint8_t in[16];
	for (NSUInteger s = 0; s < 3; s++) {
		for (size_t i = 0; i < sizeof in; i++)
			in[i] = static_cast<uint8_t>(0x10 * s + i);
		[tex replaceRegion:written
		       mipmapLevel:0
		             slice:s
		         withBytes:in
		       bytesPerRow:8
		     bytesPerImage:16];
		const NSUInteger first = offset + s * slicePitch + pitch + 4;
		CHECK(std::memcmp(bytes + first, in, 8) == 0);
		CHECK(std::memcmp(bytes + first + pitch, in + 8, 8) == 0);
		CHECK(bytes[first - 1] == 0 && bytes[first + 8] == 0);
		auto range =
		    expectMessage<RgfxBufferDidModifyRange>(fx.ring, RGFX_CMD_BUFFER_DID_MODIFY_RANGE);
		CHECK(range.obj_id == managed.rgfxObjId && range.location == first &&
		      range.length == pitch + 8);
		CHECK(ringEmpty(fx.ring));
		uint8_t out[16] = {};
		[tex getBytes:out
		      bytesPerRow:8
		    bytesPerImage:16
		       fromRegion:written
		      mipmapLevel:0
		            slice:s];
		CHECK(std::memcmp(out, in, sizeof in) == 0);
	}
	// Nothing lies between the slices' rows.
	CHECK(bytes[offset + 3 * pitch] == 0 && bytes[offset + slicePitch - 1] == 0);

	// A view of slices 1 and 2, or of all three, is nil and sends nothing.
	CHECK([tex newTextureViewWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                             textureType:MTLTextureType2DArray
	                                  levels:NSMakeRange(0, 1)
	                                  slices:NSMakeRange(1, 2)] == nil);
	CHECK([tex newTextureViewWithPixelFormat:MTLPixelFormatBGRA8Unorm] == nil);
	CHECK(ringEmpty(fx.ring));
	// A view of slice 2 alone: its slice 0 is the array's slice 2.
	id<MTLTexture> view = [tex newTextureViewWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                             textureType:MTLTextureType2D
	                                                  levels:NSMakeRange(0, 1)
	                                                  slices:NSMakeRange(2, 1)];
	CHECK(view != nil);
	drainRing(fx.ring);
	uint8_t out[16] = {};
	[view getBytes:out
	      bytesPerRow:8
	    bytesPerImage:16
	       fromRegion:written
	      mipmapLevel:0
	            slice:0];
	CHECK(std::memcmp(out, in, sizeof in) == 0); // slice 2's, written last
	std::memset(in, 0xee, sizeof in);
	[view replaceRegion:written mipmapLevel:0 slice:0 withBytes:in bytesPerRow:8 bytesPerImage:16];
	CHECK(bytes[offset + 2 * slicePitch + pitch + 4] == 0xee);
	drainRing(fx.ring);

	// Past the array: nothing is copied or posted.
	std::vector<uint8_t> before(bytes, bytes + length);
	[tex replaceRegion:written mipmapLevel:0 slice:3 withBytes:in bytesPerRow:8 bytesPerImage:16];
	CHECK(std::memcmp(bytes, before.data(), length) == 0 && ringEmpty(fx.ring));
	std::memset(out, 0x55, sizeof out);
	[tex getBytes:out bytesPerRow:8 bytesPerImage:16 fromRegion:written mipmapLevel:0 slice:3];
	CHECK(out[0] == 0x55 && out[15] == 0x55);
	drainRing(fx.ring);
}

// A device that answers only its linear texture alignment: 256 bytes, as the AMD 5300M's and the
// lane's.
@interface RgfxLinearAlignmentDevice : NSObject
- (NSUInteger)minimumLinearTextureAlignmentForPixelFormat:(MTLPixelFormat)format;
@end
@implementation RgfxLinearAlignmentDevice
- (NSUInteger)minimumLinearTextureAlignmentForPixelFormat:(MTLPixelFormat)format
{
	return 256;
}
@end

// A 2D texture over a buffer, by the public form and the linear SPI, over a Shared and a Managed
// buffer, is made at offsets and pitches that are multiples of the device's linear texture
// alignment and is nil, posting nothing, off them, as the AMD 5300M answers (`oracle_storage AMD
// linalign`: 468 x 36 BGRA8 at offsets 0, 0x40 and 0x100 and pitches 1872, 1920 and 2048). A
// texture buffer takes no part of the rule, nor the tiled SPI, which takes any offset and pitch.
void test_buffer_texture_linear_alignment()
{
	Fixture fx;
	std::vector<uint8_t> mem(0x800000, 0);
	RgfxRegion region(mem.data(), mem.size());
	id<MTLDevice> device = (id<MTLDevice>)[RgfxLinearAlignmentDevice new];
	MTLTextureDescriptor *desc =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:468
	                                                      height:36
	                                                   mipmapped:NO];
	desc.usage = MTLTextureUsageShaderRead;
	const NSUInteger offsets[] = {0, 0x40, 0x100}, pitches[] = {1872, 1920, 2048};
	RgfxRegion::Block block = region.allocate(0x12000);
	CHECK(block.ok);
	for (const MTLResourceOptions storage :
	     {MTLResourceStorageModeShared, MTLResourceStorageModeManaged}) {
		@autoreleasepool {
			RgfxBuffer *buf = [RgfxBuffer
			    bufferOverHostMemoryWithContents:const_cast<uint8_t *>(block.ptr)
			                        regionOffset:block.offset
			                              length:0x12000
			                             options:storage
			                         deallocator:nil
			                              device:device
			                          serializer:&fx.ser];
			CHECK(buf != nil);
			drainRing(fx.ring);
			desc.storageMode = buf.storageMode;
			id<RgfxMetalBufferInternals> spi = (id<RgfxMetalBufferInternals>)buf;
			for (const NSUInteger offset : offsets) {
				for (const NSUInteger pitch : pitches) {
					drainRing(fx.ring); // the last place's textures' destroys
					const bool aligned = offset % 256 == 0 && pitch % 256 == 0;
					id<MTLTexture> plain = [buf newTextureWithDescriptor:desc
					                                             offset:offset
					                                        bytesPerRow:pitch];
					CHECK((plain != nil) == aligned);
					if (aligned) {
						CHECK(expectMessage<RgfxTextureFromBuffer>(fx.ring,
						                                           RGFX_CMD_TEXTURE_FROM_BUFFER)
						          .bytes_per_row == pitch);
					}
					id<MTLTexture> linear = [spi newLinearTextureWithDescriptor:desc
					                                                     offset:offset
					                                                bytesPerRow:pitch
					                                              bytesPerImage:pitch * 36];
					CHECK((linear != nil) == aligned);
					if (aligned) {
						CHECK(expectMessage<RgfxLinearTextureFromBuffer>(
						          fx.ring, RGFX_CMD_LINEAR_TEXTURE_FROM_BUFFER)
						          .offset == offset);
					}
					CHECK(ringEmpty(fx.ring));
				}
			}
			if (storage == MTLResourceStorageModeShared) {
				// The tiled SPI: a Managed descriptor over the Shared buffer, at any place.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
				desc.storageMode = MTLStorageModeManaged;
#pragma clang diagnostic pop
				drainRing(fx.ring);
				id<MTLTexture> tiled = [spi newTiledTextureWithDescriptor:desc
				                                                   offset:0x40
				                                              bytesPerRow:1920];
				CHECK(tiled != nil);
				CHECK(expectMessage<RgfxTextureFromBuffer>(fx.ring, RGFX_CMD_TEXTURE_FROM_BUFFER)
				          .bytes_per_row == 1920);
			}
			// A texture buffer of 64 texels at offset 0x40.
			MTLTextureDescriptor *tb =
			    [MTLTextureDescriptor textureBufferDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
			                                                           width:64
			                                                 resourceOptions:storage
			                                                           usage:MTLTextureUsageShaderRead];
			CHECK([spi newLinearTextureWithDescriptor:tb offset:0x40 bytesPerRow:256 bytesPerImage:0] !=
			      nil);
			drainRing(fx.ring);
		}
		drainRing(fx.ring);
	}
}

// didModifyData of a texture over a Managed buffer, the plain form's, the linear form's and a view's,
// tells the host the whole buffer (BUFFER_DID_MODIFY_RANGE of all of it), as Metal's uploads it to
// the GPU's copy (the AMD 5300M's two copies, `oracle_storage AMD didmodify length` and `nocopy`);
// a no-copy buffer's block takes the app's bytes first. Over a Shared buffer it posts nothing.
void test_buffer_texture_did_modify_data_tells_the_host()
{
	Fixture fx;
	std::vector<uint8_t> mem(0x800000, 0);
	RgfxRegion region(mem.data(), mem.size());
	// A no-copy buffer's GPU copy is a range of a slab of the region's.
	RgfxRegionSlabSource slabSource(region, fx.ser);
	RgfxSlabs slabs(slabSource);
	const NSUInteger length = 0x2000, offset = 0x100, pitch = 0x100;
	MTLTextureDescriptor *desc =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:8
	                                                      height:4
	                                                   mipmapped:NO];
	desc.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
	auto overBuffer = [&](RgfxBuffer *buf, bool linear) -> id<MTLTexture> {
		desc.storageMode = buf.storageMode;
		id<RgfxMetalBufferInternals> spi = (id<RgfxMetalBufferInternals>)buf;
		id<MTLTexture> tex = linear ? [spi newLinearTextureWithDescriptor:desc
		                                                           offset:offset
		                                                      bytesPerRow:pitch
		                                                    bytesPerImage:pitch * 4]
		                            : [buf newTextureWithDescriptor:desc
		                                                     offset:offset
		                                                bytesPerRow:pitch];
		CHECK(tex != nil && tex.buffer == buf);
		drainRing(fx.ring);
		return tex;
	};
	auto modified = [&](id<MTLTexture> tex) {
		[(id<RgfxMetalTextureInternals>)tex didModifyData];
	};
	auto expectWhole = [&](RgfxBuffer *buf) {
		auto range =
		    expectMessage<RgfxBufferDidModifyRange>(fx.ring, RGFX_CMD_BUFFER_DID_MODIFY_RANGE);
		CHECK(range.obj_id == buf.rgfxObjId && range.location == 0 && range.length == length);
		CHECK(ringEmpty(fx.ring));
	};
	for (const bool linear : {false, true}) {
		@autoreleasepool {
			RgfxRegion::Block block = region.allocate(length);
			CHECK(block.ok);
			auto over = [&](MTLResourceOptions storage) {
				RgfxBuffer *b = [RgfxBuffer
				    bufferOverHostMemoryWithContents:const_cast<uint8_t *>(block.ptr)
				                        regionOffset:block.offset
				                              length:length
				                             options:storage
				                         deallocator:nil
				                              device:nil
				                          serializer:&fx.ser];
				CHECK(b != nil);
				drainRing(fx.ring);
				return b;
			};
			RgfxBuffer *managed = over(MTLResourceStorageModeManaged);
			id<MTLTexture> tex = overBuffer(managed, linear);
			modified(tex);
			expectWhole(managed);
			id<MTLTexture> view = [tex newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm];
			CHECK(view != nil);
			drainRing(fx.ring);
			modified(view);
			expectWhole(managed);

			RgfxBuffer *shared = over(MTLResourceStorageModeShared);
			id<MTLTexture> overShared = overBuffer(shared, linear);
			modified(overShared);
			CHECK(ringEmpty(fx.ring));
		}
		drainRing(fx.ring);
	}

	// A no-copy Managed buffer: its contents are the app's memory, which its block takes before
	// the host is told.
	@autoreleasepool {
		std::vector<uint8_t> app(length + 0x20, 0);
		uint8_t *p = app.data() + 0x13;
		RgfxBuffer *noCopy = [RgfxBuffer bufferWithBytesNoCopy:p
		                                                length:length
		                                               options:MTLResourceStorageModeManaged
		                                           deallocator:nil
		                                                region:&region
		                                                 slabs:&slabs
		                                                device:nil
		                                            serializer:&fx.ser];
		CHECK(noCopy != nil && [noCopy contents] == p);
		auto made =
		    expectMessage<RgfxBufferOverHostMemory>(fx.ring, RGFX_CMD_BUFFER_OVER_HOST_MEMORY);
		drainRing(fx.ring);
		id<MTLTexture> tex = overBuffer(noCopy, true);
		p[offset + 5] = 0xd7;
		modified(tex);
		CHECK(mem[made.region_offset + offset + 5] == 0xd7);
		expectWhole(noCopy);
	}
	drainRing(fx.ring);
}

// A Managed texture's CPU copy is a block of the region, made over a Shared buffer and named by
// its MANAGED_TEXTURE_CREATE: replaceRegion: writes it at the layout's rows and posts the region
// (TEXTURE_REPLACE_REGION); getBytes: reads it back with no message; a region past the level
// writes nothing and posts nothing. A format no CPU copy holds makes no Managed texture.
void test_managed_texture_cpu_copy()
{
	Fixture fx;
	std::vector<uint8_t> mem(0x800000, 0xff); // pre-dirtied, as BAR memory
	RgfxRegion region(mem.data(), mem.size());
	@autoreleasepool {
		MTLTextureDescriptor *desc =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
		                                                       width:100
		                                                      height:60
		                                                   mipmapped:YES];
		desc.mipmapLevelCount = 3;
		desc.storageMode = MTLStorageModeManaged;
		RgfxTexture *tex = [RgfxTexture managedTextureWithDescriptor:desc
		                                                      region:&region
		                                                       slabs:nullptr
		                                                      device:nil
		                                                  serializer:&fx.ser];
		CHECK(tex != nil);
		CHECK([tex storageMode] == MTLStorageModeManaged);
		auto cpu =
		    expectMessage<RgfxBufferOverHostMemory>(fx.ring, RGFX_CMD_BUFFER_OVER_HOST_MEMORY);
		CHECK(cpu.length == 42240 && cpu.options == MTLResourceStorageModeShared);
		auto create = expectMessage<RgfxManagedTextureCreate>(fx.ring,
		                                                      RGFX_CMD_MANAGED_TEXTURE_CREATE);
		CHECK(create.obj_id == tex.rgfxObjId && create.cpu_buffer_obj_id == cpu.obj_id);
		CHECK(create.resource_id == [tex gpuResourceID]._impl && create.gpu_address != 0);
		CHECK(create.bytes_per_block == 4 && create.block_width == 1 &&
		      create.block_height == 1 && create.flags == RGFX_MANAGED_TEXTURE_LINEAR);
		CHECK(create.desc.mipmap_level_count == 3 &&
		      create.desc.options == static_cast<uint32_t>(desc.resourceOptions));
		uint8_t *copy = mem.data() + cpu.region_offset;
		CHECK(copy[0] == 0 && copy[42239] == 0); // a new Managed texture reads zero

		// Level 1, a 4x3 region at (2, 1): its rows at the layout's 256-byte pitch.
		uint8_t pixels[3][16];
		for (int r = 0; r < 3; r++)
			for (int i = 0; i < 16; i++)
				pixels[r][i] = static_cast<uint8_t>(r * 16 + i + 1);
		[tex replaceRegion:MTLRegionMake2D(2, 1, 4, 3)
		       mipmapLevel:1
		         withBytes:pixels
		       bytesPerRow:16];
		for (int r = 0; r < 3; r++)
			CHECK(std::memcmp(copy + 30720 + (1 + r) * 256 + 8, pixels[r], 16) == 0);
		CHECK(copy[30720 + 256 + 7] == 0 && copy[30720 + 256 + 24] == 0);
		auto replaced = expectMessage<RgfxTextureReplaceRegion>(fx.ring,
		                                                        RGFX_CMD_TEXTURE_REPLACE_REGION);
		CHECK(replaced.obj_id == tex.rgfxObjId && replaced.level == 1 && replaced.slice == 0);
		CHECK(replaced.origin[0] == 2 && replaced.origin[1] == 1 && replaced.origin[2] == 0);
		CHECK(replaced.size[0] == 4 && replaced.size[1] == 3 && replaced.size[2] == 1);

		// getBytes: reads the copy, here as the GPU's synchronize would have left it.
		copy[30720 + 2 * 256 + 16] = 0xab;
		uint8_t back[2][8] = {};
		[tex getBytes:back bytesPerRow:8 fromRegion:MTLRegionMake2D(3, 1, 2, 2) mipmapLevel:1];
		CHECK(back[0][0] == pixels[0][4] && back[1][0] == pixels[1][4] && back[1][4] == 0xab);
		RgfxMsgHeader extra = {};
		std::vector<uint8_t> extraBytes;
		CHECK(!fx.ring.next(&extra, &extraBytes)); // getBytes: posts nothing

		// A region past the level writes nothing and posts nothing.
		const uint8_t before = copy[30720 + 49 * 4];
		[tex replaceRegion:MTLRegionMake2D(49, 0, 2, 1)
		       mipmapLevel:1
		         withBytes:pixels
		       bytesPerRow:16];
		CHECK(copy[30720 + 49 * 4] == before);
		CHECK(!fx.ring.next(&extra, &extraBytes));

		// A depth texture keeps no CPU copy.
		MTLTextureDescriptor *depth =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
		                                                       width:64
		                                                      height:64
		                                                   mipmapped:NO];
		depth.storageMode = MTLStorageModeManaged;
		CHECK([RgfxTexture managedTextureWithDescriptor:depth
		                                         region:&region
		                                          slabs:nullptr
		                                         device:nil
		                                     serializer:&fx.ser] == nil);
		CHECK(!fx.ring.next(&extra, &extraBytes));
	}
}

// A device that answers whether its GPU shares the CPU's memory, and makes no queue and no
// buffer: a CPU copy through it (rgfxCopyRegion:) asks for its queue, counted here, and copies
// nothing.
@interface RgfxUnifiedAnswerDevice : NSObject
@property(nonatomic) BOOL unified;
@property(nonatomic) NSUInteger queuesAsked;
- (BOOL)hasUnifiedMemory;
- (id<MTLCommandQueue>)newCommandQueue;
- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options;
@end
@implementation RgfxUnifiedAnswerDevice
- (BOOL)hasUnifiedMemory
{
	return _unified;
}
- (id<MTLCommandQueue>)newCommandQueue
{
	_queuesAsked++;
	return nil;
}
- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options
{
	return nil;
}
@end

// On a GPU that shares the CPU's memory, Metal's CPU reads of a Managed texture see the GPU's
// writes with no synchronize (the Intel 630 and the M3, dgpu-oracle.md Q2). One the host keeps in
// two copies there (any shape but one level, one sample, 2D, of a linear format) answers getBytes:
// from its GPU copy, copied on the device, and not from its CPU copy; its replaceRegion: still
// writes the CPU copy and posts the region, with no device copy. One the host lays over its CPU
// copy, and any on a GPU of its own memory, reads its CPU copy, and answers it as its
// virtualAddress: a CPU write there is what getBytes: answers next, and posts nothing, as on both
// x86 Macs (oracle_bundle texmem); the one reading its GPU copy answers NULL.
void test_managed_texture_reads_on_unified_memory()
{
	Fixture fx;
	std::vector<uint8_t> mem(0x800000, 0xff);
	RgfxRegion region(mem.data(), mem.size());
	struct Case {
		BOOL unified;
		NSUInteger levels;
		NSUInteger arrayLength;
		bool readsGpuCopy;
	};
	const Case cases[] = {
	    {YES, 3, 1, true}, {YES, 1, 2, true},  {YES, 1, 1, false},
	    {NO, 3, 1, false}, {NO, 1, 1, false},
	};
	for (const Case &c : cases) {
		drainRing(fx.ring); // the last case's destroys
		@autoreleasepool {
			RgfxUnifiedAnswerDevice *device = [RgfxUnifiedAnswerDevice new];
			device.unified = c.unified;
			MTLTextureDescriptor *desc =
			    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
			                                                       width:64
			                                                      height:16
			                                                   mipmapped:c.levels > 1];
			desc.mipmapLevelCount = c.levels;
			if (c.arrayLength > 1) {
				desc.textureType = MTLTextureType2DArray;
				desc.arrayLength = c.arrayLength;
			}
			desc.storageMode = MTLStorageModeManaged;
			RgfxTexture *tex = [RgfxTexture managedTextureWithDescriptor:desc
			                                                      region:&region
			                                                       slabs:nullptr
			                                                      device:(id<MTLDevice>)device
			                                                  serializer:&fx.ser];
			CHECK(tex != nil);
			auto cpu =
			    expectMessage<RgfxBufferOverHostMemory>(fx.ring, RGFX_CMD_BUFFER_OVER_HOST_MEMORY);
			(void)expectMessage<RgfxManagedTextureCreate>(fx.ring,
			                                              RGFX_CMD_MANAGED_TEXTURE_CREATE);
			uint8_t *copy = mem.data() + cpu.region_offset;

			// replaceRegion: writes the CPU copy and posts its region, on every GPU.
			const uint8_t pixel[4] = {1, 2, 3, 4};
			[tex replaceRegion:MTLRegionMake2D(0, 0, 1, 1)
			       mipmapLevel:0
			         withBytes:pixel
			       bytesPerRow:4];
			CHECK(std::memcmp(copy, pixel, 4) == 0);
			(void)expectMessage<RgfxTextureReplaceRegion>(fx.ring,
			                                              RGFX_CMD_TEXTURE_REPLACE_REGION);
			CHECK(device.queuesAsked == 0);

			// getBytes: of the pixel the CPU copy holds.
			uint8_t back[4] = {};
			[tex getBytes:back bytesPerRow:4 fromRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0];
			if (c.readsGpuCopy) {
				// On the device, which here copies nothing back.
				CHECK(device.queuesAsked == 1);
				CHECK(back[0] == 0 && back[3] == 0);
			} else {
				CHECK(device.queuesAsked == 0);
				CHECK(std::memcmp(back, pixel, 4) == 0);
			}
			uint8_t *va = static_cast<uint8_t *>([(id<RgfxMetalTextureInternals>)tex virtualAddress]);
			if (c.readsGpuCopy) {
				CHECK(va == nullptr);
			} else {
				// The CPU copy, where replaceRegion: wrote, and where getBytes: reads a CPU write.
				CHECK(va == copy);
				const uint8_t written[4] = {9, 8, 7, 6};
				std::memcpy(va + 4, written, 4); // texel 1, 0 of level 0
				uint8_t seen[4] = {};
				[tex getBytes:seen
				    bytesPerRow:4
				     fromRegion:MTLRegionMake2D(1, 0, 1, 1)
				    mipmapLevel:0];
				CHECK(std::memcmp(seen, written, 4) == 0);
			}
			RgfxMsgHeader extra = {};
			std::vector<uint8_t> extraBytes;
			CHECK(!fx.ring.next(&extra, &extraBytes)); // neither read posts a message
		}
	}
	drainRing(fx.ring);
}

// Where the device took RGFX_FEATURE_MANAGED_TEXTURE_SLABS, a Managed texture's CPU copy is a
// Shared buffer of the slabs, as a device buffer of its size is: two copies share one slab's block
// of the region, not a block each, each names its own buffer in its MANAGED_TEXTURE_CREATE, reads
// zero though the block held other bytes, and replaceRegion: writes the slab's bytes.
void test_managed_texture_cpu_copy_in_slabs()
{
	Fixture fx;
	std::vector<uint8_t> mem(RGFX_ALLOCATIONS_OFFSET + 0x400000, 0xff); // pre-dirtied, as BAR memory
	RgfxRegion region(mem.data(), mem.size());
	BlockLog blocks;
	region.setBlockSource(&BlockLog::alloc, &BlockLog::release, &blocks);
	RgfxRegionSlabSource slabSource(region, fx.ser);
	RgfxSlabs slabs(slabSource);
	@autoreleasepool {
		MTLTextureDescriptor *desc =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
		                                                       width:100
		                                                      height:60
		                                                   mipmapped:YES];
		desc.mipmapLevelCount = 3;
		desc.storageMode = MTLStorageModeManaged;
		RgfxTexture *a = [RgfxTexture managedTextureWithDescriptor:desc
		                                                    region:&region
		                                                     slabs:&slabs
		                                                    device:nil
		                                                serializer:&fx.ser];
		RgfxTexture *b = [RgfxTexture managedTextureWithDescriptor:desc
		                                                    region:&region
		                                                     slabs:&slabs
		                                                    device:nil
		                                                serializer:&fx.ser];
		CHECK(a != nil && b != nil && [a storageMode] == MTLStorageModeManaged);
		auto heap = expectMessage<RgfxHeapOverHostMemory>(fx.ring, RGFX_CMD_HEAP_OVER_HOST_MEMORY);
		auto cpuA = expectMessage<RgfxBufferAtOffset>(fx.ring, RGFX_CMD_BUFFER_AT_OFFSET);
		auto createA = expectMessage<RgfxManagedTextureCreate>(fx.ring,
		                                                       RGFX_CMD_MANAGED_TEXTURE_CREATE);
		auto cpuB = expectMessage<RgfxBufferAtOffset>(fx.ring, RGFX_CMD_BUFFER_AT_OFFSET);
		auto createB = expectMessage<RgfxManagedTextureCreate>(fx.ring,
		                                                       RGFX_CMD_MANAGED_TEXTURE_CREATE);
		CHECK(cpuA.heap_obj_id == heap.obj_id && cpuB.heap_obj_id == heap.obj_id);
		CHECK(cpuA.length == 42240 && cpuB.length == 42240 && cpuA.offset != cpuB.offset);
		CHECK(cpuA.options == MTLResourceStorageModeShared);
		CHECK(createA.obj_id == a.rgfxObjId && createA.cpu_buffer_obj_id == cpuA.obj_id);
		CHECK(createB.obj_id == b.rgfxObjId && createB.cpu_buffer_obj_id == cpuB.obj_id);
		CHECK(blocks.allocated.size() == 1 && blocks.allocated[0] == heap.region_offset);
		uint8_t *copy = mem.data() + heap.region_offset + cpuA.offset;
		CHECK(copy[0] == 0 && copy[42239] == 0); // a new Managed texture reads zero

		// Level 1, a 4x1 region at (2, 1), at the layout's 256-byte pitch.
		uint8_t pixels[16];
		for (int i = 0; i < 16; i++)
			pixels[i] = static_cast<uint8_t>(i + 1);
		[a replaceRegion:MTLRegionMake2D(2, 1, 4, 1)
		     mipmapLevel:1
		       withBytes:pixels
		     bytesPerRow:16];
		CHECK(std::memcmp(copy + 30720 + 256 + 8, pixels, 16) == 0);
		auto replaced = expectMessage<RgfxTextureReplaceRegion>(fx.ring,
		                                                        RGFX_CMD_TEXTURE_REPLACE_REGION);
		CHECK(replaced.obj_id == a.rgfxObjId && replaced.level == 1);
	}
}

// --- host objects go as the app lets them go ----------------------------------------------

// The obj_id of the OBJECT_DESTROY next on `ring`.
uint64_t nextDestroyed(TestRing &ring)
{
	return expectMessage<RgfxObjectDestroy>(ring, RGFX_CMD_OBJECT_DESTROY).obj_id;
}

// What a commit syncs of the IOSurface views (rgfxIOSurfaceSyncBeforeCommit) for a command
// buffer whose encoders named `textures`.
void syncAsCommitted(std::initializer_list<id> textures)
{
	TestRing ring(0x1000);
	RgfxRingWriter writer(ring.base(), ring.dataSize, 1, TestRing::doorbell, nullptr);
	RgfxEncoder enc(writer, 1, 2);
	for (id t : textures)
		rgfxIOSurfaceNoteUse(&enc, t);
	rgfxIOSurfaceSyncBeforeCommit(&enc);
}

bool ringEmpty(TestRing &ring)
{
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	return !ring.next(&h, &bytes);
}

// Reads every message waiting on `ring` (the creates a test does not check).
void drainRing(TestRing &ring)
{
	while (!ringEmpty(ring)) {
	}
}

// newBufferWithBytesNoCopy: over app memory outside the region (RgfxBuffer.h): `contents` is the
// app's pointer and the buffer lies over a block of its own, seeded from the memory. A command
// buffer that uses it takes the CPU's writes into the block at its commit, and one that may write
// it gives the GPU's back at its completion, keeping the CPU's beside them; one that only reads
// it gives nothing back, nor does memory mapped read-only. A texture over it (CA's shape: offset
// 0x100, 1920-byte rows) writes through it. The deallocator gets the pointer and length once the
// host has dropped the buffer. Metal's refusals are nil: a length of 0, Private storage, a range
// not wholly mapped. A null pointer is contents of the buffer's own; bytes in the region are
// used in place.
void test_no_copy_buffers()
{
	@autoreleasepool {
	QueueFixture f;
	(void)expectQueueCreate(f.control);
	std::vector<uint8_t> mem(RGFX_ALLOCATIONS_OFFSET + 0x200000, 0xff);
	RgfxRegion region(mem.data(), mem.size());
	RgfxRegionSlabSource slabSource(region, f.ser);
	RgfxSlabs slabs(slabSource);
	const MTLResourceOptions shared = MTLResourceStorageModeShared;
	vm_address_t app = 0;
	CHECK(vm_allocate(mach_task_self(), &app, 0x40000, VM_FLAGS_ANYWHERE) == KERN_SUCCESS);
	uint8_t *p = reinterpret_cast<uint8_t *>(app);
	std::memset(p, 0x11, 0x40000);
	// What the deallocator got, which a block writes and a lambda reads.
	struct Freed {
		std::atomic<int> count{0};
		void *at = nullptr;
		NSUInteger length = 0;
	} freedState;
	Freed *freedLog = &freedState;
	auto completeOn = [&](id<MTLCommandBuffer> cb) {
		RgfxCbComplete done = cbComplete(f.nextSeqno(), ((RgfxResource *)cb).rgfxObjId);
		CHECK(f.completion.postFromHost(&done, sizeof(done)));
		[cb waitUntilCompleted];
		CHECK(cb.status == MTLCommandBufferStatusCompleted);
	};
	uint8_t *block = nullptr;
	@autoreleasepool {
		RgfxBuffer *b = [RgfxBuffer bufferWithBytesNoCopy:p
		                                           length:0x14000
		                                          options:shared
		                                      deallocator:^(void *q, NSUInteger l) {
			                                      freedLog->at = q;
			                                      freedLog->length = l;
			                                      freedLog->count++;
		                                      }
		                                           region:&region
		                                            slabs:&slabs
		                                           device:nil
		                                       serializer:&f.ser];
		CHECK(b != nil && [b contents] == p && b.length == 0x14000 && [b rgfxShadow] != nullptr);
		CHECK(b.allocatedSize == 0x14000); // the pages its bytes meet, as the x86 Macs answer
		// Its GPU copy is a range of a slab: a heap over a block, the buffer placed in it.
		auto heap =
		    expectMessage<RgfxHeapOverHostMemory>(f.control, RGFX_CMD_HEAP_OVER_HOST_MEMORY);
		auto create = expectMessage<RgfxBufferAtOffset>(f.control, RGFX_CMD_BUFFER_AT_OFFSET);
		CHECK(create.obj_id == b.rgfxObjId && create.length == 0x14000 &&
		      create.heap_obj_id == heap.obj_id && b.gpuAddress == heap.gpu_address + create.offset);
		CHECK(heap.region_offset >= region.floor());
		block = mem.data() + heap.region_offset + create.offset;
		CHECK(std::memcmp(block, p, 0x14000) == 0);

		// A writing command buffer: the CPU's bytes go in at the commit.
		p[0x10] = 0xa1;
		p[0x13fff] = 0xa2;
		id<MTLCommandBuffer> cb = [f.queue commandBuffer];
		id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
		[blit fillBuffer:b range:NSMakeRange(0x8000, 0x100) value:0x77];
		[blit endEncoding];
		[cb commit];
		CHECK(block[0x10] == 0xa1 && block[0x13fff] == 0xa2);
		// The host fills; the CPU writes beside it meanwhile.
		std::memset(block + 0x8000, 0x77, 0x100);
		p[0x8100] = 0xa3;
		CHECK(p[0x8000] == 0x11);
		completeOn(cb);
		CHECK(p[0x8000] == 0x77 && p[0x80ff] == 0x77 && p[0x8100] == 0xa3 && p[0x10] == 0xa1);

		// One that only reads it gives nothing back.
		RgfxRegion::Block other = region.allocate(0x1000);
		RgfxBuffer *dst = [RgfxBuffer bufferOverHostMemoryWithContents:const_cast<uint8_t *>(other.ptr)
		                                                  regionOffset:other.offset
		                                                        length:0x1000
		                                                       options:shared
		                                                   deallocator:nil
		                                                        device:nil
		                                                    serializer:&f.ser];
		id<MTLCommandBuffer> reads = [f.queue commandBuffer];
		blit = [reads blitCommandEncoder];
		[blit copyFromBuffer:b sourceOffset:0 toBuffer:dst destinationOffset:0 size:0x100];
		[blit endEncoding];
		[reads commit];
		block[0x9000] = 0xb1; // not this buffer's to give back
		completeOn(reads);
		CHECK(p[0x9000] == 0x11);
		block[0x9000] = 0x11;

		// A texture over it, CA's shape, as a blit's destination.
		MTLTextureDescriptor *td =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
		                                                       width:468
		                                                      height:36
		                                                   mipmapped:NO];
		td.storageMode = MTLStorageModeShared;
		id<MTLTexture> tex = [b newTextureWithDescriptor:td offset:0x100 bytesPerRow:1920];
		CHECK(tex != nil && tex.buffer == b);
		id<MTLCommandBuffer> draws = [f.queue commandBuffer];
		blit = [draws blitCommandEncoder];
		[blit copyFromBuffer:dst
		           sourceOffset:0
		      sourceBytesPerRow:1920
		    sourceBytesPerImage:1920
		             sourceSize:MTLSizeMake(4, 1, 1)
		              toTexture:tex
		       destinationSlice:0
		       destinationLevel:0
		      destinationOrigin:MTLOriginMake(0, 0, 0)];
		[blit endEncoding];
		[draws commit];
		block[0x100] = 0xc1;
		completeOn(draws);
		CHECK(p[0x100] == 0xc1);
		tex = nil;
		dst = nil;
		CHECK(freedState.count == 0);
	}
	CHECK(eventually([&] { return freedState.count == 1; }));
	CHECK(freedState.at == p && freedState.length == 0x14000);

	drainRing(f.control);

	// Metal's refusals.
	auto noCopy = [&](void *ptr, NSUInteger length, MTLResourceOptions options) {
		return [RgfxBuffer bufferWithBytesNoCopy:ptr
		                                  length:length
		                                 options:options
		                             deallocator:nil
		                                  region:&region
		                                   slabs:&slabs
		                                  device:nil
		                              serializer:&f.ser];
	};
	CHECK(noCopy(p, 0, shared) == nil);
	CHECK(noCopy(p, 0x1000, MTLResourceStorageModePrivate) == nil);
	vm_address_t holed = 0;
	CHECK(vm_allocate(mach_task_self(), &holed, 0x10000, VM_FLAGS_ANYWHERE) == KERN_SUCCESS);
	CHECK(vm_deallocate(mach_task_self(), holed + 0x8000, 0x8000) == KERN_SUCCESS);
	CHECK(noCopy(reinterpret_cast<void *>(holed + 0x10), 0x8000, shared) == nil);
	@autoreleasepool {
		// Any alignment, Managed too: its didModifyRange: takes the CPU's bytes first.
		RgfxBuffer *odd = noCopy(reinterpret_cast<void *>(holed + 0x13), 0x100,
		                         MTLResourceStorageModeManaged);
		CHECK(odd != nil && [odd contents] == reinterpret_cast<void *>(holed + 0x13));
		auto c = expectMessage<RgfxBufferOverHostMemory>(f.control, RGFX_CMD_BUFFER_OVER_HOST_MEMORY);
		reinterpret_cast<uint8_t *>(holed)[0x14] = 0xd1;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
		[odd didModifyRange:NSMakeRange(1, 1)];
#pragma clang diagnostic pop
		CHECK(mem[c.region_offset + 1] == 0xd1);
		auto range =
		    expectMessage<RgfxBufferDidModifyRange>(f.control, RGFX_CMD_BUFFER_DID_MODIFY_RANGE);
		CHECK(range.obj_id == odd.rgfxObjId && range.location == 1 && range.length == 1);

		// Read-only memory: made, never written back.
		CHECK(vm_protect(mach_task_self(), holed, 0x4000, false, VM_PROT_READ) == KERN_SUCCESS);
		RgfxBuffer *ro = noCopy(reinterpret_cast<void *>(holed), 0x4000, shared);
		CHECK(ro != nil && [ro rgfxShadow] != nullptr && ![ro rgfxShadow]->writable());

		// A null pointer: zeroed contents of its own, in the region, with no shadow. The
		// deallocator gets null.
		std::atomic<void *> nullFreedState{reinterpret_cast<void *>(1)};
		std::atomic<void *> *nullFreed = &nullFreedState;
		@autoreleasepool {
			RgfxBuffer *own = [RgfxBuffer bufferWithBytesNoCopy:nullptr
			                                             length:0x2000
			                                            options:shared
			                                        deallocator:^(void *q, NSUInteger) {
				                                        nullFreed->store(q);
			                                        }
			                                             region:&region
			                                              slabs:&slabs
			                                             device:nil
			                                         serializer:&f.ser];
			CHECK(own != nil && [own rgfxShadow] == nullptr &&
			      region.contains([own contents], 0x2000));
			CHECK(static_cast<uint8_t *>([own contents])[0x1fff] == 0);
		}
		CHECK(eventually([&] { return nullFreedState.load() == nullptr; }));

		// Bytes already in the region: in place.
		RgfxRegion::Block inRegion = region.allocate(0x1000);
		RgfxBuffer *placed = noCopy(const_cast<uint8_t *>(inRegion.ptr), 0x1000, shared);
		CHECK(placed != nil && [placed rgfxShadow] == nullptr);
		drainRing(f.control);
		(void)ro;
	}
	drainRing(f.control);
	vm_deallocate(mach_task_self(), holed, 0x8000);
	vm_deallocate(mach_task_self(), app, 0x40000);
	}
	CHECK(!RgfxMemoryShadow::anyLive());
}

// newBufferWithBytesNoCopy: over writable Shared app memory where the region has pages blocks
// (RgfxBuffer.h): the buffer lies over the app's own pages, no block or shadow of its own, its
// create a BUFFER_OVER_PAGES over the block the kext gave for exactly the app's bytes, at the
// pointer's offset in its page, which its GPU address keeps. Its block goes back once the host
// has dropped it, before the deallocator. Writable Managed memory lies there too, one copy: the
// host's buffer a Shared one, the buffer still Managed, its didModifyRange: and a blit's
// synchronize of it, or of a texture over it, sending nothing, where a Managed buffer's shadow
// sends both. Read-only memory, a null pointer and bytes in the region never ask for a block, and
// a block not given, or one the region does not hold, falls back to the shadow.
void test_no_copy_buffers_in_place()
{
	@autoreleasepool {
	QueueFixture f;
	(void)expectQueueCreate(f.control);
	std::vector<uint8_t> mem(RGFX_ALLOCATIONS_OFFSET + 0x200000, 0xff);
	RgfxRegion region(mem.data(), mem.size());
	BlockLog blocks;
	region.setBlockSource(&BlockLog::alloc, &BlockLog::release, &blocks);
	RgfxRegionSlabSource slabSource(region, f.ser);
	RgfxSlabs slabs(slabSource);
	struct Pages {
		std::vector<std::tuple<const void *, uint64_t, bool>> asked;
		uint64_t offset = RGFX_ALLOCATIONS_OFFSET + 0x100000;
		uint64_t size = 0x1000;
		bool gives = true;
		static bool block(void *c, const void *address, uint64_t length, bool writable,
		                  uint64_t *offset, uint64_t *size)
		{
			Pages *p = static_cast<Pages *>(c);
			p->asked.emplace_back(address, length, writable);
			*offset = p->offset;
			*size = p->size;
			return p->gives;
		}
	} pages;
	CHECK(!region.hasPagesBlocks());
	region.setPagesBlockSource(&Pages::block, &pages);
	CHECK(region.hasPagesBlocks());
	const MTLResourceOptions shared = MTLResourceStorageModeShared;
	vm_address_t app = 0;
	CHECK(vm_allocate(mach_task_self(), &app, 0x40000, VM_FLAGS_ANYWHERE) == KERN_SUCCESS);
	uint8_t *p = reinterpret_cast<uint8_t *>(app) + 0x1123;
	std::atomic<int> freedState{0};
	std::atomic<int> *freed = &freedState;
	size_t releasedBefore = 0;
	@autoreleasepool {
		RgfxBuffer *b = [RgfxBuffer bufferWithBytesNoCopy:p
		                                           length:0x14000
		                                          options:shared
		                                      deallocator:^(void *, NSUInteger) {
			                                      freed->fetch_add(1);
		                                      }
		                                           region:&region
		                                            slabs:&slabs
		                                           device:nil
		                                       serializer:&f.ser];
		CHECK(b != nil && [b contents] == p && b.length == 0x14000 && [b rgfxShadow] == nullptr);
		// The pages its bytes meet, 0x123 into the first: as the x86 Macs answer (oracle_bundle
		// nocopy.off16 answers 8192 for 4096 bytes 16 into a page).
		CHECK(b.allocatedSize == 0x15000);
		CHECK(pages.asked.size() == 1 && std::get<0>(pages.asked[0]) == p &&
		      std::get<1>(pages.asked[0]) == 0x14000 && std::get<2>(pages.asked[0]));
		CHECK(blocks.allocated.empty()); // no block of its own
		auto create = expectMessage<RgfxBufferOverPages>(f.control, RGFX_CMD_BUFFER_OVER_PAGES);
		CHECK(create.obj_id == b.rgfxObjId && create.length == 0x14000);
		CHECK(create.block_offset == pages.offset && create.block_size == pages.size);
		CHECK(create.offset == 0x123 && create.gpu_address == b.gpuAddress);
		CHECK(b.gpuAddress % 0x1000 == 0x123 && b.gpuAddress >= RGFX_CONTEXT_VA_BASE);
		releasedBefore = blocks.released.size();
	}
	CHECK(eventually([&] { return freedState.load() == 1; }));
	CHECK(blocks.released.size() == releasedBefore + 1 &&
	      blocks.released.back() == pages.offset);
	auto destroy = expectMessage<RgfxObjectDestroy>(f.control, RGFX_CMD_OBJECT_DESTROY);
	(void)destroy;
	drainRing(f.control);

	auto noCopy = [&](void *ptr, NSUInteger length, MTLResourceOptions options) {
		return [RgfxBuffer bufferWithBytesNoCopy:ptr
		                                  length:length
		                                 options:options
		                             deallocator:nil
		                                  region:&region
		                                   slabs:&slabs
		                                  device:nil
		                              serializer:&f.ser];
	};
	@autoreleasepool {
		// Managed, writable: in place, one copy.
		pages.asked.clear();
		const MTLResourceOptions managedOptions =
		    MTLResourceStorageModeManaged | MTLResourceCPUCacheModeWriteCombined;
		RgfxBuffer *managed = noCopy(p, 0x2000, managedOptions);
		CHECK(managed != nil && [managed contents] == p && [managed rgfxShadow] == nullptr);
		CHECK(managed.storageMode == MTLStorageModeManaged &&
		      managed.cpuCacheMode == MTLCPUCacheModeWriteCombined);
		CHECK(pages.asked.size() == 1 && std::get<0>(pages.asked[0]) == p &&
		      std::get<2>(pages.asked[0]));
		auto over = expectMessage<RgfxBufferOverPages>(f.control, RGFX_CMD_BUFFER_OVER_PAGES);
		CHECK(over.obj_id == managed.rgfxObjId && over.length == 0x2000 && over.offset == 0x123);
		CHECK((over.options & MTLResourceStorageModeMask) == MTLResourceStorageModeShared &&
		      (over.options & MTLResourceCPUCacheModeMask) == MTLResourceCPUCacheModeWriteCombined);
		MTLTextureDescriptor *td =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
		                                                       width:16
		                                                      height:16
		                                                   mipmapped:NO];
		td.storageMode = MTLStorageModeManaged;
		td.usage = MTLTextureUsageShaderRead;
		id<MTLTexture> overManaged = [managed newTextureWithDescriptor:td offset:0 bytesPerRow:256];
		CHECK(overManaged != nil && overManaged.storageMode == MTLStorageModeManaged);
		drainRing(f.control);
		[managed didModifyRange:NSMakeRange(0, 0x100)];
		uint32_t px[4] = {1, 2, 3, 4};
		[overManaged replaceRegion:MTLRegionMake2D(0, 1, 4, 1) mipmapLevel:0 withBytes:px
		               bytesPerRow:16];
		CHECK(std::memcmp(p + 256, px, sizeof px) == 0);
		RgfxMsgHeader h = {};
		std::vector<uint8_t> sent;
		CHECK(!f.control.next(&h, &sent)); // no BUFFER_DID_MODIFY_RANGE: one copy
		// Read-only Managed memory keeps its shadow, a Managed buffer of two copies, whose
		// didModifyRange: and synchronize the host carries.
		uint8_t *roManaged = reinterpret_cast<uint8_t *>(app) + 0x38000;
		CHECK(vm_protect(mach_task_self(), reinterpret_cast<vm_address_t>(roManaged), 0x4000,
		                 false, VM_PROT_READ) == KERN_SUCCESS);
		RgfxBuffer *twoCopies = noCopy(roManaged, 0x1000, MTLResourceStorageModeManaged);
		CHECK(twoCopies != nil && [twoCopies rgfxShadow] != nullptr && pages.asked.size() == 1);
		drainRing(f.control);
		[twoCopies didModifyRange:NSMakeRange(0, 0x10)];
		auto told =
		    expectMessage<RgfxBufferDidModifyRange>(f.control, RGFX_CMD_BUFFER_DID_MODIFY_RANGE);
		CHECK(told.obj_id == twoCopies.rgfxObjId && told.length == 0x10);
		id<MTLCommandBuffer> cb = [f.queue commandBuffer];
		id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
		[blit synchronizeResource:managed];
		[blit synchronizeResource:overManaged];
		[blit synchronizeTexture:overManaged slice:0 level:0];
		[blit synchronizeResource:twoCopies];
		[blit endEncoding];
		[cb commit];
		CHECK(f.submit.next(&h, &sent) && h.type == RGFX_CMD_COMMAND_BUFFER);
		RgfxCommandBuffer cbh = {};
		std::memcpy(&cbh, sent.data(), sizeof(cbh));
		OpWalk walk(sent.data() + RGFX_COMMAND_BUFFER_OPS_OFFSET, cbh.op_bytes);
		std::vector<uint64_t> synced;
		while (!walk.done()) {
			uint16_t tag = 0;
			uint32_t len = 0;
			const uint8_t *body = nullptr;
			size_t bodyLength = 0;
			CHECK(walk.next(&tag, &len, &body, &bodyLength));
			if (tag == RGFX_OP_BLIT_SYNCHRONIZE_RESOURCE) {
				RgfxBlitOpSynchronizeResource m = {};
				std::memcpy(&m, body, std::min(bodyLength, sizeof m));
				synced.push_back(m.resource_obj_id);
			}
			CHECK(tag != RGFX_OP_BLIT_SYNCHRONIZE_TEXTURE);
		}
		CHECK(synced.size() == 1 && synced[0] == twoCopies.rgfxObjId);
		RgfxCbComplete done = cbComplete(h.seqno, ((RgfxResource *)cb).rgfxObjId);
		CHECK(f.completion.postFromHost(&done, sizeof(done)));
		[cb waitUntilCompleted];
		overManaged = nil;
		drainRing(f.control);
	}
	drainRing(f.control);
	@autoreleasepool {
		// Never asked: a null pointer, bytes in the region, read-only memory.
		pages.asked.clear();
		RgfxBuffer *own = noCopy(nullptr, 0x1000, shared);
		CHECK(own != nil && [own rgfxShadow] == nullptr);
		RgfxRegion::Block inRegion = region.allocate(0x1000);
		RgfxBuffer *placed = noCopy(const_cast<uint8_t *>(inRegion.ptr), 0x1000, shared);
		CHECK(placed != nil && [placed rgfxShadow] == nullptr);
		uint8_t *ro = reinterpret_cast<uint8_t *>(app) + 0x30000;
		CHECK(vm_protect(mach_task_self(), reinterpret_cast<vm_address_t>(ro), 0x4000, false,
		                 VM_PROT_READ) == KERN_SUCCESS);
		RgfxBuffer *readOnly = noCopy(ro, 0x4000, shared);
		CHECK(readOnly != nil && [readOnly rgfxShadow] != nullptr);
		CHECK(pages.asked.empty());

		// Asked, and no block given, or one the region does not hold: a shadow.
		pages.gives = false;
		RgfxBuffer *refused = noCopy(p, 0x1000, shared);
		CHECK(refused != nil && [refused rgfxShadow] != nullptr && pages.asked.size() == 1);
		pages.gives = true;
		pages.offset = mem.size();
		const size_t released = blocks.released.size();
		RgfxBuffer *outside = noCopy(p, 0x1000, shared);
		CHECK(outside != nil && [outside rgfxShadow] != nullptr && pages.asked.size() == 2);
		CHECK(blocks.released.size() == released + 1 && blocks.released.back() == mem.size());
		drainRing(f.control);
	}
	drainRing(f.control);
	vm_deallocate(mach_task_self(), app, 0x40000);
	}
	CHECK(!RgfxMemoryShadow::anyLive());
}


// A no-copy buffer's GPU copy (its shadow) is a range of a slab of its options, as a device
// buffer's is (RgfxBuffer.h): with the process's superpages as the slabs' first source, the
// copies of read-only memory, Shared and Managed, lie in them, seeded from the memory, eight 4 KiB
// copies a page apart in one heap over one pages block, and the region gives none of them a block
// of its own (each took one before: the region held every app-sized copy).
void test_no_copy_shadows_lie_in_slabs()
{
	@autoreleasepool {
	QueueFixture f;
	(void)expectQueueCreate(f.control);
	std::vector<uint8_t> mem(RGFX_ALLOCATIONS_OFFSET + 0x200000, 0xff);
	RgfxRegion region(mem.data(), mem.size());
	BlockLog blocks;
	region.setBlockSource(&BlockLog::alloc, &BlockLog::release, &blocks);
	// The kernel's superpages, zeroed, and the kext's pages blocks of them, each a block of the
	// region's a page long.
	struct Vm {
		std::map<uint8_t *, std::unique_ptr<uint8_t[]>> held;
		std::vector<const void *> wired;
		BlockLog *blocks = nullptr;
		static int superpages(void *c, uint64_t size, uint8_t **bytes)
		{
			std::unique_ptr<uint8_t[]> memory(new uint8_t[size]());
			*bytes = memory.get();
			static_cast<Vm *>(c)->held[*bytes] = std::move(memory);
			return KERN_SUCCESS;
		}
		static void unmap(void *c, uint8_t *bytes, uint64_t) { static_cast<Vm *>(c)->held.erase(bytes); }
		static bool block(void *c, const void *address, uint64_t, bool, uint64_t *offset,
		                  uint64_t *size)
		{
			Vm *vm = static_cast<Vm *>(c);
			vm->wired.push_back(address);
			*size = 0x1000;
			return BlockLog::alloc(vm->blocks, 0x1000, 0x1000, offset);
		}
	} vm;
	vm.blocks = &blocks;
	region.setPagesBlockSource(&Vm::block, &vm);
	RgfxPagesMemory calls;
	calls.context = &vm;
	calls.superpages = &Vm::superpages;
	calls.unmap = &Vm::unmap;
	// As a device that took MANAGED_AT_OFFSET and HEAP_OVER_PAGES makes them.
	RgfxRegionSlabSource regionSource(region, f.ser, true);
	RgfxPagesSlabSource pagesSource(region, f.ser, calls, 16ull << 30, true);
	RgfxSlabs slabs(regionSource);
	slabs.setFirst(&pagesSource);
	vm_address_t app = 0;
	CHECK(vm_allocate(mach_task_self(), &app, 0x10000, VM_FLAGS_ANYWHERE) == KERN_SUCCESS);
	uint8_t *p = reinterpret_cast<uint8_t *>(app);
	for (size_t i = 0; i < 0x10000; i++)
		p[i] = static_cast<uint8_t>(i * 7 + 3);
	CHECK(vm_protect(mach_task_self(), app, 0xa000, false, VM_PROT_READ) == KERN_SUCCESS);
	auto noCopy = [&](void *ptr, NSUInteger length, MTLResourceOptions options) {
		return [RgfxBuffer bufferWithBytesNoCopy:ptr
		                                  length:length
		                                 options:options
		                             deallocator:nil
		                                  region:&region
		                                   slabs:&slabs
		                                  device:nil
		                              serializer:&f.ser];
	};
	@autoreleasepool {
		NSMutableArray<RgfxBuffer *> *held = [NSMutableArray array];
		for (int i = 0; i < 8; i++) {
			RgfxBuffer *b = noCopy(p + i * 0x1000, 0x1000, MTLResourceStorageModeShared);
			CHECK(b != nil && [b contents] == p + i * 0x1000 && [b rgfxShadow] != nullptr &&
			      ![b rgfxShadow]->writable());
			[held addObject:b];
		}
		CHECK(vm.held.size() == 1 && vm.wired.size() == 1 && blocks.allocated.size() == 1);
		uint8_t *superpage = vm.held.begin()->first;
		auto heap = expectMessage<RgfxHeapOverPages>(f.control, RGFX_CMD_HEAP_OVER_PAGES);
		CHECK(heap.block_offset == blocks.allocated[0] && heap.size_bytes == 2ull << 20);
		for (int i = 0; i < 8; i++) {
			auto at = expectMessage<RgfxBufferAtOffset>(f.control, RGFX_CMD_BUFFER_AT_OFFSET);
			CHECK(at.obj_id == held[i].rgfxObjId && at.heap_obj_id == heap.obj_id &&
			      at.offset == static_cast<uint64_t>(i) * 0x1000 && at.length == 0x1000);
			CHECK(std::memcmp(superpage + at.offset, p + i * 0x1000, 0x1000) == 0);
		}
		// A Managed one's CPU copy lies in a superpage of its own class, placed in its heap.
		RgfxBuffer *managed = noCopy(p + 0x8000, 0x2000, MTLResourceStorageModeManaged);
		CHECK(managed != nil && [managed rgfxShadow] != nullptr &&
		      ![managed rgfxShadow]->writable());
		CHECK(vm.held.size() == 2 && blocks.allocated.size() == 2);
		auto mheap = expectMessage<RgfxHeapOverPages>(f.control, RGFX_CMD_HEAP_OVER_PAGES);
		auto mat =
		    expectMessage<RgfxBufferManagedAtOffset>(f.control, RGFX_CMD_BUFFER_MANAGED_AT_OFFSET);
		CHECK(mat.obj_id == managed.rgfxObjId && mat.heap_obj_id == mheap.obj_id &&
		      mat.offset == 0 && mat.length == 0x2000);
		uint8_t *copy = nullptr;
		for (auto &h : vm.held)
			if (h.first != superpage)
				copy = h.first;
		CHECK(copy != nullptr && std::memcmp(copy, p + 0x8000, 0x2000) == 0);
		// The region's only blocks are the two pages blocks.
		CHECK(blocks.allocated.size() == vm.wired.size());
	}
	drainRing(f.control);
	vm_deallocate(mach_task_self(), app, 0x10000);
	}
	CHECK(!RgfxMemoryShadow::anyLive());
}

// A no-copy buffer bound to a shader is one the command buffer may write, so its completion
// writes it back; a texture over it is one only where its usage has ShaderWrite. A bound buffer
// with no copy of app memory notes nothing beside them. One used to read is written back where a
// kernel after it has an argument buffer, through which it may write it, and not where none has.
void test_binds_note_shadow_writes()
{
	@autoreleasepool {
	QueueFixture f;
	(void)expectQueueCreate(f.control);
	std::vector<uint8_t> mem(RGFX_ALLOCATIONS_OFFSET + 0x200000, 0xff);
	RgfxRegion region(mem.data(), mem.size());
	RgfxRegionSlabSource slabSource(region, f.ser);
	RgfxSlabs slabs(slabSource);
	const MTLResourceOptions shared = MTLResourceStorageModeShared;
	vm_address_t app = 0;
	CHECK(vm_allocate(mach_task_self(), &app, 0x20000, VM_FLAGS_ANYWHERE) == KERN_SUCCESS);
	uint8_t *p = reinterpret_cast<uint8_t *>(app);
	std::memset(p, 0x11, 0x20000);
	auto completeOn = [&](id<MTLCommandBuffer> cb) {
		RgfxCbComplete done = cbComplete(f.nextSeqno(), ((RgfxResource *)cb).rgfxObjId);
		CHECK(f.completion.postFromHost(&done, sizeof(done)));
		[cb waitUntilCompleted];
		CHECK(cb.status == MTLCommandBufferStatusCompleted);
	};
	@autoreleasepool {
		RgfxBuffer *b = [RgfxBuffer bufferWithBytesNoCopy:p
		                                           length:0x10000
		                                          options:shared
		                                      deallocator:nil
		                                           region:&region
		                                            slabs:&slabs
		                                           device:nil
		                                       serializer:&f.ser];
		CHECK(b != nil && [b rgfxShadow] != nullptr);
		auto heap =
		    expectMessage<RgfxHeapOverHostMemory>(f.control, RGFX_CMD_HEAP_OVER_HOST_MEMORY);
		auto create = expectMessage<RgfxBufferAtOffset>(f.control, RGFX_CMD_BUFFER_AT_OFFSET);
		CHECK(create.obj_id == b.rgfxObjId && create.heap_obj_id == heap.obj_id);
		uint8_t *block = mem.data() + heap.region_offset + create.offset;
		RgfxRegion::Block other = region.allocate(0x1000);
		RgfxBuffer *plain = [RgfxBuffer bufferOverHostMemoryWithContents:const_cast<uint8_t *>(other.ptr)
		                                                    regionOffset:other.offset
		                                                          length:0x1000
		                                                         options:shared
		                                                     deallocator:nil
		                                                          device:nil
		                                                      serializer:&f.ser];
		CHECK(plain != nil && [plain rgfxShadow] == nullptr);
		MTLTextureDescriptor *td =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
		                                                       width:16
		                                                      height:16
		                                                   mipmapped:NO];
		td.storageMode = MTLStorageModeShared;
		td.usage = MTLTextureUsageShaderRead;
		id<MTLTexture> reads = [b newTextureWithDescriptor:td offset:0x1000 bytesPerRow:64];
		td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
		id<MTLTexture> writes = [b newTextureWithDescriptor:td offset:0x2000 bytesPerRow:64];
		CHECK(reads != nil && writes != nil);

		// Binds `bind`'s objects on a compute encoder, then the host writes `at` meanwhile:
		// whether the app's bytes there have it after the completion.
		auto writtenBack = [&](auto bind, size_t at) {
			id<MTLCommandBuffer> cb = [f.queue commandBuffer];
			id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
			bind(ce);
			[ce endEncoding];
			[cb commit];
			block[at] = 0xe1;
			completeOn(cb);
			const bool back = p[at] == 0xe1;
			block[at] = p[at] = 0x11;
			return back;
		};
		CHECK(writtenBack([&](id<MTLComputeCommandEncoder> ce) {
			[ce setBuffer:b offset:0 atIndex:0];
		}, 0x20));
		CHECK(!writtenBack([&](id<MTLComputeCommandEncoder> ce) {
			[ce setBuffer:plain offset:0 atIndex:0];
			[ce setTexture:reads atIndex:0];
			[ce setBuffer:plain offset:0 atIndex:1];
		}, 0x1010));
		CHECK(writtenBack([&](id<MTLComputeCommandEncoder> ce) {
			[ce setTexture:reads atIndex:0];
			[ce setTexture:writes atIndex:1];
		}, 0x2010));
		id<MTLTexture> both[2] = {reads, nil};
		CHECK(!writtenBack([&](id<MTLComputeCommandEncoder> ce) {
			[ce setTextures:both withRange:NSMakeRange(0, 2)];
		}, 0x1020));

		// A kernel of `arguments`, its AIR as read.
		auto kernel = [&](std::vector<RgfxAirArgument> arguments) {
			RgfxFunction *fn = [RgfxFunction functionWithLibrary:nil
			                                                name:@"k"
			                                           constants:nullptr
			                                               count:0
			                                              device:nil
			                                          serializer:&f.ser];
			auto air = std::make_shared<RgfxAirFunction>();
			air->name = "k";
			air->stage = RgfxAirStage::Kernel;
			air->arguments = std::move(arguments);
			[fn rgfxSetAirFunction:air];
			NSError *err = nil;
			RgfxComputePipelineState *ps =
			    [RgfxComputePipelineState computePipelineStateWithFunction:fn
			                                                       device:nil
			                                                   serializer:&f.ser
			                                                        error:&err];
			CHECK(ps != nil);
			drainRing(f.control);
			return ps;
		};
		RgfxComputePipelineState *throughArgs =
		    kernel({{RgfxAirKind::IndirectBuffer, 0, 1, RgfxAirAccess::Read, false, false}});
		RgfxComputePipelineState *noArgs =
		    kernel({{RgfxAirKind::Texture, 0, 1, RgfxAirAccess::Read, false, false}});
		const MTLSize one = MTLSizeMake(1, 1, 1);
		CHECK(writtenBack([&](id<MTLComputeCommandEncoder> ce) {
			[ce useResource:b usage:MTLResourceUsageRead];
			[ce setComputePipelineState:throughArgs];
			[ce dispatchThreads:one threadsPerThreadgroup:one];
		}, 0x30));
		CHECK(!writtenBack([&](id<MTLComputeCommandEncoder> ce) {
			[ce useResource:b usage:MTLResourceUsageRead];
			[ce setComputePipelineState:noArgs];
			[ce dispatchThreads:one threadsPerThreadgroup:one];
		}, 0x40));
		CHECK(!writtenBack([&](id<MTLComputeCommandEncoder> ce) {
			[ce setComputePipelineState:throughArgs];
			[ce useResource:b usage:MTLResourceUsageRead];
		}, 0x50));
		reads = nil;
		writes = nil;
		plain = nil;
	}
	drainRing(f.control);
	vm_deallocate(mach_task_self(), app, 0x20000);
	}
	CHECK(!RgfxMemoryShadow::anyLive());
}

// A Managed texture's CPU copy goes with it: the texture's destroy, then its copy's buffer's,
// then the copy's block goes back.
void test_managed_texture_copy_block_after_destroys()
{
	Fixture fx;
	std::vector<uint8_t> mem(RGFX_ALLOCATIONS_OFFSET + 0x100000, 0);
	RgfxRegion region(mem.data(), mem.size());
	struct Log {
		BlockLog blocks;
		TestRing *ring;
		std::vector<uint64_t> seenAtRelease; // the objects destroyed on the ring by each release
		static bool alloc(void *c, uint64_t size, uint64_t align, uint64_t *offset)
		{
			return BlockLog::alloc(&static_cast<Log *>(c)->blocks, size, align, offset);
		}
		static void release(void *c, uint64_t offset)
		{
			Log *log = static_cast<Log *>(c);
			BlockLog::release(&log->blocks, offset);
			RgfxMsgHeader h = {};
			std::vector<uint8_t> bytes;
			while (log->ring->next(&h, &bytes)) {
				RgfxObjectDestroy d = {};
				if (h.type == RGFX_CMD_OBJECT_DESTROY && bytes.size() >= sizeof(d)) {
					std::memcpy(&d, bytes.data(), sizeof(d));
					log->seenAtRelease.push_back(d.obj_id);
				}
			}
		}
	} log;
	log.ring = &fx.ring;
	region.setBlockSource(&Log::alloc, &Log::release, &log);
	uint64_t texId = 0, bufId = 0;
	@autoreleasepool {
		MTLTextureDescriptor *td =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
		                                                       width:64
		                                                      height:64
		                                                   mipmapped:NO];
		td.storageMode = MTLStorageModeManaged;
		RgfxTexture *t = [RgfxTexture managedTextureWithDescriptor:td
		                                                    region:&region
		                                                     slabs:nullptr
		                                                    device:nil
		                                                serializer:&fx.ser];
		CHECK(t != nil);
		texId = t.rgfxObjId;
		bufId = expectMessage<RgfxBufferOverHostMemory>(fx.ring, RGFX_CMD_BUFFER_OVER_HOST_MEMORY)
		            .obj_id;
		drainRing(fx.ring);
		CHECK(log.blocks.allocated.size() == 1 && log.blocks.released.empty());
	}
	CHECK(log.blocks.released == log.blocks.allocated);
	CHECK(log.seenAtRelease == std::vector<uint64_t>({texId, bufId}));
	CHECK(ringEmpty(fx.ring));
}

// A queue's ring carries only its command buffers: every create and destroy of the objects they
// use rides the control ring (ring 0), which the kext checks, and nothing else rides the queue's,
// whose host side refuses any other type. Objects of every kind are made, used by a command
// buffer committed on the queue, and let go.
void test_queue_ring_carries_only_command_buffers()
{
	QueueFixture f;
	int calls = 0;
	f.ids.setResourceIdSource(&oneChunk, &calls);
	id<MTLCommandBuffer> cb = nil;
	@autoreleasepool {
		MTLHeapDescriptor *hd = [MTLHeapDescriptor new];
		hd.size = 0x10000;
		hd.type = MTLHeapTypePlacement;
		RgfxHeap *heap = [RgfxHeap heapWithDescriptor:hd device:nil serializer:&f.ser];
		id<MTLBuffer> placed = [heap newBufferWithLength:0x100
		                                         options:MTLResourceStorageModePrivate
		                                          offset:0];
		RgfxBuffer *buffer = [RgfxBuffer bufferWithLength:0x1000
		                                          options:MTLResourceStorageModePrivate
		                                           device:nil
		                                       serializer:&f.ser];
		MTLTextureDescriptor *td =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
		                                                       width:8
		                                                      height:8
		                                                   mipmapped:NO];
		RgfxTexture *texture = [RgfxTexture textureWithDescriptor:td
		                                                     heap:RGFX_OBJ_NONE
		                                               heapOffset:0
		                                                   device:nil
		                                               serializer:&f.ser];
		id<MTLSamplerState> sampler =
		    [RgfxSampler samplerWithDescriptor:[MTLSamplerDescriptor new] device:nil serializer:&f.ser];
		id<MTLDepthStencilState> depth =
		    [RgfxDepthStencilState depthStencilStateWithDescriptor:[MTLDepthStencilDescriptor new]
		                                                    device:nil
		                                                serializer:&f.ser];
		id<MTLEvent> event = [RgfxEvent eventWithDevice:nil serializer:&f.ser];
		id<MTLFence> fence = [RgfxFence fenceWithDevice:nil serializer:&f.ser];
		CHECK(heap != nil && placed != nil && buffer != nil && texture != nil && sampler != nil &&
		      depth != nil && event != nil && fence != nil);
		cb = [f.queue commandBuffer];
		id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
		[blit fillBuffer:buffer range:NSMakeRange(0, 0x100) value:7];
		[blit copyFromBuffer:buffer sourceOffset:0 toBuffer:placed destinationOffset:0 size:0x100];
		[blit updateFence:fence];
		[blit endEncoding];
		[cb encodeSignalEvent:event value:1];
		[cb commit];
	}
	const std::set<uint32_t> queueTypes = {RGFX_CMD_COMMAND_BUFFER,
	                                       RGFX_CMD_COMMAND_BUFFER_IN_REGION};
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	int buffers = 0;
	uint64_t seqno = 0;
	while (f.submit.next(&h, &bytes)) {
		CHECK(queueTypes.count(h.type) == 1);
		seqno = h.seqno;
		buffers++;
	}
	CHECK(buffers == 1);
	// The buffer completes, so the destroys of what it used, held back for it, go too.
	RgfxCbComplete done = cbComplete(seqno, ((RgfxResource *)cb).rgfxObjId);
	CHECK(f.completion.postFromHost(&done, sizeof(done)));
	[cb waitUntilCompleted];
	@autoreleasepool {
		cb = nil;
	}
	std::set<uint32_t> lifecycle;
	while (f.control.next(&h, &bytes)) {
		CHECK(queueTypes.count(h.type) == 0);
		lifecycle.insert(h.type);
	}
	for (uint32_t type : {RGFX_CMD_HEAP_CREATE, RGFX_CMD_BUFFER_AT_OFFSET, RGFX_CMD_BUFFER_CREATE,
	                      RGFX_CMD_TEXTURE_CREATE, RGFX_CMD_SAMPLER_CREATE,
	                      RGFX_CMD_DEPTH_STENCIL_CREATE, RGFX_CMD_EVENT_CREATE,
	                      RGFX_CMD_FENCE_CREATE, RGFX_CMD_OBJECT_DESTROY})
		CHECK(lifecycle.count(type) == 1);
}

// A heap of no bytes is nil, as Metal answers one, and so is a heap or device buffer for which
// the context's range has no GPU address left: neither posts anything, so the host is never
// asked for an object the app does not hold.
void test_heaps_with_no_bytes_or_address_are_nil()
{
	Fixture fx;
	@autoreleasepool {
		MTLHeapDescriptor *d = [MTLHeapDescriptor new];
		d.size = 0;
		d.type = MTLHeapTypePlacement;
		CHECK([RgfxHeap heapWithDescriptor:d device:nil serializer:&fx.ser] == nil);
		d.storageMode = MTLStorageModeShared;
		CHECK([RgfxHeap heapOverHostMemoryWithDescriptor:d
		                                    regionOffset:RGFX_CONTEXT_ALLOCATIONS_OFFSET
		                                        contents:nullptr
		                                     deallocator:nil
		                                          device:nil
		                                      serializer:&fx.ser] == nil);
		d.size = RGFX_CONTEXT_VA_SPAN;
		d.storageMode = MTLStorageModePrivate;
		CHECK([RgfxHeap heapWithDescriptor:d device:nil serializer:&fx.ser] == nil);
		CHECK([RgfxBuffer bufferWithLength:RGFX_CONTEXT_VA_SPAN
		                           options:MTLResourceStorageModePrivate
		                            device:nil
		                        serializer:&fx.ser] == nil);
		CHECK(ringEmpty(fx.ring));
		d.size = 0x10000;
		CHECK([RgfxHeap heapWithDescriptor:d device:nil serializer:&fx.ser] != nil);
		CHECK(expectMessage<RgfxHeapCreate>(fx.ring, RGFX_CMD_HEAP_CREATE).size_bytes == 0x10000);
	}
	drainRing(fx.ring);
}

// A Shared heap's placed buffers alias its bytes, as Metal's do: a buffer is the heap's bytes at
// its offset, two at one offset are the same bytes, and each is at the heap's GPU address plus
// its offset; one that runs past the heap is nil, and posts nothing. A Private heap's placed
// buffer is at its address plus the offset too.
void test_shared_heap_buffers_alias_its_bytes()
{
	Fixture fx;
	static uint8_t bytes[0x10000];
	MTLHeapDescriptor *d = [MTLHeapDescriptor new];
	d.size = sizeof(bytes);
	d.storageMode = MTLStorageModeShared;
	d.type = MTLHeapTypePlacement;
	@autoreleasepool {
		RgfxHeap *heap = [RgfxHeap heapOverHostMemoryWithDescriptor:d
		                                              regionOffset:RGFX_CONTEXT_ALLOCATIONS_OFFSET
		                                                  contents:bytes
		                                               deallocator:nil
		                                                    device:nil
		                                                serializer:&fx.ser];
		CHECK(heap != nil);
		const uint64_t at =
		    expectMessage<RgfxHeapOverHostMemory>(fx.ring, RGFX_CMD_HEAP_OVER_HOST_MEMORY)
		        .gpu_address;
		CHECK(at != 0);
		id<MTLBuffer> a = [heap newBufferWithLength:0x1000
		                                    options:MTLResourceStorageModeShared
		                                     offset:0x2000];
		id<MTLBuffer> b = [heap newBufferWithLength:0x800
		                                    options:MTLResourceStorageModeShared
		                                     offset:0x2000];
		CHECK(a != nil && b != nil);
		auto placed = expectMessage<RgfxBufferAtOffset>(fx.ring, RGFX_CMD_BUFFER_AT_OFFSET);
		CHECK(placed.heap_obj_id == heap.rgfxObjId && placed.offset == 0x2000 &&
		      placed.length == 0x1000);
		drainRing(fx.ring);
		CHECK([a contents] == bytes + 0x2000 && [b contents] == [a contents]);
		CHECK(a.gpuAddress == at + 0x2000 && b.gpuAddress == at + 0x2000);
		static_cast<uint8_t *>([a contents])[5] = 0xc3;
		CHECK(static_cast<uint8_t *>([b contents])[5] == 0xc3 && bytes[0x2005] == 0xc3);
		CHECK([heap newBufferWithLength:0x2000
		                        options:MTLResourceStorageModeShared
		                         offset:sizeof(bytes) - 0x1000] == nil);
		CHECK([heap newBufferWithLength:1
		                        options:MTLResourceStorageModeShared
		                         offset:sizeof(bytes) + 1] == nil);
		CHECK(ringEmpty(fx.ring));

		MTLHeapDescriptor *pd = [MTLHeapDescriptor new];
		pd.size = 0x10000;
		pd.storageMode = MTLStorageModePrivate;
		pd.type = MTLHeapTypePlacement;
		RgfxHeap *priv = [RgfxHeap heapWithDescriptor:pd device:nil serializer:&fx.ser];
		const uint64_t privAt =
		    expectMessage<RgfxHeapCreate>(fx.ring, RGFX_CMD_HEAP_CREATE).gpu_address;
		id<MTLBuffer> c = [priv newBufferWithLength:0x100
		                                    options:MTLResourceStorageModePrivate
		                                     offset:0x4000];
		CHECK(c != nil && c.gpuAddress == privAt + 0x4000);
		drainRing(fx.ring);
	}
	drainRing(fx.ring);
}

// What a fake device answers a heap texture size query: 264192 bytes at 128, as Metal answers a
// 256 x 256 RGBA8Unorm texture (M3 Max, macOS 27).
bool askTextureSize(void *, const void *, uint32_t, void *reply, uint32_t capacity,
                    uint32_t *reply_size)
{
	RgfxSizeAndAlign answer = {};
	answer.header = {RGFX_EVT_SIZE_AND_ALIGN, sizeof(answer), 1};
	answer.status = RGFX_STATUS_OK;
	answer.size = 264192;
	answer.align = 128;
	CHECK(capacity >= sizeof(answer));
	std::memcpy(reply, &answer, sizeof(answer));
	*reply_size = sizeof(answer);
	return true;
}

// An Automatic heap places what it makes as Metal's does, in the sequence measured on the M3
// Max under macOS 27 (RgfxHeapRanges.hpp): its size rounded up to the 16 KiB page, buffers at
// their length and textures at the device's size, each at the lowest free offset at its
// alignment, routed as a placement at that offset, and answering heapOffset 0; usedSize and
// maxAvailableSizeWithAlignment: follow, a resource gives its bytes back when freed or made
// aliasable (once), and one no free run holds is nil and posts nothing. A Placement heap
// answers 0 for both and places nothing itself.
void test_automatic_heap_places_as_metal()
{
	Fixture fx;
	fx.ser.setAsker(&askTextureSize, nullptr);
	MTLHeapDescriptor *d = [MTLHeapDescriptor new];
	d.size = 1000000;
	d.storageMode = MTLStorageModePrivate;
	d.type = MTLHeapTypeAutomatic;
	const MTLResourceOptions o = MTLResourceStorageModePrivate;
	@autoreleasepool {
		RgfxHeap *heap = [RgfxHeap heapWithDescriptor:d device:nil serializer:&fx.ser];
		CHECK(heap != nil);
		const RgfxHeapCreate made = expectMessage<RgfxHeapCreate>(fx.ring, RGFX_CMD_HEAP_CREATE);
		CHECK(made.heap_type == MTLHeapTypeAutomatic && made.size_bytes == 1000000);
		CHECK(heap.size == 1015808 && heap.currentAllocatedSize == 1015808);
		CHECK(heap.usedSize == 0 && [heap maxAvailableSizeWithAlignment:65536] == 1015808);
		auto placed = [&](uint64_t offset, uint64_t length) {
			auto m = expectMessage<RgfxBufferAtOffset>(fx.ring, RGFX_CMD_BUFFER_AT_OFFSET);
			CHECK(m.heap_obj_id == heap.rgfxObjId && m.offset == offset && m.length == length);
		};
		id<MTLBuffer> b1 = [heap newBufferWithLength:100 options:o];
		placed(0, 100);
		CHECK(b1 != nil && b1.heap == heap && b1.heapOffset == 0 && b1.gpuAddress == made.gpu_address);
		CHECK(heap.usedSize == 100 && [heap maxAvailableSizeWithAlignment:256] == 1015552);
		id<MTLBuffer> b2 = [heap newBufferWithLength:100 options:o];
		placed(256, 100);
		id<MTLBuffer> b3 = [heap newBufferWithLength:70000 options:o];
		placed(512, 70000);
		CHECK(b3.gpuAddress == made.gpu_address + 512 && b3.heapOffset == 0);
		CHECK(heap.usedSize == 70200 && [heap maxAvailableSizeWithAlignment:1] == 945296);
		CHECK([heap maxAvailableSizeWithAlignment:4096] == 942080);
		CHECK([heap maxAvailableSizeWithAlignment:65536] == 884736);
		MTLTextureDescriptor *td =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
		                                                       width:256
		                                                      height:256
		                                                   mipmapped:NO];
		td.storageMode = MTLStorageModePrivate;
		id<MTLTexture> t1 = [heap newTextureWithDescriptor:td];
		auto tex = expectMessage<RgfxTextureCreate>(fx.ring, RGFX_CMD_TEXTURE_CREATE);
		CHECK(t1 != nil && t1.heap == heap && t1.heapOffset == 0 && t1.allocatedSize == 264192);
		CHECK(tex.heap_obj_id == heap.rgfxObjId && tex.heap_offset == 70528);
		CHECK(heap.usedSize == 334392 && [heap maxAvailableSizeWithAlignment:256] == 680960);
		CHECK(!RgfxUnsupportedWasLogged([RgfxHeap class], @selector(newTextureWithDescriptor:)));

		b1 = nil; // no command buffer in flight: its bytes go back at once
		CHECK(expectMessage<RgfxObjectDestroy>(fx.ring, RGFX_CMD_OBJECT_DESTROY).obj_id != 0);
		CHECK(heap.usedSize == 334292);
		id<MTLBuffer> b4 = [heap newBufferWithLength:50 options:o];
		placed(0, 50);
		CHECK(!b2.isAliasable);
		[b2 makeAliasable];
		CHECK(b2.isAliasable && heap.usedSize == 334242);
		[b2 makeAliasable];
		CHECK(heap.usedSize == 334242);
		id<MTLBuffer> b5 = [heap newBufferWithLength:60 options:o];
		placed(256, 60);
		b2 = nil; // made aliasable, so its bytes went back already
		drainRing(fx.ring);
		CHECK(heap.usedSize == 334302);
		const NSUInteger most = [heap maxAvailableSizeWithAlignment:256];
		CHECK(most == 680960);
		CHECK([heap newBufferWithLength:most + 1 options:o] == nil && ringEmpty(fx.ring));
		id<MTLBuffer> b7 = [heap newBufferWithLength:most options:o];
		placed(334848, most);
		CHECK(heap.usedSize == 1015262 && [heap maxAvailableSizeWithAlignment:1] == 206);
		CHECK([heap newBufferWithLength:16 options:o] == nil);
		CHECK([heap newTextureWithDescriptor:td] == nil);
		CHECK([heap newBufferWithLength:0 options:o] == nil && ringEmpty(fx.ring));
		CHECK([heap maxAvailableSizeWithAlignment:3] == 0);
		(void)b3, (void)b4, (void)b5, (void)b7, (void)t1;
	}
	drainRing(fx.ring);

	// With no answer from the device, a texture takes the bound of Metal's size, logged.
	@autoreleasepool {
		Fixture bare;
		RgfxHeap *heap = [RgfxHeap heapWithDescriptor:d device:nil serializer:&bare.ser];
		MTLTextureDescriptor *td =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
		                                                       width:256
		                                                      height:256
		                                                   mipmapped:NO];
		RgfxHeapSizeAlign bound = {};
		CHECK(rgfxHeapTextureBound(MTLPixelFormatRGBA8Unorm, MTLTextureType2D, 256, 256, 1, 1, 1,
		                           1, &bound));
		id<MTLTexture> t = [heap newTextureWithDescriptor:td];
		CHECK(t != nil && t.allocatedSize == bound.size && heap.usedSize == bound.size);
		CHECK(RgfxUnsupportedWasLogged([RgfxHeap class], @selector(newTextureWithDescriptor:)));
		drainRing(bare.ring);
	}

	// A Placement heap places nothing itself, and answers 0 used and 0 available.
	@autoreleasepool {
		Fixture px;
		MTLHeapDescriptor *pd = [MTLHeapDescriptor new];
		pd.size = 20000;
		pd.type = MTLHeapTypePlacement;
		RgfxHeap *heap = [RgfxHeap heapWithDescriptor:pd device:nil serializer:&px.ser];
		CHECK(heap.size == 32768 && heap.currentAllocatedSize == 32768);
		id<MTLBuffer> b = [heap newBufferWithLength:4096 options:0 offset:0];
		CHECK(b != nil && heap.usedSize == 0 && [heap maxAvailableSizeWithAlignment:256] == 0);
		CHECK([heap newBufferWithLength:64 options:0] == nil);
		drainRing(px.ring);
	}
}

// An Automatic Shared heap's buffers are its bytes at the offsets it picks.
void test_automatic_shared_heap_buffers_are_its_bytes()
{
	Fixture fx;
	static uint8_t bytes[0x8000];
	MTLHeapDescriptor *d = [MTLHeapDescriptor new];
	d.size = sizeof(bytes);
	d.storageMode = MTLStorageModeShared;
	d.type = MTLHeapTypeAutomatic;
	@autoreleasepool {
		RgfxHeap *heap = [RgfxHeap heapOverHostMemoryWithDescriptor:d
		                                              regionOffset:RGFX_CONTEXT_ALLOCATIONS_OFFSET
		                                                  contents:bytes
		                                               deallocator:nil
		                                                    device:nil
		                                                serializer:&fx.ser];
		const uint64_t at =
		    expectMessage<RgfxHeapOverHostMemory>(fx.ring, RGFX_CMD_HEAP_OVER_HOST_MEMORY)
		        .gpu_address;
		id<MTLBuffer> a = [heap newBufferWithLength:300 options:MTLResourceStorageModeShared];
		id<MTLBuffer> b = [heap newBufferWithLength:300 options:MTLResourceStorageModeShared];
		CHECK(a != nil && b != nil);
		CHECK([a contents] == bytes && [b contents] == bytes + 512);
		CHECK(a.gpuAddress == at && b.gpuAddress == at + 512);
		CHECK([heap newBufferWithLength:sizeof(bytes) options:MTLResourceStorageModeShared] ==
		      nil);
		drainRing(fx.ring);
	}
	drainRing(fx.ring);
}

// A heap's GPU address goes back once its destroy is retired: churned 1 GiB heaps, each made
// while the one before it lives, take spans that never meet a live one and are picked again
// once the picks have gone round the context's range, so the churn never runs out. One a
// command buffer in flight holds back keeps its span till the buffer completes.
void test_addresses_reused_after_destroy()
{
	Fixture fx;
	MTLHeapDescriptor *d = [MTLHeapDescriptor new];
	d.size = uint64_t{1} << 30;
	d.type = MTLHeapTypePlacement;
	const uint64_t span = RgfxIdSpace::addressSpan(d.size);
	auto heapAddress = [&] {
		return expectMessage<RgfxHeapCreate>(fx.ring, RGFX_CMD_HEAP_CREATE).gpu_address;
	};
	std::set<uint64_t> seen;
	RgfxHeap *held = nil;
	uint64_t heldAt = 0;
	bool reused = false;
	// Once round the range, and a few more.
	const uint64_t rounds = RGFX_CONTEXT_VA_SPAN / span + 8;
	for (uint64_t i = 0; i < rounds; i++) {
		@autoreleasepool {
			RgfxHeap *heap = [RgfxHeap heapWithDescriptor:d device:nil serializer:&fx.ser];
			uint64_t at = heapAddress();
			CHECK(heap != nil && at != 0 && at % RgfxIdSpace::kAddressGranule == 0);
			CHECK(at + span <= heldAt || heldAt + span <= at); // apart from the live one's span
			reused |= !seen.insert(at).second;
			held = heap; // the one before goes
			heldAt = at;
			drainRing(fx.ring);
		}
	}
	CHECK(reused && seen.size() < rounds && held != nil);
	@autoreleasepool {
		held = nil;
	}
	drainRing(fx.ring);

	// A heap a command buffer in flight keeps (its retained references), though the app let it
	// go, keeps its span till the buffer lets it go: with every other span taken, only its own is
	// picked once it has.
	uint64_t keptAt = 0;
	NSMutableArray *inFlight = [NSMutableArray array]; // what the command buffer keeps
	@autoreleasepool {
		RgfxHeap *kept = [RgfxHeap heapWithDescriptor:d device:nil serializer:&fx.ser];
		keptAt = heapAddress();
		CHECK(kept != nil);
		[inFlight addObject:kept];
		kept = nil;
	}
	CHECK(ringEmpty(fx.ring)); // its destroy waits
	int taken = 0;
	for (uint64_t at; (at = fx.ids.allocAddress(d.size, 1)) != 0; taken++)
		CHECK(at != keptAt);
	CHECK(taken > 0);
	@autoreleasepool {
		[inFlight removeAllObjects];
	}
	CHECK(nextDestroyed(fx.ring) != 0 && ringEmpty(fx.ring));
	CHECK(fx.ids.allocAddress(d.size, 1) == keptAt);
}

// Each host object the bundle makes is destroyed as the app lets it go, at once with no
// command buffer in flight: samplers, depth-stencil states, events, fences, shared events,
// buffers, heaps, libraries, and render and compute pipelines.
void test_objects_destroyed_as_let_go()
{
	Fixture fx;
	auto made = [&](id object) {
		CHECK(object != nil);
		uint64_t objId = ((RgfxResource *)object).rgfxObjId;
		CHECK(objId != 0);
		drainRing(fx.ring);
		return objId;
	};
	uint64_t objId = 0;
	@autoreleasepool {
		objId = made([RgfxSampler samplerWithDescriptor:[MTLSamplerDescriptor new]
		                                         device:nil
		                                     serializer:&fx.ser]);
	}
	CHECK(nextDestroyed(fx.ring) == objId && ringEmpty(fx.ring));
	@autoreleasepool {
		objId = made([RgfxDepthStencilState
		    depthStencilStateWithDescriptor:[MTLDepthStencilDescriptor new]
		                             device:nil
		                         serializer:&fx.ser]);
	}
	CHECK(nextDestroyed(fx.ring) == objId && ringEmpty(fx.ring));
	@autoreleasepool {
		objId = made([RgfxEvent eventWithDevice:nil serializer:&fx.ser]);
	}
	CHECK(nextDestroyed(fx.ring) == objId && ringEmpty(fx.ring));
	@autoreleasepool {
		objId = made([RgfxFence fenceWithDevice:nil serializer:&fx.ser]);
	}
	CHECK(nextDestroyed(fx.ring) == objId && ringEmpty(fx.ring));
	@autoreleasepool {
		objId = made([RgfxSharedEvent sharedEventWithInitialValue:3 device:nil serializer:&fx.ser]);
	}
	CHECK(nextDestroyed(fx.ring) == objId && ringEmpty(fx.ring));
	@autoreleasepool {
		objId = made([RgfxBuffer bufferWithLength:256 options:0 device:nil serializer:&fx.ser]);
	}
	CHECK(nextDestroyed(fx.ring) == objId && ringEmpty(fx.ring));
	@autoreleasepool {
		MTLHeapDescriptor *d = [MTLHeapDescriptor new];
		d.size = 0x10000;
		objId = made([RgfxHeap heapWithDescriptor:d device:nil serializer:&fx.ser]);
	}
	CHECK(nextDestroyed(fx.ring) == objId && ringEmpty(fx.ring));

	const uint8_t libBytes[] = {1, 2, 3, 4};
	uint32_t status = 0;
	uint64_t libId = 0;
	@autoreleasepool {
		libId = made([RgfxLibrary libraryWithData:libBytes
		                                   length:sizeof(libBytes)
		                                      key:nullptr
		                                   device:nil
		                               serializer:&fx.ser
		                                   status:&status]);
	}
	CHECK(nextDestroyed(fx.ring) == libId && ringEmpty(fx.ring));

	RgfxLibrary *lib = nil;
	@autoreleasepool {
		lib = [RgfxLibrary libraryWithData:libBytes
		                            length:sizeof(libBytes)
		                               key:nullptr
		                            device:nil
		                        serializer:&fx.ser
		                            status:&status];
		drainRing(fx.ring);
	}
	@autoreleasepool {
		MTLRenderPipelineDescriptor *desc = [MTLRenderPipelineDescriptor new];
		desc.vertexFunction = [lib newFunctionWithName:@"vmain"];
		desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
		NSError *err = nil;
		objId = made([RgfxRenderPipelineState renderPipelineStateWithDescriptor:desc
		                                                                 device:nil
		                                                             serializer:&fx.ser
		                                                                  error:&err]);
	}
	CHECK(nextDestroyed(fx.ring) == objId && ringEmpty(fx.ring));
	@autoreleasepool {
		NSError *err = nil;
		objId = made([RgfxComputePipelineState
		    computePipelineStateWithFunction:[lib newFunctionWithName:@"kmain"]
		                              device:nil
		                          serializer:&fx.ser
		                               error:&err]);
	}
	CHECK(nextDestroyed(fx.ring) == objId && ringEmpty(fx.ring));
	libId = lib.rgfxObjId;
	@autoreleasepool {
		lib = nil;
	}
	CHECK(nextDestroyed(fx.ring) == libId && ringEmpty(fx.ring)); // with its last function
}

// A function keeps its library on the host, so a pipeline made from it after the app let the
// library go still names a library the host holds; the pipeline needs the library no more.
void test_function_keeps_its_library()
{
	Fixture fx;
	const uint8_t libBytes[] = {5, 6, 7};
	uint32_t status = 0;
	uint64_t libId = 0;
	id<MTLFunction> fn = nil;
	@autoreleasepool {
		RgfxLibrary *lib = [RgfxLibrary libraryWithData:libBytes
		                                        length:sizeof(libBytes)
		                                           key:nullptr
		                                        device:nil
		                                    serializer:&fx.ser
		                                        status:&status];
		libId = lib.rgfxObjId;
		fn = [lib newFunctionWithName:@"kmain"];
		drainRing(fx.ring);
	}
	CHECK(ringEmpty(fx.ring)); // the function keeps it
	RgfxComputePipelineState *ps = nil;
	@autoreleasepool {
		NSError *err = nil;
		ps = [RgfxComputePipelineState computePipelineStateWithFunction:fn
		                                                         device:nil
		                                                     serializer:&fx.ser
		                                                          error:&err];
		CHECK(ps != nil);
		auto bytes = nextMessage(fx.ring, RGFX_CMD_COMPUTE_PIPELINE_CREATE);
		RgfxComputePipelineCreate create;
		std::memcpy(&create, bytes.data(), sizeof(create));
		CHECK(create.function_library_obj_id == libId);
	}
	@autoreleasepool {
		fn = nil;
	}
	CHECK(nextDestroyed(fx.ring) == libId && ringEmpty(fx.ring)); // with its last function
	uint64_t pipeId = ps.rgfxObjId;
	@autoreleasepool {
		ps = nil;
	}
	CHECK(nextDestroyed(fx.ring) == pipeId && ringEmpty(fx.ring));
}

// A buffer or texture placed in a heap keeps the heap (its `heap`): the heap goes from the
// host only after the last of them.
void test_heap_outlives_its_resources()
{
	Fixture fx;
	int calls = 0;
	fx.ids.setResourceIdSource(&oneChunk, &calls);
	RgfxHeap *heap = nil;
	id<MTLBuffer> buf = nil;
	id<MTLTexture> tex = nil;
	@autoreleasepool {
		MTLHeapDescriptor *d = [MTLHeapDescriptor new];
		d.size = 0x100000;
		d.type = MTLHeapTypePlacement; // only a Placement heap places at an offset
		heap = [RgfxHeap heapWithDescriptor:d device:nil serializer:&fx.ser];
		buf = [heap newBufferWithLength:256 options:0 offset:0];
		MTLTextureDescriptor *td =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
		                                                       width:8
		                                                      height:8
		                                                   mipmapped:NO];
		tex = [heap newTextureWithDescriptor:td offset:0x10000];
		CHECK(buf != nil && tex != nil);
		drainRing(fx.ring);
	}
	uint64_t heapId = heap.rgfxObjId;
	uint64_t bufId = ((RgfxResource *)buf).rgfxObjId;
	uint64_t texId = ((RgfxResource *)tex).rgfxObjId;
	@autoreleasepool {
		CHECK(buf.heap == heap && tex.heap == heap);
		heap = nil;
	}
	CHECK(ringEmpty(fx.ring)); // its resources keep it
	@autoreleasepool {
		buf = nil;
	}
	CHECK(nextDestroyed(fx.ring) == bufId && ringEmpty(fx.ring));
	@autoreleasepool {
		tex = nil;
	}
	CHECK(nextDestroyed(fx.ring) == texId);
	CHECK(nextDestroyed(fx.ring) == heapId && ringEmpty(fx.ring));
}

// A texture answers isFramebufferOnly from its descriptor's private framebufferOnly, which a
// CAMetalLayer sets for its drawables, and its views answer their texture's: NO unless set, as
// Metal's (measured on the M3 Max and the AMD 5300M).
void test_texture_framebuffer_only()
{
	Fixture fx;
	@autoreleasepool {
		MTLTextureDescriptor *desc =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
		                                                       width:16
		                                                      height:16
		                                                   mipmapped:NO];
		desc.usage = MTLTextureUsageRenderTarget;
		desc.storageMode = MTLStorageModePrivate;
		RgfxTexture *plain = [RgfxTexture textureWithDescriptor:desc
		                                                   heap:RGFX_OBJ_NONE
		                                             heapOffset:0
		                                                 device:nil
		                                             serializer:&fx.ser];
		CHECK(plain != nil && !plain.isFramebufferOnly);
		SEL set = @selector(setFramebufferOnly:);
		CHECK([desc respondsToSelector:set]);
		((void (*)(id, SEL, BOOL))objc_msgSend)(desc, set, YES);
		RgfxTexture *drawable = [RgfxTexture textureWithDescriptor:desc
		                                                      heap:RGFX_OBJ_NONE
		                                                heapOffset:0
		                                                    device:nil
		                                                serializer:&fx.ser];
		CHECK(drawable != nil && drawable.isFramebufferOnly);
		id<MTLTexture> view = [drawable newTextureViewWithPixelFormat:MTLPixelFormatBGRA8Unorm_sRGB];
		CHECK(view != nil && view.isFramebufferOnly);
		CHECK(![plain newTextureViewWithPixelFormat:MTLPixelFormatBGRA8Unorm_sRGB].isFramebufferOnly);
		drainRing(fx.ring);
	}
}

// A texture made with MTLTextureUsageUnknown answers usage 0x17, as Metal's do (measured on the
// M3 Max, the AMD 5300M and the Intel UHD 630), its views too, and its create carries the 0 the
// descriptor gave; a swizzled view keeps 0x11 of it, as of a texture made with 0x17.
void test_texture_unknown_usage()
{
	Fixture fx;
	@autoreleasepool {
		for (MTLStorageMode storage : {MTLStorageModePrivate, MTLStorageModeShared}) {
			// The pool ends the texture and its views before the ring is drained, so the next
			// create is the ring's next message.
			@autoreleasepool {
			MTLTextureDescriptor *desc =
			    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
			                                                       width:16
			                                                      height:16
			                                                   mipmapped:NO];
			desc.usage = MTLTextureUsageUnknown;
			desc.storageMode = storage;
			RgfxTexture *tex = [RgfxTexture textureWithDescriptor:desc
			                                                 heap:RGFX_OBJ_NONE
			                                           heapOffset:0
			                                               device:nil
			                                           serializer:&fx.ser];
			CHECK(tex != nil && tex.usage == 0x17);
			auto made = expectMessage<RgfxTextureCreate>(fx.ring, RGFX_CMD_TEXTURE_CREATE);
			CHECK(made.obj_id == tex.rgfxObjId && made.desc.usage == 0);
			id<MTLTexture> view = [tex newTextureViewWithPixelFormat:MTLPixelFormatBGRA8Unorm_sRGB];
			CHECK(view != nil && view.usage == 0x17);
			id<MTLTexture> swizzled = [tex
			    newTextureViewWithPixelFormat:MTLPixelFormatBGRA8Unorm
			                      textureType:MTLTextureType2D
			                           levels:NSMakeRange(0, 1)
			                           slices:NSMakeRange(0, 1)
			                          swizzle:MTLTextureSwizzleChannelsMake(
			                                      MTLTextureSwizzleBlue, MTLTextureSwizzleGreen,
			                                      MTLTextureSwizzleRed, MTLTextureSwizzleAlpha)];
			CHECK(swizzled != nil && (swizzled.usage & 0xff) == 0x11);
			}
			drainRing(fx.ring);
		}
		// A usage the descriptor names is answered as given.
		MTLTextureDescriptor *desc =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
		                                                       width:16
		                                                      height:16
		                                                   mipmapped:NO];
		desc.usage = MTLTextureUsageRenderTarget;
		desc.storageMode = MTLStorageModePrivate;
		RgfxTexture *rt = [RgfxTexture textureWithDescriptor:desc
		                                                heap:RGFX_OBJ_NONE
		                                          heapOffset:0
		                                              device:nil
		                                          serializer:&fx.ser];
		CHECK(rt != nil && rt.usage == MTLTextureUsageRenderTarget);
		CHECK([rt newTextureViewWithPixelFormat:MTLPixelFormatBGRA8Unorm_sRGB].usage ==
		      MTLTextureUsageRenderTarget);
		drainRing(fx.ring);
	}
}

// A texture's views (RGFX_CMD_TEXTURE_VIEW): each form posts its view of the texture with the
// fields it names, and the view answers its own shape and its texture's memory, as Metal's do
// (measured on the M3 Max under macOS 27.0.1). A view Metal makes none of, or asserts at,
// answers nil and posts nothing.
void test_texture_views()
{
	Fixture fx;
	@autoreleasepool {
	// A Private 2D array of four slices and seven levels, of usage 0x17.
	MTLTextureDescriptor *desc =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                                                       width:64
	                                                      height:32
	                                                   mipmapped:YES];
	desc.textureType = MTLTextureType2DArray;
	desc.arrayLength = 4;
	desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite |
	             MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
	desc.storageMode = MTLStorageModePrivate;
	RgfxTexture *tex = [RgfxTexture textureWithDescriptor:desc
	                                                 heap:RGFX_OBJ_NONE
	                                           heapOffset:0
	                                               device:nil
	                                           serializer:&fx.ser];
	CHECK(tex != nil && tex.mipmapLevelCount == 7);
	drainRing(fx.ring);

	// In a format alone: the whole texture in its type.
	id<MTLTexture> srgb = [tex newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm_sRGB];
	CHECK(srgb != nil && srgb.parentTexture == tex);
	CHECK(srgb.pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB);
	CHECK(srgb.textureType == MTLTextureType2DArray && srgb.width == 64 && srgb.height == 32);
	CHECK(srgb.mipmapLevelCount == 7 && srgb.arrayLength == 4 && srgb.usage == 0x17);
	CHECK(srgb.parentRelativeLevel == 0 && srgb.parentRelativeSlice == 0);
	CHECK(srgb.storageMode == MTLStorageModePrivate);
	CHECK(srgb.gpuResourceID._impl != 0 && srgb.gpuResourceID._impl != tex.gpuResourceID._impl);
	auto m = expectMessage<RgfxTextureView>(fx.ring, RGFX_CMD_TEXTURE_VIEW);
	CHECK(m.obj_id == ((RgfxResource *)srgb).rgfxObjId && m.parent_obj_id == tex.rgfxObjId);
	CHECK(m.resource_id == srgb.gpuResourceID._impl && m.pixel_format == 71 && m.flags == 0);
	CHECK(m.texture_type == 0 && m.level_count == 0 && m.slice_count == 0 && m.swizzle == 0);

	// Of a type, levels and slices: the texture's size at the view's first level.
	id<MTLTexture> face = [tex newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                                             textureType:MTLTextureType2D
	                                                  levels:NSMakeRange(1, 2)
	                                                  slices:NSMakeRange(2, 1)];
	CHECK(face != nil && face.textureType == MTLTextureType2D);
	CHECK(face.width == 32 && face.height == 16 && face.mipmapLevelCount == 2);
	CHECK(face.arrayLength == 1 && face.parentRelativeLevel == 1 && face.parentRelativeSlice == 2);
	m = expectMessage<RgfxTextureView>(fx.ring, RGFX_CMD_TEXTURE_VIEW);
	CHECK(m.flags == RGFX_TEXTURE_VIEW_SUBRANGE && m.texture_type == MTLTextureType2D);
	CHECK(m.level_start == 1 && m.level_count == 2 && m.slice_start == 2 && m.slice_count == 1);
	CHECK(m.swizzle == 0);

	// Of a swizzle other than the default: it keeps only ShaderRead and PixelFormatView of the
	// usage; of the default swizzle, all of it.
	MTLTextureSwizzleChannels rrr1 = MTLTextureSwizzleChannelsMake(
	    MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleOne);
	id<MTLTexture> red = [tex newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                                            textureType:MTLTextureType2DArray
	                                                 levels:NSMakeRange(0, 7)
	                                                 slices:NSMakeRange(1, 3)
	                                                swizzle:rrr1];
	CHECK(red != nil && red.usage == 0x11 && red.arrayLength == 3);
	CHECK(red.swizzle.red == MTLTextureSwizzleRed && red.swizzle.alpha == MTLTextureSwizzleOne);
	m = expectMessage<RgfxTextureView>(fx.ring, RGFX_CMD_TEXTURE_VIEW);
	CHECK(m.flags == (RGFX_TEXTURE_VIEW_SUBRANGE | RGFX_TEXTURE_VIEW_SWIZZLE));
	CHECK(m.swizzle == 0x01020202 && m.slice_start == 1 && m.slice_count == 3);
	id<MTLTexture> plain = [tex newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                                              textureType:MTLTextureType2D
	                                                   levels:NSMakeRange(0, 1)
	                                                   slices:NSMakeRange(0, 1)
	                                                  swizzle:MTLTextureSwizzleChannelsDefault];
	CHECK(plain != nil && plain.usage == 0x17);
	m = expectMessage<RgfxTextureView>(fx.ring, RGFX_CMD_TEXTURE_VIEW);
	CHECK(m.swizzle == 0x05040302);

	// By a descriptor: made as the longest form.
	MTLTextureViewDescriptor *vd = [MTLTextureViewDescriptor new];
	vd.pixelFormat = MTLPixelFormatRGBA8Unorm;
	vd.textureType = MTLTextureType2D;
	vd.levelRange = NSMakeRange(3, 1);
	vd.sliceRange = NSMakeRange(3, 1);
	id<MTLTexture> byDesc = [tex newTextureViewWithDescriptor:vd];
	CHECK(byDesc != nil && byDesc.width == 8 && byDesc.parentRelativeSlice == 3);
	m = expectMessage<RgfxTextureView>(fx.ring, RGFX_CMD_TEXTURE_VIEW);
	CHECK(m.flags == (RGFX_TEXTURE_VIEW_SUBRANGE | RGFX_TEXTURE_VIEW_SWIZZLE));
	CHECK(m.level_start == 3 && m.slice_start == 3 && m.swizzle == 0x05040302);
	CHECK(byDesc.minLOD == 0.0f);
	if ([vd respondsToSelector:@selector(minLOD)]) {
		vd.minLOD = 1.5f; // a lowest level of detail the device carries not
		CHECK([tex newTextureViewWithDescriptor:vd] == nil && ringEmpty(fx.ring));
	}

	// A view of a view: of it, at levels and slices relative to it.
	id<MTLTexture> inner = [red newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                                              textureType:MTLTextureType2D
	                                                   levels:NSMakeRange(1, 1)
	                                                   slices:NSMakeRange(1, 1)];
	CHECK(inner.parentTexture == red && inner.parentRelativeLevel == 1);
	CHECK(inner.parentRelativeSlice == 1 && inner.width == 32);
	// It keeps the swizzle of the view it is made of, as it names none.
	CHECK(inner.swizzle.alpha == MTLTextureSwizzleOne);
	m = expectMessage<RgfxTextureView>(fx.ring, RGFX_CMD_TEXTURE_VIEW);
	CHECK(m.parent_obj_id == ((RgfxResource *)red).rgfxObjId);

	// Views Metal asserts at where no oracle measured whether it does: nil, and nothing posted
	// (those every x86 Mac's Metal ends the process at are test_metal_ended_views_end_the_process).
	CHECK([tex newTextureViewWithPixelFormat:MTLPixelFormatBC1_RGBA] == nil);
	CHECK([tex newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                             textureType:MTLTextureType2D
	                                  levels:NSMakeRange(0, 0)
	                                  slices:NSMakeRange(0, 1)] == nil);
	CHECK([tex newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                             textureType:MTLTextureType2D
	                                  levels:NSMakeRange(0, 1)
	                                  slices:NSMakeRange(0, 2)] == nil);
	CHECK(ringEmpty(fx.ring));
	// A format the bundle cannot size is the host's to refuse: posted.
	CHECK([tex newTextureViewWithPixelFormat:MTLPixelFormatR32Float] != nil);
	CHECK([tex newTextureViewWithPixelFormat:MTLPixelFormatDepth32Float] != nil);
	drainRing(fx.ring);

	// Cube and 3D views.
	MTLTextureDescriptor *cd =
	    [MTLTextureDescriptor textureCubeDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                                                         size:16
	                                                    mipmapped:YES];
	RgfxTexture *cube = [RgfxTexture textureWithDescriptor:cd
	                                                  heap:RGFX_OBJ_NONE
	                                            heapOffset:0
	                                                device:nil
	                                            serializer:&fx.ser];
	id<MTLTexture> faces = [cube newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                                               textureType:MTLTextureType2DArray
	                                                    levels:NSMakeRange(0, 2)
	                                                    slices:NSMakeRange(0, 6)];
	CHECK(faces != nil && faces.arrayLength == 6 && faces.mipmapLevelCount == 2);
	id<MTLTexture> face3 = [cube newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                                               textureType:MTLTextureType2D
	                                                    levels:NSMakeRange(1, 1)
	                                                    slices:NSMakeRange(3, 1)];
	CHECK(face3 != nil && face3.width == 8 && face3.parentRelativeSlice == 3);
	MTLTextureDescriptor *dd = [MTLTextureDescriptor new];
	dd.textureType = MTLTextureType3D;
	dd.pixelFormat = MTLPixelFormatRGBA8Unorm;
	dd.width = 16;
	dd.height = 8;
	dd.depth = 4;
	dd.mipmapLevelCount = 3;
	RgfxTexture *vol = [RgfxTexture textureWithDescriptor:dd
	                                                 heap:RGFX_OBJ_NONE
	                                           heapOffset:0
	                                               device:nil
	                                           serializer:&fx.ser];
	id<MTLTexture> level1 = [vol newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                                               textureType:MTLTextureType3D
	                                                    levels:NSMakeRange(1, 2)
	                                                    slices:NSMakeRange(0, 1)];
	CHECK(level1 != nil && level1.width == 8 && level1.height == 4 && level1.depth == 2);
	drainRing(fx.ring);

	// A view of a texture over a buffer: the buffer's, and its CPU copies are the texture's.
	std::vector<uint8_t> memory(16 + 32 * 3, 0xee);
	RgfxBuffer *buf = [RgfxBuffer bufferOverHostMemoryWithContents:memory.data()
	                                                  regionOffset:0x10000
	                                                        length:memory.size()
	                                                       options:MTLResourceStorageModeShared
	                                                   deallocator:nil
	                                                        device:nil
	                                                    serializer:&fx.ser];
	MTLTextureDescriptor *bd =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:4
	                                                      height:3
	                                                   mipmapped:NO];
	bd.storageMode = MTLStorageModeShared;
	RgfxTexture *over = [RgfxTexture textureWithDescriptor:bd
	                                                buffer:buf
	                                                offset:16
	                                           bytesPerRow:32
	                                                device:nil
	                                            serializer:&fx.ser];
	id<MTLTexture> asWords = [over newTextureViewWithPixelFormat:MTLPixelFormatR32Uint];
	CHECK(asWords.buffer == buf && asWords.bufferOffset == 16 && asWords.bufferBytesPerRow == 32);
	CHECK(asWords.allocatedSize == over.allocatedSize);
	CHECK(over.allocatedSize == buf.allocatedSize); // the buffer's, as Metal answers
	const uint32_t words[2] = {0x11223344, 0x55667788};
	[asWords replaceRegion:MTLRegionMake2D(1, 2, 2, 1) mipmapLevel:0 withBytes:words bytesPerRow:8];
	CHECK(std::memcmp(&memory[16 + 32 * 2 + 4], words, 8) == 0);
	uint32_t back[2] = {};
	[asWords getBytes:back bytesPerRow:8 fromRegion:MTLRegionMake2D(1, 2, 2, 1) mipmapLevel:0];
	CHECK(back[0] == words[0] && back[1] == words[1]);
	// A level the view lacks: nothing copied.
	[asWords replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:1 withBytes:words bytesPerRow:4];
	CHECK(memory[16] == 0xee);
	drainRing(fx.ring);

	// A view of a heap texture: the heap's, at the texture's offset, aliased as the texture
	// is; making the view aliasable changes nothing, and its purgeable state stays NonVolatile.
	MTLHeapDescriptor *hd = [MTLHeapDescriptor new];
	hd.size = 0x100000;
	hd.type = MTLHeapTypePlacement;
	RgfxHeap *heap = [RgfxHeap heapWithDescriptor:hd device:nil serializer:&fx.ser];
	id<MTLTexture> placed = [heap newTextureWithDescriptor:bd offset:0x10000];
	id<MTLTexture> placedView = [placed newTextureViewWithPixelFormat:MTLPixelFormatBGRA8Unorm_sRGB];
	CHECK(placedView.heap == heap && placedView.heapOffset == 0x10000);
	CHECK(placed.isAliasable && placedView.isAliasable); // a Placement heap's
	CHECK([placedView setPurgeableState:MTLPurgeableStateVolatile] == MTLPurgeableStateNonVolatile);
	CHECK([placed setPurgeableState:MTLPurgeableStateKeepCurrent] == MTLPurgeableStateNonVolatile);
	drainRing(fx.ring);
	}
}

// A view keeps its texture: the texture's destroy is posted only after the view's.
void test_view_keeps_its_texture()
{
	Fixture fx;
	RgfxTexture *tex = nil;
	id<MTLTexture> view = nil;
	@autoreleasepool {
		MTLTextureDescriptor *d =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
		                                                       width:8
		                                                      height:8
		                                                   mipmapped:NO];
		tex = [RgfxTexture textureWithDescriptor:d
		                                    heap:RGFX_OBJ_NONE
		                              heapOffset:0
		                                  device:nil
		                              serializer:&fx.ser];
		view = [tex newTextureViewWithPixelFormat:MTLPixelFormatBGRA8Unorm_sRGB];
		CHECK(view != nil);
		drainRing(fx.ring);
	}
	uint64_t texId = tex.rgfxObjId;
	uint64_t viewId = ((RgfxResource *)view).rgfxObjId;
	@autoreleasepool {
		tex = nil;
	}
	CHECK(ringEmpty(fx.ring)); // the view keeps it
	@autoreleasepool {
		view = nil;
	}
	CHECK(nextDestroyed(fx.ring) == viewId);
	CHECK(nextDestroyed(fx.ring) == texId && ringEmpty(fx.ring));
}

// The region memory under a buffer or heap over host memory goes back only once the host has
// dropped the object, after any command buffer in flight that keeps it has let it go.
void test_region_memory_given_back_after_destroy()
{
	Fixture fx;
	TestRing *ring = &fx.ring;
	__block int released = 0;
	__block bool destroyedFirst = false;
	void (^deallocator)(void) = ^{
		RgfxMsgHeader h = {};
		std::vector<uint8_t> bytes;
		destroyedFirst = ring->next(&h, &bytes) && h.type == RGFX_CMD_OBJECT_DESTROY;
		released++;
	};
	static uint8_t page[0x1000];
	NSMutableArray *inFlight = [NSMutableArray array]; // what a command buffer in flight keeps
	@autoreleasepool {
		RgfxBuffer *b = [RgfxBuffer bufferOverHostMemoryWithContents:page
		                                                regionOffset:RGFX_CONTEXT_ALLOCATIONS_OFFSET
		                                                      length:sizeof(page)
		                                                     options:MTLResourceStorageModeShared
		                                                 deallocator:deallocator
		                                                      device:nil
		                                                  serializer:&fx.ser];
		CHECK(b != nil);
		MTLHeapDescriptor *d = [MTLHeapDescriptor new];
		d.size = 0x10000;
		d.storageMode = MTLStorageModeShared;
		RgfxHeap *h = [RgfxHeap heapOverHostMemoryWithDescriptor:d
		                                            regionOffset:RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x1000
		                                                contents:nullptr
		                                             deallocator:deallocator
		                                                  device:nil
		                                              serializer:&fx.ser];
		CHECK(h != nil);
		[inFlight addObject:b];
		[inFlight addObject:h];
		drainRing(fx.ring);
	}
	CHECK(released == 0 && ringEmpty(fx.ring)); // the buffer in flight keeps them
	@autoreleasepool {
		[inFlight removeAllObjects];
	}
	CHECK(released == 2 && destroyedFirst);
	CHECK(ringEmpty(fx.ring));
}

// The bytes of a surface's shared-block header (RgfxIOSurface.mm's RgfxSurfaceHeader), which
// the writers follow: a copy's seed and serial, and a cached block's CPU generation at 16.
constexpr uint64_t kSurfaceHeader = 24;

// An IOSurface texture's copy goes as the texture goes: the texture, then the copy's buffer,
// from the host, and only then the copy's block back to the region.
void test_iosurface_copy_block_after_destroys()
{
	Fixture fx;
	int calls = 0;
	fx.ids.setResourceIdSource(&oneChunk, &calls);
	std::vector<uint8_t> mem(RGFX_ALLOCATIONS_OFFSET + 0x100000, 0);
	RgfxRegion region(mem.data(), mem.size());
	struct Log {
		BlockLog blocks;
		TestRing *ring;
		std::vector<uint64_t> seenAtRelease; // the objects destroyed on the ring by each release
		static bool alloc(void *c, uint64_t size, uint64_t align, uint64_t *offset)
		{
			return BlockLog::alloc(&static_cast<Log *>(c)->blocks, size, align, offset);
		}
		static void release(void *c, uint64_t offset)
		{
			Log *log = static_cast<Log *>(c);
			BlockLog::release(&log->blocks, offset);
			RgfxMsgHeader h = {};
			std::vector<uint8_t> bytes;
			while (log->ring->next(&h, &bytes)) {
				RgfxObjectDestroy d = {};
				if (h.type == RGFX_CMD_OBJECT_DESTROY && bytes.size() >= sizeof(d)) {
					std::memcpy(&d, bytes.data(), sizeof(d));
					log->seenAtRelease.push_back(d.obj_id);
				}
			}
		}
	} log;
	log.ring = &fx.ring;
	region.setBlockSource(&Log::alloc, &Log::release, &log);
	NSDictionary *props = @{
		(id)kIOSurfaceWidth : @16,
		(id)kIOSurfaceHeight : @16,
		(id)kIOSurfaceBytesPerElement : @4,
		(id)kIOSurfacePixelFormat : @(static_cast<uint32_t>('BGRA')),
	};
	IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)props);
	CHECK(surface != nullptr);
	uint64_t texId = 0, bufId = 0;
	@autoreleasepool {
		MTLTextureDescriptor *td =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
		                                                       width:16
		                                                      height:16
		                                                   mipmapped:NO];
		RgfxIOSurfaceTexture *t = [RgfxIOSurfaceTexture textureWithDescriptor:td
		                                                            iosurface:surface
		                                                                plane:0
		                                                               region:&region
		                                                               device:nil
		                                                           serializer:&fx.ser];
		CHECK(t != nil);
		texId = t.rgfxObjId;
		bufId = expectMessage<RgfxBufferOverHostMemory>(fx.ring, RGFX_CMD_BUFFER_OVER_HOST_MEMORY)
		            .obj_id;
		drainRing(fx.ring);
		CHECK(log.blocks.allocated.size() == 1);
	}
	CHECK(log.blocks.released == log.blocks.allocated);
	CHECK(log.seenAtRelease == std::vector<uint64_t>({texId, bufId}));
	CHECK(ringEmpty(fx.ring));
	CFRelease(surface);
}

// A surface plane's GPU view is the plane's shared block where the kext gives one: synced from
// the pages only where no holder has synced it since their last CPU write (its header names the
// block and the seed), let go through the surface source as the texture goes; and the process's
// own block where the kext gives none.
void test_iosurface_shared_block()
{
	Fixture fx;
	int calls = 0;
	fx.ids.setResourceIdSource(&oneChunk, &calls);
	std::vector<uint8_t> mem(RGFX_ALLOCATIONS_OFFSET + 0x100000, 0);
	RgfxRegion region(mem.data(), mem.size());
	BlockLog blocks;
	region.setBlockSource(&BlockLog::alloc, &BlockLog::release, &blocks);
	struct Source {
		uint64_t offset = RGFX_ALLOCATIONS_OFFSET + 0x40000;
		uint64_t serial = 0;
		bool fresh = true;
		bool give = true;
		const void *plane = nullptr;
		uint64_t length = 0, size = 0;
		int acquired = 0;
		std::vector<uint64_t> released;
		static bool acquire(void *c, const void *plane, uint64_t length, uint64_t size,
		                    uint64_t *offset, uint64_t *serial, bool *fresh)
		{
			Source *src = static_cast<Source *>(c);
			src->plane = plane;
			src->length = length;
			src->size = size;
			src->acquired++;
			*offset = src->offset;
			*serial = src->serial;
			*fresh = src->fresh;
			return src->give;
		}
		static void release(void *c, uint64_t offset)
		{
			static_cast<Source *>(c)->released.push_back(offset);
		}
	} src;
	region.setSurfaceSource(&Source::acquire, &Source::release, &src);
	// 16 x 16 BGRA: rows 256 bytes apart in the block, its header at 4096, the block 8192 bytes.
	const uint64_t kPitch = 256, kHeader = 4096;
	auto fill = [](IOSurfaceRef s, uint32_t v) {
		IOSurfaceLock(s, 0, nullptr);
		auto *base = static_cast<uint8_t *>(IOSurfaceGetBaseAddress(s));
		for (int y = 0; y < 16; y++)
			for (int x = 0; x < 16; x++)
				reinterpret_cast<uint32_t *>(base + y * IOSurfaceGetBytesPerRow(s))[x] = v;
		IOSurfaceUnlock(s, 0, nullptr);
	};
	auto made = [&](uint32_t v) {
		NSDictionary *props = @{
			(id)kIOSurfaceWidth : @16,
			(id)kIOSurfaceHeight : @16,
			(id)kIOSurfaceBytesPerElement : @4,
			(id)kIOSurfacePixelFormat : @(static_cast<uint32_t>('BGRA')),
		};
		IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef)props);
		fill(s, v);
		return s;
	};
	uint8_t *block = mem.data() + src.offset;
	auto blockIs = [&](uint32_t v) {
		for (int y = 0; y < 16; y++)
			for (int x = 0; x < 16; x++)
				if (reinterpret_cast<uint32_t *>(block + y * kPitch)[x] != v)
					return false;
		return true;
	};
	auto header = [&]() { return reinterpret_cast<uint64_t *>(block + kHeader); };
	auto headerSeed = [&]() { return *reinterpret_cast<uint32_t *>(block + kHeader + 8); };
	auto texture = [&](IOSurfaceRef s) {
		MTLTextureDescriptor *td =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
		                                                       width:16
		                                                      height:16
		                                                   mipmapped:NO];
		return [RgfxIOSurfaceTexture textureWithDescriptor:td
		                                        iosurface:s
		                                            plane:0
		                                           region:&region
		                                           device:nil
		                                       serializer:&fx.ser];
	};

	// The first holder: a fresh block, synced from the pages, its header its own.
	src.serial = 41;
	IOSurfaceRef first = made(0x11223344);
	@autoreleasepool {
		RgfxIOSurfaceTexture *t = texture(first);
		CHECK(t != nil && src.acquired == 1 && blocks.allocated.empty());
		CHECK(src.plane == IOSurfaceGetBaseAddress(first) &&
		      src.length == IOSurfaceGetBytesPerRow(first) * 16 && src.size == 8192);
		auto create =
		    expectMessage<RgfxBufferOverHostMemory>(fx.ring, RGFX_CMD_BUFFER_OVER_HOST_MEMORY);
		CHECK(create.region_offset == src.offset && create.length == 8192);
		CHECK(blockIs(0x11223344));
		CHECK(*header() == 41 && headerSeed() == IOSurfaceGetSeed(first));
		drainRing(fx.ring);
	}
	CHECK(src.released == std::vector<uint64_t>({src.offset}));

	// Another process synced it with the pages already: no copy, until a CPU writes them.
	src.serial = 42;
	src.fresh = false;
	src.released.clear();
	IOSurfaceRef second = made(0x55555555);
	std::memset(block, 0xaa, kHeader);
	*header() = 42;
	*reinterpret_cast<uint32_t *>(block + kHeader + 8) = IOSurfaceGetSeed(second);
	@autoreleasepool {
		RgfxIOSurfaceTexture *t = texture(second);
		CHECK(t != nil && blockIs(0xaaaaaaaa));
		syncAsCommitted({t});
		CHECK(blockIs(0xaaaaaaaa));
		fill(second, 0x66666666);
		// A commit that names no view of the surface leaves it; one that names a view syncs it.
		syncAsCommitted({});
		CHECK(blockIs(0xaaaaaaaa));
		id<MTLTexture> view = [t newTextureViewWithPixelFormat:MTLPixelFormatBGRA8Unorm];
		CHECK(view != nil);
		syncAsCommitted({view});
		CHECK(blockIs(0x66666666) && headerSeed() == IOSurfaceGetSeed(second));
		drainRing(fx.ring);
	}
	CHECK(src.released.size() == 1);

	// A block made again holds an older block's header: synced from the pages.
	src.serial = 43;
	IOSurfaceRef third = made(0x77777777);
	@autoreleasepool {
		RgfxIOSurfaceTexture *t = texture(third);
		CHECK(t != nil && blockIs(0x77777777) && *header() == 43);
		drainRing(fx.ring);
	}

	// No shared block: the process's own.
	src.give = false;
	IOSurfaceRef fourth = made(0x12121212);
	@autoreleasepool {
		RgfxIOSurfaceTexture *t = texture(fourth);
		CHECK(t != nil && blocks.allocated.size() == 1);
		drainRing(fx.ring);
	}
	CHECK(blocks.released == blocks.allocated && src.released.size() == 2);
	drainRing(fx.ring); // the textures' and buffers' destroys
	for (IOSurfaceRef s : {first, second, third, fourth})
		CFRelease(s);
}

// A surface with planes has one copy holding every plane (RgfxIOSurface.h): each plane's rows in
// plane order, 256 bytes apart, then the header and the writers, one shared block found by plane
// 0's pages. A texture over any plane lies in that copy's one buffer at its plane's offset, and its
// create carries the surface's facts: its OSType, planes, size, plane 1's place in the copy and its
// YCbCr matrix. A texture from an origin inside a plane lies there in the copy. A surface without
// planes keeps its one plane at the copy's start, with no chroma and, without the matrix value,
// none.
void test_iosurface_planes()
{
	Fixture fx;
	int calls = 0;
	fx.ids.setResourceIdSource(&oneChunk, &calls);
	std::vector<uint8_t> mem(RGFX_ALLOCATIONS_OFFSET + 0x100000, 0);
	RgfxRegion region(mem.data(), mem.size());
	BlockLog blocks;
	region.setBlockSource(&BlockLog::alloc, &BlockLog::release, &blocks);
	struct Source {
		uint64_t offset = RGFX_ALLOCATIONS_OFFSET + 0x40000;
		const void *plane = nullptr;
		uint64_t length = 0, size = 0;
		int acquired = 0;
		static bool acquire(void *c, const void *plane, uint64_t length, uint64_t size,
		                    uint64_t *offset, uint64_t *serial, bool *fresh)
		{
			Source *src = static_cast<Source *>(c);
			src->plane = plane;
			src->length = length;
			src->size = size;
			src->acquired++;
			*offset = src->offset;
			*serial = 5;
			*fresh = true;
			return true;
		}
		static void release(void *, uint64_t) {}
	} src;
	region.setSurfaceSource(&Source::acquire, &Source::release, &src);
	// A 64 x 48 'x420' surface as CoreVideo lays one out: plane 0 of 16-bit luma, plane 1 of
	// 32-bit chroma pairs at half size, both 128 bytes a row, the matrix CoreVideo leaves there
	// for a BT.2020 attachment.
	const uint32_t kX420 = 0x78343230;
	NSDictionary *props = @{
		(id)kIOSurfaceWidth : @64,
		(id)kIOSurfaceHeight : @48,
		(id)kIOSurfacePixelFormat : @(kX420),
		(id)kIOSurfaceAllocSize : @(128 * 48 + 128 * 24),
		(id)kIOSurfacePlaneInfo : @[
			@{
				(id)kIOSurfacePlaneWidth : @64,
				(id)kIOSurfacePlaneHeight : @48,
				(id)kIOSurfacePlaneBytesPerElement : @2,
				(id)kIOSurfacePlaneBytesPerRow : @128,
				(id)kIOSurfacePlaneOffset : @0,
				(id)kIOSurfacePlaneSize : @(128 * 48),
			},
			@{
				(id)kIOSurfacePlaneWidth : @32,
				(id)kIOSurfacePlaneHeight : @24,
				(id)kIOSurfacePlaneBytesPerElement : @4,
				(id)kIOSurfacePlaneBytesPerRow : @128,
				(id)kIOSurfacePlaneOffset : @(128 * 48),
				(id)kIOSurfacePlaneSize : @(128 * 24),
			},
		],
	};
	IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)props);
	CHECK(surface != nullptr && IOSurfaceGetPlaneCount(surface) == 2);
	IOSurfaceSetValue(surface, CFSTR("IOSurfaceYCbCrMatrix"), CFSTR("ITU_R_2020"));
	IOSurfaceLock(surface, 0, nullptr);
	for (size_t p = 0; p < 2; p++) {
		auto *base = static_cast<uint8_t *>(IOSurfaceGetBaseAddressOfPlane(surface, p));
		for (size_t r = 0; r < IOSurfaceGetHeightOfPlane(surface, p); r++)
			for (size_t i = 0; i < 128; i++)
				base[r * 128 + i] = static_cast<uint8_t>(p * 100 + r * 7 + i);
	}
	IOSurfaceUnlock(surface, 0, nullptr);
	// The copy: plane 0's rows from 0, plane 1's from 12288, 256 bytes apart; the header at
	// 18432, the writers after it, the block 20480 bytes.
	const uint64_t kPitch = 256, kChroma = 12288, kSize = 20480;
	const uint64_t kWriters = 18432 + kSurfaceHeader;
	uint8_t *block = mem.data() + src.offset;
	auto copied = [&](size_t p, uint64_t at, size_t rows) {
		for (size_t r = 0; r < rows; r++)
			for (size_t i = 0; i < 128; i++)
				if (block[at + r * kPitch + i] != static_cast<uint8_t>(p * 100 + r * 7 + i))
					return false;
		return true;
	};
	auto overPlane = [&](IOSurfaceRef s, MTLPixelFormat format, NSUInteger w, NSUInteger h,
	                     NSUInteger plane) {
		MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
		                                                                              width:w
		                                                                             height:h
		                                                                          mipmapped:NO];
		return [RgfxIOSurfaceTexture textureWithDescriptor:td
		                                        iosurface:s
		                                            plane:plane
		                                           region:&region
		                                           device:nil
		                                       serializer:&fx.ser];
	};
	@autoreleasepool {
		RgfxIOSurfaceTexture *luma = overPlane(surface, MTLPixelFormatR16Unorm, 64, 48, 0);
		CHECK(luma != nil && src.acquired == 1);
		CHECK(src.plane == IOSurfaceGetBaseAddressOfPlane(surface, 0) && src.length == 128 * 48 &&
		      src.size == kSize);
		auto buffer =
		    expectMessage<RgfxBufferOverHostMemory>(fx.ring, RGFX_CMD_BUFFER_OVER_HOST_MEMORY);
		CHECK(buffer.region_offset == src.offset && buffer.length == kSize);
		auto first = expectMessage<RgfxSurfaceTextureMessage>(fx.ring,
		                                                      RGFX_CMD_TEXTURE_FROM_IOSURFACE);
		CHECK(first.texture.buffer_obj_id == buffer.obj_id && first.texture.offset == 0 &&
		      first.texture.bytes_per_row == kPitch && first.texture.plane == 0 &&
		      first.texture.writers_offset == kWriters);
		CHECK(first.facts.surface_pixel_format == kX420 && first.facts.plane_count == 2 &&
		      first.facts.width == 64 && first.facts.height == 48 &&
		      first.facts.chroma_offset == kChroma && first.facts.chroma_bytes_per_row == kPitch &&
		      first.facts.ycbcr_matrix == RGFX_YCBCR_MATRIX_ITU_R_2020 &&
		      first.facts.reserved_24 == 0);
		CHECK(copied(0, 0, 48) && copied(1, kChroma, 24));

		// Plane 1: the same copy, at its offset; no second block or buffer.
		RgfxIOSurfaceTexture *chroma = overPlane(surface, MTLPixelFormatRG16Unorm, 32, 24, 1);
		CHECK(chroma != nil && src.acquired == 1 && blocks.allocated.empty());
		auto second = expectMessage<RgfxSurfaceTextureMessage>(fx.ring,
		                                                       RGFX_CMD_TEXTURE_FROM_IOSURFACE);
		CHECK(second.texture.buffer_obj_id == buffer.obj_id &&
		      second.texture.offset == kChroma && second.texture.bytes_per_row == kPitch &&
		      second.texture.plane == 1 &&
		      std::memcmp(&second.facts, &first.facts, sizeof(first.facts)) == 0);
		CHECK(chroma.iosurfacePlane == 1 && luma.iosurfacePlane == 0);
		CHECK([(id<RgfxMetalTextureInternals>)chroma gpuAddress] ==
		      [(id<RgfxMetalTextureInternals>)luma gpuAddress] + kChroma);
		CHECK(chroma.allocatedSize == kSize && luma.allocatedSize == kSize);

		// The CPU's access to plane 1 reads and writes its rows in the copy and in the surface.
		uint32_t texel = 0;
		[chroma getBytes:&texel bytesPerRow:4 fromRegion:MTLRegionMake2D(1, 2, 1, 1) mipmapLevel:0];
		CHECK(std::memcmp(&texel, block + kChroma + 2 * kPitch + 4, 4) == 0);
		const uint32_t written = 0xa1b2c3d4;
		[chroma replaceRegion:MTLRegionMake2D(1, 2, 1, 1)
		          mipmapLevel:0
		            withBytes:&written
		          bytesPerRow:4];
		CHECK(std::memcmp(block + kChroma + 2 * kPitch + 4, &written, 4) == 0);
		IOSurfaceLock(surface, kIOSurfaceLockReadOnly, nullptr);
		auto *plane1 = static_cast<uint8_t *>(IOSurfaceGetBaseAddressOfPlane(surface, 1));
		CHECK(std::memcmp(plane1 + 2 * 128 + 4, &written, 4) == 0);
		IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, nullptr);

		// A plane the surface lacks: no texture, nothing sent.
		CHECK(overPlane(surface, MTLPixelFormatR16Unorm, 64, 48, 2) == nil);
		CHECK(ringEmpty(fx.ring));

		// Textures inside each plane, from an origin of its elements (a tile's place in its
		// canvas): in the same buffer at the plane's offset and the origin's rows and elements in
		// the copy, the GPU address there too; the CPU's access reads and writes nothing. One that
		// would reach past its plane's rows or elements: none, nothing sent.
		auto inside = [&](MTLPixelFormat format, NSUInteger x, NSUInteger y, NSUInteger w, NSUInteger h,
		                  NSUInteger plane) {
			MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
			                                                                              width:w
			                                                                             height:h
			                                                                          mipmapped:NO];
			return [RgfxIOSurfaceTexture textureWithDescriptor:td
			                                        iosurface:surface
			                                            plane:plane
			                                           origin:MTLOriginMake(x, y, 0)
			                                           region:&region
			                                           device:nil
			                                       serializer:&fx.ser];
		};
		RgfxIOSurfaceTexture *tile = inside(MTLPixelFormatR16Unorm, 32, 16, 32, 32, 0);
		CHECK(tile != nil && src.acquired == 1 && blocks.allocated.empty());
		auto third = expectMessage<RgfxSurfaceTextureMessage>(fx.ring, RGFX_CMD_TEXTURE_FROM_IOSURFACE);
		CHECK(third.texture.buffer_obj_id == buffer.obj_id &&
		      third.texture.offset == 16 * kPitch + 32 * 2 && third.texture.bytes_per_row == kPitch &&
		      third.texture.plane == 0);
		CHECK([(id<RgfxMetalTextureInternals>)tile gpuAddress] ==
		      [(id<RgfxMetalTextureInternals>)luma gpuAddress] + 16 * kPitch + 32 * 2);
		RgfxIOSurfaceTexture *tileChroma = inside(MTLPixelFormatRG16Unorm, 16, 8, 16, 16, 1);
		CHECK(tileChroma != nil);
		auto fourth = expectMessage<RgfxSurfaceTextureMessage>(fx.ring, RGFX_CMD_TEXTURE_FROM_IOSURFACE);
		CHECK(fourth.texture.buffer_obj_id == buffer.obj_id &&
		      fourth.texture.offset == kChroma + 8 * kPitch + 16 * 4 &&
		      fourth.texture.bytes_per_row == kPitch && fourth.texture.plane == 1);
		CHECK([(id<RgfxMetalTextureInternals>)tileChroma gpuAddress] ==
		      [(id<RgfxMetalTextureInternals>)luma gpuAddress] + kChroma + 8 * kPitch + 16 * 4);
		const uint8_t before = block[16 * kPitch + 32 * 2];
		const uint16_t sample = 0xbeef;
		[tile replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:&sample bytesPerRow:2];
		uint16_t read = 0x1234;
		[tile getBytes:&read bytesPerRow:2 fromRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0];
		CHECK(block[16 * kPitch + 32 * 2] == before && read == 0x1234);
		// Past plane 0's 48 rows, past its 64 elements, past plane 1's 24 rows.
		CHECK(inside(MTLPixelFormatR16Unorm, 32, 16, 32, 33, 0) == nil);
		CHECK(inside(MTLPixelFormatR16Unorm, 34, 16, 32, 32, 0) == nil);
		CHECK(inside(MTLPixelFormatRG16Unorm, 16, 8, 16, 17, 1) == nil);
		CHECK(ringEmpty(fx.ring));
		drainRing(fx.ring);
	}
	drainRing(fx.ring); // the textures' and the buffer's destroys

	// A surface without planes: its plane at the copy's start, no chroma, no matrix value.
	NSDictionary *flat = @{
		(id)kIOSurfaceWidth : @16,
		(id)kIOSurfaceHeight : @16,
		(id)kIOSurfaceBytesPerElement : @4,
		(id)kIOSurfacePixelFormat : @(static_cast<uint32_t>('BGRA')),
	};
	IOSurfaceRef plain = IOSurfaceCreate((__bridge CFDictionaryRef)flat);
	CHECK(plain != nullptr);
	@autoreleasepool {
		RgfxIOSurfaceTexture *t = overPlane(plain, MTLPixelFormatBGRA8Unorm, 16, 16, 0);
		CHECK(t != nil && src.length == IOSurfaceGetBytesPerRow(plain) * 16 && src.size == 8192);
		(void)expectMessage<RgfxBufferOverHostMemory>(fx.ring, RGFX_CMD_BUFFER_OVER_HOST_MEMORY);
		auto m = expectMessage<RgfxSurfaceTextureMessage>(fx.ring, RGFX_CMD_TEXTURE_FROM_IOSURFACE);
		CHECK(m.texture.offset == 0 && m.texture.bytes_per_row == kPitch && m.texture.plane == 0);
		CHECK(m.facts.surface_pixel_format == static_cast<uint32_t>('BGRA') &&
		      m.facts.plane_count == 0 && m.facts.width == 16 && m.facts.height == 16 &&
		      m.facts.chroma_offset == 0 && m.facts.chroma_bytes_per_row == 0 &&
		      m.facts.ycbcr_matrix == RGFX_YCBCR_MATRIX_NONE);
		CHECK(overPlane(plain, MTLPixelFormatBGRA8Unorm, 16, 16, 1) == nil);
		drainRing(fx.ring);
	}
	drainRing(fx.ring);
	CFRelease(surface);
	CFRelease(plain);
}

// A surface copy's writers (RgfxIOSurface.h): a block made for this process starts with none
// whatever it held, and its texture names them to the host; each commit of a buffer that draws
// the copy names its queue and seqno there, in the queue's own slot, raised; another queue takes
// an empty slot, never one another process holds, or one of ours whose queue is gone; with none
// left, nothing is named; and the slots of ours go empty as the copy goes.
void test_iosurface_writers()
{
	// Never freed, as Fixture's: the queues and their buffers may go after the test.
	TestRing &control = *new TestRing(0x8000);
	control.setConsumerIdle(1);
	RgfxRingWriter &controlWriter = *new RgfxRingWriter(
	    control.base(), control.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell, nullptr);
	RgfxIdSpace &ids = *new RgfxIdSpace();
	int &calls = *new int(0);
	ids.setResourceIdSource(&oneChunk, &calls);
	RgfxSerializer &ser = *new RgfxSerializer(controlWriter, ids);
	TestRing &submit = *new TestRing(0x8000);
	submit.setConsumerIdle(1);
	RgfxRingWriter &submitWriter =
	    *new RgfxRingWriter(submit.base(), submit.dataSize, 1, TestRing::doorbell, nullptr);
	std::vector<uint8_t> &mem = *new std::vector<uint8_t>(RGFX_ALLOCATIONS_OFFSET + 0x100000, 0);
	RgfxRegion &region = *new RgfxRegion(mem.data(), mem.size());
	struct Source {
		uint64_t offset = RGFX_ALLOCATIONS_OFFSET + 0x40000;
		static bool acquire(void *c, const void *, uint64_t, uint64_t, uint64_t *offset,
		                    uint64_t *serial, bool *fresh)
		{
			*offset = static_cast<Source *>(c)->offset;
			*serial = 7;
			*fresh = true;
			return true;
		}
		static void release(void *, uint64_t) {}
	} &src = *new Source();
	region.setSurfaceSource(&Source::acquire, &Source::release, &src);
	auto queueMade = [&]() {
		drainRing(control); // a queue let go before has its destroys there
		RgfxCommandQueue *q = [RgfxCommandQueue queueWithSubmitRing:&submitWriter
		                                              submitOffset:0
		                                            completeOffset:0
		                                          completeDataSize:0
		                                                       ids:&ids
		                                                  maxCount:0
		                                                    device:nil
		                                                serializer:&ser];
		CHECK(q != nil);
		(void)expectQueueCreate(control);
		return q;
	};
	// 16 x 16 BGRA: rows 256 bytes apart, the header at 4096, the writers after it.
	const uint64_t kWriters = 4096 + kSurfaceHeader;
	auto *writers = reinterpret_cast<RgfxSurfaceWriters *>(mem.data() + src.offset + kWriters);
	// The block held an older one's writers.
	for (auto &w : writers->writers)
		w = {0x5000000000 + 9, 77};
	NSDictionary *props = @{
		(id)kIOSurfaceWidth : @16,
		(id)kIOSurfaceHeight : @16,
		(id)kIOSurfaceBytesPerElement : @4,
		(id)kIOSurfacePixelFormat : @(static_cast<uint32_t>('BGRA')),
	};
	IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)props);
	CHECK(surface != nullptr);
	// A buffer drawing the texture, committed on `q`: the seqno it was posted at.
	auto draw = [&](RgfxCommandQueue *q, id<MTLTexture> t) {
		id<MTLCommandBuffer> cb = [q commandBuffer];
		MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
		rp.colorAttachments[0].texture = t;
		rp.colorAttachments[0].loadAction = MTLLoadActionClear;
		rp.colorAttachments[0].storeAction = MTLStoreActionStore;
		[[cb renderCommandEncoderWithDescriptor:rp] endEncoding];
		[cb commit];
		RgfxMsgHeader h = {};
		std::vector<uint8_t> bytes;
		CHECK(submit.next(&h, &bytes) && h.type == RGFX_CMD_COMMAND_BUFFER);
		return h.seqno;
	};
	auto slot = [&](int i) {
		return std::make_pair(writers->writers[i].queue_obj_id, writers->writers[i].seqno);
	};
	using Slot = std::pair<uint64_t, uint64_t>;
	const Slot kEmpty = {0, 0}, kOther = {0x6000000000 + 3, 12};
	RgfxCommandQueue *first = queueMade();
	@autoreleasepool {
		MTLTextureDescriptor *td =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
		                                                       width:16
		                                                      height:16
		                                                   mipmapped:NO];
		td.usage = MTLTextureUsageRenderTarget;
		RgfxIOSurfaceTexture *t = [RgfxIOSurfaceTexture textureWithDescriptor:td
		                                                            iosurface:surface
		                                                                plane:0
		                                                               region:&region
		                                                               device:nil
		                                                           serializer:&ser];
		CHECK(t != nil);
		(void)expectMessage<RgfxBufferOverHostMemory>(control, RGFX_CMD_BUFFER_OVER_HOST_MEMORY);
		auto made = expectMessage<RgfxSurfaceTextureMessage>(control,
		                                                     RGFX_CMD_TEXTURE_FROM_IOSURFACE)
		                .texture;
		CHECK(made.writers_offset == kWriters && made.iosurface_id == IOSurfaceGetID(surface));
		for (int i = 0; i < RGFX_SURFACE_WRITER_SLOTS; i++)
			CHECK(slot(i) == kEmpty);

		// The first queue's buffers: its slot, raised.
		uint64_t a = draw(first, t);
		CHECK(slot(0) == Slot(first.rgfxObjId, a));
		uint64_t b = draw(first, t);
		CHECK(b > a && slot(0) == Slot(first.rgfxObjId, b));
		// Another process holds slot 1: a second queue takes slot 2.
		writers->writers[1] = {kOther.first, kOther.second};
		@autoreleasepool {
			RgfxCommandQueue *second = queueMade();
			uint64_t c = draw(second, t);
			CHECK(slot(1) == kOther && slot(2) == Slot(second.rgfxObjId, c));
			second = nil;
		}
		// Once the second queue is gone, a third takes its slot again.
		RgfxCommandQueue *third = queueMade();
		uint64_t d = draw(third, t);
		CHECK(slot(2) == Slot(third.rgfxObjId, d) && slot(3) == kEmpty);
		// Another process takes the last; a fourth queue finds none, and names nothing.
		writers->writers[3] = {kOther.first + 1, kOther.second};
		RgfxCommandQueue *fourth = queueMade();
		(void)draw(fourth, t);
		CHECK(slot(0) == Slot(first.rgfxObjId, b) && slot(1) == kOther &&
		      slot(2) == Slot(third.rgfxObjId, d) && slot(3) == Slot(kOther.first + 1, 12));
		drainRing(control);
	}
	// The copy goes: our slots empty, the other process's kept.
	CHECK(slot(0) == kEmpty && slot(1) == kOther && slot(2) == kEmpty &&
	      slot(3) == Slot(kOther.first + 1, 12));
	drainRing(control);
	CFRelease(surface);
}

// A process's commit order (RgfxCommitOrder.hpp; design.md, "Commit order"): a queue that takes
// part names, before its buffer goes on its ring, once and before its first other op
// (RGFX_OP_COMMIT_ORDER), the latest buffer each other queue that takes part put on its ring since
// this queue last named that queue: never its own queue, nor a queue that takes no part, nor one
// that put nothing on its ring since. A buffer that finds nothing new carries no op.
void test_commit_order()
{
	// Never freed, as Fixture's: the queues and their buffers may go after the test.
	TestRing &control = *new TestRing(0x8000);
	control.setConsumerIdle(1);
	RgfxRingWriter &controlWriter = *new RgfxRingWriter(
	    control.base(), control.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell, nullptr);
	RgfxIdSpace &ids = *new RgfxIdSpace();
	int &calls = *new int(0);
	ids.setResourceIdSource(&oneChunk, &calls);
	RgfxSerializer &ser = *new RgfxSerializer(controlWriter, ids);
	TestRing &submit = *new TestRing(0x8000);
	submit.setConsumerIdle(1);
	RgfxRingWriter &submitWriter =
	    *new RgfxRingWriter(submit.base(), submit.dataSize, 1, TestRing::doorbell, nullptr);
	auto order = std::make_shared<RgfxCommitOrder>();
	auto queueMade = [&](bool joins) {
		drainRing(control);
		RgfxCommandQueue *q = [RgfxCommandQueue queueWithSubmitRing:&submitWriter
		                                              submitOffset:0
		                                            completeOffset:0
		                                          completeDataSize:0
		                                                       ids:&ids
		                                                  maxCount:0
		                                                    device:nil
		                                                serializer:&ser];
		CHECK(q != nil);
		(void)expectQueueCreate(control);
		if (joins)
			[q rgfxJoinCommitOrder:order];
		return q;
	};
	using Named = std::vector<std::pair<uint64_t, uint64_t>>;
	// What a commit carried: its seqno, whether it named its order, and the buffers it named.
	struct Committed {
		uint64_t seqno = 0;
		bool order = false;
		Named named;
	};
	auto commit = [&](RgfxCommandQueue *q) {
		id<MTLCommandBuffer> cb = [q commandBuffer];
		[cb commit];
		RgfxMsgHeader h = {};
		std::vector<uint8_t> bytes;
		CHECK(submit.next(&h, &bytes) && h.type == RGFX_CMD_COMMAND_BUFFER);
		Committed out;
		out.seqno = h.seqno;
		size_t at = RGFX_COMMAND_BUFFER_OPS_OFFSET;
		bool first = true;
		while (at + sizeof(RgfxOpHeader) <= bytes.size()) {
			RgfxOpHeader op = {};
			std::memcpy(&op, bytes.data() + at, sizeof(op));
			CHECK(op.len >= RGFX_OP_BODY_OFFSET && at + op.len <= bytes.size());
			if (op.tag == RGFX_OP_COMMIT_ORDER) {
				const size_t body = op.len - RGFX_OP_BODY_OFFSET;
				CHECK(first && !out.order && body % sizeof(RgfxSurfaceOrderWait) == 0);
				out.order = true;
				for (size_t i = 0; i < body / sizeof(RgfxSurfaceOrderWait); i++) {
					RgfxSurfaceOrderWait w = {};
					std::memcpy(&w, bytes.data() + at + RGFX_OP_BODY_OFFSET + i * sizeof(w),
					            sizeof(w));
					out.named.push_back({w.queue_obj_id, w.seqno});
				}
				std::sort(out.named.begin(), out.named.end());
				CHECK(!out.named.empty());
			}
			first = false;
			at += op.len;
		}
		drainRing(control);
		return out;
	};
	auto sorted = [](Named n) {
		std::sort(n.begin(), n.end());
		return n;
	};
	@autoreleasepool {
		RgfxCommandQueue *a = queueMade(true), *b = queueMade(true), *c = queueMade(true);
		RgfxCommandQueue *apart = queueMade(false);
		// Nothing on another queue's ring yet: no op.
		Committed a1 = commit(a);
		CHECK(!a1.order);
		// A buffer after it on another queue names it.
		Committed b1 = commit(b);
		CHECK(b1.order && b1.named == Named({{a.rgfxObjId, a1.seqno}}));
		// Nothing new since: no op.
		Committed b2 = commit(b);
		CHECK(!b2.order);
		// Never its own queue: a names b's latest, once.
		Committed a2 = commit(a);
		CHECK(a2.order && a2.named == Named({{b.rgfxObjId, b2.seqno}}));
		// A queue that takes no part names nothing and is named by none.
		Committed x1 = commit(apart);
		CHECK(!x1.order);
		// A third queue's first buffer names each other's latest.
		Committed c1 = commit(c);
		CHECK(c1.order &&
		      c1.named == sorted({{a.rgfxObjId, a2.seqno}, {b.rgfxObjId, b2.seqno}}));
		Committed b3 = commit(b);
		CHECK(b3.order &&
		      b3.named == sorted({{a.rgfxObjId, a2.seqno}, {c.rgfxObjId, c1.seqno}}));
		CHECK(!commit(apart).order);
	}
	drainRing(control);
}

// Two queues posting at once never name each other's buffers both ways: a post reads the other
// queue's latest before its buffer goes on its ring and raises its own after, so where one waits
// for the other, the other was on its ring before the read, and its own read came before this
// one's raise.
void test_commit_order_names_no_cycle()
{
	RgfxCommitOrder order;
	constexpr int kPosts = 20000;
	struct Post {
		uint64_t seqno;
		uint64_t named; // the other queue's seqno it names, 0 for none
	};
	std::vector<Post> posts[2];
	auto run = [&](int me) {
		const size_t slot = order.join(0x100 + static_cast<uint64_t>(me));
		CHECK(slot != RgfxCommitOrder::kNoSlot);
		RgfxCommitNamed named;
		uint64_t last = 0;
		for (int i = 1; i <= kPosts; i++) {
			std::vector<RgfxSurfaceOrderWait> waits;
			order.name(slot, named, waits);
			CHECK(waits.size() <= 1);
			// What it names is cumulative: the latest named stands for every one before it.
			if (!waits.empty())
				last = waits[0].seqno;
			posts[me].push_back({static_cast<uint64_t>(i), last});
			order.posted(slot, static_cast<uint64_t>(i));
		}
	};
	std::thread t0(run, 0), t1(run, 1);
	t0.join();
	t1.join();
	// Post p of queue 0 waits for queue 1 through p.named; post q of queue 1 waits for queue 0
	// through q.named. A cycle: q.seqno <= p.named and p.seqno <= q.named.
	size_t j = 0;
	for (const Post &p : posts[0]) {
		// The posts of queue 1 that p waits for are 1..p.named; any of them that waits for p
		// names queue 0 at p.seqno or past. The named seqnos only rise, so the first of them that
		// can is the one to check: queue 1's posts up to p.named, the last of which names most.
		if (p.named == 0)
			continue;
		while (j + 1 < posts[1].size() && posts[1][j + 1].seqno <= p.named)
			j++;
		CHECK(posts[1][j].seqno <= p.named);
		CHECK(posts[1][j].named < p.seqno);
	}
}

// A process that names its command buffers' order (rgfxIOSurfaceSetOrder; RgfxIOSurface.h): the
// post of a buffer that uses a surface names, once and before its first other op
// (RGFX_OP_SURFACE_ORDER), the buffers of other queues it runs after, as the surface's block names
// them at the post: of a surface it only reads, the writers; of one it writes, the writers and
// the readers; each queue once, at its highest seqno; never its own queue, nor a slot whose seqno
// is 0. Once posted, a buffer that only reads the surface raises its queue's reader slot, and one
// that writes it its writer slot. A surface named by no other queue gives the op with no wait; a
// surface no buffer of another queue used names nothing of the other's; a buffer that uses no
// surface carries no op, and nor does any buffer without the order, whose reads raise no slot.
// A buffer committed behind one not yet committed names them as it is posted, after that one.
// A block made for this process starts with no readers, whatever it held.
void test_iosurface_order()
{
	// Never freed, as Fixture's: the queues and their buffers may go after the test.
	TestRing &control = *new TestRing(0x8000);
	control.setConsumerIdle(1);
	RgfxRingWriter &controlWriter = *new RgfxRingWriter(
	    control.base(), control.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell, nullptr);
	RgfxIdSpace &ids = *new RgfxIdSpace();
	int &calls = *new int(0);
	ids.setResourceIdSource(&oneChunk, &calls);
	RgfxSerializer &ser = *new RgfxSerializer(controlWriter, ids);
	TestRing &submit = *new TestRing(0x8000);
	submit.setConsumerIdle(1);
	RgfxRingWriter &submitWriter =
	    *new RgfxRingWriter(submit.base(), submit.dataSize, 1, TestRing::doorbell, nullptr);
	std::vector<uint8_t> &mem = *new std::vector<uint8_t>(RGFX_ALLOCATIONS_OFFSET + 0x100000, 0);
	RgfxRegion &region = *new RgfxRegion(mem.data(), mem.size());
	struct Source {
		uint64_t next = RGFX_ALLOCATIONS_OFFSET + 0x40000;
		static bool acquire(void *c, const void *, uint64_t, uint64_t size, uint64_t *offset,
		                    uint64_t *serial, bool *fresh)
		{
			Source *src = static_cast<Source *>(c);
			*offset = src->next;
			src->next += (size + 0xfff) & ~uint64_t(0xfff);
			*serial = 7;
			*fresh = true;
			return true;
		}
		static void release(void *, uint64_t) {}
	} &src = *new Source();
	region.setSurfaceSource(&Source::acquire, &Source::release, &src);
	auto queueMade = [&]() {
		drainRing(control);
		RgfxCommandQueue *q = [RgfxCommandQueue queueWithSubmitRing:&submitWriter
		                                              submitOffset:0
		                                            completeOffset:0
		                                          completeDataSize:0
		                                                       ids:&ids
		                                                  maxCount:0
		                                                    device:nil
		                                                serializer:&ser];
		CHECK(q != nil);
		(void)expectQueueCreate(control);
		return q;
	};
	using Slot = std::pair<uint64_t, uint64_t>;
	using Waits = std::vector<Slot>;
	// What a commit carried: its seqno, whether it named its order, and the waits it named.
	struct Committed {
		uint64_t seqno = 0;
		bool order = false;
		Waits waits;
	};
	// The next buffer on the queues' ring, as posted.
	auto posted = [&]() {
		RgfxMsgHeader h = {};
		std::vector<uint8_t> bytes;
		CHECK(submit.next(&h, &bytes) && h.type == RGFX_CMD_COMMAND_BUFFER);
		Committed out;
		out.seqno = h.seqno;
		size_t at = RGFX_COMMAND_BUFFER_OPS_OFFSET;
		bool first = true;
		while (at + sizeof(RgfxOpHeader) <= bytes.size()) {
			RgfxOpHeader op = {};
			std::memcpy(&op, bytes.data() + at, sizeof(op));
			CHECK(op.len >= RGFX_OP_BODY_OFFSET && at + op.len <= bytes.size());
			if (op.tag == RGFX_OP_SURFACE_ORDER) {
				const size_t body = op.len - RGFX_OP_BODY_OFFSET;
				CHECK(first && !out.order && body % sizeof(RgfxSurfaceOrderWait) == 0);
				out.order = true;
				for (size_t i = 0; i < body / sizeof(RgfxSurfaceOrderWait); i++) {
					RgfxSurfaceOrderWait w = {};
					std::memcpy(&w, bytes.data() + at + RGFX_OP_BODY_OFFSET + i * sizeof(w),
					            sizeof(w));
					out.waits.push_back({w.queue_obj_id, w.seqno});
				}
				std::sort(out.waits.begin(), out.waits.end());
			}
			first = false;
			at += op.len;
		}
		drainRing(control);
		return out;
	};
	auto commit = [&](id<MTLCommandBuffer> cb) {
		[cb commit];
		return posted();
	};
	auto sorted = [](Waits w) {
		std::sort(w.begin(), w.end());
		return w;
	};
	NSDictionary *props = @{
		(id)kIOSurfaceWidth : @16,
		(id)kIOSurfaceHeight : @16,
		(id)kIOSurfaceBytesPerElement : @4,
		(id)kIOSurfacePixelFormat : @(static_cast<uint32_t>('BGRA')),
	};
	IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)props);
	IOSurfaceRef unrelated = IOSurfaceCreate((__bridge CFDictionaryRef)props);
	CHECK(surface != nullptr && unrelated != nullptr);
	// 16 x 16 BGRA: rows 256 bytes apart, the header at 4096, the writers after it, then the
	// readers; the surface's block first, the unrelated one's 8192 bytes on.
	const uint64_t kWriters = 4096 + kSurfaceHeader;
	auto *writers = reinterpret_cast<RgfxSurfaceWriters *>(mem.data() + src.next + kWriters);
	RgfxSurfaceWriters *readers = writers + 1;
	auto *otherReaders =
	    reinterpret_cast<RgfxSurfaceWriters *>(mem.data() + src.next + 0x2000 + kWriters) + 1;
	// The block held an older one's readers.
	for (auto &r : readers->writers)
		r = {0x5000000000 + 9, 77};
	auto slot = [](const RgfxSurfaceWriters *rec, int i) {
		return Slot(rec->writers[i].queue_obj_id, rec->writers[i].seqno);
	};
	const Slot kEmpty = {0, 0};
	// Other processes' buffers, as their slots name them.
	const Slot kOtherWriter = {0x6000000000 + 3, 12}, kOtherReader = {0x7000000000 + 5, 30};
	auto texture = [&](IOSurfaceRef s) {
		MTLTextureDescriptor *td =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
		                                                       width:16
		                                                      height:16
		                                                   mipmapped:NO];
		td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
		RgfxIOSurfaceTexture *t = [RgfxIOSurfaceTexture textureWithDescriptor:td
		                                                            iosurface:s
		                                                                plane:0
		                                                               region:&region
		                                                               device:nil
		                                                           serializer:&ser];
		CHECK(t != nil);
		drainRing(control);
		return t;
	};
	id<MTLBuffer> buffer = [RgfxBuffer bufferWithLength:16 * 16 * 4
	                                            options:MTLResourceStorageModeShared
	                                             device:nil
	                                         serializer:&ser];
	CHECK(buffer != nil);
	drainRing(control);
	// A buffer of `q` that reads `t` (a blit from it), and one that writes it (an attachment).
	auto reading = [&](RgfxCommandQueue *q, id<MTLTexture> t) {
		id<MTLCommandBuffer> cb = [q commandBuffer];
		id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
		[be copyFromTexture:t
		             sourceSlice:0
		             sourceLevel:0
		            sourceOrigin:MTLOriginMake(0, 0, 0)
		              sourceSize:MTLSizeMake(16, 16, 1)
		                toBuffer:buffer
		       destinationOffset:0
		  destinationBytesPerRow:64
		destinationBytesPerImage:64 * 16];
		[be endEncoding];
		return cb;
	};
	auto read = [&](RgfxCommandQueue *q, id<MTLTexture> t) { return commit(reading(q, t)); };
	auto write = [&](RgfxCommandQueue *q, id<MTLTexture> t) {
		id<MTLCommandBuffer> cb = [q commandBuffer];
		MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
		rp.colorAttachments[0].texture = t;
		rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
		rp.colorAttachments[0].storeAction = MTLStoreActionStore;
		[[cb renderCommandEncoderWithDescriptor:rp] endEncoding];
		return commit(cb);
	};
	rgfxIOSurfaceSetOrder(true);
	RgfxCommandQueue *a = queueMade(), *b = queueMade();
	@autoreleasepool {
		RgfxIOSurfaceTexture *t = texture(surface);
		RgfxIOSurfaceTexture *u = texture(unrelated);
		for (int i = 0; i < RGFX_SURFACE_WRITER_SLOTS; i++)
			CHECK(slot(writers, i) == kEmpty && slot(readers, i) == kEmpty);

		// Named by no queue: the op, with no wait; once posted, its queue among the readers.
		Committed r1 = read(a, t);
		CHECK(r1.order && r1.waits.empty());
		CHECK(slot(readers, 0) == Slot(a.rgfxObjId, r1.seqno) && slot(writers, 0) == kEmpty);

		// Other processes' buffers: a writer and a reader. A read waits for the writer alone
		// (RAW), never for its own queue's reader slot, and raises that slot.
		writers->writers[1] = {kOtherWriter.first, kOtherWriter.second};
		readers->writers[1] = {kOtherReader.first, kOtherReader.second};
		Committed r2 = read(a, t);
		CHECK(r2.order && r2.waits == Waits{kOtherWriter});
		CHECK(r2.seqno > r1.seqno && slot(readers, 0) == Slot(a.rgfxObjId, r2.seqno));

		// A write of another queue waits for the writers (WAW) and the readers (WAR), its own
		// process's included; once posted, its queue among the writers.
		Committed w1 = write(b, t);
		CHECK(w1.order && w1.waits == sorted({kOtherWriter, kOtherReader,
		                                       Slot(a.rgfxObjId, r2.seqno)}));
		CHECK(slot(writers, 0) == Slot(b.rgfxObjId, w1.seqno) && slot(writers, 1) == kOtherWriter);

		// A read after it waits for it (RAW), and for the other process's writer.
		Committed r3 = read(a, t);
		CHECK(r3.order && r3.waits == sorted({kOtherWriter, Slot(b.rgfxObjId, w1.seqno)}));
		CHECK(slot(readers, 0) == Slot(a.rgfxObjId, r3.seqno));

		// A queue in more than one slot, of the writers and the readers, is named once, at its
		// highest seqno; a slot being given (seqno 0) is not named.
		writers->writers[2] = {kOtherWriter.first, 15};
		readers->writers[2] = {kOtherWriter.first, 20};
		readers->writers[3] = {0x8000000000 + 1, 0};
		Committed w2 = write(b, t);
		CHECK(w2.order && w2.waits == sorted({Slot(kOtherWriter.first, 20), kOtherReader,
		                                       Slot(a.rgfxObjId, r3.seqno)}));
		CHECK(slot(writers, 0) == Slot(b.rgfxObjId, w2.seqno));
		Committed r4 = read(a, t);
		CHECK(r4.waits == sorted({Slot(kOtherWriter.first, 15), Slot(b.rgfxObjId, w2.seqno)}));

		// An unrelated surface names nothing of the first's: read by one queue, no wait; written
		// by the other, a wait for that read alone.
		Committed r5 = read(a, u);
		CHECK(r5.order && r5.waits.empty());
		CHECK(slot(otherReaders, 0) == Slot(a.rgfxObjId, r5.seqno));
		Committed w3 = write(b, u);
		CHECK(w3.order && w3.waits == Waits{Slot(a.rgfxObjId, r5.seqno)});

		// A buffer that uses no surface carries no op.
		id<MTLCommandBuffer> cb = [a commandBuffer];
		id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
		[be fillBuffer:buffer range:NSMakeRange(0, 64) value:1];
		[be endEncoding];
		CHECK(!commit(cb).order);

		// A buffer committed behind one enqueued before it and not yet committed names its
		// order as it is posted, once that one is: a write of another queue posted meanwhile is
		// among what it runs after, and names none of it.
		id<MTLCommandBuffer> ahead = [a commandBuffer];
		[ahead enqueue];
		id<MTLCommandBuffer> held = reading(a, t);
		[held commit];
		{
			RgfxMsgHeader h = {};
			std::vector<uint8_t> bytes;
			CHECK(!submit.next(&h, &bytes));
		}
		Committed w4 = write(b, t);
		CHECK(w4.order && w4.waits == sorted({Slot(kOtherWriter.first, 20), kOtherReader,
		                                       Slot(a.rgfxObjId, r4.seqno)}));
		[ahead commit];
		CHECK(!posted().order);
		Committed r7 = posted();
		CHECK(r7.order && r7.waits == sorted({Slot(kOtherWriter.first, 15),
		                                       Slot(b.rgfxObjId, w4.seqno)}));
		CHECK(slot(readers, 0) == Slot(a.rgfxObjId, r7.seqno));

		// Without the order: no op, and a read raises no reader slot.
		rgfxIOSurfaceSetOrder(false);
		Committed r6 = read(a, t);
		CHECK(!r6.order && slot(readers, 0) == Slot(a.rgfxObjId, r7.seqno));
		CHECK(!write(b, t).order);
		rgfxIOSurfaceSetOrder(true);
		drainRing(control);
	}
	// The copies go: our slots empty, the other processes' kept.
	CHECK(slot(readers, 0) == kEmpty && slot(readers, 1) == kOtherReader);
	rgfxIOSurfaceSetOrder(false);
	drainRing(control);
	CFRelease(surface);
	CFRelease(unrelated);
}

// A queue's slot of a surface's block names its buffer only while the buffer runs
// (rgfxIOSurfaceNoteFinished; RgfxIOSurface.h). A read of a copy that another process writes
// waits for that process's writer and, while it runs, holds its reader slot, which a write of
// another process reads at its post; once the host has finished it, the slot is empty, so a
// write after it waits for nothing of it. A later buffer of the queue keeps the slot as an
// earlier one finishes. Where other processes' queues hold the other reader slots, a fifth
// queue's read finds none while this process's read runs, and takes the slot it leaves.
void test_iosurface_slots_free_at_completion()
{
	// Never freed, as Fixture's: the queues and their buffers may go after the test.
	TestRing &control = *new TestRing(0x8000);
	control.setConsumerIdle(1);
	RgfxRingWriter &controlWriter = *new RgfxRingWriter(
	    control.base(), control.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell, nullptr);
	RgfxIdSpace &ids = *new RgfxIdSpace();
	int &calls = *new int(0);
	ids.setResourceIdSource(&oneChunk, &calls);
	RgfxSerializer &ser = *new RgfxSerializer(controlWriter, ids);
	TestRing &submit = *new TestRing(0x8000);
	submit.setConsumerIdle(1);
	RgfxRingWriter &submitWriter =
	    *new RgfxRingWriter(submit.base(), submit.dataSize, 1, TestRing::doorbell, nullptr);
	TestRing &completion = *new TestRing(0x8000);
	RgfxReplyReader &reader = *new RgfxReplyReader(completion.base(), completion.dataSize);
	std::vector<uint8_t> &mem = *new std::vector<uint8_t>(RGFX_ALLOCATIONS_OFFSET + 0x100000, 0);
	RgfxRegion &region = *new RgfxRegion(mem.data(), mem.size());
	struct Source {
		uint64_t offset = RGFX_ALLOCATIONS_OFFSET + 0x40000;
		static bool acquire(void *c, const void *, uint64_t, uint64_t, uint64_t *offset,
		                    uint64_t *serial, bool *fresh)
		{
			*offset = static_cast<Source *>(c)->offset;
			*serial = 7;
			*fresh = true;
			return true;
		}
		static void release(void *, uint64_t) {}
	} &src = *new Source();
	region.setSurfaceSource(&Source::acquire, &Source::release, &src);
	auto queueMade = [&]() {
		drainRing(control);
		RgfxCommandQueue *q = [RgfxCommandQueue queueWithSubmitRing:&submitWriter
		                                              submitOffset:0
		                                            completeOffset:0
		                                          completeDataSize:0
		                                                       ids:&ids
		                                                  maxCount:0
		                                                    device:nil
		                                                serializer:&ser];
		CHECK(q != nil);
		q.replyReader = &reader;
		(void)expectQueueCreate(control);
		return q;
	};
	using Slot = std::pair<uint64_t, uint64_t>;
	using Waits = std::vector<Slot>;
	struct Committed {
		uint64_t seqno = 0;
		Waits waits;
	};
	// The buffer `cb` committed, as posted: its seqno and the waits its order named.
	auto commit = [&](id<MTLCommandBuffer> cb) {
		[cb commit];
		RgfxMsgHeader h = {};
		std::vector<uint8_t> bytes;
		CHECK(submit.next(&h, &bytes) && h.type == RGFX_CMD_COMMAND_BUFFER);
		Committed out;
		out.seqno = h.seqno;
		RgfxOpHeader op = {};
		CHECK(bytes.size() >= RGFX_COMMAND_BUFFER_OPS_OFFSET + sizeof(op));
		std::memcpy(&op, bytes.data() + RGFX_COMMAND_BUFFER_OPS_OFFSET, sizeof(op));
		CHECK(op.tag == RGFX_OP_SURFACE_ORDER);
		for (size_t i = 0; i < (op.len - RGFX_OP_BODY_OFFSET) / sizeof(RgfxSurfaceOrderWait);
		     i++) {
			RgfxSurfaceOrderWait w = {};
			std::memcpy(&w,
			            bytes.data() + RGFX_COMMAND_BUFFER_OPS_OFFSET + RGFX_OP_BODY_OFFSET +
			                i * sizeof(w),
			            sizeof(w));
			out.waits.push_back({w.queue_obj_id, w.seqno});
		}
		std::sort(out.waits.begin(), out.waits.end());
		drainRing(control);
		return out;
	};
	// The host's completion of `cb`, posted at `seqno`, delivered.
	auto completed = [&](id<MTLCommandBuffer> cb, uint64_t seqno) {
		RgfxCbComplete done = {};
		done.header.type = RGFX_EVT_CB_COMPLETE;
		done.header.size = sizeof(done);
		done.header.seqno = seqno;
		done.cb_obj_id = ((RgfxResource *)cb).rgfxObjId;
		done.status = RGFX_CB_OK;
		CHECK(completion.postFromHost(&done, sizeof(done)));
		CHECK(reader.pump() == 1);
		CHECK(cb.status == MTLCommandBufferStatusCompleted);
	};
	NSDictionary *props = @{
		(id)kIOSurfaceWidth : @16,
		(id)kIOSurfaceHeight : @16,
		(id)kIOSurfaceBytesPerElement : @4,
		(id)kIOSurfacePixelFormat : @(static_cast<uint32_t>('BGRA')),
	};
	IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)props);
	CHECK(surface != nullptr);
	// 16 x 16 BGRA: rows 256 bytes apart, the header at 4096, the writers after it, then the
	// readers.
	const uint64_t kWriters = 4096 + kSurfaceHeader;
	auto *writers = reinterpret_cast<RgfxSurfaceWriters *>(mem.data() + src.offset + kWriters);
	RgfxSurfaceWriters *readers = writers + 1;
	auto slot = [](const RgfxSurfaceWriters *rec, int i) {
		return Slot(rec->writers[i].queue_obj_id, rec->writers[i].seqno);
	};
	const Slot kEmpty = {0, 0};
	// Other processes' buffers, as their slots name them: a writer, and three readers.
	const Slot kOtherWriter = {0x6000000000 + 3, 12};
	const Slot kOtherReaders[3] = {
	    {0x7000000000 + 5, 30}, {0x7000000000 + 6, 31}, {0x7000000000 + 7, 32}};
	id<MTLBuffer> buffer = [RgfxBuffer bufferWithLength:16 * 16 * 4
	                                            options:MTLResourceStorageModeShared
	                                             device:nil
	                                         serializer:&ser];
	CHECK(buffer != nil);
	drainRing(control);
	// A buffer of `q` that reads `t` (a blit from it), and one that writes it (an attachment).
	auto reading = [&](RgfxCommandQueue *q, id<MTLTexture> t) {
		id<MTLCommandBuffer> cb = [q commandBuffer];
		id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
		[be copyFromTexture:t
		             sourceSlice:0
		             sourceLevel:0
		            sourceOrigin:MTLOriginMake(0, 0, 0)
		              sourceSize:MTLSizeMake(16, 16, 1)
		                toBuffer:buffer
		       destinationOffset:0
		  destinationBytesPerRow:64
		destinationBytesPerImage:64 * 16];
		[be endEncoding];
		return cb;
	};
	auto writing = [&](RgfxCommandQueue *q, id<MTLTexture> t) {
		id<MTLCommandBuffer> cb = [q commandBuffer];
		MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
		rp.colorAttachments[0].texture = t;
		rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
		rp.colorAttachments[0].storeAction = MTLStoreActionStore;
		[[cb renderCommandEncoderWithDescriptor:rp] endEncoding];
		return cb;
	};
	rgfxIOSurfaceSetOrder(true);
	RgfxCommandQueue *a = queueMade(), *b = queueMade(), *c = queueMade();
	@autoreleasepool {
		MTLTextureDescriptor *td =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
		                                                       width:16
		                                                      height:16
		                                                   mipmapped:NO];
		td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
		RgfxIOSurfaceTexture *t = [RgfxIOSurfaceTexture textureWithDescriptor:td
		                                                            iosurface:surface
		                                                                plane:0
		                                                               region:&region
		                                                               device:nil
		                                                           serializer:&ser];
		CHECK(t != nil);
		drainRing(control);

		// Another process writes the copy; this process's read waits for it, and while it runs
		// its queue holds a reader slot, which a write of another queue names at its post.
		writers->writers[1] = {kOtherWriter.first, kOtherWriter.second};
		id<MTLCommandBuffer> ra1 = reading(a, t);
		Committed r1 = commit(ra1);
		CHECK(r1.waits == Waits{kOtherWriter});
		CHECK(slot(readers, 0) == Slot(a.rgfxObjId, r1.seqno));
		id<MTLCommandBuffer> wb1 = writing(b, t);
		Committed w1 = commit(wb1);
		Waits w1Waits = {kOtherWriter, Slot(a.rgfxObjId, r1.seqno)};
		std::sort(w1Waits.begin(), w1Waits.end());
		CHECK(w1.waits == w1Waits);
		CHECK(slot(writers, 0) == Slot(b.rgfxObjId, w1.seqno));

		// Finished, the read leaves its slot empty, and the write still running keeps its own;
		// finished too, the write leaves its slot empty. A write after them waits for the other
		// process's writer alone.
		completed(ra1, r1.seqno);
		CHECK(slot(readers, 0) == kEmpty && slot(writers, 0) == Slot(b.rgfxObjId, w1.seqno));
		completed(wb1, w1.seqno);
		CHECK(slot(writers, 0) == kEmpty && slot(writers, 1) == kOtherWriter);
		id<MTLCommandBuffer> wc1 = writing(c, t);
		Committed w2 = commit(wc1);
		CHECK(w2.waits == Waits{kOtherWriter});
		completed(wc1, w2.seqno);
		CHECK(slot(writers, 0) == kEmpty);

		// A later buffer of the queue keeps the slot as an earlier one finishes.
		id<MTLCommandBuffer> ra2 = reading(a, t), ra3 = reading(a, t);
		Committed r2 = commit(ra2), r3 = commit(ra3);
		CHECK(slot(readers, 0) == Slot(a.rgfxObjId, r3.seqno));
		completed(ra2, r2.seqno);
		CHECK(slot(readers, 0) == Slot(a.rgfxObjId, r3.seqno));
		completed(ra3, r3.seqno);
		CHECK(slot(readers, 0) == kEmpty);

		// Every reader slot held: three by other processes' queues, one by this process's read
		// while it runs. A fifth queue's read finds none; once the read has finished, the fifth
		// queue's next read takes its slot.
		for (int i = 0; i < 3; i++)
			readers->writers[i + 1] = {kOtherReaders[i].first, kOtherReaders[i].second};
		id<MTLCommandBuffer> ra4 = reading(a, t);
		Committed r4 = commit(ra4);
		CHECK(slot(readers, 0) == Slot(a.rgfxObjId, r4.seqno));
		id<MTLCommandBuffer> rc1 = reading(c, t);
		Committed r5 = commit(rc1);
		for (int i = 0; i < RGFX_SURFACE_WRITER_SLOTS; i++)
			CHECK(slot(readers, i).first != c.rgfxObjId);
		completed(rc1, r5.seqno);
		completed(ra4, r4.seqno);
		CHECK(slot(readers, 0) == kEmpty);
		id<MTLCommandBuffer> rc2 = reading(c, t);
		Committed r6 = commit(rc2);
		CHECK(slot(readers, 0) == Slot(c.rgfxObjId, r6.seqno));
		completed(rc2, r6.seqno);
		CHECK(slot(readers, 0) == kEmpty);
		for (int i = 0; i < 3; i++)
			CHECK(slot(readers, i + 1) == kOtherReaders[i]);
		drainRing(control);
	}
	rgfxIOSurfaceSetOrder(false);
	drainRing(control);
	CFRelease(surface);
}

// A command buffer held at its post, as Metal holds one from submission (RgfxCmdBuffer's
// rgfxPost): one that waits on an event not at its value as far as the process knows, or that is
// posted behind a held buffer of its queue, names its accesses of surfaces at its CB_SCHEDULED,
// not at its post: until then no slot of its surfaces' blocks names it, so a read of another
// queue or process posted meanwhile waits for none of it, and the kext is not told of its writes,
// so a CPU lock does not wait for them. Scheduled, it names them, and a read posted after waits
// for it. A wait the process knows is met (a plain event a finished buffer of the process
// signalled, a shared event at its value) holds nothing.
void test_iosurface_held_buffer_names_at_scheduling()
{
	// Never freed, as Fixture's: the queues and their buffers may go after the test.
	TestRing &control = *new TestRing(0x8000);
	control.setConsumerIdle(1);
	RgfxRingWriter &controlWriter = *new RgfxRingWriter(
	    control.base(), control.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell, nullptr);
	RgfxIdSpace &ids = *new RgfxIdSpace();
	int &calls = *new int(0);
	ids.setResourceIdSource(&oneChunk, &calls);
	RgfxSerializer &ser = *new RgfxSerializer(controlWriter, ids);
	TestRing &submit = *new TestRing(0x8000);
	submit.setConsumerIdle(1);
	RgfxRingWriter &submitWriter =
	    *new RgfxRingWriter(submit.base(), submit.dataSize, 1, TestRing::doorbell, nullptr);
	TestRing &completion = *new TestRing(0x8000);
	RgfxReplyReader &reader = *new RgfxReplyReader(completion.base(), completion.dataSize);
	std::vector<uint8_t> &mem = *new std::vector<uint8_t>(RGFX_ALLOCATIONS_OFFSET + 0x100000, 0);
	RgfxRegion &region = *new RgfxRegion(mem.data(), mem.size());
	struct Source {
		uint64_t offset = RGFX_ALLOCATIONS_OFFSET + 0x40000;
		std::vector<std::pair<uint64_t, uint64_t>> named; // the writes named to the kext
		static bool acquire(void *c, const void *, uint64_t, uint64_t, uint64_t *offset,
		                    uint64_t *serial, bool *fresh)
		{
			*offset = static_cast<Source *>(c)->offset;
			*serial = 7;
			*fresh = true;
			return true;
		}
		static void release(void *, uint64_t) {}
		static void writes(void *c, uint64_t queue, uint64_t seqno, const uint64_t *, uint32_t)
		{
			static_cast<Source *>(c)->named.emplace_back(queue, seqno);
		}
		static void retire(void *, uint64_t, uint64_t) {}
	} &src = *new Source();
	region.setSurfaceSource(&Source::acquire, &Source::release, &src);
	region.setSurfaceWriteSource(&Source::writes, &Source::retire, &src);
	auto queueMade = [&]() {
		drainRing(control);
		RgfxCommandQueue *q = [RgfxCommandQueue queueWithSubmitRing:&submitWriter
		                                              submitOffset:0
		                                            completeOffset:0
		                                          completeDataSize:0
		                                                       ids:&ids
		                                                  maxCount:0
		                                                    device:nil
		                                                serializer:&ser];
		CHECK(q != nil);
		q.replyReader = &reader;
		(void)expectQueueCreate(control);
		return q;
	};
	using Slot = std::pair<uint64_t, uint64_t>;
	using Waits = std::vector<Slot>;
	struct Committed {
		uint64_t seqno = 0;
		Waits waits; // its order's, where it names one
	};
	// The buffer `cb` committed, as posted.
	auto commit = [&](id<MTLCommandBuffer> cb) {
		[cb commit];
		RgfxMsgHeader h = {};
		std::vector<uint8_t> bytes;
		CHECK(submit.next(&h, &bytes) && h.type == RGFX_CMD_COMMAND_BUFFER);
		Committed out;
		out.seqno = h.seqno;
		RgfxOpHeader op = {};
		CHECK(bytes.size() >= RGFX_COMMAND_BUFFER_OPS_OFFSET + sizeof(op));
		std::memcpy(&op, bytes.data() + RGFX_COMMAND_BUFFER_OPS_OFFSET, sizeof(op));
		if (op.tag == RGFX_OP_SURFACE_ORDER) {
			for (size_t i = 0; i < (op.len - RGFX_OP_BODY_OFFSET) / sizeof(RgfxSurfaceOrderWait);
			     i++) {
				RgfxSurfaceOrderWait w = {};
				std::memcpy(&w,
				            bytes.data() + RGFX_COMMAND_BUFFER_OPS_OFFSET + RGFX_OP_BODY_OFFSET +
				                i * sizeof(w),
				            sizeof(w));
				out.waits.push_back({w.queue_obj_id, w.seqno});
			}
		}
		drainRing(control);
		return out;
	};
	// The host's CB_SCHEDULED and CB_COMPLETE of `cb`, posted at `seqno`, taken.
	auto scheduled = [&](id<MTLCommandBuffer> cb, uint64_t seqno) {
		RgfxCbScheduled s = cbScheduled(seqno, ((RgfxResource *)cb).rgfxObjId);
		CHECK(completion.postFromHost(&s, sizeof(s)));
		CHECK(reader.pump() == 1);
		CHECK(cb.status == MTLCommandBufferStatusScheduled);
	};
	auto completed = [&](id<MTLCommandBuffer> cb, uint64_t seqno) {
		RgfxCbComplete done = {};
		done.header.type = RGFX_EVT_CB_COMPLETE;
		done.header.size = sizeof(done);
		done.header.seqno = seqno;
		done.cb_obj_id = ((RgfxResource *)cb).rgfxObjId;
		done.status = RGFX_CB_OK;
		CHECK(completion.postFromHost(&done, sizeof(done)));
		CHECK(reader.pump() == 1);
		CHECK(cb.status == MTLCommandBufferStatusCompleted);
	};
	NSDictionary *props = @{
		(id)kIOSurfaceWidth : @16,
		(id)kIOSurfaceHeight : @16,
		(id)kIOSurfaceBytesPerElement : @4,
		(id)kIOSurfacePixelFormat : @(static_cast<uint32_t>('BGRA')),
	};
	IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)props);
	CHECK(surface != nullptr);
	// 16 x 16 BGRA: rows 256 bytes apart, the header at 4096, then the writers.
	auto *writers =
	    reinterpret_cast<RgfxSurfaceWriters *>(mem.data() + src.offset + 4096 + kSurfaceHeader);
	auto slot = [](const RgfxSurfaceWriters *rec, int i) {
		return Slot(rec->writers[i].queue_obj_id, rec->writers[i].seqno);
	};
	const Slot kEmpty = {0, 0};
	id<MTLBuffer> buffer = [RgfxBuffer bufferWithLength:16 * 16 * 4
	                                            options:MTLResourceStorageModeShared
	                                             device:nil
	                                         serializer:&ser];
	CHECK(buffer != nil);
	drainRing(control);
	// A buffer of `q` that reads `t` (a blit from it), and one that writes it (an attachment),
	// each waiting first on `event` at `value` where one is given.
	auto reading = [&](RgfxCommandQueue *q, id<MTLTexture> t) {
		id<MTLCommandBuffer> cb = [q commandBuffer];
		id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
		[be copyFromTexture:t
		             sourceSlice:0
		             sourceLevel:0
		            sourceOrigin:MTLOriginMake(0, 0, 0)
		              sourceSize:MTLSizeMake(16, 16, 1)
		                toBuffer:buffer
		       destinationOffset:0
		  destinationBytesPerRow:64
		destinationBytesPerImage:64 * 16];
		[be endEncoding];
		return cb;
	};
	auto writing = [&](RgfxCommandQueue *q, id<MTLTexture> t, id<MTLEvent> event, uint64_t value) {
		id<MTLCommandBuffer> cb = [q commandBuffer];
		if (event != nil)
			[cb encodeWaitForEvent:event value:value];
		MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
		rp.colorAttachments[0].texture = t;
		rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
		rp.colorAttachments[0].storeAction = MTLStoreActionStore;
		[[cb renderCommandEncoderWithDescriptor:rp] endEncoding];
		return cb;
	};
	rgfxIOSurfaceSetOrder(true);
	RgfxCommandQueue *a = queueMade(), *b = queueMade();
	@autoreleasepool {
		RgfxEvent *event = [RgfxEvent eventWithDevice:nil serializer:&ser];
		RgfxSharedEvent *shared = [RgfxSharedEvent sharedEventWithInitialValue:5
		                                                                device:nil
		                                                            serializer:&ser];
		CHECK(event != nil && shared != nil);
		drainRing(control);
		MTLTextureDescriptor *td =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
		                                                       width:16
		                                                      height:16
		                                                   mipmapped:NO];
		td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
		RgfxIOSurfaceTexture *t = [RgfxIOSurfaceTexture textureWithDescriptor:td
		                                                            iosurface:surface
		                                                                plane:0
		                                                               region:&region
		                                                               device:nil
		                                                           serializer:&ser];
		CHECK(t != nil);
		drainRing(control);

		// A write that waits on a plain event nothing has signalled: posted with its surface's
		// slot and the kext's writes naming none of it, so a read of another queue waits for
		// nothing; a write posted behind it on its queue, which waits for nothing itself, is held
		// as well.
		id<MTLCommandBuffer> w1 = writing(a, t, event, 1);
		Committed c1 = commit(w1);
		CHECK(slot(writers, 0) == kEmpty && src.named.empty());
		id<MTLCommandBuffer> r1 = reading(b, t);
		Committed c2 = commit(r1);
		CHECK(c2.waits.empty());
		id<MTLCommandBuffer> w2 = writing(a, t, nil, 0);
		Committed c3 = commit(w2);
		CHECK(slot(writers, 0) == kEmpty && src.named.empty());

		// Scheduled, the first names its accesses, and a read posted after waits for it; then
		// the second.
		scheduled(w1, c1.seqno);
		CHECK(slot(writers, 0) == Slot(a.rgfxObjId, c1.seqno));
		CHECK(src.named == (std::vector<Slot>{Slot(a.rgfxObjId, c1.seqno)}));
		id<MTLCommandBuffer> r2 = reading(b, t);
		Committed c4 = commit(r2);
		CHECK(c4.waits == Waits{Slot(a.rgfxObjId, c1.seqno)});
		scheduled(w2, c3.seqno);
		CHECK(slot(writers, 0) == Slot(a.rgfxObjId, c3.seqno));
		CHECK(src.named.size() == 2 && src.named[1] == Slot(a.rgfxObjId, c3.seqno));
		completed(w1, c1.seqno);
		completed(w2, c3.seqno);
		completed(r1, c2.seqno);
		completed(r2, c4.seqno);
		CHECK(slot(writers, 0) == kEmpty);

		// A plain event a finished buffer of the process signalled, and a shared event at its
		// value: a write that waits on either names its accesses at its post.
		id<MTLCommandBuffer> signal = [b commandBuffer];
		[signal encodeSignalEvent:event value:2];
		Committed c5 = commit(signal);
		completed(signal, c5.seqno);
		id<MTLCommandBuffer> w3 = writing(a, t, event, 2);
		Committed c6 = commit(w3);
		CHECK(slot(writers, 0) == Slot(a.rgfxObjId, c6.seqno));
		id<MTLCommandBuffer> w4 = writing(a, t, shared, 5);
		Committed c7 = commit(w4);
		CHECK(slot(writers, 0) == Slot(a.rgfxObjId, c7.seqno));
		completed(w3, c6.seqno);
		completed(w4, c7.seqno);

		// A shared event below its value holds the write until it is scheduled, though the CPU
		// sets the value meanwhile; then a write that waits for nothing names at its post.
		id<MTLCommandBuffer> w5 = writing(a, t, shared, 6);
		Committed c8 = commit(w5);
		CHECK(slot(writers, 0) == kEmpty);
		shared.signaledValue = 6;
		drainRing(control);
		CHECK(slot(writers, 0) == kEmpty);
		scheduled(w5, c8.seqno);
		CHECK(slot(writers, 0) == Slot(a.rgfxObjId, c8.seqno));
		completed(w5, c8.seqno);
		id<MTLCommandBuffer> w6 = writing(a, t, nil, 0);
		Committed c9 = commit(w6);
		CHECK(slot(writers, 0) == Slot(a.rgfxObjId, c9.seqno));
		completed(w6, c9.seqno);
		drainRing(control);
	}
	rgfxIOSurfaceSetOrder(false);
	drainRing(control);
	CFRelease(surface);
}

// The kext learns of a command buffer's writes of surfaces' shared blocks (RgfxIOSurface.h), so
// a CPU lock of one waits for them: a commit of a buffer that draws a surface's copy names its
// queue, its seqno and the block and surface once, before the commit returns, and records the
// surface's seed in the copy's header; a buffer that writes no surface names nothing; the
// buffer's completion writes the copy back to the pages with no lock of the surface (its seed
// stays), and then retires the queue's writes up to its seqno.
void test_iosurface_writes_named()
{
	// Never freed, as Fixture's: the queue and its buffers may go after the test.
	TestRing &control = *new TestRing(0x8000);
	control.setConsumerIdle(1);
	RgfxRingWriter &controlWriter = *new RgfxRingWriter(
	    control.base(), control.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell, nullptr);
	RgfxIdSpace &ids = *new RgfxIdSpace();
	int &calls = *new int(0);
	ids.setResourceIdSource(&oneChunk, &calls);
	RgfxSerializer &ser = *new RgfxSerializer(controlWriter, ids);
	TestRing &submit = *new TestRing(0x8000);
	submit.setConsumerIdle(1);
	RgfxRingWriter &submitWriter =
	    *new RgfxRingWriter(submit.base(), submit.dataSize, 1, TestRing::doorbell, nullptr);
	TestRing &completion = *new TestRing(0x8000);
	RgfxReplyReader &reader = *new RgfxReplyReader(completion.base(), completion.dataSize);
	std::vector<uint8_t> &mem = *new std::vector<uint8_t>(RGFX_ALLOCATIONS_OFFSET + 0x100000, 0);
	RgfxRegion &region = *new RgfxRegion(mem.data(), mem.size());
	struct Named {
		uint64_t queue, seqno;
		std::vector<uint64_t> words;
	};
	struct Source {
		uint64_t offset = RGFX_ALLOCATIONS_OFFSET + 0x40000;
		std::vector<Named> named;
		std::vector<std::pair<uint64_t, uint64_t>> retired;
		static bool acquire(void *c, const void *, uint64_t, uint64_t, uint64_t *offset,
		                    uint64_t *serial, bool *fresh)
		{
			*offset = static_cast<Source *>(c)->offset;
			*serial = 7;
			*fresh = true;
			return true;
		}
		static void release(void *, uint64_t) {}
		static void writes(void *c, uint64_t queue, uint64_t seqno, const uint64_t *words,
		                   uint32_t count)
		{
			static_cast<Source *>(c)->named.push_back(
			    {queue, seqno, std::vector<uint64_t>(words, words + 2 * count)});
		}
		static void retire(void *c, uint64_t queue, uint64_t seqno)
		{
			static_cast<Source *>(c)->retired.emplace_back(queue, seqno);
		}
	} &src = *new Source();
	region.setSurfaceSource(&Source::acquire, &Source::release, &src);
	region.setSurfaceWriteSource(&Source::writes, &Source::retire, &src);
	RgfxCommandQueue *queue = [RgfxCommandQueue queueWithSubmitRing:&submitWriter
	                                                  submitOffset:0
	                                                completeOffset:0
	                                              completeDataSize:0
	                                                           ids:&ids
	                                                      maxCount:0
	                                                        device:nil
	                                                    serializer:&ser];
	CHECK(queue != nil);
	queue.replyReader = &reader;
	(void)expectQueueCreate(control);
	// 16 x 16 BGRA: rows 256 bytes apart, the header at 4096: its serial, then its seed.
	const uint8_t *header = mem.data() + src.offset + 4096;
	auto headerSerial = [&]() { return *reinterpret_cast<const uint64_t *>(header); };
	auto headerSeed = [&]() { return *reinterpret_cast<const uint32_t *>(header + 8); };
	NSDictionary *props = @{
		(id)kIOSurfaceWidth : @16,
		(id)kIOSurfaceHeight : @16,
		(id)kIOSurfaceBytesPerElement : @4,
		(id)kIOSurfacePixelFormat : @(static_cast<uint32_t>('BGRA')),
	};
	IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)props);
	CHECK(surface != nullptr);
	auto completed = [&](id<MTLCommandBuffer> cb, uint64_t seqno) {
		RgfxCbComplete done = {};
		done.header.type = RGFX_EVT_CB_COMPLETE;
		done.header.size = sizeof(done);
		done.header.seqno = seqno;
		done.cb_obj_id = ((RgfxResource *)cb).rgfxObjId;
		done.status = RGFX_CB_OK;
		CHECK(completion.postFromHost(&done, sizeof(done)));
		CHECK(reader.pump() == 1);
		CHECK(cb.status == MTLCommandBufferStatusCompleted);
	};
	@autoreleasepool {
		MTLTextureDescriptor *td =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
		                                                       width:16
		                                                      height:16
		                                                   mipmapped:NO];
		td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
		RgfxIOSurfaceTexture *t = [RgfxIOSurfaceTexture textureWithDescriptor:td
		                                                            iosurface:surface
		                                                                plane:0
		                                                               region:&region
		                                                               device:nil
		                                                           serializer:&ser];
		CHECK(t != nil);
		drainRing(control);

		// A buffer that draws the copy twice: named once, before the commit returns.
		id<MTLCommandBuffer> cb = [queue commandBuffer];
		for (int i = 0; i < 2; i++) {
			MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
			rp.colorAttachments[0].texture = t;
			rp.colorAttachments[0].loadAction = MTLLoadActionClear;
			rp.colorAttachments[0].storeAction = MTLStoreActionStore;
			[[cb renderCommandEncoderWithDescriptor:rp] endEncoding];
		}
		[cb commit];
		RgfxMsgHeader h = {};
		std::vector<uint8_t> bytes;
		CHECK(submit.next(&h, &bytes) && h.type == RGFX_CMD_COMMAND_BUFFER);
		CHECK(src.named.size() == 1);
		CHECK(src.named[0].queue == queue.rgfxObjId && src.named[0].seqno == h.seqno);
		const std::vector<uint64_t> words = {src.offset,
		                                     static_cast<uint64_t>(IOSurfaceGetID(surface))};
		CHECK(src.named[0].words == words);
		uint32_t seed = IOSurfaceGetSeed(surface);
		CHECK(headerSerial() == 7 && headerSeed() == seed);
		CHECK(src.retired.empty());

		// Its completion writes the copy back with no lock, then retires the queue's writes.
		completed(cb, h.seqno);
		CHECK(IOSurfaceGetSeed(surface) == seed);
		CHECK(src.retired.size() == 1 && src.retired[0].first == queue.rgfxObjId &&
		      src.retired[0].second == h.seqno);

		// A buffer that writes no surface: nothing named, nothing retired.
		id<MTLCommandBuffer> none = [queue commandBuffer];
		[[none blitCommandEncoder] endEncoding];
		[none commit];
		CHECK(submit.next(&h, &bytes) && h.type == RGFX_CMD_COMMAND_BUFFER);
		completed(none, h.seqno);
		CHECK(src.named.size() == 1 && src.retired.size() == 1);
		drainRing(control);
	}
	drainRing(control);
	CFRelease(surface);
}

// A buffer names a surface to the kext only where it writes it (RgfxSurfaceBindings): a texture
// bound where the draw's or dispatch's function writes, by its AIR arguments, or, where they are
// unknown or it writes a texture at an index function constants set, one whose usage has
// ShaderWrite or is unknown; a resource used with Write usage; and
// one used to read, where a function after it has an argument buffer that may write a texture,
// or arguments unknown. A texture a function only samples or reads, one bound with no dispatch
// after, and one used to read with no such function after, are not named, so no CPU lock waits
// for them.
void test_iosurface_writes_by_function()
{
	// Never freed, as Fixture's: the queue and its buffers may go after the test.
	TestRing &control = *new TestRing(0x8000);
	control.setConsumerIdle(1);
	RgfxRingWriter &controlWriter = *new RgfxRingWriter(
	    control.base(), control.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell, nullptr);
	RgfxIdSpace &ids = *new RgfxIdSpace();
	int &calls = *new int(0);
	ids.setResourceIdSource(&oneChunk, &calls);
	RgfxSerializer &ser = *new RgfxSerializer(controlWriter, ids);
	TestRing &submit = *new TestRing(0x8000);
	submit.setConsumerIdle(1);
	RgfxRingWriter &submitWriter =
	    *new RgfxRingWriter(submit.base(), submit.dataSize, 1, TestRing::doorbell, nullptr);
	TestRing &completion = *new TestRing(0x8000);
	RgfxReplyReader &reader = *new RgfxReplyReader(completion.base(), completion.dataSize);
	std::vector<uint8_t> &mem = *new std::vector<uint8_t>(RGFX_ALLOCATIONS_OFFSET + 0x100000, 0);
	RgfxRegion &region = *new RgfxRegion(mem.data(), mem.size());
	struct Source {
		uint64_t offset = RGFX_ALLOCATIONS_OFFSET + 0x40000;
		int named = 0;
		static bool acquire(void *c, const void *, uint64_t, uint64_t, uint64_t *offset,
		                    uint64_t *serial, bool *fresh)
		{
			*offset = static_cast<Source *>(c)->offset;
			*serial = 7;
			*fresh = true;
			return true;
		}
		static void release(void *, uint64_t) {}
		static void writes(void *c, uint64_t, uint64_t, const uint64_t *, uint32_t count)
		{
			static_cast<Source *>(c)->named += static_cast<int>(count);
		}
		static void retire(void *, uint64_t, uint64_t) {}
	} &src = *new Source();
	region.setSurfaceSource(&Source::acquire, &Source::release, &src);
	region.setSurfaceWriteSource(&Source::writes, &Source::retire, &src);
	RgfxCommandQueue *queue = [RgfxCommandQueue queueWithSubmitRing:&submitWriter
	                                                  submitOffset:0
	                                                completeOffset:0
	                                              completeDataSize:0
	                                                           ids:&ids
	                                                      maxCount:0
	                                                        device:nil
	                                                    serializer:&ser];
	CHECK(queue != nil);
	queue.replyReader = &reader;
	(void)expectQueueCreate(control);
	NSDictionary *props = @{
		(id)kIOSurfaceWidth : @16,
		(id)kIOSurfaceHeight : @16,
		(id)kIOSurfaceBytesPerElement : @4,
		(id)kIOSurfacePixelFormat : @(static_cast<uint32_t>('BGRA')),
	};
	IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)props);
	CHECK(surface != nullptr);
	// A function of `stage` whose arguments are `textures`, each (index, access).
	auto function = [&](RgfxAirStage stage,
	                    std::vector<std::pair<uint32_t, RgfxAirAccess>> textures) -> RgfxFunction * {
		RgfxFunction *fn = [RgfxFunction functionWithLibrary:nil
		                                                name:@"f"
		                                           constants:nullptr
		                                               count:0
		                                              device:nil
		                                          serializer:&ser];
		auto air = std::make_shared<RgfxAirFunction>();
		air->name = "f";
		air->stage = stage;
		for (const auto &[index, access] : textures)
			air->arguments.push_back({RgfxAirKind::Texture, index, 1, access, false, false});
		[fn rgfxSetAirFunction:air];
		return fn;
	};
	// A kernel of textures at indices function constants set, each of `accesses`, and one written
	// at texture 1.
	auto anyIndex = [&](std::vector<RgfxAirAccess> accesses) -> RgfxFunction * {
		RgfxFunction *fn = [RgfxFunction functionWithLibrary:nil
		                                                name:@"c"
		                                           constants:nullptr
		                                               count:0
		                                              device:nil
		                                          serializer:&ser];
		auto air = std::make_shared<RgfxAirFunction>();
		air->name = "c";
		air->stage = RgfxAirStage::Kernel;
		for (RgfxAirAccess access : accesses)
			air->arguments.push_back({RgfxAirKind::Texture, 0, 1, access, false, true});
		air->arguments.push_back({RgfxAirKind::Texture, 1, 1, RgfxAirAccess::Write, false, false});
		[fn rgfxSetAirFunction:air];
		return fn;
	};
	// A function of `stage` of one argument buffer at buffer 0, whose struct writes a texture or
	// not.
	auto argumentBuffer = [&](RgfxAirStage stage, bool writesTextures) -> RgfxFunction * {
		RgfxFunction *fn = [RgfxFunction functionWithLibrary:nil
		                                                name:@"a"
		                                           constants:nullptr
		                                               count:0
		                                              device:nil
		                                          serializer:&ser];
		auto air = std::make_shared<RgfxAirFunction>();
		air->name = "a";
		air->stage = stage;
		air->arguments.push_back(
		    {RgfxAirKind::IndirectBuffer, 0, 1, RgfxAirAccess::Read, writesTextures, false});
		[fn rgfxSetAirFunction:air];
		return fn;
	};
	auto compute = [&](RgfxFunction *fn) {
		NSError *err = nil;
		RgfxComputePipelineState *ps =
		    [RgfxComputePipelineState computePipelineStateWithFunction:fn
		                                                       device:nil
		                                                   serializer:&ser
		                                                        error:&err];
		CHECK(ps != nil);
		return ps;
	};
	auto render = [&](RgfxFunction *vertex, RgfxFunction *fragment) {
		MTLRenderPipelineDescriptor *desc = [[MTLRenderPipelineDescriptor alloc] init];
		desc.vertexFunction = vertex;
		desc.fragmentFunction = fragment;
		desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
		NSError *err = nil;
		RgfxRenderPipelineState *ps = [RgfxRenderPipelineState renderPipelineStateWithDescriptor:desc
		                                                                                 device:nil
		                                                                             serializer:&ser
		                                                                                  error:&err];
		CHECK(ps != nil);
		return ps;
	};
	// Commits `cb`, completes it, and answers how many surfaces it named.
	auto named = [&](id<MTLCommandBuffer> cb) {
		const int before = src.named;
		[cb commit];
		RgfxMsgHeader h = {};
		std::vector<uint8_t> bytes;
		CHECK(submit.next(&h, &bytes) && h.type == RGFX_CMD_COMMAND_BUFFER);
		RgfxCbComplete done = {};
		done.header.type = RGFX_EVT_CB_COMPLETE;
		done.header.size = sizeof(done);
		done.header.seqno = h.seqno;
		done.cb_obj_id = ((RgfxResource *)cb).rgfxObjId;
		done.status = RGFX_CB_OK;
		CHECK(completion.postFromHost(&done, sizeof(done)));
		CHECK(reader.pump() == 1);
		CHECK(cb.status == MTLCommandBufferStatusCompleted);
		drainRing(control);
		return src.named - before;
	};
	const MTLSize one = MTLSizeMake(1, 1, 1);
	@autoreleasepool {
		MTLTextureDescriptor *td =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
		                                                       width:16
		                                                      height:16
		                                                   mipmapped:NO];
		td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
		RgfxIOSurfaceTexture *t = [RgfxIOSurfaceTexture textureWithDescriptor:td
		                                                            iosurface:surface
		                                                                plane:0
		                                                               region:&region
		                                                               device:nil
		                                                           serializer:&ser];
		CHECK(t != nil);
		id<MTLTexture> view = [t newTextureViewWithPixelFormat:MTLPixelFormatBGRA8Unorm];
		CHECK(view != nil);
		RgfxComputePipelineState *samples =
		    compute(function(RgfxAirStage::Kernel, {{0, RgfxAirAccess::Sample}}));
		RgfxComputePipelineState *writesOne = compute(function(
		    RgfxAirStage::Kernel, {{0, RgfxAirAccess::Read}, {1, RgfxAirAccess::Write}}));
		RgfxComputePipelineState *unknown = compute([RgfxFunction functionWithLibrary:nil
		                                                                         name:@"u"
		                                                                    constants:nullptr
		                                                                        count:0
		                                                                       device:nil
		                                                                   serializer:&ser]);
		drainRing(control);

		// A kernel that samples it, or reads it: not named.
		id<MTLCommandBuffer> cb = [queue commandBuffer];
		id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
		[ce setComputePipelineState:samples];
		[ce setTexture:t atIndex:0];
		[ce dispatchThreadgroups:one threadsPerThreadgroup:one];
		[ce setComputePipelineState:writesOne];
		[ce dispatchThreads:one threadsPerThreadgroup:one];
		[ce endEncoding];
		CHECK(named(cb) == 0);

		// A kernel that writes it, through a view, after one that reads it: named once.
		cb = [queue commandBuffer];
		ce = [cb computeCommandEncoder];
		[ce setComputePipelineState:writesOne];
		[ce setTextures:(const id<MTLTexture>[]){t, view} withRange:NSMakeRange(0, 2)];
		[ce dispatchThreadgroups:one threadsPerThreadgroup:one];
		[ce dispatchThreadgroups:one threadsPerThreadgroup:one];
		[ce endEncoding];
		CHECK(named(cb) == 1);

		// Bound where the kernel writes, with no dispatch after: not named.
		cb = [queue commandBuffer];
		ce = [cb computeCommandEncoder];
		[ce setComputePipelineState:writesOne];
		[ce setTexture:t atIndex:1];
		[ce endEncoding];
		CHECK(named(cb) == 0);

		// A kernel whose arguments are unknown: its ShaderWrite usage counts.
		cb = [queue commandBuffer];
		ce = [cb computeCommandEncoder];
		[ce setComputePipelineState:unknown];
		[ce setTexture:t atIndex:0];
		[ce dispatchThreads:one threadsPerThreadgroup:one];
		[ce endEncoding];
		CHECK(named(cb) == 1);

		// Used to read: not named; used to write: named.
		cb = [queue commandBuffer];
		ce = [cb computeCommandEncoder];
		[ce useResource:t usage:MTLResourceUsageRead];
		[ce endEncoding];
		CHECK(named(cb) == 0);
		cb = [queue commandBuffer];
		ce = [cb computeCommandEncoder];
		[ce useResource:view usage:MTLResourceUsageRead | MTLResourceUsageWrite];
		[ce endEncoding];
		CHECK(named(cb) == 1);

		// Unknown usage allows every use: where the kernel's arguments are unknown, named.
		td.usage = MTLTextureUsageUnknown;
		RgfxIOSurfaceTexture *anyUse = [RgfxIOSurfaceTexture textureWithDescriptor:td
		                                                                 iosurface:surface
		                                                                     plane:0
		                                                                    region:&region
		                                                                    device:nil
		                                                                serializer:&ser];
		CHECK(anyUse != nil);
		cb = [queue commandBuffer];
		ce = [cb computeCommandEncoder];
		[ce setComputePipelineState:unknown];
		[ce setTexture:anyUse atIndex:0];
		[ce dispatchThreads:one threadsPerThreadgroup:one];
		[ce endEncoding];
		CHECK(named(cb) == 1);

		// A kernel that writes a texture at an index function constants set: one bound at 0 whose
		// usage has ShaderWrite is named, one whose usage has not, not; a kernel that only reads
		// at such an index: not named.
		td.usage = MTLTextureUsageShaderRead;
		RgfxIOSurfaceTexture *readOnly = [RgfxIOSurfaceTexture textureWithDescriptor:td
		                                                                   iosurface:surface
		                                                                       plane:0
		                                                                      region:&region
		                                                                      device:nil
		                                                                  serializer:&ser];
		CHECK(readOnly != nil);
		RgfxComputePipelineState *writesAnywhere =
		    compute(anyIndex({RgfxAirAccess::Read, RgfxAirAccess::Write}));
		RgfxComputePipelineState *readsAnywhere = compute(anyIndex({RgfxAirAccess::Read}));
		drainRing(control);
		for (const auto &[ps, texture, want] :
		     {std::tuple<RgfxComputePipelineState *, id<MTLTexture>, int>{writesAnywhere, t, 1},
		      {writesAnywhere, readOnly, 0},
		      {readsAnywhere, t, 0}}) {
			cb = [queue commandBuffer];
			ce = [cb computeCommandEncoder];
			[ce setComputePipelineState:ps];
			[ce setTexture:texture atIndex:0];
			[ce dispatchThreads:one threadsPerThreadgroup:one];
			[ce endEncoding];
			CHECK(named(cb) == want);
		}

		// Used to read, then a kernel whose argument buffer may write a texture: named, as a
		// discrete Mac's driver lands the write whatever usage was given; one whose argument
		// buffer writes none: not named; one whose arguments are unknown: named.
		RgfxComputePipelineState *argsWrite = compute(argumentBuffer(RgfxAirStage::Kernel, true));
		RgfxComputePipelineState *argsRead = compute(argumentBuffer(RgfxAirStage::Kernel, false));
		drainRing(control);
		for (const auto &[ps, want] :
		     {std::pair<RgfxComputePipelineState *, int>{argsWrite, 1}, {argsRead, 0}, {unknown, 1}}) {
			cb = [queue commandBuffer];
			ce = [cb computeCommandEncoder];
			[ce useResource:t usage:MTLResourceUsageRead];
			[ce setComputePipelineState:ps];
			[ce dispatchThreads:one threadsPerThreadgroup:one];
			[ce endEncoding];
			CHECK(named(cb) == want);
		}

		// A draw whose functions sample it: not named; one whose fragment function writes it at
		// the index it is bound: named. A pipeline with no fragment function binds nothing there.
		RgfxRenderPipelineState *sampling =
		    render(function(RgfxAirStage::Vertex, {{0, RgfxAirAccess::Sample}}),
		           function(RgfxAirStage::Fragment, {{0, RgfxAirAccess::Sample}}));
		RgfxRenderPipelineState *writing =
		    render(function(RgfxAirStage::Vertex, {}),
		           function(RgfxAirStage::Fragment,
		                    {{0, RgfxAirAccess::Sample}, {2, RgfxAirAccess::Write}}));
		RgfxRenderPipelineState *vertexOnly =
		    render(function(RgfxAirStage::Vertex, {{0, RgfxAirAccess::Sample}}), nil);
		drainRing(control);
		MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
		rp.renderTargetWidth = 16;
		rp.renderTargetHeight = 16;
		cb = [queue commandBuffer];
		id<MTLRenderCommandEncoder> re = [cb renderCommandEncoderWithDescriptor:rp];
		[re setRenderPipelineState:sampling];
		[re setVertexTexture:t atIndex:0];
		[re setFragmentTexture:view atIndex:0];
		[re drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
		[re setRenderPipelineState:vertexOnly];
		[re setFragmentTexture:t atIndex:2];
		[re drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
		[re useResource:t usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
		[re endEncoding];
		CHECK(named(cb) == 0);
		cb = [queue commandBuffer];
		re = [cb renderCommandEncoderWithDescriptor:rp];
		[re setRenderPipelineState:writing];
		[re setFragmentTextures:(const id<MTLTexture>[]){view, nil, t} withRange:NSMakeRange(0, 3)];
		[re drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
		[re endEncoding];
		CHECK(named(cb) == 1);
		cb = [queue commandBuffer];
		re = [cb renderCommandEncoderWithDescriptor:rp];
		[re useResource:t usage:MTLResourceUsageWrite];
		[re endEncoding];
		CHECK(named(cb) == 1);

		// Used to read by the stage whose function's argument buffer may write a texture: named
		// at the draw; by the other stage alone: not named.
		RgfxRenderPipelineState *fragmentArgs = render(
		    function(RgfxAirStage::Vertex, {}), argumentBuffer(RgfxAirStage::Fragment, true));
		drainRing(control);
		for (const auto &[stages, want] : {std::pair<MTLRenderStages, int>{MTLRenderStageFragment, 1},
		                                   {MTLRenderStageVertex, 0}}) {
			cb = [queue commandBuffer];
			re = [cb renderCommandEncoderWithDescriptor:rp];
			[re setRenderPipelineState:fragmentArgs];
			[re useResource:t usage:MTLResourceUsageRead stages:stages];
			[re drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
			[re endEncoding];
			CHECK(named(cb) == want);
		}
	}
	drainRing(control);
	CFRelease(surface);
}

// A surface whose planes the host's textures lie over, where the device uses surfaces in place
// (RgfxIOSurface.h): its GPU view is its own pages. The kext's block in place is asked for by the
// surface's base, its whole allocation in whole pages and the part the processes map, one page
// for the header and the writers; a BUFFER_OVER_SURFACE_PAGES over the allocation names the
// block, and each texture lies over its plane's own offset and rows there, its writers in the
// block, its facts plane 1's own. Nothing is copied: no block of the process's own and no copy
// block are made, syncs leave the block's bytes be, and the texture's CPU access reads and writes
// the surface's pages. The texture has no GPU address; the block goes back as the texture goes.
// A surface whose rows miss the host's alignment is copied, as before.
void test_iosurface_in_place()
{
	Fixture fx;
	int calls = 0;
	fx.ids.setResourceIdSource(&oneChunk, &calls);
	std::vector<uint8_t> mem(RGFX_ALLOCATIONS_OFFSET + 0x100000, 0);
	RgfxRegion region(mem.data(), mem.size());
	BlockLog blocks;
	region.setBlockSource(&BlockLog::alloc, &BlockLog::release, &blocks);
	struct Source {
		uint64_t offset = RGFX_ALLOCATIONS_OFFSET + 0x40000;
		uint64_t copyOffset = RGFX_ALLOCATIONS_OFFSET + 0x60000;
		const void *base = nullptr;
		uint64_t length = 0, size = 0;
		uint32_t surfaceId = 0;
		bool cacheable = false;
		int asked = 0, copies = 0;
		std::vector<uint64_t> released;
		static RgfxRegion::PagesAnswer pages(void *c, const void *base, uint64_t length,
		                                     uint64_t size, uint32_t surfaceId, bool cacheable,
		                                     uint64_t *offset, uint64_t *serial, bool *fresh,
		                                     bool *cached)
		{
			Source *src = static_cast<Source *>(c);
			src->surfaceId = surfaceId;
			src->cacheable = cacheable;
			*cached = false;
			src->base = base;
			src->length = length;
			src->size = size;
			src->asked++;
			*offset = src->offset;
			*serial = 9;
			*fresh = true;
			return RgfxRegion::kPagesGiven;
		}
		static bool acquire(void *c, const void *, uint64_t, uint64_t, uint64_t *offset,
		                    uint64_t *serial, bool *fresh)
		{
			Source *src = static_cast<Source *>(c);
			src->copies++;
			*offset = src->copyOffset;
			*serial = 10;
			*fresh = true;
			return true;
		}
		static void release(void *c, uint64_t offset)
		{
			static_cast<Source *>(c)->released.push_back(offset);
		}
	} src;
	region.setSurfaceSource(&Source::acquire, &Source::release, &src);
	region.setSurfacePagesSource(&Source::pages, 16, &src);
	// The block in place: its header first, untouched, then the writers and the readers,
	// emptied, and nothing more of the page written.
	const uint64_t kWriters = kSurfaceHeader;
	uint8_t *block = mem.data() + src.offset;
	auto blockLeft = [&]() {
		for (uint64_t i = 0; i < 0x1000; i++) {
			bool writer = i >= kWriters && i < kWriters + 2 * sizeof(RgfxSurfaceWriters);
			if (block[i] != (writer ? 0 : 0xcd))
				return false;
		}
		return true;
	};
	auto overPlane = [&](IOSurfaceRef s, MTLPixelFormat format, NSUInteger w, NSUInteger h,
	                     NSUInteger plane) {
		MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
		                                                                              width:w
		                                                                             height:h
		                                                                          mipmapped:NO];
		return [RgfxIOSurfaceTexture textureWithDescriptor:td
		                                        iosurface:s
		                                            plane:plane
		                                           region:&region
		                                           device:nil
		                                       serializer:&fx.ser];
	};
	auto allocBytes = [](IOSurfaceRef s) {
		return (static_cast<uint64_t>(IOSurfaceGetAllocSize(s)) + 0xfff) & ~uint64_t(0xfff);
	};

	// A 16 x 16 BGRA surface.
	NSDictionary *props = @{
		(id)kIOSurfaceWidth : @16,
		(id)kIOSurfaceHeight : @16,
		(id)kIOSurfaceBytesPerElement : @4,
		(id)kIOSurfacePixelFormat : @(static_cast<uint32_t>('BGRA')),
	};
	IOSurfaceRef flat = IOSurfaceCreate((__bridge CFDictionaryRef)props);
	CHECK(flat != nullptr);
	const uint64_t rowBytes = IOSurfaceGetBytesPerRow(flat);
	auto *pixels = static_cast<uint8_t *>(IOSurfaceGetBaseAddress(flat));
	IOSurfaceLock(flat, 0, nullptr);
	for (uint64_t i = 0; i < rowBytes * 16; i++)
		pixels[i] = static_cast<uint8_t>(i * 3);
	IOSurfaceUnlock(flat, 0, nullptr);
	std::memset(block, 0xcd, 0x1000);
	@autoreleasepool {
		RgfxIOSurfaceTexture *t = overPlane(flat, MTLPixelFormatBGRA8Unorm, 16, 16, 0);
		CHECK(t != nil && src.asked == 1 && src.copies == 0 && blocks.allocated.empty());
		CHECK(src.base == pixels && src.length == allocBytes(flat) && src.size == 0x1000);
		// One plane, no YCbCr: asked cached, by its IOSurfaceID.
		CHECK(src.surfaceId == IOSurfaceGetID(flat) && src.cacheable);
		auto buffer =
		    expectMessage<RgfxBufferOverSurfacePages>(fx.ring, RGFX_CMD_BUFFER_OVER_SURFACE_PAGES);
		CHECK(buffer.obj_id != 0 && buffer.block_offset == src.offset &&
		      buffer.block_size == 0x1000 && buffer.length == allocBytes(flat) &&
		      (buffer.options & MTLResourceStorageModeMask) == MTLResourceStorageModeShared &&
		      buffer.reserved_34 == 0);
		auto made =
		    expectMessage<RgfxSurfaceTextureMessage>(fx.ring, RGFX_CMD_TEXTURE_FROM_IOSURFACE);
		CHECK(made.texture.buffer_obj_id == buffer.obj_id && made.texture.offset == 0 &&
		      made.texture.bytes_per_row == rowBytes && made.texture.plane == 0 &&
		      made.texture.writers_offset == kWriters &&
		      made.texture.iosurface_id == IOSurfaceGetID(flat));
		CHECK(made.facts.plane_count == 0 && made.facts.chroma_offset == 0 &&
		      made.facts.chroma_bytes_per_row == 0);
		CHECK(ringEmpty(fx.ring) && blockLeft());
		CHECK([(id<RgfxMetalTextureInternals>)t gpuAddress] == 0 &&
		      t.allocatedSize == allocBytes(flat));

		// A CPU write of the surface reaches the GPU's view with no copy at a commit.
		IOSurfaceLock(flat, 0, nullptr);
		pixels[5] = 0x77;
		IOSurfaceUnlock(flat, 0, nullptr);
		syncAsCommitted({t});
		CHECK(blockLeft());

		// The texture's CPU access is the surface's pages.
		uint32_t texel = 0;
		[t getBytes:&texel bytesPerRow:4 fromRegion:MTLRegionMake2D(1, 2, 1, 1) mipmapLevel:0];
		CHECK(std::memcmp(&texel, pixels + 2 * rowBytes + 4, 4) == 0);
		const uint32_t written = 0x0badf00d;
		[t replaceRegion:MTLRegionMake2D(1, 2, 1, 1)
		     mipmapLevel:0
		       withBytes:&written
		     bytesPerRow:4];
		CHECK(std::memcmp(pixels + 2 * rowBytes + 4, &written, 4) == 0 && blockLeft());
		CHECK(ringEmpty(fx.ring));
	}
	rgfxIOSurfaceKeepFlush(); // the views kept past their textures go too
	drainRing(fx.ring); // the texture's and the buffer's destroys
	CHECK(src.released == std::vector<uint64_t>({src.offset}));

	// A surface with planes: one buffer over its allocation, each plane's texture at the plane's
	// own place and rows, the facts naming plane 1's.
	const uint32_t kX420 = 0x78343230;
	NSDictionary *planar = @{
		(id)kIOSurfaceWidth : @64,
		(id)kIOSurfaceHeight : @48,
		(id)kIOSurfacePixelFormat : @(kX420),
		(id)kIOSurfaceAllocSize : @(128 * 48 + 128 * 24),
		(id)kIOSurfacePlaneInfo : @[
			@{
				(id)kIOSurfacePlaneWidth : @64,
				(id)kIOSurfacePlaneHeight : @48,
				(id)kIOSurfacePlaneBytesPerElement : @2,
				(id)kIOSurfacePlaneBytesPerRow : @128,
				(id)kIOSurfacePlaneOffset : @0,
				(id)kIOSurfacePlaneSize : @(128 * 48),
			},
			@{
				(id)kIOSurfacePlaneWidth : @32,
				(id)kIOSurfacePlaneHeight : @24,
				(id)kIOSurfacePlaneBytesPerElement : @4,
				(id)kIOSurfacePlaneBytesPerRow : @128,
				(id)kIOSurfacePlaneOffset : @(128 * 48),
				(id)kIOSurfacePlaneSize : @(128 * 24),
			},
		],
	};
	IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)planar);
	CHECK(surface != nullptr && IOSurfaceGetPlaneCount(surface) == 2);
	auto *base = static_cast<uint8_t *>(IOSurfaceGetBaseAddress(surface));
	auto *chromaRows = static_cast<uint8_t *>(IOSurfaceGetBaseAddressOfPlane(surface, 1));
	const uint64_t kChroma = static_cast<uint64_t>(chromaRows - base);
	CHECK(kChroma == 128 * 48);
	IOSurfaceLock(surface, 0, nullptr);
	for (uint64_t i = 0; i < 128 * 24; i++)
		chromaRows[i] = static_cast<uint8_t>(i * 5);
	IOSurfaceUnlock(surface, 0, nullptr);
	std::memset(block, 0xcd, 0x1000);
	src.released.clear();
	@autoreleasepool {
		RgfxIOSurfaceTexture *luma = overPlane(surface, MTLPixelFormatR16Unorm, 64, 48, 0);
		CHECK(luma != nil && src.asked == 2 && src.copies == 0 && blocks.allocated.empty());
		CHECK(src.base == base && src.length == allocBytes(surface) && src.size == 0x1000);
		// Two planes: asked cached, as every surface is.
		CHECK(src.surfaceId == IOSurfaceGetID(surface) && src.cacheable);
		auto buffer =
		    expectMessage<RgfxBufferOverSurfacePages>(fx.ring, RGFX_CMD_BUFFER_OVER_SURFACE_PAGES);
		CHECK(buffer.block_offset == src.offset && buffer.length == allocBytes(surface));
		auto first =
		    expectMessage<RgfxSurfaceTextureMessage>(fx.ring, RGFX_CMD_TEXTURE_FROM_IOSURFACE);
		CHECK(first.texture.buffer_obj_id == buffer.obj_id && first.texture.offset == 0 &&
		      first.texture.bytes_per_row == 128 && first.texture.writers_offset == kWriters);
		CHECK(first.facts.surface_pixel_format == kX420 && first.facts.plane_count == 2 &&
		      first.facts.width == 64 && first.facts.height == 48 &&
		      first.facts.chroma_offset == kChroma && first.facts.chroma_bytes_per_row == 128);
		RgfxIOSurfaceTexture *chroma = overPlane(surface, MTLPixelFormatRG16Unorm, 32, 24, 1);
		CHECK(chroma != nil && src.asked == 2);
		auto second =
		    expectMessage<RgfxSurfaceTextureMessage>(fx.ring, RGFX_CMD_TEXTURE_FROM_IOSURFACE);
		CHECK(second.texture.buffer_obj_id == buffer.obj_id &&
		      second.texture.offset == kChroma && second.texture.bytes_per_row == 128 &&
		      second.texture.plane == 1 &&
		      std::memcmp(&second.facts, &first.facts, sizeof(first.facts)) == 0);
		uint32_t texel = 0;
		[chroma getBytes:&texel bytesPerRow:4 fromRegion:MTLRegionMake2D(3, 1, 1, 1) mipmapLevel:0];
		CHECK(std::memcmp(&texel, chromaRows + 128 + 12, 4) == 0);
		CHECK(ringEmpty(fx.ring) && blockLeft());
	}
	rgfxIOSurfaceKeepFlush(); // the views kept past their textures go too
	drainRing(fx.ring);
	CHECK(src.released == std::vector<uint64_t>({src.offset}));

	// Rows the host's textures cannot lie over: a copy, as before, and no ask in place.
	region.setSurfacePagesSource(&Source::pages, rowBytes * 2, &src);
	IOSurfaceRef misaligned = IOSurfaceCreate((__bridge CFDictionaryRef)props);
	CHECK(misaligned != nullptr && IOSurfaceGetBytesPerRow(misaligned) == rowBytes);
	@autoreleasepool {
		RgfxIOSurfaceTexture *t = overPlane(misaligned, MTLPixelFormatBGRA8Unorm, 16, 16, 0);
		CHECK(t != nil && src.asked == 2 && src.copies == 1);
		auto create =
		    expectMessage<RgfxBufferOverHostMemory>(fx.ring, RGFX_CMD_BUFFER_OVER_HOST_MEMORY);
		CHECK(create.region_offset == src.copyOffset && create.length == 8192);
		auto made =
		    expectMessage<RgfxSurfaceTextureMessage>(fx.ring, RGFX_CMD_TEXTURE_FROM_IOSURFACE);
		CHECK(made.texture.buffer_obj_id == create.obj_id && made.texture.bytes_per_row == 256 &&
		      made.texture.writers_offset == 4096 + kSurfaceHeader);
		rgfxIOSurfaceKeepFlush(); // the views kept past their textures go too
		drainRing(fx.ring);
	}
	rgfxIOSurfaceKeepFlush(); // the views kept past their textures go too
	drainRing(fx.ring);
	for (IOSurfaceRef s : {flat, surface, misaligned})
		CFRelease(s);
	rgfxIOSurfaceKeepFlush(); // nothing kept outlives the test's region
}

// Where the device uses no surface in place, or the kext gives no block in place, a surface's
// GPU view is a copy, as before: the ask in place is made, then the kext's copy block is asked
// for by plane 0's pages, a BUFFER_OVER_HOST_MEMORY lies over it, and it is synced from the
// pages.
// The views in place a process makes are kept past their textures (the keep, RgfxIOSurface.mm):
// a texture made over the same surface soon after asks the kext for nothing, and the view's block
// goes back once the keep lets the view go: past its time, at a flush (the process's memory
// pressure), past the keep's count of views, or at once where a texture over it was set
// volatile. In WindowServer it also stays while another process holds its surface, by the kext's
// count of the surface's clients, and goes all the same where the kext has no room for a block.
// A kept view leaves the surface's use count alone.
void test_iosurface_keep()
{
	Fixture fx;
	int calls = 0;
	fx.ids.setResourceIdSource(&oneChunk, &calls);
	std::vector<uint8_t> mem(RGFX_ALLOCATIONS_OFFSET + 0x100000, 0);
	RgfxRegion region(mem.data(), mem.size());
	BlockLog blocks;
	region.setBlockSource(&BlockLog::alloc, &BlockLog::release, &blocks);
	struct Source {
		int asked = 0;
		bool cached = false; // whether the blocks it gives are cached
		std::vector<uint64_t> released;
		// The kext's room: none until `roomAt` blocks are back, where `full`.
		bool full = false;
		size_t roomAt = 0;
		int noRoom = 0;
		static RgfxRegion::PagesAnswer pages(void *c, const void *, uint64_t, uint64_t, uint32_t,
		                                     bool, uint64_t *offset, uint64_t *serial,
		                                     bool *fresh, bool *cached)
		{
			Source *src = static_cast<Source *>(c);
			if (src->full && src->released.size() < src->roomAt) {
				src->noRoom++;
				return RgfxRegion::kPagesNoRoom;
			}
			// A block of its own for each ask.
			*offset = RGFX_ALLOCATIONS_OFFSET + 0x40000 + 0x1000 * static_cast<uint64_t>(src->asked);
			src->asked++;
			*serial = 9;
			*fresh = true;
			*cached = src->cached;
			return RgfxRegion::kPagesGiven;
		}
		static bool acquire(void *, const void *, uint64_t, uint64_t, uint64_t *, uint64_t *,
		                    bool *)
		{
			return false;
		}
		static void release(void *c, uint64_t offset)
		{
			static_cast<Source *>(c)->released.push_back(offset);
		}
	} src;
	region.setSurfaceSource(&Source::acquire, &Source::release, &src);
	region.setSurfacePagesSource(&Source::pages, 16, &src);
	NSDictionary *props = @{
		(id)kIOSurfaceWidth : @16,
		(id)kIOSurfaceHeight : @16,
		(id)kIOSurfaceBytesPerElement : @4,
		(id)kIOSurfacePixelFormat : @(static_cast<uint32_t>('BGRA')),
	};
	auto over = [&](IOSurfaceRef s) {
		MTLTextureDescriptor *td =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
		                                                      width:16
		                                                     height:16
		                                                  mipmapped:NO];
		return [RgfxIOSurfaceTexture textureWithDescriptor:td
		                                        iosurface:s
		                                            plane:0
		                                           region:&region
		                                           device:nil
		                                       serializer:&fx.ser];
	};
	IOSurfaceRef a = IOSurfaceCreate((__bridge CFDictionaryRef)props);
	CHECK(a != nullptr);

	// Kept past its texture: the next texture over the surface asks nothing, and the surface is
	// not in use.
	@autoreleasepool {
		CHECK(over(a) != nil && src.asked == 1);
	}
	drainRing(fx.ring);
	CHECK(src.released.empty() && !IOSurfaceIsInUse(a));
	@autoreleasepool {
		CHECK(over(a) != nil && src.asked == 1);
	}
	drainRing(fx.ring);
	CHECK(src.released.empty());

	// Past its time it goes.
	rgfxIOSurfaceKeepSweepAt(UINT64_MAX);
	drainRing(fx.ring);
	CHECK(src.released.size() == 1);

	// Short of its time it stays, and its next texture asks nothing.
	@autoreleasepool {
		CHECK(over(a) != nil && src.asked == 2);
	}
	rgfxIOSurfaceKeepSweepAt(0);
	drainRing(fx.ring);
	CHECK(src.released.size() == 1);
	@autoreleasepool {
		CHECK(over(a) != nil && src.asked == 2);
	}
	rgfxIOSurfaceKeepSweepAt(UINT64_MAX);
	drainRing(fx.ring);
	CHECK(src.released.size() == 2);

	// Set volatile through a texture over it: let go with the texture, the state answered as
	// before.
	@autoreleasepool {
		RgfxIOSurfaceTexture *t = over(a);
		CHECK(t != nil && src.asked == 3);
		CHECK([t setPurgeableState:MTLPurgeableStateVolatile] == MTLPurgeableStateNonVolatile);
		CHECK([t setPurgeableState:MTLPurgeableStateKeepCurrent] == MTLPurgeableStateVolatile);
	}
	drainRing(fx.ring);
	CHECK(src.released.size() == 3);

	// A flush, as at memory pressure, lets every kept view go.
	@autoreleasepool {
		CHECK(over(a) != nil && src.asked == 4);
	}
	drainRing(fx.ring);
	CHECK(src.released.size() == 3);
	rgfxIOSurfaceKeepFlush();
	drainRing(fx.ring);
	CHECK(src.released.size() == 4);

	// Past eight views the least recently textured goes.
	std::vector<IOSurfaceRef> many;
	for (int i = 0; i < 9; i++) {
		many.push_back(IOSurfaceCreate((__bridge CFDictionaryRef)props));
		@autoreleasepool {
			CHECK(over(many.back()) != nil);
		}
	}
	drainRing(fx.ring);
	CHECK(src.asked == 13 && src.released.size() == 5);
	CHECK(src.released.back() == RGFX_ALLOCATIONS_OFFSET + 0x40000 + 0x1000 * 4);
	rgfxIOSurfaceKeepFlush();
	drainRing(fx.ring);
	CHECK(src.released.size() == 13);
	for (IOSurfaceRef s : many)
		CFRelease(s);

	// Held elsewhere in the process (the test holds `a`), a view stays past 50 ms, its next
	// texture asking nothing, as a ring of drawing buffers turning at a few frames a second needs;
	// past a second it goes all the same.
	const uint64_t ms = 1000ull * 1000;
	uint64_t made = 0;
	@autoreleasepool {
		CHECK(over(a) != nil && src.asked == 14);
		made = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
	}
	rgfxIOSurfaceKeepSweepAt(made + 200 * ms);
	drainRing(fx.ring);
	CHECK(src.released.size() == 13);
	@autoreleasepool {
		CHECK(over(a) != nil && src.asked == 14);
		made = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
	}
	rgfxIOSurfaceKeepSweepAt(made + 500 * ms);
	drainRing(fx.ring);
	CHECK(src.released.size() == 13);
	rgfxIOSurfaceKeepSweepAt(made + 1500 * ms);
	drainRing(fx.ring);
	CHECK(src.released.size() == 14);

	// A surface the view alone holds goes past 50 ms, and with it the surface.
	IOSurfaceRef b = IOSurfaceCreate((__bridge CFDictionaryRef)props);
	CHECK(b != nullptr);
	const IOSurfaceID bId = IOSurfaceGetID(b);
	@autoreleasepool {
		CHECK(over(b) != nil && src.asked == 15);
		made = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
	}
	CHECK(CFGetRetainCount(b) == 2);
	CFRelease(b);
	rgfxIOSurfaceKeepSweepAt(made + 200 * ms);
	drainRing(fx.ring);
	CHECK(src.released.size() == 15);
	IOSurfaceRef gone = IOSurfaceLookup(bId);
	CHECK(gone == nullptr);
	if (gone != nullptr)
		CFRelease(gone);

	// WindowServer's keep (set here) keeps a cached block's view while another process holds its
	// surface, by the kext's count of the surface's clients in the block's header: past a second,
	// and a texture over it then asks nothing, though it comes back long past the 50 ms keep; once
	// the count is down to this process, the view goes kKeepNs past its texture, the surface with
	// it.
	rgfxIOSurfaceKeepCountClients(true);
	src.cached = true;
	auto clientsAt = [&](uint64_t block) {
		return reinterpret_cast<volatile uint32_t *>(mem.data() + block +
		                                             kRgfxSurfaceClientsOffset);
	};
	IOSurfaceRef c = IOSurfaceCreate((__bridge CFDictionaryRef)props);
	CHECK(c != nullptr);
	const IOSurfaceID cId = IOSurfaceGetID(c);
	const uint64_t cBlock = RGFX_ALLOCATIONS_OFFSET + 0x40000 + 0x1000 * 15;
	@autoreleasepool {
		CHECK(over(c) != nil && src.asked == 16);
		made = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
	}
	*clientsAt(cBlock) = 2;
	CHECK(CFGetRetainCount(c) == 2);
	CFRelease(c);
	rgfxIOSurfaceKeepSweepAt(made + 1500 * ms);
	drainRing(fx.ring);
	CHECK(src.released.size() == 15);
	@autoreleasepool {
		CHECK(over(c) != nil && src.asked == 16);
		made = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
	}
	rgfxIOSurfaceKeepSweepAt(made + 5000 * ms);
	drainRing(fx.ring);
	CHECK(src.released.size() == 15);
	*clientsAt(cBlock) = 1;
	rgfxIOSurfaceKeepSweepAt(made + 10 * ms);
	drainRing(fx.ring);
	CHECK(src.released.size() == 15);
	rgfxIOSurfaceKeepSweepAt(made + 200 * ms);
	drainRing(fx.ring);
	CHECK(src.released.size() == 16 && src.released.back() == cBlock);
	gone = IOSurfaceLookup(cId);
	CHECK(gone == nullptr);
	if (gone != nullptr)
		CFRelease(gone);

	// Another process's keep reads no count: its view goes kKeepNs past its texture once the
	// view alone holds the surface in the process, though another process holds it too.
	rgfxIOSurfaceKeepCountClients(false);
	IOSurfaceRef d = IOSurfaceCreate((__bridge CFDictionaryRef)props);
	CHECK(d != nullptr);
	const IOSurfaceID dId = IOSurfaceGetID(d);
	const uint64_t dBlock = RGFX_ALLOCATIONS_OFFSET + 0x40000 + 0x1000 * 16;
	@autoreleasepool {
		CHECK(over(d) != nil && src.asked == 17);
		made = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
	}
	*clientsAt(dBlock) = 2;
	CFRelease(d);
	rgfxIOSurfaceKeepSweepAt(made + 200 * ms);
	drainRing(fx.ring);
	CHECK(src.released.size() == 17 && src.released.back() == dBlock);
	gone = IOSurfaceLookup(dId);
	CHECK(gone == nullptr);
	if (gone != nullptr)
		CFRelease(gone);

	// WindowServer at the region's limit: where the kext has no room for a block, the region's
	// reclaim lets go of every kept view, the least recently textured first, those another
	// process's hold keeps among them, and the ask is made once more and given its block.
	rgfxIOSurfaceKeepCountClients(true);
	region.setReclaim([](void *) { return rgfxIOSurfaceKeepReclaim(); }, nullptr);
	const uint64_t eBlock = RGFX_ALLOCATIONS_OFFSET + 0x40000 + 0x1000 * 17;
	const uint64_t fBlock = RGFX_ALLOCATIONS_OFFSET + 0x40000 + 0x1000 * 18;
	IOSurfaceRef e = IOSurfaceCreate((__bridge CFDictionaryRef)props);
	IOSurfaceRef f = IOSurfaceCreate((__bridge CFDictionaryRef)props);
	CHECK(e != nullptr && f != nullptr);
	@autoreleasepool {
		CHECK(over(e) != nil && over(f) != nil && src.asked == 19);
		made = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
	}
	*clientsAt(eBlock) = 2;
	*clientsAt(fBlock) = 2;
	CFRelease(e);
	CFRelease(f);
	rgfxIOSurfaceKeepSweepAt(made + 1500 * ms);
	drainRing(fx.ring);
	CHECK(src.released.size() == 17);
	src.full = true;
	src.roomAt = 19;
	IOSurfaceRef g = IOSurfaceCreate((__bridge CFDictionaryRef)props);
	CHECK(g != nullptr);
	@autoreleasepool {
		CHECK(over(g) != nil);
	}
	drainRing(fx.ring);
	CHECK(src.noRoom == 1 && src.asked == 20);
	CHECK(src.released.size() == 19 && src.released[17] == eBlock && src.released[18] == fBlock);
	src.full = false;
	region.setReclaim(nullptr, nullptr);
	rgfxIOSurfaceKeepCountClients(false);
	CFRelease(g);
	src.cached = false;
	CFRelease(a);
	rgfxIOSurfaceKeepFlush(); // nothing kept outlives the test's region
}

void test_iosurface_in_place_fallback()
{
	Fixture fx;
	int calls = 0;
	fx.ids.setResourceIdSource(&oneChunk, &calls);
	std::vector<uint8_t> mem(RGFX_ALLOCATIONS_OFFSET + 0x100000, 0);
	RgfxRegion region(mem.data(), mem.size());
	BlockLog blocks;
	region.setBlockSource(&BlockLog::alloc, &BlockLog::release, &blocks);
	struct Source {
		uint64_t offset = RGFX_ALLOCATIONS_OFFSET + 0x40000;
		RgfxRegion::PagesAnswer answer = RgfxRegion::kPagesUnsupported;
		int asked = 0, copies = 0;
		std::vector<uint64_t> released;
		static RgfxRegion::PagesAnswer pages(void *c, const void *, uint64_t, uint64_t, uint32_t,
		                                     bool, uint64_t *, uint64_t *, bool *, bool *)
		{
			Source *src = static_cast<Source *>(c);
			src->asked++;
			return src->answer;
		}
		static bool acquire(void *c, const void *, uint64_t, uint64_t, uint64_t *offset,
		                    uint64_t *serial, bool *fresh)
		{
			Source *src = static_cast<Source *>(c);
			// A serial of its own each time, as a block made again has.
			*serial = 10 + ++src->copies;
			*offset = src->offset;
			*fresh = true;
			return true;
		}
		static void release(void *c, uint64_t offset)
		{
			static_cast<Source *>(c)->released.push_back(offset);
		}
	} src;
	region.setSurfaceSource(&Source::acquire, &Source::release, &src);
	region.setSurfacePagesSource(&Source::pages, 16, &src);
	// 16 x 16 BGRA: the copy's rows 256 bytes apart, its header at 4096, the block 8192 bytes.
	const uint64_t kPitch = 256;
	uint8_t *block = mem.data() + src.offset;
	auto blockIs = [&](uint32_t v) {
		for (int y = 0; y < 16; y++)
			for (int x = 0; x < 16; x++)
				if (reinterpret_cast<uint32_t *>(block + y * kPitch)[x] != v)
					return false;
		return true;
	};
	int asked = 0, copies = 0;
	for (RgfxRegion::PagesAnswer answer :
	     {RgfxRegion::kPagesUnsupported, RgfxRegion::kPagesRefused}) {
		src.answer = answer;
		NSDictionary *props = @{
			(id)kIOSurfaceWidth : @16,
			(id)kIOSurfaceHeight : @16,
			(id)kIOSurfaceBytesPerElement : @4,
			(id)kIOSurfacePixelFormat : @(static_cast<uint32_t>('BGRA')),
		};
		IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef)props);
		CHECK(s != nullptr);
		IOSurfaceLock(s, 0, nullptr);
		auto *base = static_cast<uint8_t *>(IOSurfaceGetBaseAddress(s));
		for (int y = 0; y < 16; y++)
			for (int x = 0; x < 16; x++)
				reinterpret_cast<uint32_t *>(base + y * IOSurfaceGetBytesPerRow(s))[x] =
				    0x2468ace0 + answer;
		IOSurfaceUnlock(s, 0, nullptr);
		@autoreleasepool {
			MTLTextureDescriptor *td =
			    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
			                                                       width:16
			                                                      height:16
			                                                   mipmapped:NO];
			RgfxIOSurfaceTexture *t = [RgfxIOSurfaceTexture textureWithDescriptor:td
			                                                            iosurface:s
			                                                                plane:0
			                                                               region:&region
			                                                               device:nil
			                                                           serializer:&fx.ser];
			CHECK(t != nil && src.asked == ++asked && src.copies == ++copies &&
			      blocks.allocated.empty());
			auto create =
			    expectMessage<RgfxBufferOverHostMemory>(fx.ring, RGFX_CMD_BUFFER_OVER_HOST_MEMORY);
			CHECK(create.region_offset == src.offset && create.length == 8192);
			auto made = expectMessage<RgfxSurfaceTextureMessage>(fx.ring,
			                                                     RGFX_CMD_TEXTURE_FROM_IOSURFACE);
			CHECK(made.texture.buffer_obj_id == create.obj_id && made.texture.offset == 0 &&
			      made.texture.bytes_per_row == kPitch &&
			      made.texture.writers_offset == 4096 + kSurfaceHeader);
			CHECK(blockIs(0x2468ace0 + answer) && ringEmpty(fx.ring));
			CHECK([(id<RgfxMetalTextureInternals>)t gpuAddress] != 0 && t.allocatedSize == 8192);
		}
		rgfxIOSurfaceKeepFlush(); // the views kept past their textures go too
		drainRing(fx.ring);
		CHECK(src.released.size() == static_cast<size_t>(copies) &&
		      src.released.back() == src.offset);
		CFRelease(s);
	}
	rgfxIOSurfaceKeepFlush(); // nothing kept outlives the test's region
}

// Core Image's blur of a large image, as a guest boot met it (newTextureWithDescriptor:iosurface:
// plane: answered nil, the region too small for their copies): fifteen one-plane RGhA surfaces of
// 6094 x 1566 alive at once, a 6016 x 6016 '420f' source as IOSurfaceCreate lays it out and as
// CoreVideo does (plane 0 past a 512-byte header, measured on the M3 Max under macOS 27), and
// 3020 x 3020 RGBA and BGRA outputs, each in the shape and rows IOSurface gives it here. In place,
// none needs a region block for its pixels: every plane asked for gets a texture, in place, over a
// region with no room for a copy of any of them, at the plane's own offset from the surface's
// base and with its own rows.
void test_iosurface_in_place_core_image_set()
{
	Fixture fx;
	int calls = 0;
	fx.ids.setResourceIdSource(&oneChunk, &calls);
	std::vector<uint8_t> mem(RGFX_ALLOCATIONS_OFFSET + 0x100000, 0);
	RgfxRegion region(mem.data(), mem.size());
	struct Source {
		uint64_t next = RGFX_ALLOCATIONS_OFFSET + 0x80000;
		const void *base = nullptr;
		uint64_t length = 0, size = 0, offset = 0;
		uint32_t surfaceId = 0;
		bool cacheable = false;
		int asked = 0, copies = 0;
		std::vector<uint64_t> released;
		static RgfxRegion::PagesAnswer pages(void *c, const void *base, uint64_t length,
		                                     uint64_t size, uint32_t surfaceId, bool cacheable,
		                                     uint64_t *offset, uint64_t *serial, bool *fresh,
		                                     bool *cached)
		{
			Source *src = static_cast<Source *>(c);
			src->surfaceId = surfaceId;
			src->cacheable = cacheable;
			*cached = false;
			src->base = base;
			src->length = length;
			src->size = size;
			src->offset = src->next;
			src->next += size;
			*offset = src->offset;
			*serial = ++src->asked;
			*fresh = true;
			return RgfxRegion::kPagesGiven;
		}
		static bool acquire(void *c, const void *, uint64_t, uint64_t, uint64_t *, uint64_t *,
		                    bool *)
		{
			static_cast<Source *>(c)->copies++;
			return false;
		}
		static void release(void *c, uint64_t offset)
		{
			static_cast<Source *>(c)->released.push_back(offset);
		}
	} src;
	region.setSurfaceSource(&Source::acquire, &Source::release, &src);
	region.setSurfacePagesSource(&Source::pages, 16, &src);
	auto make = [](uint32_t width, uint32_t height, uint32_t bytesPerElement, uint32_t format) {
		NSDictionary *props = @{
			(id)kIOSurfaceWidth : @(width),
			(id)kIOSurfaceHeight : @(height),
			(id)kIOSurfaceBytesPerElement : @(bytesPerElement),
			(id)kIOSurfacePixelFormat : @(format),
		};
		return IOSurfaceCreate((__bridge CFDictionaryRef)props);
	};
	const uint32_t kRGhA = 0x52476841, kRGBA = 0x52474241, kBGRA = 0x42475241, k420f = 0x34323066;
	std::vector<IOSurfaceRef> surfaces;
	for (int i = 0; i < 15; i++)
		surfaces.push_back(make(6094, 1566, 8, kRGhA));
	NSDictionary *biPlanar = @{
		(id)kIOSurfaceWidth : @6016,
		(id)kIOSurfaceHeight : @6016,
		(id)kIOSurfacePixelFormat : @(k420f),
		(id)kIOSurfacePlaneInfo : @[
			@{
				(id)kIOSurfacePlaneWidth : @6016,
				(id)kIOSurfacePlaneHeight : @6016,
				(id)kIOSurfacePlaneBytesPerElement : @1,
			},
			@{
				(id)kIOSurfacePlaneWidth : @3008,
				(id)kIOSurfacePlaneHeight : @3008,
				(id)kIOSurfacePlaneBytesPerElement : @2,
			},
		],
	};
	surfaces.push_back(IOSurfaceCreate((__bridge CFDictionaryRef)biPlanar));
	CVPixelBufferRef pixelBuffer = nullptr;
	NSDictionary *attrs = @{(id)kCVPixelBufferIOSurfacePropertiesKey : @{}};
	CHECK(CVPixelBufferCreate(nullptr, 6016, 6016, k420f, (__bridge CFDictionaryRef)attrs,
	                          &pixelBuffer) == kCVReturnSuccess);
	surfaces.push_back((IOSurfaceRef)CFRetain(CVPixelBufferGetIOSurface(pixelBuffer)));
	surfaces.push_back(make(3020, 3020, 4, kRGBA));
	surfaces.push_back(make(3020, 3020, 4, kBGRA));
	for (IOSurfaceRef s : surfaces)
		CHECK(s != nullptr);

	auto pageUp = [](uint64_t n) { return (n + 0xfff) & ~uint64_t(0xfff); };
	std::vector<uint64_t> blocks;
	@autoreleasepool {
		NSMutableArray *alive = [NSMutableArray array];
		for (IOSurfaceRef s : surfaces) {
			uint32_t format = IOSurfaceGetPixelFormat(s);
			size_t planes = std::max<size_t>(IOSurfaceGetPlaneCount(s), 1);
			auto *base = static_cast<uint8_t *>(IOSurfaceGetBaseAddress(s));
			uint64_t buffer = 0;
			for (size_t k = 0; k < planes; k++) {
				MTLPixelFormat pf = format == kRGhA   ? MTLPixelFormatRGBA16Float
				                    : format == kRGBA ? MTLPixelFormatRGBA8Unorm
				                    : format == kBGRA ? MTLPixelFormatBGRA8Unorm
				                    : k == 0          ? MTLPixelFormatR8Unorm
				                                      : MTLPixelFormatRG8Unorm;
				size_t w = planes == 1 ? IOSurfaceGetWidth(s) : IOSurfaceGetWidthOfPlane(s, k);
				size_t h = planes == 1 ? IOSurfaceGetHeight(s) : IOSurfaceGetHeightOfPlane(s, k);
				size_t rowBytes = planes == 1 && IOSurfaceGetPlaneCount(s) == 0
				                      ? IOSurfaceGetBytesPerRow(s)
				                      : IOSurfaceGetBytesPerRowOfPlane(s, k);
				auto *rows = static_cast<uint8_t *>(IOSurfaceGetPlaneCount(s) == 0
				                                        ? IOSurfaceGetBaseAddress(s)
				                                        : IOSurfaceGetBaseAddressOfPlane(s, k));
				MTLTextureDescriptor *td =
				    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pf
				                                                       width:w
				                                                      height:h
				                                                   mipmapped:NO];
				td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite |
				           MTLTextureUsageRenderTarget;
				td.storageMode = MTLStorageModeShared;
				RgfxIOSurfaceTexture *t = [RgfxIOSurfaceTexture textureWithDescriptor:td
				                                                            iosurface:s
				                                                                plane:k
				                                                               region:&region
				                                                               device:nil
				                                                           serializer:&fx.ser];
				CHECK(t != nil);
				if (t == nil)
					continue;
				[alive addObject:t];
				if (k == 0) {
					// The surface's first texture asks for its block in place, by its base and
					// whole allocation, and lays the buffer over them.
					CHECK(src.base == base && src.length == pageUp(IOSurfaceGetAllocSize(s)) &&
					      src.size == 0x1000);
					auto over = expectMessage<RgfxBufferOverSurfacePages>(
					    fx.ring, RGFX_CMD_BUFFER_OVER_SURFACE_PAGES);
					CHECK(over.block_offset == src.offset && over.block_size == 0x1000 &&
					      over.length == pageUp(IOSurfaceGetAllocSize(s)));
					buffer = over.obj_id;
					blocks.push_back(src.offset);
				}
				auto made = expectMessage<RgfxSurfaceTextureMessage>(
				    fx.ring, RGFX_CMD_TEXTURE_FROM_IOSURFACE);
				CHECK(made.texture.buffer_obj_id == buffer &&
				      made.texture.offset == static_cast<uint64_t>(rows - base) &&
				      made.texture.bytes_per_row == rowBytes && made.texture.plane == k &&
				      made.texture.writers_offset == kSurfaceHeader &&
				      made.texture.iosurface_id == IOSurfaceGetID(s));
				if (planes == 2) {
					auto *chroma = static_cast<uint8_t *>(IOSurfaceGetBaseAddressOfPlane(s, 1));
					CHECK(made.facts.plane_count == 2 &&
					      made.facts.chroma_offset == static_cast<uint64_t>(chroma - base) &&
					      made.facts.chroma_bytes_per_row == IOSurfaceGetBytesPerRowOfPlane(s, 1));
				}
				std::string line;
				CHECK(rgfxIOSurfaceNilLog(s, "region", &line) == 0);
			}
		}
		CHECK(alive.count == 15 + 2 + 2 + 1 + 1);
		CHECK(src.asked == static_cast<int>(surfaces.size()) && src.copies == 0);
		// No block of the region's own, for a copy or anything else.
		CHECK(region.used() == 0 && ringEmpty(fx.ring));
	}
	rgfxIOSurfaceKeepFlush(); // the views kept past their textures go too
	drainRing(fx.ring); // the textures' and the buffers' destroys
	std::sort(src.released.begin(), src.released.end());
	CHECK(src.released == blocks);
	for (IOSurfaceRef s : surfaces)
		CFRelease(s);
	CVPixelBufferRelease(pixelBuffer);
	rgfxIOSurfaceKeepFlush(); // nothing kept outlives the test's region
}

// A surface whose block the kext made cached, its GPU copy in a device cache on the host
// (RgfxIOSurface.h): every surface is asked cached, of any planes and format, but a video
// decoder's output made under RgfxAskUncachedScope, and a block is cached as the kext answers,
// whoever made it. A command buffer names each cached surface it uses
// before its first encoder, once, with the CPU generation the block's header holds at its commit
// and whether it writes the surface (RGFX_OP_SURFACE_USE): a bound texture, an attachment, a
// blit's source or destination, a resource it uses; views name their surface. A buffer that uses
// none names nothing, and nor does one using a surface not cached. The texture's own CPU access
// takes the surface's lock, a write a read-write one, which a texture not cached does not.
void test_iosurface_cached()
{
	// Never freed, as Fixture's: the queue and its buffers may go after the test.
	TestRing &control = *new TestRing(0x8000);
	control.setConsumerIdle(1);
	RgfxRingWriter &controlWriter = *new RgfxRingWriter(
	    control.base(), control.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell, nullptr);
	RgfxIdSpace &ids = *new RgfxIdSpace();
	int &calls = *new int(0);
	ids.setResourceIdSource(&oneChunk, &calls);
	RgfxSerializer &ser = *new RgfxSerializer(controlWriter, ids);
	TestRing &submit = *new TestRing(0x8000);
	submit.setConsumerIdle(1);
	RgfxRingWriter &submitWriter =
	    *new RgfxRingWriter(submit.base(), submit.dataSize, 1, TestRing::doorbell, nullptr);
	TestRing &completion = *new TestRing(0x8000);
	RgfxReplyReader &reader = *new RgfxReplyReader(completion.base(), completion.dataSize);
	std::vector<uint8_t> &mem = *new std::vector<uint8_t>(RGFX_ALLOCATIONS_OFFSET + 0x100000, 0);
	RgfxRegion &region = *new RgfxRegion(mem.data(), mem.size());
	struct Source {
		uint64_t next = RGFX_ALLOCATIONS_OFFSET + 0x40000;
		bool answerCached = true;
		uint32_t surfaceId = 0;
		bool cacheable = false;
		// The block the kext revives at the next ask, and the blocks let go.
		uint64_t revive = 0;
		std::vector<uint64_t> released;
		static RgfxRegion::PagesAnswer pages(void *c, const void *, uint64_t, uint64_t size,
		                                     uint32_t surfaceId, bool cacheable, uint64_t *offset,
		                                     uint64_t *serial, bool *fresh, bool *cached)
		{
			Source *src = static_cast<Source *>(c);
			src->surfaceId = surfaceId;
			src->cacheable = cacheable;
			if (src->revive) {
				*offset = src->revive;
				src->revive = 0;
			} else {
				*offset = src->next;
				src->next += size;
			}
			*serial = 3;
			*fresh = true;
			*cached = src->answerCached;
			return RgfxRegion::kPagesGiven;
		}
		static void release(void *c, uint64_t offset)
		{
			static_cast<Source *>(c)->released.push_back(offset);
		}
	} &src = *new Source();
	region.setSurfaceSource(nullptr, &Source::release, &src);
	region.setSurfacePagesSource(&Source::pages, 16, &src);
	RgfxCommandQueue *queue = [RgfxCommandQueue queueWithSubmitRing:&submitWriter
	                                                  submitOffset:0
	                                                completeOffset:0
	                                              completeDataSize:0
	                                                           ids:&ids
	                                                      maxCount:0
	                                                        device:nil
	                                                    serializer:&ser];
	CHECK(queue != nil);
	queue.replyReader = &reader;
	(void)expectQueueCreate(control);
	// Commits `cb`, completes it, and answers its SURFACE_USE ops, which must all come before its
	// first other op.
	auto uses = [&](id<MTLCommandBuffer> cb) {
		[cb commit];
		RgfxMsgHeader h = {};
		std::vector<uint8_t> bytes;
		CHECK(submit.next(&h, &bytes) && h.type == RGFX_CMD_COMMAND_BUFFER);
		std::vector<RgfxOpSurfaceUse> out;
		bool before = true;
		size_t at = RGFX_COMMAND_BUFFER_OPS_OFFSET;
		while (at + sizeof(RgfxOpHeader) <= bytes.size()) {
			RgfxOpHeader op = {};
			std::memcpy(&op, bytes.data() + at, sizeof(op));
			CHECK(op.len >= RGFX_OP_BODY_OFFSET && at + op.len <= bytes.size());
			if (op.tag == RGFX_OP_SURFACE_USE) {
				CHECK(before && op.len == RGFX_OP_BODY_OFFSET + sizeof(RgfxOpSurfaceUse));
				RgfxOpSurfaceUse u = {};
				std::memcpy(&u, bytes.data() + at + RGFX_OP_BODY_OFFSET, sizeof(u));
				CHECK(u.reserved_14 == 0);
				out.push_back(u);
			} else {
				before = false;
			}
			at += op.len;
		}
		RgfxCbComplete done = {};
		done.header.type = RGFX_EVT_CB_COMPLETE;
		done.header.size = sizeof(done);
		done.header.seqno = h.seqno;
		done.cb_obj_id = ((RgfxResource *)cb).rgfxObjId;
		done.status = RGFX_CB_OK;
		CHECK(completion.postFromHost(&done, sizeof(done)));
		CHECK(reader.pump() == 1);
		CHECK(cb.status == MTLCommandBufferStatusCompleted);
		rgfxIOSurfaceKeepFlush(); // the views kept past their textures go too
		drainRing(control);
		return out;
	};
	auto is = [](const std::vector<RgfxOpSurfaceUse> &got, uint64_t offset, uint64_t gen,
	             uint32_t flags) {
		return got.size() == 1 && got[0].block_offset == offset && got[0].cpu_gen == gen &&
		       got[0].flags == flags;
	};
	auto overSurface = [&](IOSurfaceRef s, MTLTextureUsage usage) {
		MTLTextureDescriptor *td =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
		                                                       width:16
		                                                      height:16
		                                                   mipmapped:NO];
		td.usage = usage;
		return [RgfxIOSurfaceTexture textureWithDescriptor:td
		                                        iosurface:s
		                                            plane:0
		                                           region:&region
		                                           device:nil
		                                       serializer:&ser];
	};
	auto pixelBuffer = [](OSType format) {
		CVPixelBufferRef pb = nullptr;
		NSDictionary *attrs = @{(id)kCVPixelBufferIOSurfacePropertiesKey : @{}};
		CHECK(CVPixelBufferCreate(nullptr, 16, 16, format, (__bridge CFDictionaryRef)attrs, &pb) ==
		          kCVReturnSuccess &&
		      pb != nullptr && CVPixelBufferGetIOSurface(pb) != nullptr);
		return pb;
	};
	const MTLSize one = MTLSizeMake(1, 1, 1);
	CVPixelBufferRef bgra = pixelBuffer(kCVPixelFormatType_32BGRA);
	CVPixelBufferRef yuv = pixelBuffer(kCVPixelFormatType_422YpCbCr8);
	IOSurfaceRef surface = CVPixelBufferGetIOSurface(bgra);
	uint64_t dormant = 0;
	@autoreleasepool {
		RgfxIOSurfaceTexture *t =
		    overSurface(surface, MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget);
		CHECK(t != nil && src.surfaceId == IOSurfaceGetID(surface) && src.cacheable);
		const uint64_t block = src.next - 0x1000;
		dormant = block;
		auto *header = reinterpret_cast<uint64_t *>(mem.data() + block);
		header[2] = 7; // the kext's CPU generation, at kRgfxSurfaceCpuGenOffset
		id<MTLTexture> view = [t newTextureViewWithPixelFormat:MTLPixelFormatBGRA8Unorm];
		CHECK(view != nil);
		rgfxIOSurfaceKeepFlush(); // the views kept past their textures go too
		drainRing(control);

		// A 4:2:2 YCbCr surface of one plane is asked cached too, and one the kext answers not
		// cached is named by no buffer.
		src.answerCached = false;
		RgfxIOSurfaceTexture *video = overSurface(CVPixelBufferGetIOSurface(yuv),
		                                          MTLTextureUsageShaderRead);
		CHECK(video != nil && src.cacheable);
		// A video decoder's output, its plane's texture made under RgfxAskUncachedScope: not asked
		// cached; and one of two planes of YCbCr made after the scope is.
		auto lumaOf = [&](CVPixelBufferRef pb) {
			MTLTextureDescriptor *td =
			    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
			                                                       width:16
			                                                      height:16
			                                                   mipmapped:NO];
			td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
			return [RgfxIOSurfaceTexture textureWithDescriptor:td
			                                        iosurface:CVPixelBufferGetIOSurface(pb)
			                                            plane:0
			                                           region:&region
			                                           device:nil
			                                       serializer:&ser];
		};
		CVPixelBufferRef decoded = pixelBuffer(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange);
		CVPixelBufferRef other = pixelBuffer(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange);
		{
			RgfxAskUncachedScope ask;
			CHECK(lumaOf(decoded) != nil && !src.cacheable &&
			      src.surfaceId == IOSurfaceGetID(CVPixelBufferGetIOSurface(decoded)));
		}
		CHECK(lumaOf(other) != nil && src.cacheable &&
		      src.surfaceId == IOSurfaceGetID(CVPixelBufferGetIOSurface(other)));
		CVPixelBufferRelease(decoded); // the views hold their surfaces
		CVPixelBufferRelease(other);
		src.answerCached = true;
		rgfxIOSurfaceKeepFlush(); // the views kept past their textures go too
		drainRing(control);

		// Sampled in a kernel through a view, then bound again: named once, not written.
		RgfxComputePipelineState *ps = [RgfxComputePipelineState
		    computePipelineStateWithFunction:[RgfxFunction functionWithLibrary:nil
		                                                                  name:@"k"
		                                                             constants:nullptr
		                                                                 count:0
		                                                                device:nil
		                                                            serializer:&ser]
		                              device:nil
		                          serializer:&ser
		                               error:nil];
		CHECK(ps != nil);
		rgfxIOSurfaceKeepFlush(); // the views kept past their textures go too
		drainRing(control);
		id<MTLCommandBuffer> cb = [queue commandBuffer];
		id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
		[ce setComputePipelineState:ps];
		[ce setTexture:view atIndex:0];
		[ce setTexture:video atIndex:1];
		[ce dispatchThreadgroups:one threadsPerThreadgroup:one];
		[ce setTexture:t atIndex:2];
		[ce dispatchThreadgroups:one threadsPerThreadgroup:one];
		[ce endEncoding];
		CHECK(is(uses(cb), block, 7, 0));

		// Its attachment, at the generation the header holds now: written.
		header[2] = 8;
		MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
		rp.colorAttachments[0].texture = t;
		rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
		rp.colorAttachments[0].storeAction = MTLStoreActionStore;
		cb = [queue commandBuffer];
		[[cb renderCommandEncoderWithDescriptor:rp] endEncoding];
		CHECK(is(uses(cb), block, 8, RGFX_SURFACE_USE_WRITES));

		// A blit's source, and a resource used with Write usage.
		id<MTLBuffer> buffer = [RgfxBuffer bufferWithLength:16 * 16 * 4
		                                            options:MTLResourceStorageModeShared
		                                             device:nil
		                                         serializer:&ser];
		CHECK(buffer != nil);
		rgfxIOSurfaceKeepFlush(); // the views kept past their textures go too
		drainRing(control);
		cb = [queue commandBuffer];
		id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
		[be copyFromTexture:t
		             sourceSlice:0
		             sourceLevel:0
		            sourceOrigin:MTLOriginMake(0, 0, 0)
		              sourceSize:MTLSizeMake(16, 16, 1)
		                toBuffer:buffer
		       destinationOffset:0
		  destinationBytesPerRow:64
		destinationBytesPerImage:64 * 16];
		[be endEncoding];
		CHECK(is(uses(cb), block, 8, 0));
		cb = [queue commandBuffer];
		ce = [cb computeCommandEncoder];
		[ce useResource:view usage:MTLResourceUsageRead | MTLResourceUsageWrite];
		[ce endEncoding];
		CHECK(is(uses(cb), block, 8, RGFX_SURFACE_USE_WRITES));

		// A buffer that uses no cached surface names none.
		cb = [queue commandBuffer];
		ce = [cb computeCommandEncoder];
		[ce setComputePipelineState:ps];
		[ce setTexture:video atIndex:0];
		[ce dispatchThreadgroups:one threadsPerThreadgroup:one];
		[ce endEncoding];
		CHECK(uses(cb).empty());

		// The texture's CPU access locks the surface: a read leaves its seed, a write moves it,
		// as a read-write lock's unlock does; one not cached takes no lock.
		const uint32_t seed = IOSurfaceGetSeed(surface);
		uint32_t texel = 0;
		[t getBytes:&texel bytesPerRow:4 fromRegion:MTLRegionMake2D(1, 1, 1, 1) mipmapLevel:0];
		CHECK(IOSurfaceGetSeed(surface) == seed);
		[t replaceRegion:MTLRegionMake2D(1, 1, 1, 1) mipmapLevel:0 withBytes:&texel bytesPerRow:4];
		CHECK(IOSurfaceGetSeed(surface) != seed);
		IOSurfaceRef plain = CVPixelBufferGetIOSurface(yuv);
		const uint32_t plainSeed = IOSurfaceGetSeed(plain);
		[video replaceRegion:MTLRegionMake2D(0, 0, 1, 1)
		         mipmapLevel:0
		           withBytes:&texel
		         bytesPerRow:4];
		CHECK(IOSurfaceGetSeed(plain) == plainSeed);
	}
	rgfxIOSurfaceKeepFlush(); // the views kept past their textures go too
	drainRing(control);

	// Its textures gone, the process lets the block go, and the kext keeps it dormant: the next
	// texture over the surface is given the same block again, fresh and cached. Its writers start
	// empty, its header's CPU generation is the kext's, untouched, and the next buffer names it.
	auto letGo = [&] {
		return std::count(src.released.begin(), src.released.end(), dormant);
	};
	CHECK(letGo() == 1);
	auto *header = reinterpret_cast<uint64_t *>(mem.data() + dormant);
	header[2] = 9;
	mem[dormant + kSurfaceHeader] = 0xff;
	src.revive = dormant;
	@autoreleasepool {
		RgfxIOSurfaceTexture *t =
		    overSurface(surface, MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget);
		CHECK(t != nil && src.revive == 0 && src.next - 0x1000 != dormant);
		CHECK(mem[dormant + kSurfaceHeader] == 0 && header[2] == 9);
		MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
		rp.colorAttachments[0].texture = t;
		rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
		rp.colorAttachments[0].storeAction = MTLStoreActionStore;
		id<MTLCommandBuffer> cb = [queue commandBuffer];
		[[cb renderCommandEncoderWithDescriptor:rp] endEncoding];
		CHECK(is(uses(cb), dormant, 9, RGFX_SURFACE_USE_WRITES));
	}
	rgfxIOSurfaceKeepFlush(); // the views kept past their textures go too
	drainRing(control);
	CHECK(letGo() == 2);
	CVPixelBufferRelease(bgra);
	CVPixelBufferRelease(yuv);
	rgfxIOSurfaceKeepFlush(); // nothing kept outlives the test's region
}

// A texture newTextureWithDescriptor:iosurface:plane: cannot make is nil, as Metal answers one,
// and never silently: each cause logs one "RGFX:" line per surface, however often it is asked.
// A surface whose view is refused in place and that has no room for a copy names the checks that
// refused, the surface, the block asked and the region's free bytes; a plane past the surface's
// and a call with no descriptor name theirs. Nothing reaches the host.
void test_iosurface_nil_is_logged()
{
	Fixture fx;
	int calls = 0;
	fx.ids.setResourceIdSource(&oneChunk, &calls);
	std::vector<uint8_t> mem(RGFX_ALLOCATIONS_OFFSET + 0x100000, 0);
	RgfxRegion region(mem.data(), mem.size());
	struct Source {
		static RgfxRegion::PagesAnswer pages(void *, const void *, uint64_t, uint64_t, uint32_t,
		                                     bool, uint64_t *, uint64_t *, bool *, bool *)
		{
			return RgfxRegion::kPagesUnsupported;
		}
		static bool acquire(void *, const void *, uint64_t, uint64_t, uint64_t *, uint64_t *,
		                    bool *)
		{
			return false;
		}
		static void release(void *, uint64_t) {}
	} src;
	region.setSurfaceSource(&Source::acquire, &Source::release, &src);
	region.setSurfacePagesSource(&Source::pages, 16, &src);
	NSDictionary *props = @{
		(id)kIOSurfaceWidth : @6094,
		(id)kIOSurfaceHeight : @1566,
		(id)kIOSurfaceBytesPerElement : @8,
		(id)kIOSurfacePixelFormat : @(0x52476841u),
	};
	IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef)props);
	CHECK(s != nullptr);
	MTLTextureDescriptor *td =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
	                                                       width:6094
	                                                      height:1566
	                                                   mipmapped:NO];
	auto texture = [&](MTLTextureDescriptor *d, NSUInteger plane) {
		return [RgfxIOSurfaceTexture textureWithDescriptor:d
		                                        iosurface:s
		                                            plane:plane
		                                           region:&region
		                                           device:nil
		                                       serializer:&fx.ser];
	};
	// The copy the region has no room for: the rows 256 bytes apart, the header and the writers.
	uint64_t pitch = (IOSurfaceGetBytesPerRow(s) + 255) & ~uint64_t(255);
	uint64_t size =
	    (pitch * 1566 + kSurfaceHeader + sizeof(RgfxSurfaceWriters) + 0xfff) & ~uint64_t(0xfff);
	auto has = [](const std::string &line, const std::string &part) {
		return line.find(part) != std::string::npos;
	};
	std::string line;
	@autoreleasepool {
		CHECK(texture(td, 0) == nil && texture(td, 0) == nil);
	}
	CHECK(rgfxIOSurfaceNilLog(s, "region", &line) == 2);
	CHECK(has(line, "IOSurface " + std::to_string(IOSurfaceGetID(s)) + " 'RGhA' 6094x1566") &&
	      has(line, "0 planes; plane 0: 1566 rows of " +
	                    std::to_string(IOSurfaceGetBytesPerRow(s)) + " bytes") &&
	      has(line, "not in place (the device uses no surface in place, or refused its pages)") &&
	      has(line, "no shared copy block of " + std::to_string(size) + " bytes") &&
	      has(line, "(1048576 bytes free)"));
	@autoreleasepool {
		CHECK(texture(td, 1) == nil && texture(nil, 0) == nil);
	}
	CHECK(rgfxIOSurfaceNilLog(s, "plane", &line) == 1 &&
	      has(line, "plane 1, past its planes"));
	CHECK(rgfxIOSurfaceNilLog(s, "arguments", &line) == 1 &&
	      has(line, "no descriptor"));
	CHECK(rgfxIOSurfaceNilLog(s, "buffer", &line) == 0 && line.empty());
	// A block-compressed format, which Metal makes but the host makes no texture of over a
	// surface's rows (oracle_surftex, Q20), answers nil before the surface's pages are looked at.
	MTLTextureDescriptor *bc1 =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBC1_RGBA
	                                                       width:64
	                                                      height:64
	                                                   mipmapped:NO];
	@autoreleasepool {
		CHECK(texture(bc1, 0) == nil);
	}
	CHECK(rgfxIOSurfaceNilLog(s, "descriptor", &line) == 1 &&
	      has(line, "a block-compressed format over a surface, which the host makes no texture "
	                "of"));
	CHECK(region.used() == 0 && ringEmpty(fx.ring));
	CFRelease(s);
	rgfxIOSurfaceKeepFlush(); // nothing kept outlives the test's region
}

// A queue's command buffers keep it alive; once they are gone it ends on the device, its
// completion ring read no more, and only then does what it holds go back.
void test_queue_ends_after_its_buffers()
{
	@autoreleasepool {
		// Made in a pool of its own, so no autorelease keeps the queue past the test's let-go.
		QueueFixture *made = nullptr;
		@autoreleasepool {
			made = new QueueFixture();
		}
		QueueFixture &f = *made;
		(void)expectQueueCreate(f.control);
		uint64_t queueId = f.queue.rgfxObjId;
		std::shared_ptr<RgfxReplyReader> reader = [f.queue sharedReplyReader];
		int released = 0;
		bool endedFirst = false;
		TestRing *control = &f.control;
		[f.queue rgfxSetRelease:[&released, &endedFirst, control] {
			RgfxMsgHeader h = {};
			std::vector<uint8_t> bytes;
			endedFirst = control->next(&h, &bytes) && h.type == RGFX_CMD_OBJECT_DESTROY;
			released++;
		}];
		id<MTLCommandBuffer> cb = nil;
		@autoreleasepool {
			cb = [f.queue commandBuffer];
			[cb commit];
			f.queue = nil;
		}
		@autoreleasepool {
			CHECK(cb.commandQueue != nil); // the buffer keeps it
		}
		CHECK(ringEmpty(f.control) && released == 0);
		RgfxCbComplete done = cbComplete(f.nextSeqno(), ((RgfxResource *)cb).rgfxObjId);
		CHECK(f.completion.postFromHost(&done, sizeof(done)));
		[cb waitUntilCompleted];
		CHECK(ringEmpty(f.control) && released == 0);
		@autoreleasepool {
			cb = nil;
		}
		CHECK(released == 1 && endedFirst);
		CHECK(queueId != 0);
		// Its completion ring is read no more.
		uint64_t consumer = f.completion.consumerCount();
		RgfxCbComplete late = cbComplete(99, 1);
		CHECK(f.completion.postFromHost(&late, sizeof(late)));
		CHECK(reader->pump() == 0 && !reader->armAndCheck());
		CHECK(f.completion.consumerCount() == consumer);
	}
}

// Where a closer is set the queue ends through it (the kext's QUEUE_DESTROY) and nothing is
// posted on the ring; without one, an OBJECT_DESTROY ends it.
void test_serializer_destroys_queues()
{
	struct Closed {
		std::vector<uint64_t> ids;
		static void close(void *c, RgfxObjId objId) { static_cast<Closed *>(c)->ids.push_back(objId); }
	} closed;
	Fixture fx;
	fx.ser.destroyQueue(7);
	CHECK(nextDestroyed(fx.ring) == 7 && ringEmpty(fx.ring));
	fx.ser.setQueueCloser(&Closed::close, &closed);
	fx.ser.destroyQueue(8);
	CHECK(closed.ids == std::vector<uint64_t>({8}) && ringEmpty(fx.ring));
}

// An object a command buffer encodes stays on the host though the app let it go: until the
// buffer completes, where it is committed, and until it goes, where it never is. One it never
// encodes goes at once, whether made before the buffer or after.
void test_objects_outlive_an_uncommitted_buffer()
{
	@autoreleasepool {
		QueueFixture f;
		drainRing(f.control);
		auto sampler = [&](uint64_t *objId) {
			RgfxSampler *s = [RgfxSampler samplerWithDescriptor:[MTLSamplerDescriptor new]
			                                             device:nil
			                                         serializer:&f.ser];
			*objId = s.rgfxObjId;
			drainRing(f.control);
			return s;
		};
		auto encode = [](id<MTLCommandBuffer> cb, id<MTLSamplerState> s) {
			id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
			[enc setSamplerState:s atIndex:0];
			[enc endEncoding];
		};
		id<MTLCommandBuffer> cb = [f.queue commandBuffer];
		uint64_t samplerId = 0, unnamed = 0;
		@autoreleasepool {
			encode(cb, sampler(&samplerId));
			(void)sampler(&unnamed);
		}
		CHECK(nextDestroyed(f.control) == unnamed && ringEmpty(f.control));
		[cb commit];
		CHECK(ringEmpty(f.control));
		RgfxCbComplete done = cbComplete(f.nextSeqno(), ((RgfxResource *)cb).rgfxObjId);
		CHECK(f.completion.postFromHost(&done, sizeof(done)));
		[cb waitUntilCompleted];
		CHECK(nextDestroyed(f.control) == samplerId && ringEmpty(f.control));

		uint64_t other = 0;
		@autoreleasepool {
			id<MTLCommandBuffer> idle = [f.queue commandBuffer];
			@autoreleasepool {
				encode(idle, sampler(&other));
			}
			CHECK(ringEmpty(f.control)); // the buffer that encodes it keeps it
		}
		CHECK(nextDestroyed(f.control) == other && ringEmpty(f.control));
	}
}

// A closed reader takes nothing more off its ring: no handler runs, the consumer stays, and
// the ring is not armed.
void test_reply_reader_close()
{
	TestRing ring(0x1000);
	RgfxReplyReader reader(ring.base(), ring.dataSize);
	int fired = 0;
	reader.expectCompletion(7, [&](const RgfxCbComplete &) { fired++; });
	reader.close();
	RgfxCbComplete done = cbComplete(7, 1);
	CHECK(ring.postFromHost(&done, sizeof(done)));
	CHECK(reader.pump() == 0 && fired == 0);
	CHECK(!reader.armAndCheck());
	CHECK(ringArmed(ring) == 0);
	CHECK(ring.consumerCount() == 0);
}

void rgfxProtocolTests();
void rgfxArchiveTests();
void rgfxResourceTests();
void rgfxQueueTests(const char *self);
int rgfxQueueDiesCase(const char *name);
void rgfxCreateTests();
void rgfxKnownPipelineTests();
void rgfxIcbTests();

// A block's arguments come from its own signature, and a creation's completion handler is called
// with the arguments its shape takes: (object, error), or (object, nil reflection, error); a
// handler of another shape is not called, and that is logged.
void test_block_call()
{
	CHECK(RgfxBlockArgumentTypes(nil) == nil);
	NSObject *object = [NSObject new];
	NSError *error = RgfxMakeError(RgfxMetalErrorUnsupported, @"refused");
	SEL cmd = @selector(test_block_call_selector);

	__block id gotObject = nil;
	__block id gotError = nil;
	__block int calls = 0;
	void (^two)(id, NSError *) = ^(id o, NSError *e) {
		calls++;
		gotObject = o;
		gotError = e;
	};
	CHECK([RgfxBlockArgumentTypes(two) isEqualToArray:(@[ @"@", @"@" ])]);
	CHECK(RgfxCallCreationHandler(two, object, error, cmd));
	CHECK(calls == 1 && gotObject == object && gotError == error);

	// A block that captures an object carries copy and dispose helpers before its signature.
	__block id gotReflection = object;
	void (^three)(id, id, NSError *) = ^(id o, id reflection, NSError *e) {
		calls++;
		gotObject = o;
		gotReflection = reflection;
		gotError = e;
		(void)object;
	};
	CHECK([RgfxBlockArgumentTypes(three) isEqualToArray:(@[ @"@", @"@", @"@" ])]);
	CHECK(RgfxCallCreationHandler(three, nil, error, cmd));
	CHECK(calls == 2 && gotObject == nil && gotReflection == nil && gotError == error);

	// Not called: one object, or arguments that are not objects; nil is no call to make.
	void (^one)(NSError *) = ^(NSError *e) {
		calls++;
	};
	void (^dealloc)(void *, NSUInteger) = ^(void *p, NSUInteger n) {
		calls++;
	};
	CHECK([RgfxBlockArgumentTypes(dealloc) isEqualToArray:(@[ @"^v", @"Q" ])]);
	CHECK(!RgfxCallCreationHandler(one, object, error, cmd));
	CHECK(!RgfxCallCreationHandler(dealloc, object, error, cmd));
	CHECK(RgfxCallCreationHandler(nil, object, error, cmd));
	CHECK(calls == 2);
}

int main(int argc, char **argv)
{
	g_self = argv[0];
	if (argc == 3 && std::strcmp(argv[1], "--dies") == 0)
		return dies_case(argv[2]);
	if (argc == 3 && std::strcmp(argv[1], "--queue-dies") == 0)
		return rgfxQueueDiesCase(argv[2]);
	@autoreleasepool {
		test_buffer_routing();
		test_private_buffer_has_no_contents();
		test_device_buffers_answer_as_metal();
		test_device_buffers_take_pages_of_slabs();
		test_managed_buffers_place_their_cpu_copies_in_heaps();
		test_device_heaps_answer_as_metal();
		test_shared_heaps_and_cpu_depth_textures_end_the_process();
		test_second_buffer_is_monotonic();
		test_wrapper_guards_and_base();
		test_event_routing();
		test_fence_routing();
		test_shared_event_routing();
		test_texture_routing();
		test_texture_cpu_copies();
		test_tiled_texture_over_backing();
		test_mipmapped_buffer_texture_ends_the_process();
		test_buffer_texture_types_end_the_process();
		test_invalid_texture_descriptors_end_the_process();
		test_invalid_surface_textures_end_the_process();
		test_unalignable_linear_formats_end_the_process();
		test_memoryless_icb_ends_the_process();
		test_texture_views();
		test_metal_ended_views_end_the_process();
		test_texture_framebuffer_only();
		test_texture_unknown_usage();
		test_view_keeps_its_texture();
		test_texture_resource_ids();
		test_sampler_routing();
		test_depth_stencil_routing();
		test_heap_routing_and_suballoc();
		test_resource_guards();
		test_library_and_function_routing();
		test_libraries_of_one_key_share_their_function_table();
		test_render_pipeline_routing();
		test_compute_pipeline_routing();
		test_function_constants_resolve_as_metal();
		test_async_creators_return_before_their_handlers();
		test_async_creates_keep_what_they_use();
		test_linked_function_specialised_on_the_wire();
		test_core_image_extern_functions();
		test_function_patch_answers();
		test_stitched_library_over_shadows();
		test_stitched_dag_over_shadows();
		test_library_air_arguments();
		test_source_compiles_through_metals_frontend();
		test_stitched_libraries_in_archives();
		test_function_members_from_shadow();
		test_unread_function_list_answers_from_shadow();
		test_constants_without_cnst_resolve_from_air();
		test_stitchable_reflection_from_air();
		test_library_members_as_metal();
		test_library_type_and_install_name();
		test_command_graph();
		test_parallel_render_encoder();
		test_parallel_sub_encoders_run_in_creation_order();
		test_queue_guards();
		test_refused_queue_create_is_nil();
		test_region_allocator();
		test_queue_carries_allocated_offsets();
		test_reply_reader_completion();
		test_reply_reader_reflection();
		test_reply_reader_event();
		test_command_buffer_event_ops();
		test_message_reserve();
		test_parallel_sub_encoders_start_from_defaults();
		test_parallel_sub_encoders_on_threads();
		test_command_buffer_keeps_once();
		test_resource_ids();
		test_reply_reader_arm();
		test_reply_reader_holds_submitted();
		test_completion_thread_runs_handlers();
		test_wait_with_the_completion_thread();
		test_scheduled_before_completion();
		test_present_drawable();
		test_submitted_before_scheduled();
		test_submitted_at_the_doorbell_wakes_no_thread();
		test_wait_until_scheduled_pumps();
		test_completion_thread_stops();
		test_uncompleted_buffer_releases();
		test_command_buffer_out_of_line();
		test_body_pool_keeps_every_block();
		test_completion_expected_before_publish();
		test_command_buffer_back_pressure();
		test_texture_destroy_waits_for_buffers();
		test_managed_layout();
		test_managed_buffer_range();
		test_buffer_texture_replace_region_tells_the_host();
		test_buffer_texture_slices();
		test_buffer_texture_did_modify_data_tells_the_host();
		test_buffer_texture_linear_alignment();
		test_managed_texture_cpu_copy();
		test_managed_texture_reads_on_unified_memory();
		test_managed_texture_cpu_copy_in_slabs();
		test_objects_destroyed_as_let_go();
		test_function_keeps_its_library();
		test_heap_outlives_its_resources();
		test_region_memory_given_back_after_destroy();
		test_iosurface_copy_block_after_destroys();
		test_iosurface_shared_block();
		test_iosurface_planes();
		test_iosurface_writers();
		test_iosurface_order();
		test_iosurface_slots_free_at_completion();
		test_iosurface_held_buffer_names_at_scheduling();
		test_commit_order();
		test_commit_order_names_no_cycle();
		test_iosurface_writes_named();
		test_iosurface_writes_by_function();
		test_iosurface_in_place();
		test_iosurface_keep();
		test_iosurface_in_place_fallback();
		test_iosurface_in_place_core_image_set();
		test_iosurface_cached();
		test_iosurface_nil_is_logged();
		test_queue_ends_after_its_buffers();
		test_serializer_destroys_queues();
		test_objects_outlive_an_uncommitted_buffer();
		test_reply_reader_close();
		test_addresses_reused_after_destroy();
		test_heaps_with_no_bytes_or_address_are_nil();
		test_queue_ring_carries_only_command_buffers();
		test_shared_heap_buffers_alias_its_bytes();
		test_automatic_heap_places_as_metal();
		test_automatic_shared_heap_buffers_are_its_bytes();
		test_managed_texture_copy_block_after_destroys();
		test_no_copy_buffers();
		test_no_copy_buffers_in_place();
		test_no_copy_shadows_lie_in_slabs();
		test_binds_note_shadow_writes();
		test_block_call();
		rgfxProtocolTests(); // tests/protocol_tests.mm
		rgfxArchiveTests();  // tests/archive_tests.mm
		rgfxResourceTests(); // tests/resource_tests.mm
		rgfxQueueTests(argv[0]); // tests/queue_tests.mm
		rgfxCreateTests();        // tests/create_tests.mm
		rgfxKnownPipelineTests(); // tests/known_pipeline_tests.mm
		rgfxIcbTests();           // tests/icb_tests.mm
	}
	std::printf("shell_tests: OK (%d checks)\n", g_checks);
	return 0;
}
