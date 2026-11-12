/*
 * residency_tests: the residency set's answers (src/RgfxResidencySet.h), part of the Metal
 * shell's check (shell_tests.mm calls rgfxResidencyTests; guest/bundle/Makefile `test`). Each
 * answer checked is the M3 Max's (macOS 27.0.1, tools/oracle/oracle_residency;
 * docs/graphics-kext/dgpu-oracle.md, Q31), but where the header says the set answers otherwise:
 * the order allAllocations lists them in (the M3's changes from run to run), the bytes of a
 * small buffer and the label, which is the descriptor's.
 *
 * Our own ObjC++; built on the Mac's native arch with the NSObject-based wrappers.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <crt_externs.h>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <spawn.h>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#import "RgfxBuffer.h"
#import "RgfxHeap.h"
#import "RgfxResidencySet.h"
#import "RgfxTexture.h"

#include "RgfxObjectModel.hpp"
#include "RgfxRingWriter.hpp"
#include "RgfxSerializer.hpp"
#include "reims_graphics_wire.h"
#include "rgfx_test_ring.hpp" // CHECK, g_checks, TestRing

void rgfxResidencyTests(const char *self);
int rgfxResidencyDiesCase(const char *name);
void drainRing(TestRing &ring); // shell_tests.mm

// An allocation of no device: Metal's set takes it and counts nothing of it.
@interface RgfxTestForeignAllocation : NSObject <MTLAllocation>
@end
@implementation RgfxTestForeignAllocation
- (NSUInteger)allocatedSize
{
	return 4096;
}
@end

namespace {

// This binary's path, which the death checks run again with --residency-dies CASE.
const char *g_self = nullptr;

// The exit status of a death case that raised SIGABRT: its handler exits with it, so the death
// writes no crash report.
const int kAborted = 128 + SIGABRT;

// A ring, an id space and a serialiser for the allocations, as resource_tests.mm's fixture. The
// transport is never freed: an object a test lets go may dealloc after the test and post its
// destroy here.
struct Fixture {
	TestRing &ring;
	RgfxRingWriter &writer;
	RgfxIdSpace &ids;
	RgfxSerializer &ser;

	Fixture()
	    : ring(*new TestRing(0x40000)),
	      writer(*new RgfxRingWriter(ring.base(), ring.dataSize, RGFX_CONTROL_RING_ID,
	                                 TestRing::doorbell, nullptr)),
	      ids(*new RgfxIdSpace()), ser(*new RgfxSerializer(writer, ids))
	{
		ring.setConsumerIdle(1);
	}
};

// The allocations of oracle_residency's cases: B, a Shared buffer of 4096 bytes; T, a Private
// 64 x 64 RGBA8 texture; H, a Private heap of 1 MiB; HB, a buffer of 4096 bytes of H; TV, a view
// of T; X, a Shared buffer of 4096 bytes.
struct Allocations {
	id<MTLBuffer> B, HB, X;
	id<MTLTexture> T, TV;
	id<MTLHeap> H;
	std::vector<std::pair<const char *, id>> named;

	explicit Allocations(Fixture &fx)
	{
		B = [RgfxBuffer bufferWithLength:4096 options:0 device:nil serializer:&fx.ser];
		X = [RgfxBuffer bufferWithLength:4096 options:0 device:nil serializer:&fx.ser];
		MTLTextureDescriptor *td =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
		                                                       width:64
		                                                      height:64
		                                                   mipmapped:NO];
		td.storageMode = MTLStorageModePrivate;
		td.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
		T = [RgfxTexture textureWithDescriptor:td
		                                  heap:RGFX_OBJ_NONE
		                            heapOffset:0
		                                device:nil
		                            serializer:&fx.ser];
		TV = [T newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm];
		MTLHeapDescriptor *hd = [MTLHeapDescriptor new];
		hd.size = 1 << 20;
		hd.storageMode = MTLStorageModePrivate;
		H = [RgfxHeap heapWithDescriptor:hd device:nil serializer:&fx.ser];
		HB = [H newBufferWithLength:4096 options:MTLResourceStorageModePrivate];
		CHECK(B != nil && X != nil && T != nil && TV != nil && H != nil && HB != nil);
		named = {{"B", B}, {"T", T}, {"H", H}, {"HB", HB}, {"TV", TV}, {"X", X}};
	}

	// The names of `objects`, space-separated, in their order ("?" for one of none of them).
	std::string names(NSArray *objects) const
	{
		std::string out;
		for (id o in objects) {
			const char *name = "?";
			for (const auto &[n, a] : named) {
				if (a == o)
					name = n;
			}
			out += (out.empty() ? "" : " ") + std::string(name);
		}
		return out;
	}

	// The names `set` contains (containsAllocation:), space-separated, in the order of `named`.
	std::string contained(id<MTLResidencySet> set) const
	{
		std::string out;
		for (const auto &[n, a] : named) {
			if ([set containsAllocation:a])
				out += (out.empty() ? "" : " ") + std::string(n);
		}
		return out;
	}
};

RgfxResidencySet *newSet(NSString *label = nil)
{
	MTLResidencySetDescriptor *d = [MTLResidencySetDescriptor new];
	d.label = label;
	return [[RgfxResidencySet alloc] initWithDevice:nil descriptor:d];
}

// One step of oracle_residency's steps case: the set's count, the allocations it contains, the
// ones allAllocations lists, in its order, and its allocatedSize.
void expectStep(id<MTLResidencySet> set, const Allocations &a, NSUInteger count, const char *contains,
                const char *all, uint64_t size)
{
	const std::string has = a.contained(set), listed = a.names(set.allAllocations);
	if (set.allocationCount != count || has != contains || listed != all || set.allocatedSize != size)
		std::fprintf(stderr, "step: count %lu has [%s] all [%s] size %llu; want %lu [%s] [%s] %llu\n",
		             (unsigned long)set.allocationCount, has.c_str(), listed.c_str(),
		             (unsigned long long)set.allocatedSize, (unsigned long)count, contains, all,
		             (unsigned long long)size);
	CHECK(set.allocationCount == count);
	CHECK(has == contains);
	CHECK(listed == all);
	CHECK(set.allocatedSize == size);
}

// A fresh set: no allocation, no size, the descriptor's label (which setLabel: changes), and an
// empty allocations array; a nil descriptor and an initial capacity of NSUIntegerMax each make a
// set too.
void test_residency_set_answers()
{
	RgfxResidencySet *set = newSet(@"oracle-rs");
	CHECK(set.device == nil);
	CHECK([set.label isEqualToString:@"oracle-rs"]);
	CHECK(set.allocatedSize == 0 && set.allocationCount == 0);
	// Empty, the same array each time, as the M3 Max's (__NSArray0, oracle_residency).
	CHECK(set.allAllocations.count == 0 && set.allAllocations == set.allAllocations);
	[set setLabel:@"set-later"];
	CHECK([set.label isEqualToString:@"set-later"]);
	[set setLabel:nil];
	CHECK(set.label == nil);
	CHECK([set conformsToProtocol:@protocol(MTLResidencySet)]);

	RgfxResidencySet *none = [[RgfxResidencySet alloc] initWithDevice:nil descriptor:nil];
	CHECK(none != nil && none.label == nil && none.allocationCount == 0);
	MTLResidencySetDescriptor *huge = [MTLResidencySetDescriptor new];
	huge.initialCapacity = NSUIntegerMax;
	RgfxResidencySet *big = [[RgfxResidencySet alloc] initWithDevice:nil descriptor:huge];
	CHECK(big != nil && big.allocationCount == 0);

	// Neither a request of residency nor its end changes what the set answers.
	Fixture fx;
	@autoreleasepool {
		Allocations a(fx);
		[set endResidency];
		[set requestResidency];
		[set addAllocation:a.B];
		[set requestResidency];
		expectStep(set, a, 1, "B", "B", 0);
		[set commit];
		[set endResidency];
		[set endResidency];
		expectStep(set, a, 1, "B", "B", a.B.allocatedSize);
	}
	drainRing(fx.ring);
}

// oracle_residency's steps case, step by step, as the M3 Max answers each (the order of
// allAllocations and the bytes of a small buffer as the header gives them).
void test_residency_set_steps()
{
	Fixture fx;
	@autoreleasepool {
		Allocations a(fx);
		const uint64_t b = a.B.allocatedSize, t = a.T.allocatedSize, h = a.H.allocatedSize,
		               x = a.X.allocatedSize;
		CHECK(h == 1 << 20 && t != 0 && b != 0);
		RgfxResidencySet *s = newSet(@"oracle-rs");
		expectStep(s, a, 0, "", "", 0);
		[s addAllocation:a.B];
		expectStep(s, a, 1, "B", "B", 0);
		[s addAllocation:a.T];
		expectStep(s, a, 2, "B T", "B T", 0);
		[s addAllocation:a.H];
		expectStep(s, a, 3, "B T H", "B T H", 0);
		[s commit];
		expectStep(s, a, 3, "B T H", "B T H", b + t + h);
		[s addAllocation:a.B];
		expectStep(s, a, 3, "B T H", "B T H", b + t + h);
		[s addAllocation:a.HB];
		expectStep(s, a, 4, "B T H HB", "B T H HB", b + t + h);
		[s addAllocation:a.TV];
		expectStep(s, a, 5, "B T H HB TV", "B T H HB TV", b + t + h);
		// A heap's buffer counts its heap, a view its texture: neither adds.
		[s commit];
		expectStep(s, a, 5, "B T H HB TV", "B T H HB TV", b + t + h);
		// A remove is counted until the commit drops it.
		[s removeAllocation:a.B];
		expectStep(s, a, 5, "T H HB TV", "T H HB TV", b + t + h);
		[s commit];
		expectStep(s, a, 4, "T H HB TV", "T H HB TV", t + h);
		[s removeAllocation:a.X];
		expectStep(s, a, 4, "T H HB TV", "T H HB TV", t + h);
		[s commit];
		expectStep(s, a, 4, "T H HB TV", "T H HB TV", t + h);
		id two[] = {a.B, a.B, a.X};
		[s addAllocations:two count:3];
		expectStep(s, a, 6, "B T H HB TV X", "T H HB TV B X", t + h);
		[s commit];
		expectStep(s, a, 6, "B T H HB TV X", "T H HB TV B X", t + h + b + x);
		// T goes, but its view keeps its memory counted.
		id rm[] = {a.X, a.T};
		[s removeAllocations:rm count:2];
		expectStep(s, a, 6, "B H HB TV", "H HB TV B", t + h + b + x);
		[s commit];
		expectStep(s, a, 4, "B H HB TV", "H HB TV B", t + h + b);
		[s addAllocations:two count:0];
		[s removeAllocations:rm count:0];
		expectStep(s, a, 4, "B H HB TV", "H HB TV B", t + h + b);
		[s removeAllAllocations];
		expectStep(s, a, 4, "", "", t + h + b);
		[s commit];
		expectStep(s, a, 0, "", "", 0);
		[s commit];
		expectStep(s, a, 0, "", "", 0);
		// removeAllAllocations takes back an add, and marks what is committed.
		[s addAllocation:a.B];
		[s commit];
		[s addAllocation:a.T];
		expectStep(s, a, 2, "B T", "B T", b);
		[s removeAllAllocations];
		expectStep(s, a, 1, "", "", b);
		[s commit];
		expectStep(s, a, 0, "", "", 0);
		// An add takes a remove's mark away; a remove takes an add back.
		[s addAllocation:a.B];
		[s commit];
		[s removeAllocation:a.B];
		[s addAllocation:a.B];
		expectStep(s, a, 1, "B", "B", b);
		[s commit];
		expectStep(s, a, 1, "B", "B", b);
		[s addAllocation:a.T];
		[s removeAllocation:a.T];
		expectStep(s, a, 1, "B", "B", b);
		[s commit];
		expectStep(s, a, 1, "B", "B", b);
		[s removeAllocation:a.B];
		[s removeAllocation:a.B];
		expectStep(s, a, 1, "", "", b);
		[s commit];
		expectStep(s, a, 0, "", "", 0);
		for (const auto &[n, o] : a.named)
			[s addAllocation:o];
		expectStep(s, a, 6, "B T H HB TV X", "B T H HB TV X", 0);
		[s commit];
		expectStep(s, a, 6, "B T H HB TV X", "B T H HB TV X", b + t + h + x);
	}
	drainRing(fx.ring);
}

// allocatedSize counts each piece of memory once: a heap's resources their heap, a view its
// texture, a texture over a buffer its buffer, with or without what they lie in, as oracle_residency's
// sizes case measures on the M3 Max; an allocation of no device counts nothing, and is held as any.
void test_residency_set_sizes()
{
	Fixture fx;
	@autoreleasepool {
		Allocations a(fx);
		RgfxBuffer *lb = [RgfxBuffer bufferWithLength:65536 options:0 device:nil serializer:&fx.ser];
		MTLTextureDescriptor *td =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
		                                                       width:64
		                                                      height:64
		                                                   mipmapped:NO];
		td.storageMode = MTLStorageModeShared;
		td.usage = MTLTextureUsageShaderRead;
		RgfxTexture *bt = [RgfxTexture textureWithDescriptor:td
		                                              buffer:lb
		                                              offset:0
		                                         bytesPerRow:256
		                                              device:nil
		                                          serializer:&fx.ser];
		td.storageMode = MTLStorageModePrivate;
		id<MTLTexture> ht = [a.H newTextureWithDescriptor:td];
		CHECK(lb != nil && bt != nil && ht != nil);
		auto sizeOf = [](NSArray *objects) {
			RgfxResidencySet *s = newSet();
			for (id o in objects)
				[s addAllocation:o];
			[s commit];
			return s.allocatedSize;
		};
		const uint64_t b = a.B.allocatedSize, t = a.T.allocatedSize, h = a.H.allocatedSize;
		CHECK(sizeOf(@[ a.B ]) == b);
		CHECK(sizeOf(@[ a.B, a.X ]) == b + a.X.allocatedSize);
		CHECK(sizeOf(@[ a.HB ]) == h);
		CHECK(sizeOf(@[ ht ]) == h);
		CHECK(sizeOf(@[ a.HB, ht ]) == h);
		CHECK(sizeOf(@[ a.H, a.HB, ht ]) == h);
		CHECK(sizeOf(@[ a.T ]) == t);
		CHECK(sizeOf(@[ a.TV ]) == t);
		CHECK(sizeOf(@[ a.T, a.TV ]) == t);
		CHECK(sizeOf(@[ bt ]) == lb.allocatedSize);
		CHECK(sizeOf(@[ lb, bt ]) == lb.allocatedSize);

		RgfxTestForeignAllocation *foreign = [RgfxTestForeignAllocation new];
		RgfxResidencySet *s = newSet();
		[s addAllocation:foreign];
		CHECK(s.allocationCount == 1 && [s containsAllocation:foreign] &&
		      s.allAllocations.firstObject == foreign);
		[s commit];
		CHECK(s.allocationCount == 1 && s.allocatedSize == 0);
		[s addAllocation:a.T];
		[s commit];
		CHECK(s.allocatedSize == t);
		[s removeAllocation:foreign];
		[s commit];
		CHECK(s.allocationCount == 1 && s.allocatedSize == t);
		CHECK(![s containsAllocation:(id<MTLAllocation> _Nonnull)nil]);
	}
	drainRing(fx.ring);
}

// The set holds each allocation it holds, a removed one until the commit drops it, and lets it
// go at that commit, before the caller's pool drains, as the M3 Max's set does (oracle_residency's
// hold case, whose commits run in no pool of their own).
void test_residency_set_holds_its_allocations()
{
	Fixture fx;
	RgfxResidencySet *s = newSet();
	__weak id w = nil;
	@autoreleasepool {
		id<MTLBuffer> r = [RgfxBuffer bufferWithLength:4096 options:0 device:nil serializer:&fx.ser];
		w = r;
		[s addAllocation:r];
	}
	CHECK(w != nil && s.allocationCount == 1);
	[s commit];
	CHECK(w != nil);
	@autoreleasepool {
		[s removeAllocation:w];
	}
	CHECK(w != nil && s.allocationCount == 1);
	[s commit];
	CHECK(w == nil && s.allocationCount == 0);
	@autoreleasepool {
		id<MTLBuffer> r = [RgfxBuffer bufferWithLength:4096 options:0 device:nil serializer:&fx.ser];
		w = r;
		[s addAllocation:r];
		[s commit];
		[s removeAllAllocations];
	}
	CHECK(w != nil);
	[s commit];
	CHECK(w == nil && s.allocationCount == 0);
	// A view and its texture, both dropped at one commit, both let go at it.
	__weak id wt = nil;
	@autoreleasepool {
		MTLTextureDescriptor *d =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
		                                                       width:64
		                                                      height:64
		                                                   mipmapped:NO];
		d.storageMode = MTLStorageModePrivate;
		id<MTLTexture> t = [RgfxTexture textureWithDescriptor:d
		                                                 heap:RGFX_OBJ_NONE
		                                           heapOffset:0
		                                               device:nil
		                                           serializer:&fx.ser];
		id<MTLTexture> v = [t newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm];
		wt = t;
		w = v;
		[s addAllocation:t];
		[s addAllocation:v];
		[s commit];
		[s removeAllocation:v];
		[s removeAllocation:t];
	}
	CHECK(w != nil && wt != nil);
	[s commit];
	CHECK(w == nil && wt == nil && s.allocationCount == 0);
	// An add taken back lets the allocation go.
	@autoreleasepool {
		id<MTLBuffer> r = [RgfxBuffer bufferWithLength:4096 options:0 device:nil serializer:&fx.ser];
		w = r;
		[s addAllocation:r];
	}
	CHECK(w != nil);
	@autoreleasepool {
		[s removeAllocation:w];
	}
	CHECK(w == nil && s.allocationCount == 0);
	drainRing(fx.ring);
}

// Threads that add, remove and commit at once, as an app and QuartzCore change a layer's set,
// leave the set what their changes add up to.
void test_residency_set_threads()
{
	Fixture fx;
	@autoreleasepool {
		RgfxResidencySet *s = newSet();
		NSMutableArray *kept = [NSMutableArray array];
		for (int i = 0; i < 64; i++)
			[kept addObject:[RgfxBuffer bufferWithLength:4096 options:0 device:nil serializer:&fx.ser]];
		std::vector<std::thread> threads;
		for (int k = 0; k < 4; k++) {
			threads.emplace_back([s, kept, k] {
				for (int round = 0; round < 200; round++) {
					@autoreleasepool {
						for (int i = k; i < 64; i += 4)
							[s addAllocation:kept[i]];
						[s commit];
						(void)s.allAllocations;
						(void)s.allocatedSize;
						for (int i = k; i < 64; i += 8)
							[s removeAllocation:kept[i]];
						[s commit];
					}
				}
			});
		}
		for (std::thread &t : threads)
			t.join();
		// Each thread's last round leaves its odd half of its quarter.
		CHECK(s.allocationCount == 32);
		uint64_t size = 0;
		for (NSUInteger i = 0; i < 64; i++) {
			const bool held = (i % 8) >= 4;
			CHECK([s containsAllocation:kept[i]] == held);
			if (held)
				size += [kept[i] allocatedSize];
		}
		CHECK(s.allocatedSize == size);
	}
	drainRing(fx.ring);
}

// Runs this binary again with --residency-dies `name`: it must end the process (SIGABRT) with
// `words` on its stderr.
void checkDies(const char *name, const char *words)
{
	int fds[2];
	CHECK(pipe(fds) == 0);
	posix_spawn_file_actions_t fa;
	posix_spawn_file_actions_init(&fa);
	posix_spawn_file_actions_adddup2(&fa, fds[1], 2);
	posix_spawn_file_actions_addclose(&fa, fds[0]);
	char *args[] = {const_cast<char *>(g_self), const_cast<char *>("--residency-dies"),
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
		std::fprintf(stderr, "--residency-dies %s ended with status %d:\n%s\n", name, status,
		             err.c_str());
	CHECK(aborted);
	CHECK(err.find(words) != std::string::npos);
}

// A nil allocation added or removed ends the process, as it does on the M3 Max (SIGSEGV there:
// oracle_residency's add-nil, remove-nil, adds-nil and removes-nil).
void test_residency_set_nil_allocations_end_the_process()
{
	checkDies("add", "-[RgfxResidencySet addAllocation:] of a nil allocation ends the process");
	checkDies("adds", "-[RgfxResidencySet addAllocations:count:] of a nil allocation ends the process");
	checkDies("remove", "-[RgfxResidencySet removeAllocation:] of a nil allocation ends the process");
	checkDies("removes",
	          "-[RgfxResidencySet removeAllocations:count:] of a nil allocation ends the process");
}

} // namespace

int rgfxResidencyDiesCase(const char *name)
{
	signal(SIGABRT, [](int) { _exit(kAborted); });
	@autoreleasepool {
		RgfxResidencySet *s = newSet();
		id buffer = [RgfxTestForeignAllocation new];
		[s addAllocation:buffer];
		[s commit];
		id withNil[] = {buffer, nil};
		if (std::strcmp(name, "add") == 0)
			[s addAllocation:(id<MTLAllocation> _Nonnull)nil];
		else if (std::strcmp(name, "adds") == 0)
			[s addAllocations:withNil count:2];
		else if (std::strcmp(name, "remove") == 0)
			[s removeAllocation:(id<MTLAllocation> _Nonnull)nil];
		else if (std::strcmp(name, "removes") == 0)
			[s removeAllocations:withNil count:2];
		else
			return 2;
	}
	return 0; // it went on: the check fails
}

void rgfxResidencyTests(const char *self)
{
	g_self = self;
	test_residency_set_answers();
	test_residency_set_steps();
	test_residency_set_sizes();
	test_residency_set_holds_its_allocations();
	test_residency_set_threads();
	test_residency_set_nil_allocations_end_the_process();
}
