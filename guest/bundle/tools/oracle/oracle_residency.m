// oracle_residency: what a device answers for residency sets (MTLResidencySet, macOS 15): the
// queries that might say it makes them, a set made of a default descriptor and of one with a label
// and an initial capacity, the set's classes and members, its count, contents and size after each
// add, remove and commit of buffers, textures and heaps, its residency requests, which objects hold
// which (a set its allocations, a queue and a command buffer their sets), a queue's, a command
// buffer's and the encoders' residency calls with a command buffer run after them, a
// CAMetalLayer's set, and the calls on bad inputs. Every case past the set's making runs in a child
// process of its own, so a call that ends the process ends only its case. No Metal validation
// layer is asked for: what a guest's process meets. It runs the same on the guest's device.
//   oracle_residency DEVICE              every step
//   oracle_residency DEVICE --case NAME  the case NAME alone (what the children run)
// Steps and their lines:
//   family NAME => YES|NO                  supportsFamily: for each MTLGPUFamily
//   device member CLASS SEL                each instance method of the device's classes below
//                                          NSObject whose name holds "esidenc"
//   descriptor => CLASS label L initialCapacity N      a fresh MTLResidencySetDescriptor
//   descriptor copy => CLASS label same|copy|nil initialCapacity N
//   make NAME => made CLASS | nil ERROR; error out OUT
//                                          newResidencySetWithDescriptor:error: of the descriptor
//                                          NAME (default: a fresh one; labelled: label
//                                          "oracle-rs", initialCapacity 16; big: initialCapacity
//                                          1048576), with an error out holding an error before the
//                                          call: OUT is what it holds after (untouched, nil or an
//                                          error)
//   set NAME => device same|other label L allocatedSize S allocationCount N all CLASS [N] again
//               same|other                 a made set's answers; `again` compares the
//                                          allAllocations of a second ask with the first's
//   set NAME label => same|copy|nil, after the descriptor's label changed L
//   set NAME setLabel: => answered, label then L | not answered
//   chain => CLASS < SUPER < ...           the set's classes, up to NSObject
//   members CLASS: SEL=TYPES ...           each one's instance methods, sorted, with their types
//   alloc NAME => CLASS allocatedSize S    the allocations of the cases: B, a Shared buffer of
//                                          4096 bytes; T, a Private 64 x 64 RGBA8 texture; H, a
//                                          Private heap of 1 MiB; HB, a buffer of 4096 bytes made
//                                          of H; TV, a view of T; X, a Shared buffer of 4096 bytes
//   case NAME => exited N | died on signal N   how the child of case NAME ended, after its lines
// The cases, each on a set of its own (labelled's descriptor) unless it says otherwise:
//   steps        step OP => count N size S has B1 T0 ... all [NAMES]
//                after each OP in turn: count is allocationCount, size allocatedSize, has
//                containsAllocation: of each allocation (1 or 0), all the names of allAllocations
//                in its order (?CLASS for one of none of them)
//   sizes        size WHAT => S (allocatedSize of each: S ...)   a set's allocatedSize once the
//                allocations WHAT are added and committed: buffers alone and together, Shared
//                and Private, of 4 KiB, 64 KiB, 1 MiB and 1 MiB + 4 KiB (L2), a heap's buffer
//                and texture (HT) without the heap and with it, a texture and its view, a
//                texture over a buffer (BT over LB) without the buffer and with it
//   residency    residency OP => count N size S     requestResidency and endResidency, on an
//                empty set and on one holding allocations, before and after a commit, twice
//   retain       retain WHAT => N   an object's retainCount after each step: an allocation added to
//                a set (twice), committed, removed and that committed; a set added to a queue
//                (twice) and removed, and used by a command buffer, before and after it completes
//                and once it is released
//   hold         hold WHAT => kept|freed [count N]  whether an object stays alive once the
//                probe drops its own reference, held only by a set (an allocation added, before
//                and after a commit, and after its removal is committed), a queue (a set added,
//                then removed; then a moment later, once the queue made a command buffer, and
//                once one of its buffers completed; the same for a set added and removed twice,
//                and for a set of another queue) or a command buffer (a set it uses, before and
//                after it completes)
//   queue        queue OP => done; queue run => status S error E fill V
//                addResidencySet:, addResidencySets:count:, removeResidencySet: and
//                removeResidencySets:count:, twice, of a set never added, of none, then a command
//                buffer of the queue that fills B with a blit, and B's first byte after it
//   cb           cb OP => status S error E fill V    a command buffer that uses sets
//                (useResidencySet:, useResidencySets:count:) and fills B with a blit
//   encoders     encoder KIND => useResidencySet: YES|NO useResidencySets:count: YES|NO
//                respondsToSelector: of a render, compute and blit encoder
//   compute-use, render-use   KIND use => status S error E   useResidencySet: and
//                useResidencySets:count: on that encoder (where it answers them), then its run
//   layer        layer => residencySet CLASS|nil count N; layer drawable => made|nil texture in
//                set YES|NO count N     a CAMetalLayer of the device: its residencySet (macOS 26)
//                and that set after nextDrawable
//   layer-spy    spy SEL from IMAGE => N    the messages a residency set's class and its bases
//                (below NSObject) receive while a CAMetalLayer of the device makes its set, five
//                drawables are drawn and presented on a queue the set is added to, the layer's
//                drawable size changes, the set is removed from the queue and the layer lets the
//                device go: each member, by the image of its caller (QuartzCore, Metal, ...),
//                with how many times; `spy skipped SEL TYPES` for a member of a type it does not
//                wrap
// Each case's child ends itself after 30 s (SIGALRM, signal 14): a call that does not return.
//   the bad inputs, each printing what the call answered (or how the child ended):
//   nil-descriptor        newResidencySetWithDescriptor: of nil
//   huge-capacity         a descriptor of initialCapacity NSUIntegerMax
//   add-nil, remove-nil, contains-nil, adds-nil (addAllocations: of B, nil and T), removes-nil
//   add-foreign           an allocation of no device: an NSObject adopting MTLAllocation
//   add-freed             an allocation added and committed, then freed (where the set does not
//                         hold it), then a commit and the set's answers
//   queue-add-nil, queue-remove-nil, queue-adds-nil, queue-add-foreign (an NSObject adopting
//   MTLResidencySet), cb-use-nil, cb-uses-nil, cb-use-foreign, cb-use-after-commit,
//   cb-use-internal and cb-use-internal-nil (the command buffer's useInternalResidencySet: of a
//   set and of nil, where it answers it)
// Build: see README.md.
#import "oracle_common.h"
#import <QuartzCore/QuartzCore.h>
#include <ctype.h>
#include <dlfcn.h>
#include <objc/message.h>
#include <objc/runtime.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

// An allocation of no device, and a residency set of none.
@interface OracleForeignAllocation : NSObject <MTLAllocation>
@end
@implementation OracleForeignAllocation
- (NSUInteger)allocatedSize
{
	return 4096;
}
@end

@interface OracleForeignSet : NSObject
@end
@implementation OracleForeignSet
@end

static const char *cls(id o)
{
	return o ? class_getName(object_getClass(o)) : "nil";
}

static const char *err_text(NSError *e)
{
	if (!e)
		return "(none)";
	static char buf[512];
	snprintf(buf, sizeof(buf), "%s %ld: %s", e.domain.UTF8String, (long)e.code,
	         e.localizedDescription.UTF8String);
	for (char *p = buf; *p; p++)
		if (*p == '\n')
			*p = '|';
	return buf;
}

static void members(id obj)
{
	for (Class c = object_getClass(obj); c != Nil && c != [NSObject class];
	     c = class_getSuperclass(c)) {
		unsigned n = 0;
		Method *list = class_copyMethodList(c, &n);
		NSMutableArray<NSString *> *names = [NSMutableArray array];
		for (unsigned i = 0; i < n; i++)
			[names addObject:[NSString stringWithFormat:@"%s=%s", sel_getName(method_getName(list[i])),
			                                            method_getTypeEncoding(list[i])]];
		free(list);
		[names sortUsingSelector:@selector(compare:)];
		printf("members %s:", class_getName(c));
		for (NSString *name in names)
			printf(" %s", name.UTF8String);
		printf("\n");
	}
}

// The probe's state: the device, a queue, the allocations and their names.
static id<MTLDevice> g_dev;
static id<MTLCommandQueue> g_queue;
static NSArray *g_allocs;
static NSArray<NSString *> *g_names;

static void make_allocs(void)
{
	id<MTLBuffer> b = [g_dev newBufferWithLength:4096 options:MTLResourceStorageModeShared];
	MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                                                                               width:64
	                                                                              height:64
	                                                                           mipmapped:NO];
	td.storageMode = MTLStorageModePrivate;
	td.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
	id<MTLTexture> t = [g_dev newTextureWithDescriptor:td];
	MTLHeapDescriptor *hd = [[MTLHeapDescriptor alloc] init];
	hd.size = 1 << 20;
	hd.storageMode = MTLStorageModePrivate;
	id<MTLHeap> h = [g_dev newHeapWithDescriptor:hd];
	id<MTLBuffer> hb = [h newBufferWithLength:4096 options:MTLResourceStorageModePrivate];
	id<MTLTexture> tv = [t newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm];
	id<MTLBuffer> x = [g_dev newBufferWithLength:4096 options:MTLResourceStorageModeShared];
	g_allocs = @[ b, t, h, hb, tv, x ];
	g_names = @[ @"B", @"T", @"H", @"HB", @"TV", @"X" ];
}

static id A(NSString *name)
{
	return g_allocs[[g_names indexOfObject:name]];
}

static MTLResidencySetDescriptor *descriptor(const char *name)
{
	MTLResidencySetDescriptor *d = [[MTLResidencySetDescriptor alloc] init];
	if (!strcmp(name, "labelled")) {
		d.label = @"oracle-rs";
		d.initialCapacity = 16;
	} else if (!strcmp(name, "big")) {
		d.initialCapacity = 1 << 20;
	}
	return d;
}

static id<MTLResidencySet> make_set(void)
{
	NSError *e = nil;
	return [g_dev newResidencySetWithDescriptor:descriptor("labelled") error:&e];
}

static void step(const char *op, id<MTLResidencySet> s)
{
	printf("step %s => count %lu size %llu has", op, (unsigned long)s.allocationCount,
	       (unsigned long long)s.allocatedSize);
	for (NSUInteger i = 0; i < g_allocs.count; i++)
		printf(" %s%d", g_names[i].UTF8String, (int)[s containsAllocation:g_allocs[i]]);
	printf(" all [");
	BOOL first = YES;
	for (id a in s.allAllocations) {
		NSUInteger k = NSNotFound;
		for (NSUInteger i = 0; i < g_allocs.count; i++)
			if (g_allocs[i] == a)
				k = i;
		printf("%s%s%s", first ? "" : " ", k == NSNotFound ? "?" : "",
		       k == NSNotFound ? cls(a) : g_names[k].UTF8String);
		first = NO;
	}
	printf("]\n");
}

// A command buffer of the queue that fills B with `value` by a blit, after `use` set its sets:
// its status, error and B's first byte.
static void run_fill(const char *what, uint8_t value, void (^use)(id<MTLCommandBuffer>))
{
	id<MTLBuffer> b = A(@"B");
	memset(b.contents, 0, 16);
	id<MTLCommandBuffer> cb = [g_queue commandBuffer];
	if (use)
		use(cb);
	id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
	[bl fillBuffer:b range:NSMakeRange(0, 4096) value:value];
	[bl endEncoding];
	[cb commit];
	[cb waitUntilCompleted];
	printf("%s => status %ld error %s fill %u\n", what, (long)cb.status, err_text(cb.error),
	       ((uint8_t *)b.contents)[0]);
}

// --- the cases ----------------------------------------------------------------------------------

static void case_steps(id<MTLResidencySet> s)
{
	step("fresh", s);
	[s addAllocation:A(@"B")];
	step("add B", s);
	[s addAllocation:A(@"T")];
	step("add T", s);
	[s addAllocation:A(@"H")];
	step("add H", s);
	[s commit];
	step("commit", s);
	[s addAllocation:A(@"B")];
	step("add B again", s);
	[s addAllocation:A(@"HB")];
	step("add HB", s);
	[s addAllocation:A(@"TV")];
	step("add TV", s);
	[s commit];
	step("commit", s);
	[s removeAllocation:A(@"B")];
	step("remove B", s);
	[s commit];
	step("commit", s);
	[s removeAllocation:A(@"X")];
	step("remove X never added", s);
	[s commit];
	step("commit", s);
	id two[] = {A(@"B"), A(@"B"), A(@"X")};
	[s addAllocations:two count:3];
	step("addAllocations B B X", s);
	[s commit];
	step("commit", s);
	id rm[] = {A(@"X"), A(@"T")};
	[s removeAllocations:rm count:2];
	step("removeAllocations X T", s);
	[s commit];
	step("commit", s);
	[s addAllocations:two count:0];
	step("addAllocations count 0", s);
	[s removeAllocations:rm count:0];
	step("removeAllocations count 0", s);
	[s removeAllAllocations];
	step("removeAllAllocations", s);
	[s commit];
	step("commit", s);
	[s commit];
	step("commit again", s);
	[s addAllocation:A(@"B")];
	[s commit];
	[s addAllocation:A(@"T")];
	step("add B, commit, add T", s);
	[s removeAllAllocations];
	step("removeAllAllocations", s);
	[s commit];
	step("commit", s);
	[s addAllocation:A(@"B")];
	[s commit];
	[s removeAllocation:A(@"B")];
	[s addAllocation:A(@"B")];
	step("add B, commit, remove B, add B", s);
	[s commit];
	step("commit", s);
	[s addAllocation:A(@"T")];
	[s removeAllocation:A(@"T")];
	step("add T, remove T", s);
	[s commit];
	step("commit", s);
	[s removeAllocation:A(@"B")];
	[s removeAllocation:A(@"B")];
	step("remove B twice", s);
	[s commit];
	step("commit", s);
	for (id a in g_allocs)
		[s addAllocation:a];
	step("add each", s);
	[s commit];
	step("commit", s);
}

static void res_line(const char *op, id<MTLResidencySet> s)
{
	printf("residency %s => count %lu size %llu\n", op, (unsigned long)s.allocationCount,
	       (unsigned long long)s.allocatedSize);
}

static void case_residency(id<MTLResidencySet> s)
{
	[s endResidency];
	res_line("end on a fresh set", s);
	[s requestResidency];
	res_line("request on an empty set", s);
	[s endResidency];
	res_line("end", s);
	[s addAllocation:A(@"B")];
	[s requestResidency];
	res_line("add B, request before commit", s);
	[s commit];
	res_line("commit while requested", s);
	[s requestResidency];
	res_line("request again", s);
	[s addAllocation:A(@"T")];
	[s commit];
	res_line("add T, commit while requested", s);
	[s endResidency];
	res_line("end", s);
	[s endResidency];
	res_line("end again", s);
	run_fill("residency run after end", 0x21, ^(id<MTLCommandBuffer> cb) {
	  [cb useResidencySet:s];
	});
}

static NSUInteger rc(id o)
{
	return ((NSUInteger(*)(id, SEL))objc_msgSend)(o, sel_registerName("retainCount"));
}

static void case_retain(void)
{
	id<MTLResidencySet> s = make_set();
	id x = A(@"X");
	printf("retain allocation fresh => %lu\n", (unsigned long)rc(x));
	@autoreleasepool {
		[s addAllocation:x];
	}
	printf("retain allocation added => %lu\n", (unsigned long)rc(x));
	@autoreleasepool {
		[s addAllocation:x];
	}
	printf("retain allocation added twice => %lu\n", (unsigned long)rc(x));
	@autoreleasepool {
		[s commit];
	}
	printf("retain allocation committed => %lu\n", (unsigned long)rc(x));
	@autoreleasepool {
		[s removeAllocation:x];
	}
	printf("retain allocation removed => %lu\n", (unsigned long)rc(x));
	@autoreleasepool {
		[s commit];
	}
	printf("retain allocation removal committed => %lu\n", (unsigned long)rc(x));
	printf("retain set fresh => %lu\n", (unsigned long)rc(s));
	@autoreleasepool {
		[g_queue addResidencySet:s];
	}
	printf("retain set added to a queue => %lu\n", (unsigned long)rc(s));
	@autoreleasepool {
		[g_queue addResidencySet:s];
	}
	printf("retain set added to a queue twice => %lu\n", (unsigned long)rc(s));
	@autoreleasepool {
		[g_queue removeResidencySet:s];
	}
	printf("retain set removed from the queue => %lu\n", (unsigned long)rc(s));
	@autoreleasepool {
		[g_queue removeResidencySet:s];
	}
	printf("retain set removed from the queue again => %lu\n", (unsigned long)rc(s));
	id<MTLCommandBuffer> cb = nil;
	@autoreleasepool {
		cb = [g_queue commandBuffer];
		[cb useResidencySet:s];
	}
	printf("retain set used by a command buffer => %lu\n", (unsigned long)rc(s));
	@autoreleasepool {
		[cb commit];
		[cb waitUntilCompleted];
	}
	usleep(100000);
	printf("retain set used by a completed command buffer => %lu\n", (unsigned long)rc(s));
	cb = nil;
	usleep(100000);
	printf("retain set used by a released command buffer => %lu\n", (unsigned long)rc(s));
}

// Whether the set `*ws` stays alive once `queue` makes a command buffer, and once one of its
// buffers completes.
static void hold_until_let_go(__weak id *ws, id<MTLCommandQueue> queue, const char *what)
{
	@autoreleasepool {
		id<MTLCommandBuffer> b = [queue commandBuffer];
		(void)b;
	}
	printf("hold set %s, once the queue made a command buffer => %s\n", what,
	       *ws ? "kept" : "freed");
	if (!*ws)
		return;
	@autoreleasepool {
		id<MTLCommandBuffer> b = [queue commandBuffer];
		[b commit];
		[b waitUntilCompleted];
	}
	usleep(100000);
	printf("hold set %s, once a command buffer of the queue completed => %s\n", what,
	       *ws ? "kept" : "freed");
}

static void case_hold(void)
{
	__weak id w = nil;
	id<MTLResidencySet> s = make_set();
	if (!s) {
		printf("hold => no set\n");
		return;
	}
	@autoreleasepool {
		id<MTLBuffer> r = [g_dev newBufferWithLength:4096 options:MTLResourceStorageModeShared];
		w = r;
		[s addAllocation:r];
	}
	printf("hold allocation added => %s count %lu\n", w ? "kept" : "freed",
	       (unsigned long)s.allocationCount);
	if (!w)
		return; // the set holds a freed allocation: nothing more to ask of it
	[s commit];
	printf("hold allocation committed => %s count %lu\n", w ? "kept" : "freed",
	       (unsigned long)s.allocationCount);
	@autoreleasepool {
		[s removeAllocation:w];
	}
	printf("hold allocation removed => %s count %lu\n", w ? "kept" : "freed",
	       (unsigned long)s.allocationCount);
	[s commit];
	printf("hold allocation removal committed => %s count %lu\n", w ? "kept" : "freed",
	       (unsigned long)s.allocationCount);
	@autoreleasepool {
		id<MTLBuffer> r = [g_dev newBufferWithLength:4096 options:MTLResourceStorageModeShared];
		w = r;
		[s addAllocation:r];
		[s commit];
		[s removeAllAllocations];
	}
	printf("hold allocation after removeAllAllocations => %s count %lu\n", w ? "kept" : "freed",
	       (unsigned long)s.allocationCount);
	[s commit];
	printf("hold allocation after removeAllAllocations committed => %s count %lu\n",
	       w ? "kept" : "freed", (unsigned long)s.allocationCount);

	__weak id ws = nil;
	@autoreleasepool {
		id<MTLResidencySet> q = make_set();
		ws = q;
		[g_queue addResidencySet:q];
	}
	printf("hold set added to a queue => %s\n", ws ? "kept" : "freed");
	if (ws) {
		@autoreleasepool {
			[g_queue removeResidencySet:ws];
		}
		printf("hold set removed from the queue => %s\n", ws ? "kept" : "freed");
		usleep(100000);
		printf("hold set removed from the queue, a moment later => %s\n", ws ? "kept" : "freed");
		hold_until_let_go(&ws, g_queue, "removed from the queue");
	}
	// Added twice and removed twice, before the queue makes a buffer.
	@autoreleasepool {
		id<MTLResidencySet> q = make_set();
		ws = q;
		[g_queue addResidencySet:q];
		[g_queue addResidencySet:q];
		[g_queue removeResidencySet:q];
	}
	printf("hold set added twice, removed once => %s\n", ws ? "kept" : "freed");
	if (ws) {
		@autoreleasepool {
			id<MTLCommandBuffer> b = [g_queue commandBuffer];
			(void)b;
		}
		printf("hold set added twice, removed once, once the queue made a command buffer => %s\n",
		       ws ? "kept" : "freed");
		@autoreleasepool {
			[g_queue removeResidencySet:ws];
		}
		printf("hold set added twice, removed twice => %s\n", ws ? "kept" : "freed");
		hold_until_let_go(&ws, g_queue, "added twice, removed twice");
	}
	// Added to and removed from a queue, then another queue makes and runs a buffer.
	@autoreleasepool {
		id<MTLResidencySet> q = make_set();
		ws = q;
		[g_queue addResidencySet:q];
		[g_queue removeResidencySet:q];
	}
	printf("hold set removed, another queue => %s\n", ws ? "kept" : "freed");
	if (ws) {
		id<MTLCommandQueue> other = [g_dev newCommandQueue];
		hold_until_let_go(&ws, other, "removed, another queue");
		if (ws) {
			hold_until_let_go(&ws, g_queue, "removed, another queue, then its own");
		}
	}

	id<MTLCommandBuffer> cb = [g_queue commandBuffer];
	@autoreleasepool {
		id<MTLResidencySet> u = make_set();
		ws = u;
		[cb useResidencySet:u];
	}
	printf("hold set used by a command buffer => %s\n", ws ? "kept" : "freed");
	@autoreleasepool {
		[cb commit];
		[cb waitUntilCompleted];
	}
	usleep(100000);
	printf("hold set used by a completed command buffer => %s\n", ws ? "kept" : "freed");
	cb = nil;
	usleep(100000);
	printf("hold set used by a released command buffer => %s\n", ws ? "kept" : "freed");
}

static void case_queue(void)
{
	id<MTLResidencySet> s = make_set(), s2 = make_set(), never = make_set();
	[s addAllocation:A(@"B")];
	[s commit];
	[s2 addAllocation:A(@"T")];
	[s2 commit];
	id<MTLResidencySet> both[] = {s, s2};
	[g_queue addResidencySet:s];
	printf("queue add S => done\n");
	[g_queue addResidencySet:s];
	printf("queue add S again => done\n");
	run_fill("queue run with S added twice", 0x31, nil);
	[g_queue removeResidencySet:s];
	printf("queue remove S => done\n");
	run_fill("queue run with S removed once", 0x32, nil);
	[g_queue removeResidencySet:s];
	printf("queue remove S again => done\n");
	[g_queue removeResidencySet:never];
	printf("queue remove a set never added => done\n");
	[g_queue addResidencySets:both count:2];
	printf("queue addResidencySets S S2 => done\n");
	[g_queue addResidencySets:both count:0];
	printf("queue addResidencySets count 0 => done\n");
	run_fill("queue run with S S2", 0x33, nil);
	[g_queue removeResidencySets:both count:2];
	printf("queue removeResidencySets S S2 => done\n");
	[g_queue removeResidencySets:both count:0];
	printf("queue removeResidencySets count 0 => done\n");
	[g_queue addResidencySet:s];
	[s removeAllAllocations];
	[s commit];
	run_fill("queue run with S emptied", 0x34, nil);
	[g_queue removeResidencySet:s];
	run_fill("queue run with none", 0x35, nil);
}

static void case_cb(void)
{
	id<MTLResidencySet> s = make_set(), s2 = make_set(), empty = make_set(), pending = make_set();
	[s addAllocation:A(@"B")];
	[s commit];
	[s2 addAllocation:A(@"T")];
	[s2 commit];
	[pending addAllocation:A(@"B")];
	run_fill("cb use S", 0x41, ^(id<MTLCommandBuffer> cb) {
	  [cb useResidencySet:s];
	});
	run_fill("cb use S twice", 0x42, ^(id<MTLCommandBuffer> cb) {
	  [cb useResidencySet:s];
	  [cb useResidencySet:s];
	});
	run_fill("cb useResidencySets S S2", 0x43, ^(id<MTLCommandBuffer> cb) {
	  id<MTLResidencySet> both[] = {s, s2};
	  [cb useResidencySets:both count:2];
	});
	run_fill("cb useResidencySets count 0", 0x44, ^(id<MTLCommandBuffer> cb) {
	  id<MTLResidencySet> both[] = {s, s2};
	  [cb useResidencySets:both count:0];
	});
	run_fill("cb use an empty set", 0x45, ^(id<MTLCommandBuffer> cb) {
	  [cb useResidencySet:empty];
	});
	run_fill("cb use a set of uncommitted adds", 0x46, ^(id<MTLCommandBuffer> cb) {
	  [cb useResidencySet:pending];
	});
	run_fill("cb use S, then S changed before commit", 0x47, ^(id<MTLCommandBuffer> cb) {
	  [cb useResidencySet:s];
	  [s addAllocation:A(@"H")];
	  [s commit];
	});
	run_fill("cb use S of a queue that has it", 0x48, ^(id<MTLCommandBuffer> cb) {
	  [g_queue addResidencySet:s];
	  [cb useResidencySet:s];
	});
	[g_queue removeResidencySet:s];
}

static void case_encoders(void)
{
	id<MTLCommandBuffer> cb = [g_queue commandBuffer];
	MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
	rp.colorAttachments[0].texture = A(@"T");
	rp.colorAttachments[0].loadAction = MTLLoadActionClear;
	rp.colorAttachments[0].storeAction = MTLStoreActionStore;
	id enc[3];
	const char *kinds[3] = {"render", "compute", "blit"};
	enc[0] = [cb renderCommandEncoderWithDescriptor:rp];
	SEL one = @selector(useResidencySet:), many = @selector(useResidencySets:count:);
	printf("encoder %s %s => useResidencySet: %s useResidencySets:count: %s\n", kinds[0], cls(enc[0]),
	       [enc[0] respondsToSelector:one] ? "YES" : "NO", [enc[0] respondsToSelector:many] ? "YES" : "NO");
	[enc[0] endEncoding];
	enc[1] = [cb computeCommandEncoder];
	printf("encoder %s %s => useResidencySet: %s useResidencySets:count: %s\n", kinds[1], cls(enc[1]),
	       [enc[1] respondsToSelector:one] ? "YES" : "NO", [enc[1] respondsToSelector:many] ? "YES" : "NO");
	[enc[1] endEncoding];
	enc[2] = [cb blitCommandEncoder];
	printf("encoder %s %s => useResidencySet: %s useResidencySets:count: %s\n", kinds[2], cls(enc[2]),
	       [enc[2] respondsToSelector:one] ? "YES" : "NO", [enc[2] respondsToSelector:many] ? "YES" : "NO");
	[enc[2] endEncoding];
	[cb commit];
	[cb waitUntilCompleted];
	for (Class c = object_getClass(g_queue); c != Nil && c != [NSObject class]; c = class_getSuperclass(c)) {
		unsigned n = 0;
		Method *list = class_copyMethodList(c, &n);
		for (unsigned i = 0; i < n; i++)
			if (strcasestr(sel_getName(method_getName(list[i])), "esidenc"))
				printf("queue member %s %s\n", class_getName(c), sel_getName(method_getName(list[i])));
		free(list);
	}
	id<MTLCommandBuffer> cb2 = [g_queue commandBuffer];
	for (Class c = object_getClass(cb2); c != Nil && c != [NSObject class]; c = class_getSuperclass(c)) {
		unsigned n = 0;
		Method *list = class_copyMethodList(c, &n);
		for (unsigned i = 0; i < n; i++)
			if (strcasestr(sel_getName(method_getName(list[i])), "esidenc"))
				printf("cb member %s %s\n", class_getName(c), sel_getName(method_getName(list[i])));
		free(list);
	}
}

static void case_encoder_use(BOOL render)
{
	id<MTLResidencySet> s = make_set();
	[s addAllocation:A(@"B")];
	[s commit];
	id<MTLResidencySet> both[] = {s, s};
	id<MTLCommandBuffer> cb = [g_queue commandBuffer];
	id enc;
	if (render) {
		MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
		rp.colorAttachments[0].texture = A(@"T");
		rp.colorAttachments[0].loadAction = MTLLoadActionClear;
		rp.colorAttachments[0].storeAction = MTLStoreActionStore;
		enc = [cb renderCommandEncoderWithDescriptor:rp];
	} else {
		enc = [cb computeCommandEncoder];
	}
	const char *kind = render ? "render" : "compute";
	if (![enc respondsToSelector:@selector(useResidencySet:)]) {
		printf("%s use => not answered\n", kind);
		[enc endEncoding];
		return;
	}
	((void (*)(id, SEL, id))objc_msgSend)(enc, @selector(useResidencySet:), s);
	printf("%s useResidencySet: => done\n", kind);
	if ([enc respondsToSelector:@selector(useResidencySets:count:)]) {
		((void (*)(id, SEL, const id *, NSUInteger))objc_msgSend)(enc, @selector(useResidencySets:count:),
		                                                          both, 2);
		printf("%s useResidencySets:count: => done\n", kind);
	}
	((void (*)(id, SEL, id))objc_msgSend)(enc, @selector(useResidencySet:), nil);
	printf("%s useResidencySet: nil => done\n", kind);
	[enc endEncoding];
	[cb commit];
	[cb waitUntilCompleted];
	printf("%s use => status %ld error %s\n", kind, (long)cb.status, err_text(cb.error));
}

static void case_layer(void)
{
	CAMetalLayer *l = [CAMetalLayer layer];
	l.device = g_dev;
	l.drawableSize = CGSizeMake(64, 64);
	if (![l respondsToSelector:@selector(residencySet)]) {
		printf("layer => residencySet not answered\n");
		return;
	}
	id<MTLResidencySet> s = l.residencySet;
	printf("layer => residencySet %s count %lu\n", cls(s), (unsigned long)s.allocationCount);
	if (s)
		printf("layer set label %s size %llu\n", s.label ? s.label.UTF8String : "(nil)",
		       (unsigned long long)s.allocatedSize);
	id<CAMetalDrawable> d = [l nextDrawable];
	id<MTLResidencySet> s2 = l.residencySet;
	printf("layer drawable => %s texture in set %s count %lu same set %s\n", d ? "made" : "nil",
	       d && s2 ? ([s2 containsAllocation:d.texture] ? "YES" : "NO") : "-",
	       (unsigned long)s2.allocationCount, s == s2 ? "YES" : "NO");
	for (id a in s2.allAllocations)
		printf("layer set holds %s allocatedSize %lu%s\n", cls(a), (unsigned long)[a allocatedSize],
		       d && a == (id)d.texture ? " (the drawable's texture)" : "");
}

static void size_line(const char *what, NSArray *objs)
{
	id<MTLResidencySet> s = make_set();
	for (id a in objs)
		[s addAllocation:a];
	[s commit];
	printf("size %s => %llu (allocatedSize of each:", what, (unsigned long long)s.allocatedSize);
	for (id a in objs)
		printf(" %lu", (unsigned long)[a allocatedSize]);
	printf(")\n");
}

static void case_sizes(void)
{
	id<MTLBuffer> p = [g_dev newBufferWithLength:4096 options:MTLResourceStorageModePrivate];
	id<MTLBuffer> lb = [g_dev newBufferWithLength:65536 options:MTLResourceStorageModeShared];
	id<MTLBuffer> l = [g_dev newBufferWithLength:1 << 20 options:MTLResourceStorageModeShared];
	id<MTLBuffer> l2 = [g_dev newBufferWithLength:(1 << 20) + 4096 options:MTLResourceStorageModeShared];
	MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                                                                               width:64
	                                                                              height:64
	                                                                           mipmapped:NO];
	td.storageMode = MTLStorageModeShared;
	td.usage = MTLTextureUsageShaderRead;
	id<MTLTexture> bt = [lb newTextureWithDescriptor:td offset:0 bytesPerRow:256];
	td.storageMode = MTLStorageModePrivate;
	id<MTLTexture> ht = [(id<MTLHeap>)A(@"H") newTextureWithDescriptor:td];
	size_line("B", @[ A(@"B") ]);
	size_line("B X", @[ A(@"B"), A(@"X") ]);
	size_line("P (Private 4 KiB)", @[ p ]);
	size_line("B P", @[ A(@"B"), p ]);
	size_line("LB (Shared 64 KiB)", @[ lb ]);
	size_line("L (Shared 1 MiB)", @[ l ]);
	size_line("L2 (Shared 1 MiB + 4 KiB)", @[ l2 ]);
	size_line("L B", @[ l, A(@"B") ]);
	size_line("HB", @[ A(@"HB") ]);
	size_line("HT", @[ ht ]);
	size_line("HB HT", @[ A(@"HB"), ht ]);
	size_line("H HB HT", @[ A(@"H"), A(@"HB"), ht ]);
	size_line("T", @[ A(@"T") ]);
	size_line("TV", @[ A(@"TV") ]);
	size_line("T TV", @[ A(@"T"), A(@"TV") ]);
	size_line("BT", @[ bt ]);
	size_line("LB BT", @[ lb, bt ]);
}

// The spy: each member of the set's classes, wrapped to count its calls by its caller's image.
static NSMutableDictionary<NSString *, NSNumber *> *g_spy;
static volatile BOOL g_spying;
static Class g_spyClass;

static void spy_note(id self, SEL sel, void *ra)
{
	if (!g_spying || ![self isKindOfClass:g_spyClass])
		return;
	Dl_info info;
	const char *img = "?";
	if (dladdr(ra, &info) && info.dli_fname) {
		const char *slash = strrchr(info.dli_fname, '/');
		img = slash ? slash + 1 : info.dli_fname;
	}
	NSString *k = [NSString stringWithFormat:@"%s from %s", sel_getName(sel), img];
	@synchronized(g_spy) {
		g_spy[k] = @(g_spy[k].unsignedLongValue + 1);
	}
}

// The type encoding without its offsets and qualifiers: "v@:@" for "v24@0:8@16".
static NSString *bare_types(const char *t)
{
	NSMutableString *b = [NSMutableString string];
	for (const char *p = t; *p; p++)
		if (!isdigit((unsigned char)*p) && *p != 'r')
			[b appendFormat:@"%c", *p];
	return b;
}

static void spy_wrap(Method m)
{
	SEL sel = method_getName(m);
	IMP orig = method_getImplementation(m);
	NSString *t = bare_types(method_getTypeEncoding(m));
	id block = nil;
	if ([t isEqualToString:@"v@:"])
		block = ^(id self) {
		  spy_note(self, sel, __builtin_return_address(0));
		  ((void (*)(id, SEL))orig)(self, sel);
		};
	else if ([t isEqualToString:@"@@:"])
		block = ^id(id self) {
		  spy_note(self, sel, __builtin_return_address(0));
		  return ((id(*)(id, SEL))orig)(self, sel);
		};
	else if ([t isEqualToString:@"Q@:"])
		block = ^uint64_t(id self) {
		  spy_note(self, sel, __builtin_return_address(0));
		  return ((uint64_t(*)(id, SEL))orig)(self, sel);
		};
	else if ([t isEqualToString:@"v@:@"])
		block = ^(id self, id a) {
		  spy_note(self, sel, __builtin_return_address(0));
		  ((void (*)(id, SEL, id))orig)(self, sel, a);
		};
	else if ([t isEqualToString:@"v@:^@Q"])
		block = ^(id self, const void *a, uint64_t n) {
		  spy_note(self, sel, __builtin_return_address(0));
		  ((void (*)(id, SEL, const void *, uint64_t))orig)(self, sel, a, n);
		};
	else if ([t isEqualToString:@"B@:@"])
		block = ^BOOL(id self, id a) {
		  spy_note(self, sel, __builtin_return_address(0));
		  return ((BOOL(*)(id, SEL, id))orig)(self, sel, a);
		};
	else if ([t isEqualToString:@"Q@:@"])
		block = ^uint64_t(id self, id a) {
		  spy_note(self, sel, __builtin_return_address(0));
		  return ((uint64_t(*)(id, SEL, id))orig)(self, sel, a);
		};
	else if ([t isEqualToString:@"v@:Q"])
		block = ^(id self, uint64_t a) {
		  spy_note(self, sel, __builtin_return_address(0));
		  ((void (*)(id, SEL, uint64_t))orig)(self, sel, a);
		};
	else if ([t isEqualToString:@"B@:^@Q^@Q"])
		block = ^BOOL(id self, const void *a, uint64_t n, const void *b, uint64_t k) {
		  spy_note(self, sel, __builtin_return_address(0));
		  return ((BOOL(*)(id, SEL, const void *, uint64_t, const void *, uint64_t))orig)(self, sel, a,
		                                                                                    n, b, k);
		};
	if (!block) {
		printf("spy skipped %s %s\n", sel_getName(sel), method_getTypeEncoding(m));
		return;
	}
	method_setImplementation(m, imp_implementationWithBlock(block));
}

static void present_one(CAMetalLayer *l)
{
	@autoreleasepool {
		id<CAMetalDrawable> d = [l nextDrawable];
		if (!d)
			return;
		id<MTLCommandBuffer> cb = [g_queue commandBuffer];
		MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
		rp.colorAttachments[0].texture = d.texture;
		rp.colorAttachments[0].loadAction = MTLLoadActionClear;
		rp.colorAttachments[0].storeAction = MTLStoreActionStore;
		id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
		[enc endEncoding];
		[cb presentDrawable:d];
		[cb commit];
		[cb waitUntilCompleted];
	}
}

static void case_layer_spy(void)
{
	id<MTLResidencySet> probe = make_set();
	if (!probe) {
		printf("layer-spy => no set\n");
		return;
	}
	g_spy = [NSMutableDictionary dictionary];
	g_spyClass = object_getClass(probe);
	for (Class c = g_spyClass; c != Nil && c != [NSObject class]; c = class_getSuperclass(c)) {
		unsigned n = 0;
		Method *list = class_copyMethodList(c, &n);
		for (unsigned i = 0; i < n; i++) {
			const char *name = sel_getName(method_getName(list[i]));
			if (name[0] == '.' || !strcmp(name, "dealloc") || !strncmp(name, "init", 4))
				continue;
			spy_wrap(list[i]);
		}
		free(list);
	}
	probe = nil;
	CAMetalLayer *l = [CAMetalLayer layer];
	g_spying = YES;
	l.device = g_dev;
	l.pixelFormat = MTLPixelFormatBGRA8Unorm;
	l.drawableSize = CGSizeMake(64, 64);
	id<MTLResidencySet> s = [l respondsToSelector:@selector(residencySet)] ? l.residencySet : nil;
	if (s)
		[g_queue addResidencySet:s];
	for (int i = 0; i < 5; i++)
		present_one(l);
	l.drawableSize = CGSizeMake(128, 128);
	present_one(l);
	present_one(l);
	g_spying = NO;
	printf("layer-spy => set %s count %lu\n", cls(s), (unsigned long)s.allocationCount);
	g_spying = YES;
	if (s)
		[g_queue removeResidencySet:s];
	l.device = nil;
	l = nil;
	s = nil;
	usleep(200000);
	g_spying = NO;
	NSArray<NSString *> *keys = [g_spy.allKeys sortedArrayUsingSelector:@selector(compare:)];
	for (NSString *k in keys)
		printf("spy %s => %lu\n", k.UTF8String, g_spy[k].unsignedLongValue);
}

// --- the bad inputs -----------------------------------------------------------------------------

static int bad_input(const char *name)
{
	NSError *e = nil;
	if (!strcmp(name, "nil-descriptor")) {
		id<MTLResidencySet> s = [g_dev newResidencySetWithDescriptor:(MTLResidencySetDescriptor *_Nonnull)nil
		                                                       error:&e];
		printf("nil-descriptor => %s %s label %s\n", s ? "made" : "nil", s ? cls(s) : err_text(e),
		       s && s.label ? s.label.UTF8String : "(nil)");
		return 0;
	}
	if (!strcmp(name, "huge-capacity")) {
		MTLResidencySetDescriptor *d = [[MTLResidencySetDescriptor alloc] init];
		d.initialCapacity = NSUIntegerMax;
		id<MTLResidencySet> s = [g_dev newResidencySetWithDescriptor:d error:&e];
		printf("huge-capacity => %s %s\n", s ? "made" : "nil", s ? cls(s) : err_text(e));
		if (s) {
			[s addAllocation:A(@"B")];
			[s commit];
			step("huge-capacity add B, commit", s);
		}
		return 0;
	}
	id<MTLResidencySet> s = make_set();
	if (!s) {
		printf("%s => no set\n", name);
		return 0;
	}
	if (!strcmp(name, "add-nil")) {
		[s addAllocation:(id<MTLAllocation> _Nonnull)nil];
		step("add nil", s);
		[s commit];
		step("commit", s);
	} else if (!strcmp(name, "remove-nil")) {
		[s addAllocation:A(@"B")];
		[s commit];
		[s removeAllocation:(id<MTLAllocation> _Nonnull)nil];
		step("add B, commit, remove nil", s);
		[s commit];
		step("commit", s);
	} else if (!strcmp(name, "contains-nil")) {
		[s addAllocation:A(@"B")];
		printf("contains-nil => %d\n", (int)[s containsAllocation:(id<MTLAllocation> _Nonnull)nil]);
	} else if (!strcmp(name, "adds-nil")) {
		id list[] = {A(@"B"), nil, A(@"T")};
		[s addAllocations:list count:3];
		step("addAllocations B nil T", s);
		[s commit];
		step("commit", s);
	} else if (!strcmp(name, "removes-nil")) {
		[s addAllocation:A(@"B")];
		[s addAllocation:A(@"T")];
		[s commit];
		id list[] = {A(@"B"), nil};
		[s removeAllocations:list count:2];
		step("add B T, commit, removeAllocations B nil", s);
		[s commit];
		step("commit", s);
	} else if (!strcmp(name, "add-foreign")) {
		OracleForeignAllocation *f = [[OracleForeignAllocation alloc] init];
		[s addAllocation:f];
		step("add foreign", s);
		printf("add-foreign contains => %d\n", (int)[s containsAllocation:f]);
		[s commit];
		step("commit", s);
		run_fill("add-foreign run", 0x51, ^(id<MTLCommandBuffer> cb) {
		  [cb useResidencySet:s];
		});
	} else if (!strcmp(name, "add-freed")) {
		__weak id w = nil;
		@autoreleasepool {
			id<MTLBuffer> r = [g_dev newBufferWithLength:4096 options:MTLResourceStorageModeShared];
			w = r;
			[s addAllocation:r];
			[s commit];
		}
		printf("add-freed => %s\n", w ? "kept by the set" : "freed");
		[s commit];
		step("commit", s);
		run_fill("add-freed run", 0x52, ^(id<MTLCommandBuffer> cb) {
		  [cb useResidencySet:s];
		});
	} else if (!strcmp(name, "queue-add-nil")) {
		[g_queue addResidencySet:(id<MTLResidencySet> _Nonnull)nil];
		printf("queue-add-nil => done\n");
		run_fill("queue-add-nil run", 0x53, nil);
	} else if (!strcmp(name, "queue-remove-nil")) {
		[g_queue removeResidencySet:(id<MTLResidencySet> _Nonnull)nil];
		printf("queue-remove-nil => done\n");
		run_fill("queue-remove-nil run", 0x54, nil);
	} else if (!strcmp(name, "queue-adds-nil")) {
		id<MTLResidencySet> list[] = {s, nil};
		[g_queue addResidencySets:list count:2];
		printf("queue-adds-nil => done\n");
		run_fill("queue-adds-nil run", 0x55, nil);
	} else if (!strcmp(name, "queue-add-foreign")) {
		id f = [[OracleForeignSet alloc] init];
		[g_queue addResidencySet:f];
		printf("queue-add-foreign => done\n");
		run_fill("queue-add-foreign run", 0x56, nil);
	} else if (!strcmp(name, "cb-use-nil")) {
		run_fill("cb-use-nil run", 0x57, ^(id<MTLCommandBuffer> cb) {
		  [cb useResidencySet:(id<MTLResidencySet> _Nonnull)nil];
		});
	} else if (!strcmp(name, "cb-uses-nil")) {
		run_fill("cb-uses-nil run", 0x58, ^(id<MTLCommandBuffer> cb) {
		  id<MTLResidencySet> list[] = {s, nil};
		  [cb useResidencySets:list count:2];
		});
	} else if (!strcmp(name, "cb-use-foreign")) {
		run_fill("cb-use-foreign run", 0x59, ^(id<MTLCommandBuffer> cb) {
		  id f = [[OracleForeignSet alloc] init];
		  [cb useResidencySet:f];
		});
	} else if (!strcmp(name, "cb-use-internal") || !strcmp(name, "cb-use-internal-nil")) {
		const BOOL none = !strcmp(name, "cb-use-internal-nil");
		char what[64];
		snprintf(what, sizeof(what), "%s run", name);
		run_fill(what, none ? 0x5b : 0x5a, ^(id<MTLCommandBuffer> cb) {
		  SEL internal = sel_registerName("useInternalResidencySet:");
		  if (![(id)cb respondsToSelector:internal]) {
			  printf("%s => not answered\n", name);
			  return;
		  }
		  ((void (*)(id, SEL, id))objc_msgSend)(cb, internal, none ? nil : s);
		  printf("%s => done\n", name);
		});
	} else if (!strcmp(name, "cb-use-after-commit")) {
		id<MTLCommandBuffer> cb = [g_queue commandBuffer];
		[cb commit];
		[cb useResidencySet:s];
		printf("cb-use-after-commit => done\n");
		[cb waitUntilCompleted];
		printf("cb-use-after-commit => status %ld error %s\n", (long)cb.status, err_text(cb.error));
	} else {
		fprintf(stderr, "no case %s\n", name);
		return 2;
	}
	return 0;
}

static int run_case(const char *name)
{
	make_allocs();
	if (!strcmp(name, "hold")) {
		case_hold();
		return 0;
	}
	if (!strcmp(name, "retain")) {
		if (!make_set()) {
			printf("retain => no set\n");
			return 0;
		}
		case_retain();
		return 0;
	}
	if (!strcmp(name, "encoders")) {
		case_encoders();
		return 0;
	}
	if (!strcmp(name, "layer")) {
		case_layer();
		return 0;
	}
	if (!strcmp(name, "layer-spy")) {
		case_layer_spy();
		return 0;
	}
	if (!strcmp(name, "sizes")) {
		if (!make_set()) {
			printf("sizes => no set\n");
			return 0;
		}
		case_sizes();
		return 0;
	}
	if (!strcmp(name, "queue") || !strcmp(name, "cb") || !strcmp(name, "steps") ||
	    !strcmp(name, "residency") || !strcmp(name, "compute-use") || !strcmp(name, "render-use")) {
		id<MTLResidencySet> s = make_set();
		if (!s) {
			printf("%s => no set\n", name);
			return 0;
		}
		if (!strcmp(name, "steps"))
			case_steps(s);
		else if (!strcmp(name, "residency"))
			case_residency(s);
		else if (!strcmp(name, "queue"))
			case_queue();
		else if (!strcmp(name, "cb"))
			case_cb();
		else
			case_encoder_use(!strcmp(name, "render-use"));
		return 0;
	}
	return bad_input(name);
}

static void set_line(const char *name, id<MTLResidencySet> s, MTLResidencySetDescriptor *d)
{
	NSArray *all = s.allAllocations, *again = s.allAllocations;
	printf("set %s => device %s label %s allocatedSize %llu allocationCount %lu all %s [%lu] again %s\n",
	       name, s.device == g_dev ? "same" : "other", s.label ? s.label.UTF8String : "(nil)",
	       (unsigned long long)s.allocatedSize, (unsigned long)s.allocationCount, cls(all),
	       (unsigned long)all.count, all == again ? "same" : "other");
	NSString *before = s.label;
	const char *kind = !before ? "nil" : before == d.label ? "same" : "copy";
	d.label = @"changed";
	printf("set %s label => %s, after the descriptor's label changed %s\n", name, kind,
	       s.label ? s.label.UTF8String : "(nil)");
	if ([(id)s respondsToSelector:@selector(setLabel:)]) {
		((void (*)(id, SEL, id))objc_msgSend)(s, @selector(setLabel:), @"set-later");
		printf("set %s setLabel: => answered, label then %s\n", name,
		       s.label ? s.label.UTF8String : "(nil)");
	} else {
		printf("set %s setLabel: => not answered\n", name);
	}
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IOLBF, 0);
	@autoreleasepool {
		if (argc < 2) {
			fprintf(stderr, "usage: %s DEVICE [--case NAME]\n", argv[0]);
			return 2;
		}
		g_dev = oracle_device(argv[1]);
		g_queue = [g_dev newCommandQueue];
		if (argc >= 4 && strcmp(argv[2], "--case") == 0) {
			alarm(30);
			return run_case(argv[3]);
		}
		oracle_header("oracle_residency", g_dev);
		struct {
			const char *name;
			MTLGPUFamily family;
		} families[] = {
		    {"Apple7", MTLGPUFamilyApple7}, {"Apple8", MTLGPUFamilyApple8},
		    {"Apple9", MTLGPUFamilyApple9}, {"Apple10", (MTLGPUFamily)1010},
		    {"Mac2", MTLGPUFamilyMac2},     {"Common3", MTLGPUFamilyCommon3},
		    {"Metal3", MTLGPUFamilyMetal3}, {"Metal4", (MTLGPUFamily)5002},
		};
		for (size_t k = 0; k < sizeof(families) / sizeof(families[0]); k++)
			printf("family %s => %s\n", families[k].name,
			       [g_dev supportsFamily:families[k].family] ? "YES" : "NO");
		for (Class c = object_getClass(g_dev); c != Nil && c != [NSObject class];
		     c = class_getSuperclass(c)) {
			unsigned n = 0;
			Method *list = class_copyMethodList(c, &n);
			NSMutableArray<NSString *> *names = [NSMutableArray array];
			for (unsigned i = 0; i < n; i++) {
				const char *sel = sel_getName(method_getName(list[i]));
				if (strcasestr(sel, "esidenc"))
					[names addObject:@(sel)];
			}
			free(list);
			[names sortUsingSelector:@selector(compare:)];
			for (NSString *sel in names)
				printf("device member %s %s\n", class_getName(c), sel.UTF8String);
		}
		MTLResidencySetDescriptor *fresh = [[MTLResidencySetDescriptor alloc] init];
		printf("descriptor => %s label %s initialCapacity %lu\n", cls(fresh),
		       fresh.label ? fresh.label.UTF8String : "(nil)", (unsigned long)fresh.initialCapacity);
		MTLResidencySetDescriptor *labelled = descriptor("labelled");
		MTLResidencySetDescriptor *copy = [labelled copy];
		printf("descriptor copy => %s label %s initialCapacity %lu\n", cls(copy),
		       !copy.label ? "nil" : copy.label == labelled.label ? "same" : "copy",
		       (unsigned long)copy.initialCapacity);
		const char *shapes[] = {"default", "labelled", "big"};
		id<MTLResidencySet> made = nil;
		for (size_t k = 0; k < sizeof(shapes) / sizeof(shapes[0]); k++) {
			MTLResidencySetDescriptor *d = descriptor(shapes[k]);
			NSError *sentinel = [NSError errorWithDomain:@"oracle" code:1 userInfo:nil];
			NSError *e = sentinel;
			id<MTLResidencySet> s = [g_dev newResidencySetWithDescriptor:d error:&e];
			printf("make %s => %s %s; error out %s\n", shapes[k], s ? "made" : "nil",
			       s ? cls(s) : "", e == sentinel ? "untouched" : e ? err_text(e) : "nil");
			if (!s)
				continue;
			set_line(shapes[k], s, d);
			if (!made)
				made = s;
		}
		if (made) {
			printf("chain =>");
			for (Class c = object_getClass(made); c != Nil; c = class_getSuperclass(c))
				printf(" %s%s", class_getName(c), class_getSuperclass(c) ? " <" : "");
			printf("\n");
			members(made);
		}
		make_allocs();
		for (NSUInteger i = 0; i < g_allocs.count; i++)
			printf("alloc %s => %s allocatedSize %lu\n", g_names[i].UTF8String, cls(g_allocs[i]),
			       (unsigned long)[g_allocs[i] allocatedSize]);
		fflush(stdout);
		const char *cases[] = {"steps",          "residency",        "retain",          "hold",
		                       "queue",          "cb",               "encoders",
		                       "compute-use",    "render-use",       "layer",            "layer-spy",       "sizes",
		                       "nil-descriptor", "huge-capacity",    "add-nil",
		                       "remove-nil",     "contains-nil",     "adds-nil",
		                       "removes-nil",    "add-foreign",      "add-freed",
		                       "queue-add-nil",  "queue-remove-nil", "queue-adds-nil",
		                       "queue-add-foreign", "cb-use-nil",    "cb-uses-nil",
		                       "cb-use-foreign", "cb-use-after-commit",
		                       "cb-use-internal", "cb-use-internal-nil"};
		for (size_t k = 0; k < sizeof(cases) / sizeof(cases[0]); k++) {
			const char *name = cases[k];
			int fds[2];
			pipe(fds);
			posix_spawn_file_actions_t fa;
			posix_spawn_file_actions_init(&fa);
			posix_spawn_file_actions_adddup2(&fa, fds[1], 1);
			posix_spawn_file_actions_adddup2(&fa, fds[1], 2);
			posix_spawn_file_actions_addclose(&fa, fds[0]);
			char *args[] = {argv[0], argv[1], (char *)"--case", (char *)name, NULL};
			pid_t pid = 0;
			posix_spawn(&pid, argv[0], &fa, NULL, args, environ);
			posix_spawn_file_actions_destroy(&fa);
			close(fds[1]);
			char buf[4096];
			ssize_t n;
			while ((n = read(fds[0], buf, sizeof(buf))) > 0)
				fwrite(buf, 1, (size_t)n, stdout);
			close(fds[0]);
			int st = 0;
			waitpid(pid, &st, 0);
			if (WIFSIGNALED(st))
				printf("case %s => died on signal %d\n", name, WTERMSIG(st));
			else
				printf("case %s => exited %d\n", name, WEXITSTATUS(st));
		}
	}
	return 0;
}
