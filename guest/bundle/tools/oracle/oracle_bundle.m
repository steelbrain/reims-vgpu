// oracle_bundle: what a Mac's Metal device answers where the guest bundle's device answered
// otherwise (the bundle's divergences from Metal). Each case runs in a child process of its own,
// since some end the process at Metal's assertions; the parent prints how each child ended.
//   oracle_bundle DEVICE [CASE...]   the cases (or groups) named, or every case but the groups
// The cases, each line `CASE key=value ...`:
//   buffers      allocatedSize of buffers of each storage mode by newBufferWithLength: and
//                newBufferWithBytes:, of lengths 1 to 1 MiB + 1
//   heapbuffers  allocatedSize of buffers in a 4 MiB Private heap, and the device's
//                heapBufferSizeAndAlignWithLength:options:
//   nocopy.*     a no-copy buffer (newBufferWithBytesNoCopy:) over pages the probe maps: its
//                allocatedSize, length, contents and gpuAddress; .page (one page), .3pages,
//                .16k, .len100 (100 bytes), .off16 (16 bytes into a page), .managed (one page)
//   events       newEventWithOptions: and newSharedEventWithOptions: (SPI) for options 0 to 16:
//                the class answered, or nil
//   archive.*    newBinaryArchiveWithDescriptor: over files: .system (IconRendering's
//                binary.metallib, where the OS has it, read in place), .made (an archive the
//                device serializes, then the same bytes with the magic 0xcafebabe or 0xcbfebabe,
//                a slice's CPU type changed, a fat header over junk, no slices, a bare .metallib);
//                each line gives the archive or the error, and a pipeline's hit or miss there
//   failmiss     newFunctionWithDescriptor: with MTLFunctionOptionFailOnBinaryArchiveMiss, with
//                and without an archive holding the function
//   functypes    the TYPE tag byte of each function in oracle_bundle*.metallib, and the
//                function type and constants Metal answers for it
//   libtype      the type and installName of a library compiled from source as a dynamic library,
//                of one made from its bytes, and of oracle_bundle_dyn.metallib; the bytes' header
//   minlod       whether MTLTextureViewDescriptor answers minLOD
//   classes      the device's bufferClass (SPI) and the class of a buffer it makes
//   heapsize     heapTextureSizeAndAlignWithDescriptor: and allocatedSize of 1024 x 1024 textures
//                with allowGPUOptimizedContents YES and NO
//   libforms     newLibraryWithFunctionArray:error:, newLibraryWithMetalPackageURL:error: and
//                newLibraryWithMPSGraphPackageURL:name:error: (SPI)
//   functypes.dyn  where Metal's assertion at a dynamic library's exported function falls
//   failmiss2    newFunctionWithDescriptor: failing on a miss, by stage, by the options the
//                function was added with, and by the kernel's constant values
//   view.*       a view of a 64 x 64 Private texture (usage ShaderRead | PixelFormatView) that the
//                bundle answers nil for: .class (RGBA16Float in RGBA8Unorm), .levels (levels 2 to
//                4 of 3), .slices (slices 3 and 4 of a 4-slice array), .type (2D as 3D),
//                .slicecount (a 4-slice array as a cube), .noformat (Invalid); .ok (RGBA8Unorm in
//                R32Float) is one Metal makes
//   buftex.*     a 2D texture over a Shared buffer by the public newTextureWithDescriptor:offset:
//                bytesPerRow:: .bc1 and .depth (formats minimumLinearTextureAlignmentForPixelFormat:
//                may end the process at), .shortrow (468 x 36 BGRA8 rows 1536 bytes apart, under
//                the row's 1872); linear.shortrow, the same by the buffer's linear SPI
//   devtex.*     a 64 x 64 2D texture the device makes, usage RenderTarget: .depth.shared,
//                .depth.managed, .stencil.managed and .depth.private, by format and storage mode
//   texaddr      (a group) for a texture of each storage mode and shape (texaddr.MODE.SHAPE):
//                its class, allocatedSize, the SPI gpuAddress, virtualAddress, resourceSize and
//                allocationID (n/r where it does not answer), its gpuResourceID, a second one's,
//                and buffers' made around them; .heap (textures in Private Automatic and
//                Placement heaps), .buffer (over a Shared and a Managed buffer), .view (level 1 of
//                a Shared and a Private texture), .iosurface (over a surface)
//   texmem       (a group) texmem.MODE.SHAPE, Shared and Managed RGBA8 textures: where each texel
//                a replaceRegion: wrote lies at the texture's virtualAddress (at=-1: not found),
//                whether it is laid out linearly, whether getBytes: sees a CPU write there, and
//                what a blit's GPU write leaves there
//   nocopy.ro    (a group) .anon.MODE and .file.MODE: a no-copy buffer over read-only memory, and
//                what a GPU read, a blit fill and a kernel's store do with it
//   nocopy.copies  a no-copy Managed buffer over writable memory: whether a GPU read sees a CPU
//                write made with no didModifyRange:, and the CPU a blit's write with no synchronize
//   linalign     (a group) minimumLinearTextureAlignmentForPixelFormat: of raw formats 0 to 799,
//                1000, 1023, 4096 and 65535, one child each (linalign.N)
//   argbuf.ds.*  a depth/stencil state in an argument buffer: .id (its gpuResourceID), .encoder
//                (an encoder of one MTLDataTypeDepthStencilState descriptor), .set (the bytes
//                setDepthStencilState:atIndex: writes there), .mixed (beside a texture and a
//                pointer), .function (an argument buffer of oracle_bundle_ds.metallib holding one)
//   argbuf.binding  the bindings a compute pipeline of div_argbuf (oracle_bundle_args.metallib)
//                reflects, and newArgumentEncoderWithBufferBinding: of its argument buffer's
//                binding against the function's own encoder; .plain, of its plain buffer's; .lib,
//                of the binding the library's reflectionForFunctionWithName: answers
//   argbuf.reflection  newArgumentEncoderWithBufferIndex:reflection: at div_argbuf's argument
//                buffer, its plain buffer and an index it has no buffer at
//   iosurf.props copyIOSurfaceSharedTextureProperties: (SPI) for textures of several formats,
//                sizes and usages, and a surface made of the properties
//   iosurf.buffer  newBufferWithIOSurface: (SPI) over BGRA surfaces (.420v, a two-plane one): the
//                buffer's length, options, contents and iosurface, and CPU and GPU writes through
//                one seen through the other
//   event.port   a shared event's eventPort (SPI), its NSCoding through a keyed archiver, and its
//                handle's
//   archive.slices  binary archives of a serialized archive's slices, recombined and with their
//                bytes junked, alone and together, each opened and asked for a pipeline
//   classes.spi  which classes of the device's, its shared event's and its buffer's (and
//                IOGPU's base classes) define the SPI the cases above call
//   libforms2    newLibraryWithFunctionArray:error: of two functions, of one twice, and the
//                packages' creators over an empty directory and other URLs; libforms.kernel and
//                libforms.vertex, a function array of a kernel or a vertex function
// Build: see README.md.
#import "oracle_common.h"
#import <IOSurface/IOSurface.h>
#include <fcntl.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <objc/message.h>
#include <objc/runtime.h>
#include <spawn.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

static const char *g_argv0;

static NSString *exe_dir(void)
{
	NSString *exe = [NSString stringWithUTF8String:g_argv0];
	if (![exe containsString:@"/"])
		exe = [@"./" stringByAppendingString:exe];
	return [[exe stringByResolvingSymlinksInPath] stringByDeletingLastPathComponent];
}

static NSURL *beside(NSString *name)
{
	return [NSURL fileURLWithPath:[exe_dir() stringByAppendingPathComponent:name]];
}

static const char *err_text(NSError *e)
{
	if (e == nil)
		return "nil";
	return [[NSString stringWithFormat:@"%@ %ld %@", e.domain, (long)e.code,
	                                   [e.localizedDescription
	                                       stringByReplacingOccurrencesOfString:@"\n"
	                                                                 withString:@" | "]]
	    UTF8String];
}

static const char *cls(id o)
{
	return o ? class_getName(object_getClass(o)) : "nil";
}

// --- buffers ------------------------------------------------------------------------------

static const NSUInteger kLengths[] = {1,     16,    100,   256,   4095,  4096,  4097,
                                      16383, 16384, 16385, 65536, 65537, 1048577};
static const MTLResourceOptions kModes[] = {MTLResourceStorageModeShared,
                                            MTLResourceStorageModeManaged,
                                            MTLResourceStorageModePrivate};
static const char *const kModeNames[] = {"shared", "managed", "private"};

static void case_buffers(id<MTLDevice> d)
{
	for (int m = 0; m < 3; m++)
		for (size_t i = 0; i < sizeof kLengths / sizeof kLengths[0]; i++) {
			@autoreleasepool {
				id<MTLBuffer> b = [d newBufferWithLength:kLengths[i] options:kModes[m]];
				printf("buffers new %s length=%lu allocatedSize=%lu\n", kModeNames[m],
				       (unsigned long)kLengths[i], (unsigned long)b.allocatedSize);
			}
		}
	static uint8_t bytes[65537];
	for (int m = 0; m < 2; m++)
		for (size_t i = 0; i < sizeof kLengths / sizeof kLengths[0]; i++) {
			if (kLengths[i] > sizeof bytes)
				continue;
			@autoreleasepool {
				id<MTLBuffer> b = [d newBufferWithBytes:bytes length:kLengths[i] options:kModes[m]];
				printf("buffers bytes %s length=%lu allocatedSize=%lu\n", kModeNames[m],
				       (unsigned long)kLengths[i], (unsigned long)b.allocatedSize);
			}
		}
}

static void case_heapbuffers(id<MTLDevice> d)
{
	// Private alone: a Shared heap ends the process at the descriptor's validation on the AMD and
	// the Intel GPU ("Shared storage mode disallowed").
	const MTLStorageMode modes[] = {MTLStorageModePrivate};
	const char *names[] = {"private"};
	for (int m = 0; m < 1; m++) {
		MTLHeapDescriptor *hd = [MTLHeapDescriptor new];
		hd.size = 4u << 20;
		hd.storageMode = modes[m];
		id<MTLHeap> h = [d newHeapWithDescriptor:hd];
		printf("heapbuffers %s heap=%s size=%lu\n", names[m], cls(h), (unsigned long)h.size);
		const NSUInteger lens[] = {1, 100, 4097, 16385, 65537};
		for (int i = 0; i < 5; i++) {
			MTLResourceOptions o = (MTLResourceOptions)modes[m] << MTLResourceStorageModeShift;
			MTLSizeAndAlign sa = [d heapBufferSizeAndAlignWithLength:lens[i] options:o];
			id<MTLBuffer> b = [h newBufferWithLength:lens[i] options:o];
			printf("heapbuffers %s length=%lu sizeAndAlign=%lu,%lu allocatedSize=%lu\n", names[m],
			       (unsigned long)lens[i], (unsigned long)sa.size, (unsigned long)sa.align,
			       (unsigned long)b.allocatedSize);
		}
	}
}

static void nocopy(id<MTLDevice> d, const char *tag, size_t offset, size_t length,
                   MTLResourceOptions options)
{
	const size_t page = (size_t)getpagesize();
	uint8_t *p = mmap(NULL, 8 * page, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
	printf("%s page=%zu pointerOffset=%zu length=%zu\n", tag, page, offset, length);
	fflush(stdout);
	id<MTLBuffer> b = [d newBufferWithBytesNoCopy:p + offset
	                                       length:length
	                                      options:options
	                                  deallocator:nil];
	if (b == nil) {
		printf("%s buffer=nil\n", tag);
		return;
	}
	printf("%s buffer=%s length=%lu allocatedSize=%lu contentsOffset=%td gpuAddressModPage=%llu\n",
	       tag, cls(b), (unsigned long)b.length, (unsigned long)b.allocatedSize,
	       (uint8_t *)b.contents - p, (unsigned long long)(b.gpuAddress % page));
}

// --- events -------------------------------------------------------------------------------

static void case_events(id<MTLDevice> d)
{
	printf("events newEvent=%s newSharedEvent=%s\n", cls([d newEvent]), cls([d newSharedEvent]));
	const char *sels[] = {"newEventWithOptions:", "newSharedEventWithOptions:"};
	for (int s = 0; s < 2; s++) {
		SEL sel = sel_registerName(sels[s]);
		if (![d respondsToSelector:sel]) {
			printf("events %s n/r\n", sels[s]);
			continue;
		}
		const long opts[] = {0, 1, 2, 3, 4, 8, 16};
		for (int i = 0; i < 7; i++) {
			@try {
				id e = ((id(*)(id, SEL, long))objc_msgSend)(d, sel, opts[i]);
				printf("events %s(%ld)=%s", sels[s], opts[i], cls(e));
				if (s == 1 && e != nil)
					printf(" signaledValue=%llu", [(id<MTLSharedEvent>)e signaledValue]);
				printf("\n");
			} @catch (NSException *x) {
				printf("events %s(%ld) threw %s\n", sels[s], opts[i], x.name.UTF8String);
			}
		}
	}
}

// --- binary archives ----------------------------------------------------------------------

static uint32_t be32(const uint8_t *p)
{
	return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3];
}

static void put_be32(uint8_t *p, uint32_t v)
{
	p[0] = (uint8_t)(v >> 24);
	p[1] = (uint8_t)(v >> 16);
	p[2] = (uint8_t)(v >> 8);
	p[3] = (uint8_t)v;
}

// A fat header's fields, and each slice's first four bytes.
static void describe_fat(const char *tag, NSData *data)
{
	const uint8_t *b = data.bytes;
	if (data.length < 8) {
		printf("%s file length=%lu\n", tag, (unsigned long)data.length);
		return;
	}
	const uint32_t n = be32(b + 4);
	printf("%s file length=%lu magic=0x%08x slices=%u\n", tag, (unsigned long)data.length,
	       be32(b), n);
	for (uint32_t i = 0; i < n && i < 32 && 8 + (i + 1) * 20 <= data.length; i++) {
		const uint8_t *s = b + 8 + i * 20;
		const uint32_t off = be32(s + 8), size = be32(s + 12);
		char head[5] = "????";
		if (off + 4 <= data.length)
			for (int k = 0; k < 4; k++)
				head[k] = (b[off + k] >= 0x20 && b[off + k] < 0x7f) ? (char)b[off + k] : '.';
		printf("%s slice %u cputype=0x%08x subtype=0x%08x offset=%u size=%u align=%u head=%s\n",
		       tag, i, be32(s), be32(s + 4), off, size, be32(s + 16), head);
	}
}

static id<MTLBinaryArchive> open_archive(id<MTLDevice> d, const char *tag, NSURL *url)
{
	MTLBinaryArchiveDescriptor *ad = [MTLBinaryArchiveDescriptor new];
	ad.url = url;
	NSError *e = nil;
	id<MTLBinaryArchive> a = [d newBinaryArchiveWithDescriptor:ad error:&e];
	printf("%s archive=%s error=%s\n", tag, cls(a), err_text(e));
	fflush(stdout);
	return a;
}

// A compute pipeline of div_kernel that fails on a miss in `a`: hit (made) or the miss's error.
static void pipeline_in(id<MTLDevice> d, const char *tag, id<MTLLibrary> lib,
                        id<MTLBinaryArchive> a)
{
	if (a == nil || lib == nil)
		return;
	MTLFunctionConstantValues *cv = [MTLFunctionConstantValues new];
	bool flag = true;
	float scale = 2.0f;
	[cv setConstantValue:&flag type:MTLDataTypeBool atIndex:0];
	[cv setConstantValue:&scale type:MTLDataTypeFloat atIndex:1];
	NSError *e = nil;
	id<MTLFunction> fn = [lib newFunctionWithName:@"div_kernel" constantValues:cv error:&e];
	MTLComputePipelineDescriptor *cd = [MTLComputePipelineDescriptor new];
	cd.computeFunction = fn;
	cd.binaryArchives = @[ a ];
	printf("%s function=%s\n", tag, cls(fn));
	fflush(stdout);
	e = nil;
	id p = [d newComputePipelineStateWithDescriptor:cd
	                                        options:MTLPipelineOptionFailOnBinaryArchiveMiss
	                                     reflection:nil
	                                          error:&e];
	printf("%s failOnMiss pipeline=%s error=%s\n", tag, p ? "made" : "nil", err_text(e));
}

static void write_variant(const char *tag, NSData *base, NSURL *url, void (^edit)(NSMutableData *))
{
	NSMutableData *m = [base mutableCopy];
	edit(m);
	[m writeToURL:url atomically:YES];
	describe_fat(tag, m);
}

static void case_archive_system(id<MTLDevice> d)
{
	NSString *paths[] = {
	    @"/System/Library/PrivateFrameworks/IconRendering.framework/Resources/binary.metallib",
	    @"/System/Library/PrivateFrameworks/IconRendering.framework/Versions/A/Resources/"
	    @"binary.metallib",
	};
	for (int i = 0; i < 2; i++) {
		NSData *data = [NSData dataWithContentsOfFile:paths[i]];
		printf("archive.system path=%s present=%d\n", paths[i].UTF8String, data != nil);
		if (data == nil)
			continue;
		describe_fat("archive.system", data);
		open_archive(d, "archive.system", [NSURL fileURLWithPath:paths[i]]);
		NSError *e = nil;
		id<MTLLibrary> lib = [d newLibraryWithURL:[NSURL fileURLWithPath:paths[i]] error:&e];
		printf("archive.system asLibrary=%s functions=%lu error=%s\n", cls(lib),
		       (unsigned long)lib.functionNames.count, err_text(e));
		return;
	}
}

static void case_archive_made(id<MTLDevice> d)
{
	NSError *e = nil;
	id<MTLLibrary> lib = [d newLibraryWithURL:beside(@"oracle_bundle.metallib") error:&e];
	if (lib == nil) {
		printf("archive.made library error=%s\n", err_text(e));
		return;
	}
	id<MTLBinaryArchive> a = [d newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new]
	                                                     error:&e];
	MTLFunctionConstantValues *cv = [MTLFunctionConstantValues new];
	bool flag = true;
	float scale = 2.0f;
	[cv setConstantValue:&flag type:MTLDataTypeBool atIndex:0];
	[cv setConstantValue:&scale type:MTLDataTypeFloat atIndex:1];
	MTLComputePipelineDescriptor *cd = [MTLComputePipelineDescriptor new];
	cd.computeFunction = [lib newFunctionWithName:@"div_kernel" constantValues:cv error:&e];
	e = nil;
	BOOL added = [a addComputePipelineFunctionsWithDescriptor:cd error:&e];
	printf("archive.made add=%d error=%s\n", added, err_text(e));
	NSString *dir = NSTemporaryDirectory();
	NSURL *url = [NSURL fileURLWithPath:[dir stringByAppendingPathComponent:@"div-made.bin"]];
	e = nil;
	BOOL wrote = [a serializeToURL:url error:&e];
	printf("archive.made serialize=%d error=%s\n", wrote, err_text(e));
	NSData *base = [NSData dataWithContentsOfURL:url];
	if (base == nil)
		return;
	describe_fat("archive.made", base);
	pipeline_in(d, "archive.made", lib, open_archive(d, "archive.made", url));
	const uint32_t magic = be32(base.bytes);
	struct {
		const char *tag;
		void (^edit)(NSMutableData *);
	} variants[] = {
	    {"archive.made.cafebabe",
	     ^(NSMutableData *m) {
		     put_be32(m.mutableBytes, 0xcafebabe);
	     }},
	    {"archive.made.cbfebabe",
	     ^(NSMutableData *m) {
		     put_be32(m.mutableBytes, 0xcbfebabe);
	     }},
	    {"archive.made.cputype",
	     ^(NSMutableData *m) {
		     put_be32((uint8_t *)m.mutableBytes + 8, 0x01000007);
	     }},
	    {"archive.made.junk",
	     ^(NSMutableData *m) {
		     uint8_t *b = m.mutableBytes;
		     const uint32_t off = be32(b + 8 + 8);
		     if (off + 64 <= m.length)
			     memset(b + off, 0x5a, 64);
	     }},
	    {"archive.made.noslices",
	     ^(NSMutableData *m) {
		     put_be32((uint8_t *)m.mutableBytes + 4, 0);
	     }},
	};
	(void)magic;
	for (size_t i = 0; i < sizeof variants / sizeof variants[0]; i++) {
		NSURL *vu = [NSURL
		    fileURLWithPath:[dir stringByAppendingPathComponent:
		                             [NSString stringWithFormat:@"div-%s.bin", variants[i].tag]]];
		write_variant(variants[i].tag, base, vu, variants[i].edit);
		pipeline_in(d, variants[i].tag, lib, open_archive(d, variants[i].tag, vu));
	}
	// A fat header of one slice over 64 junk bytes, by either magic.
	const uint32_t magics[] = {0xcafebabe, 0xcbfebabe};
	for (int k = 0; k < 2; k++) {
		NSMutableData *m = [NSMutableData dataWithLength:8 + 20 + 64];
		uint8_t *b = m.mutableBytes;
		put_be32(b, magics[k]);
		put_be32(b + 4, 1);
		put_be32(b + 8, be32((const uint8_t *)base.bytes + 8));
		put_be32(b + 12, be32((const uint8_t *)base.bytes + 12));
		put_be32(b + 16, 28);
		put_be32(b + 20, 64);
		put_be32(b + 24, 0);
		memset(b + 28, 0x5a, 64);
		char tag[64];
		snprintf(tag, sizeof tag, "archive.fatjunk.%08x", magics[k]);
		NSURL *vu = [NSURL
		    fileURLWithPath:[dir stringByAppendingPathComponent:
		                             [NSString stringWithFormat:@"div-%s.bin", tag]]];
		[m writeToURL:vu atomically:YES];
		describe_fat(tag, m);
		open_archive(d, tag, vu);
	}
	open_archive(d, "archive.metallib", beside(@"oracle_bundle.metallib"));
}

// --- functions ----------------------------------------------------------------------------

static void case_failmiss(id<MTLDevice> d)
{
	NSError *e = nil;
	id<MTLLibrary> lib = [d newLibraryWithURL:beside(@"oracle_bundle.metallib") error:&e];
	if (lib == nil) {
		printf("failmiss library error=%s\n", err_text(e));
		return;
	}
	id<MTLBinaryArchive> empty =
	    [d newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new] error:&e];
	id<MTLBinaryArchive> holding =
	    [d newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new] error:&e];
	NSString *names[] = {@"div_visible", @"div_kernel", @"div_stitchable"};
	for (int n = 0; n < 3; n++) {
		MTLFunctionDescriptor *fd = [MTLFunctionDescriptor functionDescriptor];
		fd.name = names[n];
		if (n == 1) {
			MTLFunctionConstantValues *cv = [MTLFunctionConstantValues new];
			bool flag = true;
			float scale = 2.0f;
			[cv setConstantValue:&flag type:MTLDataTypeBool atIndex:0];
			[cv setConstantValue:&scale type:MTLDataTypeFloat atIndex:1];
			fd.constantValues = cv;
		}
		e = nil;
		BOOL added = [holding addFunctionWithDescriptor:fd library:lib error:&e];
		printf("failmiss %s addFunction=%d error=%s\n", names[n].UTF8String, added, err_text(e));
		const MTLFunctionOptions opts[] = {
		    MTLFunctionOptionFailOnBinaryArchiveMiss,
		    MTLFunctionOptionFailOnBinaryArchiveMiss | MTLFunctionOptionCompileToBinary};
		for (int o = 0; o < 2; o++) {
			id<MTLBinaryArchive> archives[] = {nil, empty, holding};
			const char *an[] = {"none", "empty", "holding"};
			for (int k = 0; k < 3; k++) {
				fd.options = opts[o];
				fd.binaryArchives = archives[k] ? @[ archives[k] ] : nil;
				e = nil;
				id<MTLFunction> fn = [lib newFunctionWithDescriptor:fd error:&e];
				printf("failmiss %s options=%lu archives=%s function=%s error=%s\n",
				       names[n].UTF8String, (unsigned long)opts[o], an[k], cls(fn), err_text(e));
			}
		}
	}
}

// The function list of a .metallib: each function's NAME and TYPE tag byte, and its tags' names.
static void metallib_types(NSData *data, NSMutableDictionary<NSString *, NSArray *> *out)
{
	const uint8_t *p = data.bytes;
	const size_t n = data.length;
	if (n < 0x28 || memcmp(p, "MTLB", 4) != 0)
		return;
	uint64_t q;
	memcpy(&q, p + 0x18, 8);
	if (q + 4 > n)
		return;
	uint32_t count;
	memcpy(&count, p + q, 4);
	q += 4;
	for (uint32_t i = 0; i < count && q + 4 <= n; i++) {
		uint32_t group;
		memcpy(&group, p + q, 4);
		uint64_t t = q + 4, end = q + group;
		NSString *name = nil;
		int type = -1;
		NSMutableArray<NSString *> *tags = [NSMutableArray array];
		while (t + 6 <= end && end <= n) {
			if (memcmp(p + t, "ENDT", 4) == 0)
				break;
			uint16_t size;
			memcpy(&size, p + t + 4, 2);
			[tags addObject:[[NSString alloc] initWithBytes:p + t
			                                         length:4
			                                       encoding:NSASCIIStringEncoding]];
			if (memcmp(p + t, "NAME", 4) == 0)
				name = [[NSString alloc] initWithBytes:p + t + 6
				                                length:strnlen((const char *)p + t + 6, size)
				                              encoding:NSUTF8StringEncoding];
			else if (memcmp(p + t, "TYPE", 4) == 0 && size >= 1)
				type = p[t + 6];
			t += 6 + size;
		}
		if (name)
			out[name] = @[ @(type), [tags componentsJoinedByString:@","] ];
		q = end;
	}
}

static void case_functypes(id<MTLDevice> d)
{
	NSString *libs[] = {@"oracle_bundle.metallib", @"oracle_bundle_mesh.metallib",
	                    @"oracle_bundle_rt.metallib", @"oracle_bundle_dyn.metallib"};
	for (int l = 0; l < 4; l++) {
		NSURL *url = beside(libs[l]);
		NSData *data = [NSData dataWithContentsOfURL:url];
		if (data == nil) {
			printf("functypes %s absent\n", libs[l].UTF8String);
			continue;
		}
		const uint8_t *p = data.bytes;
		printf("functypes %s header", libs[l].UTF8String);
		for (size_t k = 0; k < 0x20 && k < data.length; k++)
			printf("%s%02x", k % 4 ? "" : " ", p[k]);
		printf("\n");
		NSMutableDictionary<NSString *, NSArray *> *types = [NSMutableDictionary dictionary];
		metallib_types(data, types);
		NSError *e = nil;
		id<MTLLibrary> lib = [d newLibraryWithURL:url error:&e];
		printf("functypes %s library=%s type=%ld installName=%s functionNames=%s error=%s\n",
		       libs[l].UTF8String, cls(lib), lib ? (long)lib.type : -1L,
		       lib.installName ? lib.installName.UTF8String : "nil",
		       [[lib.functionNames sortedArrayUsingSelector:@selector(compare:)]
		           componentsJoinedByString:@","]
		           .UTF8String,
		       err_text(e));
		for (NSString *name in [types.allKeys sortedArrayUsingSelector:@selector(compare:)]) {
			if ([types[name][0] intValue] == 3) {
				// Metal asserts at a dynamic library's exported function (case functypes.dyn).
				printf("functypes %s TYPE=3 tags=%s (asked in functypes.dyn)\n", name.UTF8String,
				       [types[name][1] UTF8String]);
				continue;
			}
			id<MTLFunction> fn = [lib newFunctionWithName:name];
			printf("functypes %s TYPE=%d tags=%s function=%s functionType=%ld", name.UTF8String,
			       [types[name][0] intValue], [types[name][1] UTF8String], cls(fn),
			       fn ? (long)fn.functionType : -1L);
			NSDictionary<NSString *, MTLFunctionConstant *> *c = fn.functionConstantsDictionary;
			for (NSString *k in [c.allKeys sortedArrayUsingSelector:@selector(compare:)])
				printf(" constant=%s/%lu/%lu/%d", k.UTF8String, (unsigned long)c[k].index,
				       (unsigned long)c[k].type, (int)c[k].required);
			printf("\n");
		}
	}
}

static NSData *library_bytes(id lib)
{
	SEL sel = sel_registerName("libraryDataContents");
	if (lib == nil || ![lib respondsToSelector:sel])
		return nil;
	return ((id(*)(id, SEL))objc_msgSend)(lib, sel);
}

static void print_lib(const char *tag, id<MTLLibrary> lib, NSError *e)
{
	printf("%s library=%s type=%ld installName=%s functionNames=%s error=%s\n", tag, cls(lib),
	       lib ? (long)lib.type : -1L, lib.installName ? lib.installName.UTF8String : "nil",
	       [[lib.functionNames sortedArrayUsingSelector:@selector(compare:)]
	           componentsJoinedByString:@","]
	           .UTF8String,
	       err_text(e));
	NSData *bytes = library_bytes(lib);
	if (bytes != nil) {
		const uint8_t *p = bytes.bytes;
		printf("%s bytes=%lu header", tag, (unsigned long)bytes.length);
		for (size_t k = 0; k < 0x58 && k < bytes.length; k++)
			printf("%s%02x", k % 4 ? "" : " ", p[k]);
		printf("\n");
	}
}

static void case_libtype(id<MTLDevice> d)
{
	NSString *src = @"#include <metal_stdlib>\nusing namespace metal;\n"
	                @"float div_dyn_src(float x) { return x * 3.0f; }\n"
	                @"kernel void div_dyn_kernel(device float *o [[buffer(0)]], uint i "
	                @"[[thread_position_in_grid]]) { o[i] = float(i); }\n";
	for (int dyn = 0; dyn < 2; dyn++) {
		MTLCompileOptions *o = [MTLCompileOptions new];
		if (dyn) {
			o.libraryType = MTLLibraryTypeDynamic;
			o.installName = @"@executable_path/libdiv_src.metallib";
		}
		NSError *e = nil;
		id<MTLLibrary> lib = [d newLibraryWithSource:src options:o error:&e];
		const char *tag = dyn ? "libtype.source.dynamic" : "libtype.source.executable";
		print_lib(tag, lib, e);
		NSData *bytes = library_bytes(lib);
		if (bytes != nil) {
			dispatch_data_t dd = dispatch_data_create(bytes.bytes, bytes.length, nil,
			                                          DISPATCH_DATA_DESTRUCTOR_DEFAULT);
			e = nil;
			id<MTLLibrary> again = [d newLibraryWithData:dd error:&e];
			char t2[96];
			snprintf(t2, sizeof t2, "%s.fromData", tag);
			print_lib(t2, again, e);
		}
		if (dyn && lib != nil) {
			e = nil;
			id dl = [d newDynamicLibrary:lib error:&e];
			printf("%s newDynamicLibrary=%s installName=%s error=%s\n", tag, cls(dl),
			       dl ? [[dl installName] UTF8String] : "nil", err_text(e));
		}
	}
	NSError *e = nil;
	id<MTLLibrary> file = [d newLibraryWithURL:beside(@"oracle_bundle_dyn.metallib") error:&e];
	print_lib("libtype.file.dynamic", file, e);
}

// Where Metal's assertion at a dynamic library's exported function (TYPE 3) falls: the function
// itself, or its type.
static void case_functypes_dyn(id<MTLDevice> d)
{
	NSError *e = nil;
	id<MTLLibrary> lib = [d newLibraryWithURL:beside(@"oracle_bundle_dyn.metallib") error:&e];
	printf("functypes.dyn library=%s error=%s\n", cls(lib), err_text(e));
	fflush(stdout);
	id<MTLFunction> fn = [lib newFunctionWithName:@"_Z7div_dynf"];
	printf("functypes.dyn function=%s\n", cls(fn));
	fflush(stdout);
	printf("functypes.dyn functionType=%ld\n", fn ? (long)fn.functionType : -1L);
}

static MTLFunctionConstantValues *kernel_values(float scale)
{
	MTLFunctionConstantValues *cv = [MTLFunctionConstantValues new];
	bool flag = true;
	[cv setConstantValue:&flag type:MTLDataTypeBool atIndex:0];
	[cv setConstantValue:&scale type:MTLDataTypeFloat atIndex:1];
	return cv;
}

// newFunctionWithDescriptor: failing on a miss, by stage: each function added to an archive of its
// own with options 0 and with CompileToBinary, then looked up with FailOnBinaryArchiveMiss (4) and
// with it and CompileToBinary (5) in each; div_kernel also with other constant values.
static void case_failmiss2(id<MTLDevice> d)
{
	NSError *e = nil;
	id<MTLLibrary> lib = [d newLibraryWithURL:beside(@"oracle_bundle.metallib") error:&e];
	if (lib == nil) {
		printf("failmiss2 library error=%s\n", err_text(e));
		return;
	}
	NSString *names[] = {@"div_visible", @"div_kernel", @"div_vertex", @"div_fragment"};
	for (int n = 0; n < 4; n++) {
		id<MTLBinaryArchive> added[2];
		for (int a = 0; a < 2; a++) {
			added[a] = [d newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new] error:&e];
			MTLFunctionDescriptor *fd = [MTLFunctionDescriptor functionDescriptor];
			fd.name = names[n];
			fd.options = a ? MTLFunctionOptionCompileToBinary : MTLFunctionOptionNone;
			if (n == 1)
				fd.constantValues = kernel_values(2.0f);
			e = nil;
			BOOL ok = [added[a] addFunctionWithDescriptor:fd library:lib error:&e];
			printf("failmiss2 %s add options=%lu ok=%d error=%s\n", names[n].UTF8String,
			       (unsigned long)fd.options, ok, err_text(e));
			fflush(stdout);
		}
		const MTLFunctionOptions opts[] = {
		    MTLFunctionOptionFailOnBinaryArchiveMiss,
		    MTLFunctionOptionFailOnBinaryArchiveMiss | MTLFunctionOptionCompileToBinary};
		for (int o = 0; o < 2; o++)
			for (int a = 0; a < 2; a++)
				for (int k = 0; k < (n == 1 ? 2 : 1); k++) {
					MTLFunctionDescriptor *fd = [MTLFunctionDescriptor functionDescriptor];
					fd.name = names[n];
					fd.options = opts[o];
					fd.binaryArchives = @[ added[a] ];
					if (n == 1)
						fd.constantValues = kernel_values(k ? 3.0f : 2.0f);
					e = nil;
					id<MTLFunction> fn = [lib newFunctionWithDescriptor:fd error:&e];
					printf("failmiss2 %s options=%lu addedWith=%d values=%s function=%s error=%s\n",
					       names[n].UTF8String, (unsigned long)opts[o], a,
					       n == 1 ? (k ? "other" : "same") : "-", cls(fn), err_text(e));
					fflush(stdout);
				}
	}
}

// --- the rest -----------------------------------------------------------------------------

static void case_minlod(void)
{
	Class c = NSClassFromString(@"MTLTextureViewDescriptor");
	printf("minlod class=%s respondsToMinLOD=%d\n", c ? class_getName(c) : "nil",
	       c ? (int)[c instancesRespondToSelector:sel_registerName("minLOD")] : -1);
}

static void case_classes(id<MTLDevice> d)
{
	SEL sel = sel_registerName("bufferClass");
	if ([d respondsToSelector:sel]) {
		Class c = ((Class(*)(id, SEL))objc_msgSend)(d, sel);
		printf("classes bufferClass=%s\n", c ? class_getName(c) : "nil");
	} else
		printf("classes bufferClass=n/r\n");
	id<MTLBuffer> b = [d newBufferWithLength:64 options:MTLResourceStorageModeShared];
	printf("classes device=%s buffer=%s", cls(d), cls(b));
	for (Class c = class_getSuperclass(object_getClass(b)); c; c = class_getSuperclass(c))
		printf(" < %s", class_getName(c));
	printf("\n");
}

static void case_heapsize(id<MTLDevice> d)
{
	const MTLPixelFormat formats[] = {MTLPixelFormatDepth32Float, MTLPixelFormatDepth16Unorm,
	                                  MTLPixelFormatDepth32Float_Stencil8,
	                                  MTLPixelFormatBGRA8Unorm, MTLPixelFormatRGBA16Float};
	for (int f = 0; f < 5; f++)
		for (int opt = 1; opt >= 0; opt--) {
			MTLTextureDescriptor *td =
			    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:formats[f]
			                                                       width:1024
			                                                      height:1024
			                                                   mipmapped:NO];
			td.storageMode = MTLStorageModePrivate;
			td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
			td.allowGPUOptimizedContents = opt;
			MTLSizeAndAlign sa = [d heapTextureSizeAndAlignWithDescriptor:td];
			id<MTLTexture> t = [d newTextureWithDescriptor:td];
			printf("heapsize format=%lu allowGPUOptimizedContents=%d size=%lu align=%lu "
			       "allocatedSize=%lu\n",
			       (unsigned long)formats[f], opt, (unsigned long)sa.size,
			       (unsigned long)sa.align, (unsigned long)t.allocatedSize);
		}
}

static void case_libforms(id<MTLDevice> d)
{
	NSError *e = nil;
	id<MTLLibrary> lib = [d newLibraryWithURL:beside(@"oracle_bundle.metallib") error:&e];
	id<MTLFunction> visible = [lib newFunctionWithName:@"div_visible"];
	id<MTLFunction> stitchable = [lib newFunctionWithName:@"div_stitchable"];
	SEL arr = sel_registerName("newLibraryWithFunctionArray:error:");
	if ([d respondsToSelector:arr]) {
		// No empty array: Metal asserts at one ("Array must not be emty"), ending the process.
		NSArray *sets[] = {visible ? @[ visible ] : @[], stitchable ? @[ stitchable ] : @[]};
		const char *names[] = {"visible", "stitchable"};
		for (int i = 0; i < 2; i++) {
			if (sets[i].count == 0)
				continue;
			e = nil;
			@try {
				id l = ((id(*)(id, SEL, id, NSError **))objc_msgSend)(d, arr, sets[i], &e);
				printf("libforms functionArray(%s)=%s functions=%s error=%s\n", names[i], cls(l),
				       [[l functionNames] componentsJoinedByString:@","].UTF8String, err_text(e));
			} @catch (NSException *x) {
				printf("libforms functionArray(%s) threw %s %s\n", names[i], x.name.UTF8String,
				       x.reason.UTF8String);
			}
		}
	} else
		printf("libforms functionArray n/r\n");
	SEL pkg = sel_registerName("newLibraryWithMetalPackageURL:error:");
	if ([d respondsToSelector:pkg]) {
		NSURL *urls[] = {[NSURL fileURLWithPath:@"/nonexistent/div.mtlpackage"],
		                 beside(@"oracle_bundle.metallib")};
		for (int i = 0; i < 2; i++) {
			e = nil;
			@try {
				id l = ((id(*)(id, SEL, id, NSError **))objc_msgSend)(d, pkg, urls[i], &e);
				printf("libforms metalPackage(%s)=%s error=%s\n", urls[i].lastPathComponent.UTF8String,
				       cls(l), err_text(e));
			} @catch (NSException *x) {
				printf("libforms metalPackage(%s) threw %s %s\n",
				       urls[i].lastPathComponent.UTF8String, x.name.UTF8String,
				       x.reason.UTF8String);
			}
		}
	} else
		printf("libforms metalPackage n/r\n");
	SEL mps = sel_registerName("newLibraryWithMPSGraphPackageURL:name:error:");
	if ([d respondsToSelector:mps]) {
		e = nil;
		@try {
			id l = ((id(*)(id, SEL, id, id, NSError **))objc_msgSend)(
			    d, mps, [NSURL fileURLWithPath:@"/nonexistent/div.mpsgraphpackage"], @"div", &e);
			printf("libforms mpsGraphPackage=%s error=%s\n", cls(l), err_text(e));
		} @catch (NSException *x) {
			printf("libforms mpsGraphPackage threw %s %s\n", x.name.UTF8String,
			       x.reason.UTF8String);
		}
	} else
		printf("libforms mpsGraphPackage n/r\n");
}

// --- views and textures over buffers ------------------------------------------------------

static void view_case(id<MTLDevice> d, const char *tag, MTLPixelFormat parentFormat,
                      MTLTextureType parentType, NSUInteger parentLevels, NSUInteger parentSlices,
                      MTLPixelFormat format, MTLTextureType type, NSRange levels, NSRange slices)
{
	MTLTextureDescriptor *td = [MTLTextureDescriptor new];
	td.pixelFormat = parentFormat;
	td.textureType = parentType;
	td.width = 64;
	td.height = 64;
	td.mipmapLevelCount = parentLevels;
	td.arrayLength = parentSlices;
	td.storageMode = MTLStorageModePrivate;
	td.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
	id<MTLTexture> t = [d newTextureWithDescriptor:td];
	printf("%s parent=%s\n", tag, t ? "made" : "nil");
	fflush(stdout);
	id<MTLTexture> v = [t newTextureViewWithPixelFormat:format
	                                        textureType:type
	                                             levels:levels
	                                             slices:slices];
	printf("%s view=%s\n", tag, v ? "made" : "nil");
}

static void buffer_texture(id<MTLDevice> d, const char *tag, MTLPixelFormat format,
                           NSUInteger width, NSUInteger height, NSUInteger bytesPerRow, BOOL linear)
{
	id<MTLBuffer> b = [d newBufferWithLength:1 << 20 options:MTLResourceStorageModeShared];
	MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
	                                                                              width:width
	                                                                             height:height
	                                                                          mipmapped:NO];
	td.storageMode = MTLStorageModeShared;
	td.usage = MTLTextureUsageShaderRead;
	printf("%s format=%lu %lux%lu bytesPerRow=%lu linear=%d\n", tag, (unsigned long)format,
	       (unsigned long)width, (unsigned long)height, (unsigned long)bytesPerRow, (int)linear);
	fflush(stdout);
	id<MTLTexture> t = nil;
	if (!linear) {
		t = [b newTextureWithDescriptor:td offset:0 bytesPerRow:bytesPerRow];
	} else {
		SEL sel = sel_registerName("newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:");
		if (![(id)b respondsToSelector:sel]) {
			printf("%s n/r\n", tag);
			return;
		}
		t = ((id(*)(id, SEL, id, NSUInteger, NSUInteger, NSUInteger))objc_msgSend)(
		    b, sel, td, 0, bytesPerRow, bytesPerRow * height);
	}
	printf("%s texture=%s\n", tag, t ? "made" : "nil");
}

static void device_texture(id<MTLDevice> d, const char *tag, MTLPixelFormat format,
                           MTLStorageMode storage)
{
	MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
	                                                                              width:64
	                                                                             height:64
	                                                                          mipmapped:NO];
	td.storageMode = storage;
	td.usage = MTLTextureUsageRenderTarget;
	printf("%s format=%lu storage=%lu\n", tag, (unsigned long)format, (unsigned long)storage);
	fflush(stdout);
	id<MTLTexture> t = [d newTextureWithDescriptor:td];
	printf("%s texture=%s\n", tag, t ? "made" : "nil");
}


// --- texture addresses, read-only no-copy memory, linear alignment by raw format ----------

// A no-argument SPI getter's answer, read as 64 bits (an address or a count): `*ok` is 0 where
// the object does not answer the selector.
static uint64_t spi_u64(id o, const char *name, int *ok)
{
	SEL s = sel_registerName(name);
	if (o == nil || ![o respondsToSelector:s]) {
		*ok = 0;
		return 0;
	}
	*ok = 1;
	return ((uint64_t(*)(id, SEL))objc_msgSend)(o, s);
}

static void print_spi(id o, const char *name)
{
	int ok = 0;
	const uint64_t v = spi_u64(o, name, &ok);
	if (ok)
		printf(" %s=0x%llx", name, (unsigned long long)v);
	else
		printf(" %s=n/r", name);
}

static void describe_resource(const char *tag, const char *what, id<MTLResource> r)
{
	printf("%s %s class=%s", tag, what, cls(r));
	if (r != nil) {
		printf(" allocatedSize=%lu", (unsigned long)r.allocatedSize);
		print_spi(r, "gpuAddress");
		print_spi(r, "virtualAddress");
		print_spi(r, "resourceSize");
		print_spi(r, "allocationID");
		if ([r conformsToProtocol:@protocol(MTLTexture)])
			printf(" gpuResourceID=0x%llx",
			       (unsigned long long)((id<MTLTexture>)r).gpuResourceID._impl);
	}
	printf("\n");
	fflush(stdout);
}

static const MTLStorageMode kTexModes[] = {MTLStorageModePrivate, MTLStorageModeShared,
                                           MTLStorageModeManaged};
static const char *const kTexModeNames[] = {"private", "shared", "managed"};

// The shapes texaddr.* makes: type, format, width, height, depth, levels, slices, usage.
struct TexShape {
	const char *name;
	MTLTextureType type;
	MTLPixelFormat format;
	NSUInteger width, height, depth, levels, slices;
	MTLTextureUsage usage;
};
static const struct TexShape kTexShapes[] = {
    {"rgba8.64", MTLTextureType2D, MTLPixelFormatRGBA8Unorm, 64, 64, 1, 1, 1,
     MTLTextureUsageShaderRead},
    {"rgba8.256mips", MTLTextureType2D, MTLPixelFormatRGBA8Unorm, 256, 256, 1, 9, 1,
     MTLTextureUsageShaderRead},
    {"bgra8.1000x7", MTLTextureType2D, MTLPixelFormatBGRA8Unorm, 1000, 7, 1, 1, 1,
     MTLTextureUsageShaderRead},
    {"rgba8.64x3", MTLTextureType2DArray, MTLPixelFormatRGBA8Unorm, 64, 64, 1, 1, 3,
     MTLTextureUsageShaderRead},
    {"rgba8.3d", MTLTextureType3D, MTLPixelFormatRGBA8Unorm, 16, 16, 4, 1, 1,
     MTLTextureUsageShaderRead},
    {"r32f.rt", MTLTextureType2D, MTLPixelFormatR32Float, 64, 64, 1, 1, 1,
     MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget},
    {"rgba8.4096", MTLTextureType2D, MTLPixelFormatRGBA8Unorm, 4096, 4096, 1, 1, 1,
     MTLTextureUsageShaderRead},
};

static id<MTLTexture> make_texture(id<MTLDevice> d, const struct TexShape *s, MTLStorageMode mode)
{
	MTLTextureDescriptor *td = [MTLTextureDescriptor new];
	td.textureType = s->type;
	td.pixelFormat = s->format;
	td.width = s->width;
	td.height = s->height;
	td.depth = s->depth;
	td.mipmapLevelCount = s->levels;
	td.arrayLength = s->slices;
	td.storageMode = mode;
	td.usage = s->usage;
	return [d newTextureWithDescriptor:td];
}

// texaddr.MODE.SHAPE: the texture's address answers, and those of a Shared and a Private buffer
// made before and after it, and of a second texture of the shape made after it.
static void case_texaddr(id<MTLDevice> d, const char *tag, int mode, int shape)
{
	const struct TexShape *s = &kTexShapes[shape];
	printf("%s shape=%s storage=%s\n", tag, s->name, kTexModeNames[mode]);
	fflush(stdout);
	id<MTLBuffer> before = [d newBufferWithLength:65536 options:MTLResourceStorageModeShared];
	id<MTLBuffer> privBefore = [d newBufferWithLength:65536 options:MTLResourceStorageModePrivate];
	id<MTLTexture> t = make_texture(d, s, kTexModes[mode]);
	id<MTLTexture> t2 = make_texture(d, s, kTexModes[mode]);
	id<MTLBuffer> after = [d newBufferWithLength:65536 options:MTLResourceStorageModeShared];
	describe_resource(tag, "sharedBefore", before);
	describe_resource(tag, "privateBefore", privBefore);
	describe_resource(tag, "texture", t);
	describe_resource(tag, "second", t2);
	describe_resource(tag, "sharedAfter", after);
	int ok = 0, ok2 = 0;
	const uint64_t a1 = spi_u64(t, "gpuAddress", &ok), a2 = spi_u64(t, "gpuAddress", &ok2);
	if (ok)
		printf("%s gpuAddressStable=%d gpuAddressMod4K=%llu mod64K=%llu\n", tag, a1 == a2,
		       (unsigned long long)(a1 % 4096), (unsigned long long)(a1 % 65536));
}

// How far from `p` this process maps readable memory, in one or more regions.
static uint64_t mapped_from(const void *p, uint64_t want)
{
	mach_vm_address_t at = (mach_vm_address_t)(uintptr_t)p;
	const mach_vm_address_t end = at + want;
	while (at < end) {
		mach_vm_address_t base = at;
		mach_vm_size_t size = 0;
		vm_region_basic_info_data_64_t info;
		mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
		mach_port_t object = MACH_PORT_NULL;
		if (mach_vm_region(mach_task_self(), &base, &size, VM_REGION_BASIC_INFO_64,
		                   (vm_region_info_t)&info, &count, &object) != KERN_SUCCESS ||
		    base > at || !(info.protection & VM_PROT_READ))
			break;
		at = base + size;
	}
	return at > end ? want : at - (mach_vm_address_t)(uintptr_t)p;
}

static uint32_t texel_value(NSUInteger x, NSUInteger y, NSUInteger slice)
{
	return 0xA0000000u | ((uint32_t)slice << 20) | ((uint32_t)y << 10) | (uint32_t)x;
}

// texmem.MODE.SHAPE (RGBA8 shapes 0, 2 and 3): where each texel a replaceRegion: wrote lies in
// the memory at the texture's virtualAddress; whether getBytes: sees a write made there; and
// whether a blit's GPU write shows there (after synchronizeResource: for Managed storage).
static void case_texmem(id<MTLDevice> d, const char *tag, int mode, int shape)
{
	const struct TexShape *s = &kTexShapes[shape];
	printf("%s shape=%s storage=%s\n", tag, s->name, kTexModeNames[mode]);
	fflush(stdout);
	id<MTLTexture> t = make_texture(d, s, kTexModes[mode]);
	describe_resource(tag, "texture", t);
	if (t == nil)
		return;
	const NSUInteger w = s->width, h = s->height, slices = s->slices;
	uint32_t *px = malloc(w * h * 4);
	for (NSUInteger slice = 0; slice < slices; slice++) {
		for (NSUInteger y = 0; y < h; y++)
			for (NSUInteger x = 0; x < w; x++)
				px[y * w + x] = texel_value(x, y, slice);
		[t replaceRegion:MTLRegionMake2D(0, 0, w, h)
		     mipmapLevel:0
		           slice:slice
		       withBytes:px
		     bytesPerRow:w * 4
		   bytesPerImage:0];
	}
	free(px);
	int ok = 0;
	uint8_t *va = (uint8_t *)(uintptr_t)spi_u64(t, "virtualAddress", &ok);
	if (!ok || va == NULL) {
		printf("%s virtualAddress=%s\n", tag, ok ? "NULL" : "n/r");
		return;
	}
	int okr = 0;
	const uint64_t resourceSize = spi_u64(t, "resourceSize", &okr);
	const uint64_t want = t.allocatedSize > resourceSize ? t.allocatedSize : resourceSize;
	const uint64_t span = mapped_from(va, want);
	printf("%s virtualAddress mod4K=%llu mapped=%llu of %llu\n", tag,
	       (unsigned long long)((uintptr_t)va % 4096), (unsigned long long)span,
	       (unsigned long long)want);
	int64_t *at = malloc(w * h * slices * sizeof(int64_t));
	for (NSUInteger i = 0; i < w * h * slices; i++)
		at[i] = -1;
	uint64_t found = 0;
	for (uint64_t o = 0; o + 4 <= span; o += 4) {
		uint32_t v;
		memcpy(&v, va + o, 4);
		if ((v >> 28) != 0xA)
			continue;
		const NSUInteger x = v & 0x3ff, y = (v >> 10) & 0x3ff, slice = (v >> 20) & 0xff;
		if (x < w && y < h && slice < slices && at[(slice * h + y) * w + x] < 0) {
			at[(slice * h + y) * w + x] = (int64_t)o;
			found++;
		}
	}
	printf("%s found=%llu of %lu\n", tag, (unsigned long long)found,
	       (unsigned long)(w * h * slices));
	const NSUInteger probes[][3] = {{0, 0, 0},     {1, 0, 0},     {2, 0, 0},     {3, 0, 0},
	                                {4, 0, 0},     {8, 0, 0},     {16, 0, 0},    {32, 0, 0},
	                                {0, 1, 0},     {1, 1, 0},     {0, 2, 0},     {0, 4, 0},
	                                {0, 8, 0},     {w - 1, 0, 0}, {0, h - 1, 0}, {w - 1, h - 1, 0},
	                                {0, 0, 1},     {0, 0, 2}};
	for (size_t i = 0; i < sizeof probes / sizeof probes[0]; i++) {
		const NSUInteger x = probes[i][0], y = probes[i][1], slice = probes[i][2];
		if (slice >= slices)
			continue;
		printf("%s texel %lu,%lu,%lu at=%lld\n", tag, (unsigned long)x, (unsigned long)y,
		       (unsigned long)slice, (long long)at[(slice * h + y) * w + x]);
	}
	// Linear: every texel at base + slice * image + y * pitch + x * 4.
	const int64_t base = at[0];
	const int64_t pitch = h > 1 ? at[w] - base : 0;
	const int64_t image = slices > 1 ? at[h * w] - base : 0;
	uint64_t linear = 0;
	for (NSUInteger slice = 0; slice < slices; slice++)
		for (NSUInteger y = 0; y < h; y++)
			for (NSUInteger x = 0; x < w; x++)
				linear += at[(slice * h + y) * w + x] ==
				          base + (int64_t)slice * image + (int64_t)y * pitch + (int64_t)x * 4;
	printf("%s linear=%llu pitch=%lld image=%lld\n", tag, (unsigned long long)linear,
	       (long long)pitch, (long long)image);
	// A CPU write at texel 0,0's place, then getBytes: of it.
	if (base >= 0) {
		const uint32_t mark = 0x5A5A5A5Au;
		memcpy(va + base, &mark, 4);
		uint32_t got = 0;
		[t getBytes:&got bytesPerRow:4 fromRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0];
		printf("%s cpuWriteThenGetBytes=0x%08x\n", tag, got);
	}
	// A GPU write of texels 0..3 of row 0, then what the memory and getBytes: hold there.
	id<MTLBuffer> src = [d newBufferWithLength:16 options:MTLResourceStorageModeShared];
	uint32_t *sp = src.contents;
	for (int i = 0; i < 4; i++)
		sp[i] = 0xC0DE0000u | (uint32_t)i;
	id<MTLCommandQueue> q = [d newCommandQueue];
	id<MTLCommandBuffer> cb = [q commandBuffer];
	id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
	[be copyFromBuffer:src
	           sourceOffset:0
	      sourceBytesPerRow:16
	    sourceBytesPerImage:16
	             sourceSize:MTLSizeMake(4, 1, 1)
	              toTexture:t
	       destinationSlice:0
	       destinationLevel:0
	      destinationOrigin:MTLOriginMake(0, 0, 0)];
	if (kTexModes[mode] == MTLStorageModeManaged)
		[be synchronizeResource:t];
	[be endEncoding];
	[cb commit];
	[cb waitUntilCompleted];
	printf("%s gpuWrite status=%ld\n", tag, (long)cb.status);
	for (int i = 0; i < 4; i++) {
		uint32_t m = 0, g = 0;
		if (at[i] >= 0)
			memcpy(&m, va + at[i], 4);
		[t getBytes:&g bytesPerRow:4 fromRegion:MTLRegionMake2D(i, 0, 1, 1) mipmapLevel:0];
		printf("%s gpuWrite texel %d memory=0x%08x getBytes=0x%08x\n", tag, i, m, g);
	}
	free(at);
}

// texaddr.heap: textures in a Private heap of each type; texaddr.buffer: textures over a Shared
// and a Managed buffer; texaddr.view: a view of a Shared and a Private texture's level 1.
static void case_texaddr_heap(id<MTLDevice> d, const char *tag)
{
	const MTLHeapType types[] = {MTLHeapTypeAutomatic, MTLHeapTypePlacement};
	const char *names[] = {"automatic", "placement"};
	for (int i = 0; i < 2; i++) {
		MTLHeapDescriptor *hd = [MTLHeapDescriptor new];
		hd.size = 4u << 20;
		hd.storageMode = MTLStorageModePrivate;
		hd.type = types[i];
		id<MTLHeap> heap = [d newHeapWithDescriptor:hd];
		printf("%s %s heap=%s", tag, names[i], cls(heap));
		print_spi(heap, "gpuAddress");
		print_spi(heap, "virtualAddress");
		printf("\n");
		MTLTextureDescriptor *td =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
		                                                       width:64
		                                                      height:64
		                                                   mipmapped:NO];
		td.storageMode = MTLStorageModePrivate;
		td.usage = MTLTextureUsageShaderRead;
		for (int k = 0; k < 2; k++) {
			id<MTLTexture> t = types[i] == MTLHeapTypePlacement
			                       ? [heap newTextureWithDescriptor:td offset:65536 * (k + 1)]
			                       : [heap newTextureWithDescriptor:td];
			printf("%s %s texture %d heapOffset=%lu\n", tag, names[i], k,
			       (unsigned long)t.heapOffset);
			describe_resource(tag, names[i], t);
		}
	}
}

static void case_texaddr_buffer(id<MTLDevice> d, const char *tag)
{
	const MTLResourceOptions opts[] = {MTLResourceStorageModeShared, MTLResourceStorageModeManaged};
	const char *names[] = {"shared", "managed"};
	for (int i = 0; i < 2; i++) {
		id<MTLBuffer> b = [d newBufferWithLength:1 << 20 options:opts[i]];
		printf("%s %s buffer contents=%p", tag, names[i], b.contents);
		print_spi(b, "gpuAddress");
		print_spi(b, "virtualAddress");
		printf("\n");
		MTLTextureDescriptor *td =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
		                                                       width:64
		                                                      height:64
		                                                   mipmapped:NO];
		td.storageMode = (MTLStorageMode)(opts[i] >> MTLResourceStorageModeShift);
		td.usage = MTLTextureUsageShaderRead;
		id<MTLTexture> t = [b newTextureWithDescriptor:td offset:4096 bytesPerRow:256];
		describe_resource(tag, names[i], t);
	}
}

static void case_texaddr_view(id<MTLDevice> d, const char *tag)
{
	for (int m = 0; m < 2; m++) {
		MTLTextureDescriptor *td =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
		                                                       width:64
		                                                      height:64
		                                                   mipmapped:YES];
		td.storageMode = m == 0 ? MTLStorageModeShared : MTLStorageModePrivate;
		td.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
		id<MTLTexture> t = [d newTextureWithDescriptor:td];
		id<MTLTexture> v = [t newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm
		                                        textureType:MTLTextureType2D
		                                             levels:NSMakeRange(1, 1)
		                                             slices:NSMakeRange(0, 1)];
		describe_resource(tag, m == 0 ? "shared.parent" : "private.parent", t);
		describe_resource(tag, m == 0 ? "shared.view" : "private.view", v);
	}
}

static void case_texaddr_iosurface(id<MTLDevice> d, const char *tag)
{
	NSDictionary *props = @{
		(id)kIOSurfaceWidth : @64,
		(id)kIOSurfaceHeight : @64,
		(id)kIOSurfaceBytesPerElement : @4,
		(id)kIOSurfacePixelFormat : @((uint32_t)'BGRA'),
	};
	IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)props);
	MTLTextureDescriptor *td =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:64
	                                                      height:64
	                                                   mipmapped:NO];
	td.storageMode = MTLStorageModeManaged;
	td.usage = MTLTextureUsageShaderRead;
	id<MTLTexture> t = [d newTextureWithDescriptor:td iosurface:surface plane:0];
	printf("%s surface base=%p\n", tag, IOSurfaceGetBaseAddress(surface));
	describe_resource(tag, "texture", t);
	CFRelease(surface);
}

// nocopy.copies: a no-copy Managed buffer over 4 writable pages of 0x11: a CPU write of 0x22 to
// byte 0 with no didModifyRange:, then a blit copy of the buffer out (what the GPU read); a blit
// fill of 0x77 into bytes 256 to 511, waited on, with no synchronize (what the memory holds), then
// with one.
static void case_nocopy_copies(id<MTLDevice> d, const char *tag)
{
	const size_t page = (size_t)getpagesize(), len = 4 * page;
	uint8_t *p = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
	memset(p, 0x11, len);
	id<MTLBuffer> b = [d newBufferWithBytesNoCopy:p
	                                       length:len
	                                      options:MTLResourceStorageModeManaged
	                                  deallocator:nil];
	describe_resource(tag, "buffer", b);
	if (b == nil)
		return;
	id<MTLCommandQueue> q = [d newCommandQueue];
	id<MTLBuffer> dst = [d newBufferWithLength:len options:MTLResourceStorageModeShared];
	p[0] = 0x22;
	id<MTLCommandBuffer> cb = [q commandBuffer];
	id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
	[be copyFromBuffer:b sourceOffset:0 toBuffer:dst destinationOffset:0 size:len];
	[be endEncoding];
	[cb commit];
	[cb waitUntilCompleted];
	printf("%s cpuWriteNoCall gpuRead byte0=0x%02x byte1=0x%02x\n", tag,
	       ((uint8_t *)dst.contents)[0], ((uint8_t *)dst.contents)[1]);
	for (int sync = 0; sync < 2; sync++) {
		cb = [q commandBuffer];
		be = [cb blitCommandEncoder];
		[be fillBuffer:b range:NSMakeRange(256, 256) value:(uint8_t)(0x77 + sync)];
		if (sync)
			[be synchronizeResource:b];
		[be endEncoding];
		[cb commit];
		[cb waitUntilCompleted];
		printf("%s gpuFill synchronize=%d memory=0x%02x\n", tag, sync, p[256]);
	}
}

// nocopy.ro.KIND.MODE: a no-copy buffer over 4 pages the process maps read-only, anonymous
// memory (`anon`, written, then mprotect'd) or a file mapped shared and read-only (`file`): the
// buffer; a blit read of it; a blit fill and a kernel's store into it, each command buffer's
// status and error; what the memory, and for the file a fresh read of it, hold after.
static void case_nocopy_ro(id<MTLDevice> d, const char *tag, int file, MTLResourceOptions opts)
{
	const size_t page = (size_t)getpagesize(), len = 4 * page;
	uint8_t *p = NULL;
	char path[] = "/tmp/oracle_ro_XXXXXX";
	int fd = -1;
	if (file) {
		fd = mkstemp(path);
		uint8_t *bytes = malloc(len);
		memset(bytes, 0x11, len);
		if (fd < 0 || write(fd, bytes, len) != (ssize_t)len) {
			printf("%s file=failed\n", tag);
			return;
		}
		free(bytes);
		p = mmap(NULL, len, PROT_READ, MAP_SHARED, fd, 0);
	} else {
		p = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
		memset(p, 0x11, len);
		mprotect(p, len, PROT_READ);
	}
	printf("%s mapped=%s\n", tag, p == MAP_FAILED ? "failed" : "ok");
	fflush(stdout);
	if (p == MAP_FAILED)
		return;
	id<MTLBuffer> b = [d newBufferWithBytesNoCopy:p length:len options:opts deallocator:nil];
	describe_resource(tag, "buffer", b);
	if (b != nil) {
		id<MTLCommandQueue> q = [d newCommandQueue];
		id<MTLBuffer> dst = [d newBufferWithLength:len options:MTLResourceStorageModeShared];
		id<MTLCommandBuffer> cb = [q commandBuffer];
		id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
		[be copyFromBuffer:b sourceOffset:0 toBuffer:dst destinationOffset:0 size:len];
		[be endEncoding];
		[cb commit];
		[cb waitUntilCompleted];
		printf("%s gpuRead status=%ld error=%s first=0x%02x\n", tag, (long)cb.status,
		       err_text(cb.error), ((uint8_t *)dst.contents)[0]);
		fflush(stdout);
		cb = [q commandBuffer];
		be = [cb blitCommandEncoder];
		[be fillBuffer:b range:NSMakeRange(0, 256) value:0x77];
		if ((opts & MTLResourceStorageModeMask) == MTLResourceStorageModeManaged)
			[be synchronizeResource:b];
		[be endEncoding];
		[cb commit];
		[cb waitUntilCompleted];
		printf("%s gpuFill status=%ld error=%s memory=0x%02x\n", tag, (long)cb.status,
		       err_text(cb.error), p[0]);
		fflush(stdout);
		NSError *e = nil;
		id<MTLLibrary> lib = [d newLibraryWithURL:beside(@"oracle_bundle.metallib") error:&e];
		MTLFunctionConstantValues *cv = [MTLFunctionConstantValues new];
		bool flag = false;
		float scale = 1.0f;
		[cv setConstantValue:&flag type:MTLDataTypeBool atIndex:0];
		[cv setConstantValue:&scale type:MTLDataTypeFloat atIndex:1];
		id<MTLFunction> fn = [lib newFunctionWithName:@"div_kernel" constantValues:cv error:&e];
		id<MTLComputePipelineState> ps =
		    fn ? [d newComputePipelineStateWithFunction:fn error:&e] : nil;
		if (ps != nil) {
			cb = [q commandBuffer];
			id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
			[ce setComputePipelineState:ps];
			[ce setBuffer:b offset:512 atIndex:0];
			[ce dispatchThreads:MTLSizeMake(64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
			[ce endEncoding];
			if ((opts & MTLResourceStorageModeMask) == MTLResourceStorageModeManaged) {
				be = [cb blitCommandEncoder];
				[be synchronizeResource:b];
				[be endEncoding];
			}
			[cb commit];
			[cb waitUntilCompleted];
			uint32_t v = 0;
			memcpy(&v, p + 512, 4);
			printf("%s kernelStore status=%ld error=%s memory=0x%08x\n", tag, (long)cb.status,
			       err_text(cb.error), v);
		} else
			printf("%s kernelStore pipeline=nil error=%s\n", tag, err_text(e));
	}
	if (file) {
		uint8_t first[516] = {0};
		int again = open(path, O_RDONLY);
		const ssize_t n = again >= 0 ? pread(again, first, sizeof first, 0) : -1;
		printf("%s fileAfter read=%zd byte0=0x%02x byte512=0x%02x\n", tag, n, first[0],
		       first[512]);
		if (again >= 0)
			close(again);
		close(fd);
		unlink(path);
	}
}

// --- argument buffers (argbuf.*) ----------------------------------------------------------

static id<MTLDepthStencilState> probe_ds(id<MTLDevice> d)
{
	MTLDepthStencilDescriptor *dd = [MTLDepthStencilDescriptor new];
	dd.depthCompareFunction = MTLCompareFunctionLess;
	dd.depthWriteEnabled = YES;
	return [d newDepthStencilStateWithDescriptor:dd];
}

static void dump_bytes(const char *tag, const char *what, const void *p, size_t n)
{
	const uint8_t *b = p;
	printf("%s %s", tag, what);
	for (size_t k = 0; k < n; k++)
		printf("%s%02x", k % 8 ? "" : " ", b[k]);
	printf("\n");
}

static MTLArgumentDescriptor *arg_desc(MTLDataType type, NSUInteger index)
{
	MTLArgumentDescriptor *a = [MTLArgumentDescriptor argumentDescriptor];
	a.dataType = type;
	a.index = index;
	if (type == MTLDataTypeTexture)
		a.textureType = MTLTextureType2D;
	return a;
}

static void case_argbuf_ds_id(id<MTLDevice> d)
{
	printf("argbuf.ds.id argumentBuffersSupport=%ld\n", (long)d.argumentBuffersSupport);
	id<MTLDepthStencilState> ds = probe_ds(d);
	printf("argbuf.ds.id state=%s respondsToGpuResourceID=%d\n", cls(ds),
	       (int)[(id)ds respondsToSelector:@selector(gpuResourceID)]);
	fflush(stdout);
	if ([(id)ds respondsToSelector:@selector(gpuResourceID)]) {
		MTLResourceID rid = ds.gpuResourceID;
		printf("argbuf.ds.id gpuResourceID=0x%llx\n", (unsigned long long)rid._impl);
		id<MTLDepthStencilState> ds2 = probe_ds(d);
		printf("argbuf.ds.id second=0x%llx same=%d\n",
		       (unsigned long long)ds2.gpuResourceID._impl, ds2 == ds);
	}
}

// An encoder of `descs`, and the bytes setting `ds` at `dsIndex` writes into a buffer of 0xa5.
static void argbuf_ds_encoder(id<MTLDevice> d, const char *tag, NSArray<MTLArgumentDescriptor *> *descs,
                              NSUInteger dsIndex, BOOL set)
{
	id<MTLArgumentEncoder> enc = [d newArgumentEncoderWithArguments:descs];
	printf("%s encoder=%s encodedLength=%lu alignment=%lu\n", tag, cls(enc),
	       (unsigned long)enc.encodedLength, (unsigned long)enc.alignment);
	fflush(stdout);
	if (enc == nil || !set)
		return;
	id<MTLBuffer> b = [d newBufferWithLength:256 options:MTLResourceStorageModeShared];
	memset(b.contents, 0xa5, 256);
	[enc setArgumentBuffer:b offset:0];
	id<MTLDepthStencilState> ds = probe_ds(d);
	printf("%s state gpuResourceID=0x%llx\n", tag,
	       [(id)ds respondsToSelector:@selector(gpuResourceID)]
	           ? (unsigned long long)ds.gpuResourceID._impl
	           : 0ull);
	fflush(stdout);
	[enc setDepthStencilState:ds atIndex:dsIndex];
	dump_bytes(tag, "after set", b.contents, MIN((size_t)64, (size_t)enc.encodedLength + 16));
	fflush(stdout);
	[enc setDepthStencilState:nil atIndex:dsIndex];
	dump_bytes(tag, "after nil", b.contents, MIN((size_t)64, (size_t)enc.encodedLength + 16));
}

static void case_argbuf_ds_function(id<MTLDevice> d)
{
	NSError *e = nil;
	id<MTLLibrary> lib = [d newLibraryWithURL:beside(@"oracle_bundle_ds.metallib") error:&e];
	printf("argbuf.ds.function library=%s error=%s\n", cls(lib), err_text(e));
	fflush(stdout);
	if (lib == nil)
		return;
	id<MTLFunction> fn = [lib newFunctionWithName:@"div_ds"];
	printf("argbuf.ds.function function=%s\n", cls(fn));
	fflush(stdout);
	id<MTLArgumentEncoder> enc = [fn newArgumentEncoderWithBufferIndex:0];
	printf("argbuf.ds.function encoder=%s encodedLength=%lu alignment=%lu\n", cls(enc),
	       (unsigned long)enc.encodedLength, (unsigned long)enc.alignment);
	fflush(stdout);
	e = nil;
	id p = [d newComputePipelineStateWithFunction:fn error:&e];
	printf("argbuf.ds.function pipeline=%s error=%s\n", p ? "made" : "nil", err_text(e));
	fflush(stdout);
	if (enc == nil)
		return;
	id<MTLBuffer> b = [d newBufferWithLength:256 options:MTLResourceStorageModeShared];
	memset(b.contents, 0xa5, 256);
	[enc setArgumentBuffer:b offset:0];
	[enc setDepthStencilState:probe_ds(d) atIndex:0];
	dump_bytes("argbuf.ds.function", "after set", b.contents, 32);
}

static NSString *type_text(MTLType *t, int depth);

static NSString *struct_text(MTLStructType *s, int depth)
{
	NSMutableArray<NSString *> *parts = [NSMutableArray array];
	for (MTLStructMember *m in s.members)
		[parts addObject:[NSString stringWithFormat:@"%@@%lu:%lu#%lu%@", m.name,
		                                            (unsigned long)m.offset,
		                                            (unsigned long)m.dataType,
		                                            (unsigned long)m.argumentIndex,
		                                            depth < 3 ? type_text((MTLType *)m.arrayType ?: (MTLType *)m.structType ?: (MTLType *)m.textureReferenceType ?: (MTLType *)m.pointerType, depth + 1) : @""]];
	return [NSString stringWithFormat:@"{%@}", [parts componentsJoinedByString:@","]];
}

static NSString *type_text(MTLType *t, int depth)
{
	if (t == nil)
		return @"";
	if ([t isKindOfClass:[MTLStructType class]])
		return struct_text((MTLStructType *)t, depth);
	if ([t isKindOfClass:[MTLArrayType class]]) {
		MTLArrayType *a = (MTLArrayType *)t;
		return [NSString stringWithFormat:@"[%lu x %lu stride %lu argStride %lu%@]",
		                                  (unsigned long)a.arrayLength,
		                                  (unsigned long)a.elementType,
		                                  (unsigned long)a.stride,
		                                  (unsigned long)a.argumentIndexStride,
		                                  type_text((MTLType *)a.elementStructType ?: (MTLType *)a.elementTextureReferenceType ?: (MTLType *)a.elementPointerType, depth + 1)];
	}
	if ([t isKindOfClass:[MTLPointerType class]]) {
		MTLPointerType *p = (MTLPointerType *)t;
		return [NSString stringWithFormat:@"*(%lu align %lu size %lu access %lu argbuf %d%@)",
		                                  (unsigned long)p.elementType,
		                                  (unsigned long)p.alignment,
		                                  (unsigned long)p.dataSize, (unsigned long)p.access,
		                                  (int)p.elementIsArgumentBuffer,
		                                  type_text((MTLType *)p.elementStructType ?: (MTLType *)p.elementArrayType, depth + 1)];
	}
	if ([t isKindOfClass:[MTLTextureReferenceType class]]) {
		MTLTextureReferenceType *r = (MTLTextureReferenceType *)t;
		return [NSString stringWithFormat:@"tex(%lu %lu %lu depth %d)",
		                                  (unsigned long)r.textureDataType,
		                                  (unsigned long)r.textureType, (unsigned long)r.access,
		                                  (int)r.isDepthTexture];
	}
	return [NSString stringWithFormat:@"type(%lu)", (unsigned long)t.dataType];
}

static void print_binding(const char *tag, id<MTLBinding> b)
{
	printf("%s binding class=%s name=%s type=%ld index=%lu access=%ld used=%d argument=%d "
	       "isMTLArgument=%d",
	       tag, cls(b), b.name.UTF8String, (long)b.type, (unsigned long)b.index, (long)b.access,
	       (int)b.used, (int)b.argument, (int)[(id)b isKindOfClass:[MTLArgument class]]);
	if (b.type == MTLBindingTypeBuffer && [(id)b conformsToProtocol:@protocol(MTLBufferBinding)]) {
		id<MTLBufferBinding> bb = (id<MTLBufferBinding>)b;
		printf(" bufferDataType=%lu alignment=%lu dataSize=%lu struct=%s pointer=%s",
		       (unsigned long)bb.bufferDataType, (unsigned long)bb.bufferAlignment,
		       (unsigned long)bb.bufferDataSize, struct_text(bb.bufferStructType, 0).UTF8String,
		       type_text(bb.bufferPointerType, 0).UTF8String);
	}
	printf("\n");
}

static void print_argument(const char *tag, MTLArgument *a)
{
	if (a == nil) {
		printf("%s reflection=nil\n", tag);
		return;
	}
	printf("%s reflection class=%s conformsBufferBinding=%d name=%s type=%ld index=%lu "
	       "access=%ld active=%d bufferDataType=%lu alignment=%lu dataSize=%lu arrayLength=%lu "
	       "struct=%s pointer=%s\n",
	       tag, cls(a), (int)[a conformsToProtocol:@protocol(MTLBufferBinding)], a.name.UTF8String,
	       (long)a.type, (unsigned long)a.index, (long)a.access, (int)a.active,
	       (unsigned long)a.bufferDataType, (unsigned long)a.bufferAlignment,
	       (unsigned long)a.bufferDataSize, (unsigned long)a.arrayLength,
	       struct_text(a.bufferStructType, 0).UTF8String,
	       type_text(a.bufferPointerType, 0).UTF8String);
}

// Fills an argument buffer through `enc` with the same objects at each of div_argbuf's indices.
static NSData *fill_args(id<MTLDevice> d, id<MTLArgumentEncoder> enc, NSArray *objects)
{
	id<MTLBuffer> b = [d newBufferWithLength:4096 options:MTLResourceStorageModeShared];
	memset(b.contents, 0xa5, 4096);
	[enc setArgumentBuffer:b offset:0];
	[enc setTexture:objects[0] atIndex:0];
	[enc setBuffer:objects[1] offset:16 atIndex:1];
	[enc setSamplerState:objects[2] atIndex:2];
	float *scale = [enc constantDataAtIndex:3];
	if (scale)
		*scale = 3.5f;
	for (NSUInteger i = 0; i < 3; i++)
		[enc setTexture:objects[0] atIndex:4 + i];
	return [NSData dataWithBytes:b.contents length:MIN((NSUInteger)4096, enc.encodedLength + 8)];
}

static NSArray *args_objects(id<MTLDevice> d)
{
	MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                                                                              width:4
	                                                                             height:4
	                                                                          mipmapped:NO];
	td.storageMode = MTLStorageModePrivate;
	id<MTLTexture> t = [d newTextureWithDescriptor:td];
	id<MTLBuffer> buf = [d newBufferWithLength:256 options:MTLResourceStorageModeShared];
	MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
	sd.supportArgumentBuffers = YES;
	id<MTLSamplerState> s = [d newSamplerStateWithDescriptor:sd];
	return @[ t, buf, s ];
}

static id<MTLFunction> args_function(id<MTLDevice> d, id<MTLLibrary> *libOut)
{
	NSError *e = nil;
	id<MTLLibrary> lib = [d newLibraryWithURL:beside(@"oracle_bundle_args.metallib") error:&e];
	if (lib == nil)
		printf("argbuf library error=%s\n", err_text(e));
	if (libOut)
		*libOut = lib;
	return [lib newFunctionWithName:@"div_argbuf"];
}

// The pipeline's reflection's bindings of div_argbuf, and an encoder made from each buffer binding,
// compared with the function's own encoder.
static void case_argbuf_binding(id<MTLDevice> d, BOOL plain, BOOL fromLibrary)
{
	const char *tag = fromLibrary ? "argbuf.binding.lib" : plain ? "argbuf.binding.plain" : "argbuf.binding";
	id<MTLLibrary> lib = nil;
	id<MTLFunction> fn = args_function(d, &lib);
	if (fn == nil)
		return;
	NSArray<id<MTLBinding>> *bindings = nil;
	if (fromLibrary) {
		SEL sel = sel_registerName("reflectionForFunctionWithName:");
		if (![(id)lib respondsToSelector:sel]) {
			printf("%s library reflection n/r\n", tag);
			return;
		}
		MTLFunctionReflection *r = ((id(*)(id, SEL, id))objc_msgSend)(lib, sel, @"div_argbuf");
		printf("%s functionReflection=%s\n", tag, cls(r));
		bindings = r.bindings;
	} else {
		MTLComputePipelineReflection *refl = nil;
		NSError *e = nil;
		id p = [d newComputePipelineStateWithFunction:fn
		                                      options:MTLPipelineOptionBindingInfo |
		                                              MTLPipelineOptionBufferTypeInfo
		                                   reflection:&refl
		                                        error:&e];
		printf("%s pipeline=%s reflection=%s error=%s\n", tag, p ? "made" : "nil", cls(refl),
		       err_text(e));
		bindings = refl.bindings;
	}
	for (id<MTLBinding> b in bindings)
		print_binding(tag, b);
	fflush(stdout);
	id<MTLArgumentEncoder> own = [fn newArgumentEncoderWithBufferIndex:0];
	printf("%s functionEncoder=%s encodedLength=%lu alignment=%lu\n", tag, cls(own),
	       (unsigned long)own.encodedLength, (unsigned long)own.alignment);
	NSArray *objects = args_objects(d);
	NSData *ownBytes = own ? fill_args(d, own, objects) : nil;
	for (id<MTLBinding> b in bindings) {
		if (b.type != MTLBindingTypeBuffer || (plain ? b.index != 1 : b.index != 0))
			continue;
		printf("%s newArgumentEncoderWithBufferBinding: %s\n", tag, b.name.UTF8String);
		fflush(stdout);
		id<MTLArgumentEncoder> enc = [d newArgumentEncoderWithBufferBinding:(id<MTLBufferBinding>)b];
		printf("%s bindingEncoder=%s encodedLength=%lu alignment=%lu\n", tag, cls(enc),
		       (unsigned long)enc.encodedLength, (unsigned long)enc.alignment);
		fflush(stdout);
		if (enc == nil || own == nil)
			continue;
		NSData *bytes = fill_args(d, enc, objects);
		printf("%s sameBytes=%d\n", tag, (int)[bytes isEqualToData:ownBytes]);
		dump_bytes(tag, "function", ownBytes.bytes, MIN((size_t)96, ownBytes.length));
		dump_bytes(tag, "binding", bytes.bytes, MIN((size_t)96, bytes.length));
		id<MTLArgumentEncoder> nested = [enc newArgumentEncoderForBufferAtIndex:1];
		printf("%s nestedAt1=%s\n", tag, cls(nested));
	}
}

static void case_argbuf_reflection(id<MTLDevice> d)
{
	id<MTLFunction> fn = args_function(d, NULL);
	if (fn == nil)
		return;
	const NSUInteger indices[] = {0, 1, 5};
	for (int i = 0; i < 3; i++) {
		char tag[64];
		snprintf(tag, sizeof tag, "argbuf.reflection.%lu", (unsigned long)indices[i]);
		MTLAutoreleasedArgument arg = nil;
		id<MTLArgumentEncoder> enc = [fn newArgumentEncoderWithBufferIndex:indices[i]
		                                                         reflection:&arg];
		printf("%s encoder=%s encodedLength=%lu\n", tag, cls(enc), (unsigned long)enc.encodedLength);
		print_argument(tag, arg);
		fflush(stdout);
	}
	MTLAutoreleasedArgument a1 = nil, a2 = nil;
	id<MTLArgumentEncoder> e1 = [fn newArgumentEncoderWithBufferIndex:0 reflection:&a1];
	id<MTLArgumentEncoder> e2 = [fn newArgumentEncoderWithBufferIndex:0 reflection:&a2];
	printf("argbuf.reflection sameObject=%d isEqual=%d encoders=%s,%s\n", a1 == a2,
	       (int)[a1 isEqual:a2], cls(e1), cls(e2));
}

// --- IOSurfaces (iosurf.*) -----------------------------------------------------------------

static IOSurfaceRef probe_surface(uint32_t fourcc, size_t w, size_t h, size_t bpe)
{
	NSDictionary *p = @{
		(id)kIOSurfaceWidth : @(w),
		(id)kIOSurfaceHeight : @(h),
		(id)kIOSurfaceBytesPerElement : @(bpe),
		(id)kIOSurfacePixelFormat : @(fourcc),
	};
	return IOSurfaceCreate((__bridge CFDictionaryRef)p);
}

static NSString *dict_text(NSDictionary *dict)
{
	NSMutableArray<NSString *> *parts = [NSMutableArray array];
	for (id k in [dict.allKeys sortedArrayUsingSelector:@selector(compare:)])
		[parts addObject:[NSString stringWithFormat:@"%@=%@", k, dict[k]]];
	return [NSString stringWithFormat:@"%@{%@}", NSStringFromClass([dict class]),
	                                  [parts componentsJoinedByString:@","]];
}

static void surface_props(id<MTLDevice> d, const char *tag, MTLPixelFormat format, NSUInteger w,
                          NSUInteger h, MTLTextureUsage usage, BOOL mips)
{
	SEL sel = sel_registerName("copyIOSurfaceSharedTextureProperties:");
	MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
	                                                                              width:w
	                                                                             height:h
	                                                                          mipmapped:mips];
	td.usage = usage;
	td.storageMode = MTLStorageModeShared;
	NSDictionary *dict = nil;
	@try {
		dict = ((id(*)(id, SEL, id))objc_msgSend)(d, sel, td);
		printf("%s format=%lu %lux%lu usage=%lu mips=%d props=%s\n", tag, (unsigned long)format,
		       (unsigned long)w, (unsigned long)h, (unsigned long)usage, (int)mips,
		       dict ? dict_text(dict).UTF8String : "nil");
	} @catch (NSException *x) {
		printf("%s threw %s %s\n", tag, x.name.UTF8String, x.reason.UTF8String);
		return;
	}
	if (dict == nil)
		return;
	IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef)dict);
	printf("%s surfaceFromProps=%s", tag, s ? "made" : "nil");
	if (s) {
		printf(" alloc=%zu bytesPerRow=%zu width=%zu height=%zu", IOSurfaceGetAllocSize(s),
		       IOSurfaceGetBytesPerRow(s), IOSurfaceGetWidth(s), IOSurfaceGetHeight(s));
		td.storageMode = MTLStorageModeShared;
		id<MTLTexture> t = [d newTextureWithDescriptor:td iosurface:s plane:0];
		printf(" texture=%s", t ? "made" : "nil");
		CFRelease(s);
	}
	printf("\n");
}

static void case_iosurf_props(id<MTLDevice> d)
{
	SEL sel = sel_registerName("copyIOSurfaceSharedTextureProperties:");
	printf("iosurf.props responds=%d\n", (int)[d respondsToSelector:sel]);
	fflush(stdout);
	if (![d respondsToSelector:sel])
		return;
	const MTLTextureUsage rr = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
	surface_props(d, "iosurf.props.bgra64", MTLPixelFormatBGRA8Unorm, 64, 64, rr, NO);
	surface_props(d, "iosurf.props.bgra64.write", MTLPixelFormatBGRA8Unorm, 64, 64,
	              MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite, NO);
	surface_props(d, "iosurf.props.rgba16f", MTLPixelFormatRGBA16Float, 256, 256, rr, NO);
	surface_props(d, "iosurf.props.r8", MTLPixelFormatR8Unorm, 17, 9, rr, NO);
	surface_props(d, "iosurf.props.bgra1080", MTLPixelFormatBGRA8Unorm, 1920, 1080, rr, NO);
	surface_props(d, "iosurf.props.rgb10a2", MTLPixelFormatRGB10A2Unorm, 100, 30,
	              MTLTextureUsageShaderRead, NO);
	fflush(stdout);
	surface_props(d, "iosurf.props.mips", MTLPixelFormatBGRA8Unorm, 64, 64, rr, YES);
}

static void surface_buffer(id<MTLDevice> d, const char *tag, IOSurfaceRef s)
{
	SEL sel = sel_registerName("newBufferWithIOSurface:");
	printf("%s surface alloc=%zu bytesPerRow=%zu planes=%zu base=%p\n", tag,
	       IOSurfaceGetAllocSize(s), IOSurfaceGetBytesPerRow(s), IOSurfaceGetPlaneCount(s),
	       IOSurfaceGetBaseAddress(s));
	fflush(stdout);
	id<MTLBuffer> b = nil;
	@try {
		b = ((id(*)(id, SEL, IOSurfaceRef))objc_msgSend)(d, sel, s);
	} @catch (NSException *x) {
		printf("%s threw %s %s\n", tag, x.name.UTF8String, x.reason.UTF8String);
		return;
	}
	printf("%s buffer=%s", tag, cls(b));
	if (b == nil) {
		printf("\n");
		return;
	}
	SEL iosel = sel_registerName("iosurface");
	IOSurfaceRef back = [(id)b respondsToSelector:iosel]
	                        ? ((IOSurfaceRef(*)(id, SEL))objc_msgSend)(b, iosel)
	                        : NULL;
	printf(" length=%lu allocatedSize=%lu options=0x%lx storage=%lu contents=%p "
	       "contentsIsBase=%d iosurfaceIsSurface=%d gpuAddress=%d\n",
	       (unsigned long)b.length, (unsigned long)b.allocatedSize,
	       (unsigned long)b.resourceOptions, (unsigned long)b.storageMode, b.contents,
	       b.contents == IOSurfaceGetBaseAddress(s), back == s, b.gpuAddress != 0);
	fflush(stdout);
	// CPU writes through the surface, read through the buffer, and the reverse.
	IOSurfaceLock(s, 0, NULL);
	uint8_t *base = IOSurfaceGetBaseAddress(s);
	base[0] = 0x11;
	base[IOSurfaceGetAllocSize(s) - 1] = 0x22;
	IOSurfaceUnlock(s, 0, NULL);
	const uint8_t *c = b.contents;
	if (c)
		printf("%s cpu surface->buffer first=%02x last=%02x\n", tag, c[0],
		       c[MIN(b.length, IOSurfaceGetAllocSize(s)) - 1]);
	// The GPU fills the buffer's first 256 bytes; the surface's pages then hold them?
	id<MTLCommandQueue> q = [d newCommandQueue];
	id<MTLCommandBuffer> cb = [q commandBuffer];
	id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
	[bl fillBuffer:b range:NSMakeRange(0, 256) value:0x77];
	[bl endEncoding];
	[cb commit];
	[cb waitUntilCompleted];
	IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
	printf("%s gpu buffer->surface byte0=%02x byte255=%02x byte256=%02x status=%ld\n", tag,
	       base[0], base[255], base[256], (long)cb.status);
	IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
}

static void case_iosurf_buffer(id<MTLDevice> d, int which)
{
	SEL sel = sel_registerName("newBufferWithIOSurface:");
	printf("iosurf.buffer responds=%d\n", (int)[d respondsToSelector:sel]);
	fflush(stdout);
	if (![d respondsToSelector:sel])
		return;
	if (which == 0) {
		IOSurfaceRef s = probe_surface('BGRA', 64, 64, 4);
		surface_buffer(d, "iosurf.buffer.bgra64", s);
		CFRelease(s);
		s = probe_surface('BGRA', 100, 7, 4);
		surface_buffer(d, "iosurf.buffer.bgra100x7", s);
		CFRelease(s);
	} else {
		NSDictionary *plane0 = @{
			(id)kIOSurfacePlaneWidth : @64,
			(id)kIOSurfacePlaneHeight : @64,
			(id)kIOSurfacePlaneBytesPerElement : @1,
		};
		NSDictionary *plane1 = @{
			(id)kIOSurfacePlaneWidth : @32,
			(id)kIOSurfacePlaneHeight : @32,
			(id)kIOSurfacePlaneBytesPerElement : @2,
		};
		NSDictionary *p = @{
			(id)kIOSurfaceWidth : @64,
			(id)kIOSurfaceHeight : @64,
			(id)kIOSurfacePixelFormat : @('420v'),
			(id)kIOSurfacePlaneInfo : @[ plane0, plane1 ],
		};
		IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef)p);
		if (s == NULL) {
			printf("iosurf.buffer.420v no surface\n");
			return;
		}
		surface_buffer(d, "iosurf.buffer.420v", s);
		CFRelease(s);
	}
}

// --- shared events (event.*) ---------------------------------------------------------------

static void case_event_port(id<MTLDevice> d)
{
	id<MTLSharedEvent> se = [d newSharedEvent];
	se.signaledValue = 7;
	printf("event.port event=%s", cls(se));
	for (Class c = class_getSuperclass(object_getClass(se)); c; c = class_getSuperclass(c))
		printf(" < %s", class_getName(c));
	printf("\n");
	SEL port = sel_registerName("eventPort");
	if ([(id)se respondsToSelector:port]) {
		unsigned int p = ((unsigned int (*)(id, SEL))objc_msgSend)(se, port);
		printf("event.port eventPort=%s\n", p ? "nonzero" : "0");
	} else
		printf("event.port eventPort n/r\n");
	printf("event.port NSCoding=%d NSSecureCoding=%d supportsSecureCoding=%d "
	       "respondsEncode=%d respondsInit=%d\n",
	       (int)[(id)se conformsToProtocol:@protocol(NSCoding)],
	       (int)[(id)se conformsToProtocol:@protocol(NSSecureCoding)],
	       [[(id)se class] respondsToSelector:@selector(supportsSecureCoding)]
	           ? (int)[[(id)se class] supportsSecureCoding]
	           : -1,
	       (int)[(id)se respondsToSelector:@selector(encodeWithCoder:)],
	       (int)[[(id)se class] instancesRespondToSelector:@selector(initWithCoder:)]);
	fflush(stdout);
	for (int secure = 1; secure >= 0; secure--) {
		@try {
			NSError *e = nil;
			NSData *data = [NSKeyedArchiver archivedDataWithRootObject:se
			                                     requiringSecureCoding:secure
			                                                     error:&e];
			printf("event.port keyedArchive secure=%d data=%lu error=%s\n", secure,
			       (unsigned long)data.length, err_text(e));
			if (data != nil) {
				e = nil;
				id back = [NSKeyedUnarchiver unarchivedObjectOfClass:[(id)se class]
				                                            fromData:data
				                                               error:&e];
				printf("event.port unarchived=%s value=%llu error=%s\n", cls(back),
				       back ? [(id<MTLSharedEvent>)back signaledValue] : 0ull, err_text(e));
			}
		} @catch (NSException *x) {
			printf("event.port keyedArchive secure=%d threw %s %s\n", secure, x.name.UTF8String,
			       x.reason.UTF8String);
		}
		fflush(stdout);
	}
	MTLSharedEventHandle *h = [se newSharedEventHandle];
	printf("event.port handle=%s label=%s\n", cls(h), h.label ? h.label.UTF8String : "nil");
	@try {
		NSError *e = nil;
		NSData *data = [NSKeyedArchiver archivedDataWithRootObject:h requiringSecureCoding:YES error:&e];
		printf("event.port handleArchive data=%lu error=%s\n", (unsigned long)data.length,
		       err_text(e));
	} @catch (NSException *x) {
		printf("event.port handleArchive threw %s %s\n", x.name.UTF8String, x.reason.UTF8String);
	}
	id<MTLSharedEvent> again = h ? [d newSharedEventWithHandle:h] : nil;
	printf("event.port fromHandle=%s value=%llu sameObject=%d\n", cls(again),
	       again ? again.signaledValue : 0ull, again == se);
}

// --- binary archive slices (archive.slices) ----------------------------------------------

// A fat file of the given slices: each (cputype, subtype, bytes).
static NSData *fat_of(uint32_t magic, NSArray<NSArray *> *slices)
{
	NSMutableData *m = [NSMutableData dataWithLength:8 + 20 * slices.count];
	put_be32(m.mutableBytes, magic);
	put_be32((uint8_t *)m.mutableBytes + 4, (uint32_t)slices.count);
	for (NSUInteger i = 0; i < slices.count; i++) {
		NSData *bytes = slices[i][2];
		while (m.length % 16)
			[m increaseLengthBy:1];
		uint8_t *rec = (uint8_t *)m.mutableBytes + 8 + 20 * i;
		put_be32(rec, [slices[i][0] unsignedIntValue]);
		put_be32(rec + 4, [slices[i][1] unsignedIntValue]);
		put_be32(rec + 8, (uint32_t)m.length);
		put_be32(rec + 12, (uint32_t)bytes.length);
		put_be32(rec + 16, 4);
		[m appendData:bytes];
	}
	return m;
}

static NSData *junked(NSData *bytes, size_t from, size_t count)
{
	NSMutableData *m = [bytes mutableCopy];
	if (from < m.length)
		memset((uint8_t *)m.mutableBytes + from, 0x5a, MIN(count, m.length - from));
	return m;
}

static void slices_variant(id<MTLDevice> d, id<MTLLibrary> lib, const char *tag, NSData *file)
{
	NSURL *url = [NSURL fileURLWithPath:[NSTemporaryDirectory()
	                                        stringByAppendingPathComponent:
	                                            [NSString stringWithFormat:@"div-%s.bin", tag]]];
	[file writeToURL:url atomically:YES];
	describe_fat(tag, file);
	pipeline_in(d, tag, lib, open_archive(d, tag, url));
	fflush(stdout);
}

static void case_archive_slices(id<MTLDevice> d)
{
	NSError *e = nil;
	id<MTLLibrary> lib = [d newLibraryWithURL:beside(@"oracle_bundle.metallib") error:&e];
	id<MTLBinaryArchive> a = [d newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new]
	                                                     error:&e];
	MTLComputePipelineDescriptor *cd = [MTLComputePipelineDescriptor new];
	cd.computeFunction = [lib newFunctionWithName:@"div_kernel" constantValues:kernel_values(2.0f)
	                                        error:&e];
	[a addComputePipelineFunctionsWithDescriptor:cd error:&e];
	NSURL *url = [NSURL fileURLWithPath:[NSTemporaryDirectory()
	                                        stringByAppendingPathComponent:@"div-slices.bin"]];
	[a serializeToURL:url error:&e];
	NSData *base = [NSData dataWithContentsOfURL:url];
	if (base == nil || base.length < 48) {
		printf("archive.slices no archive error=%s\n", err_text(e));
		return;
	}
	describe_fat("archive.slices", base);
	const uint8_t *b = base.bytes;
	const uint32_t n = be32(b + 4);
	NSData *air = nil, *gpu = nil;
	uint32_t airType = 0, airSub = 0, gpuType = 0, gpuSub = 0;
	for (uint32_t i = 0; i < n; i++) {
		const uint8_t *s = b + 8 + 20 * i;
		NSData *bytes = [base subdataWithRange:NSMakeRange(be32(s + 8), be32(s + 12))];
		if (be32(s) == 0x01000017) {
			air = bytes;
			airType = be32(s);
			airSub = be32(s + 4);
		} else {
			gpu = bytes;
			gpuType = be32(s);
			gpuSub = be32(s + 4);
		}
	}
	if (air == nil || gpu == nil) {
		printf("archive.slices no AIR or GPU slice\n");
		return;
	}
	const uint32_t m = 0xcbfebabe;
	NSData *junk64 = junked([NSMutableData dataWithLength:64], 0, 64);
	NSData *junkAir = junked(air, 0, air.length);
	struct {
		const char *tag;
		NSArray *slices;
	} v[] = {
	    {"slices.both", @[ @[ @(airType), @(airSub), air ], @[ @(gpuType), @(gpuSub), gpu ] ]},
	    {"slices.air.head64.gpu", @[ @[ @(airType), @(airSub), junked(air, 0, 64) ], @[ @(gpuType), @(gpuSub), gpu ] ]},
	    {"slices.air.magic.gpu", @[ @[ @(airType), @(airSub), junked(air, 0, 4) ], @[ @(gpuType), @(gpuSub), gpu ] ]},
	    {"slices.air.tail.gpu", @[ @[ @(airType), @(airSub), junked(air, 0x100, air.length) ], @[ @(gpuType), @(gpuSub), gpu ] ]},
	    {"slices.air.all.gpu", @[ @[ @(airType), @(airSub), junkAir ], @[ @(gpuType), @(gpuSub), gpu ] ]},
	    {"slices.air64.gpu", @[ @[ @(airType), @(airSub), junk64 ], @[ @(gpuType), @(gpuSub), gpu ] ]},
	    {"slices.air.gpu.head64", @[ @[ @(airType), @(airSub), air ], @[ @(gpuType), @(gpuSub), junked(gpu, 0, 64) ] ]},
	    {"slices.air.gpu64", @[ @[ @(airType), @(airSub), air ], @[ @(gpuType), @(gpuSub), junk64 ] ]},
	    {"slices.air.only", @[ @[ @(airType), @(airSub), air ] ]},
	    {"slices.air.only.head64", @[ @[ @(airType), @(airSub), junked(air, 0, 64) ] ]},
	    {"slices.air.only.all", @[ @[ @(airType), @(airSub), junkAir ] ]},
	    {"slices.air64.only", @[ @[ @(airType), @(airSub), junk64 ] ]},
	    {"slices.gpu.only", @[ @[ @(gpuType), @(gpuSub), gpu ] ]},
	    {"slices.gpu.only.head64", @[ @[ @(gpuType), @(gpuSub), junked(gpu, 0, 64) ] ]},
	    {"slices.gpu64.only", @[ @[ @(gpuType), @(gpuSub), junk64 ] ]},
	    {"slices.air.head64.other", @[ @[ @(airType), @(airSub), junked(air, 0, 64) ], @[ @(0x01000013u), @(0x142u), gpu ] ]},
	    {"slices.air.head64.othersub", @[ @[ @(airType), @(airSub), junked(air, 0, 64) ], @[ @(gpuType), @(gpuSub ^ 0x10u), gpu ] ]},
	    {"slices.air.other64", @[ @[ @(airType), @(airSub), air ], @[ @(0x01000013u), @(0x142u), junk64 ] ]},
	    {"slices.other64.only", @[ @[ @(0x01000013u), @(0x142u), junk64 ] ]},
	    {"slices.gpu.air.head64", @[ @[ @(gpuType), @(gpuSub), gpu ], @[ @(airType), @(airSub), junked(air, 0, 64) ] ]},
	    {"slices.air.air64.gpu", @[ @[ @(airType), @(airSub), air ], @[ @(airType), @(airSub), junk64 ], @[ @(gpuType), @(gpuSub), gpu ] ]},
	};
	for (size_t i = 0; i < sizeof v / sizeof v[0]; i++)
		slices_variant(d, lib, v[i].tag, fat_of(m, v[i].slices));
}

// --- library forms, more (libforms2, libforms.kernel, libforms.vertex) ---------------------

static id lib_of_functions(id<MTLDevice> d, const char *tag, NSArray *fns)
{
	SEL arr = sel_registerName("newLibraryWithFunctionArray:error:");
	if (![d respondsToSelector:arr]) {
		printf("%s n/r\n", tag);
		return nil;
	}
	NSError *e = nil;
	id l = nil;
	@try {
		l = ((id(*)(id, SEL, id, NSError **))objc_msgSend)(d, arr, fns, &e);
	} @catch (NSException *x) {
		printf("%s threw %s %s\n", tag, x.name.UTF8String, x.reason.UTF8String);
		return nil;
	}
	print_lib(tag, l, e);
	for (NSString *name in [l functionNames]) {
		id<MTLFunction> fn = [l newFunctionWithName:name];
		printf("%s function %s=%s functionType=%ld\n", tag, name.UTF8String, cls(fn),
		       fn ? (long)fn.functionType : -1L);
	}
	return l;
}

static void case_libforms2(id<MTLDevice> d, int which)
{
	NSError *e = nil;
	id<MTLLibrary> lib = [d newLibraryWithURL:beside(@"oracle_bundle.metallib") error:&e];
	id<MTLFunction> visible = [lib newFunctionWithName:@"div_visible"];
	id<MTLFunction> stitchable = [lib newFunctionWithName:@"div_stitchable"];
	if (which == 0) {
		id l = lib_of_functions(d, "libforms2.both", @[ visible, stitchable ]);
		if (l != nil) {
			// A pipeline links the array's library's function: its function is usable?
			MTLLinkedFunctions *lf = [MTLLinkedFunctions linkedFunctions];
			lf.functions = @[ [l newFunctionWithName:@"div_visible"] ];
			printf("libforms2.both linkable=%s\n", cls(lf.functions.firstObject));
		}
		lib_of_functions(d, "libforms2.twice", @[ visible, visible ]);
		NSString *dir = [NSTemporaryDirectory() stringByAppendingPathComponent:@"div-empty.mtlpackage"];
		[[NSFileManager defaultManager] createDirectoryAtPath:dir withIntermediateDirectories:YES
		                                           attributes:nil
		                                                error:nil];
		SEL pkg = sel_registerName("newLibraryWithMetalPackageURL:error:");
		NSURL *urls[] = {[NSURL fileURLWithPath:dir], [NSURL fileURLWithPath:NSTemporaryDirectory()],
		                 [NSURL URLWithString:@"https://example.invalid/div.mtlpackage"]};
		for (int i = 0; i < 3; i++) {
			e = nil;
			@try {
				id l = ((id(*)(id, SEL, id, NSError **))objc_msgSend)(d, pkg, urls[i], &e);
				printf("libforms2 metalPackage(%s)=%s error=%s\n",
				       i == 0 ? "emptydir" : i == 1 ? "tmpdir" : "https", cls(l), err_text(e));
			} @catch (NSException *x) {
				printf("libforms2 metalPackage(%d) threw %s %s\n", i, x.name.UTF8String,
				       x.reason.UTF8String);
			}
		}
		SEL mps = sel_registerName("newLibraryWithMPSGraphPackageURL:name:error:");
		e = nil;
		@try {
			id l = ((id(*)(id, SEL, id, id, NSError **))objc_msgSend)(
			    d, mps, [NSURL fileURLWithPath:dir], @"div", &e);
			printf("libforms2 mpsGraphPackage(emptydir)=%s error=%s\n", cls(l), err_text(e));
		} @catch (NSException *x) {
			printf("libforms2 mpsGraphPackage(emptydir) threw %s %s\n", x.name.UTF8String,
			       x.reason.UTF8String);
		}
	} else {
		id<MTLFunction> fn = which == 1 ? [lib newFunctionWithName:@"div_kernel"
		                                             constantValues:kernel_values(2.0f)
		                                                      error:&e]
		                                : [lib newFunctionWithName:@"div_vertex"];
		const char *tag = which == 1 ? "libforms.kernel" : "libforms.vertex";
		if (fn == nil)
			printf("%s no function error=%s\n", tag, err_text(e));
		else
			lib_of_functions(d, tag, @[ fn ]);
	}
}

// --- who answers the SPI (classes.spi) ---------------------------------------------------

// Whether `c` itself (not a superclass) has an instance method `sel`.
static BOOL own_method(Class c, SEL sel)
{
	unsigned int n = 0;
	Method *m = class_copyMethodList(c, &n);
	BOOL found = NO;
	for (unsigned int i = 0; i < n && !found; i++)
		found = method_getName(m[i]) == sel;
	free(m);
	return found;
}

static void print_owners(const char *tag, id obj, Class extra, const char *const *sels, int count)
{
	for (int i = 0; i < count; i++) {
		SEL sel = sel_registerName(sels[i]);
		printf("%s %s responds=%d defined-by", tag, sels[i], (int)[obj respondsToSelector:sel]);
		for (Class c = object_getClass(obj); c; c = class_getSuperclass(c))
			if (own_method(c, sel))
				printf(" %s(%s)", class_getName(c), method_getTypeEncoding(class_getInstanceMethod(c, sel)));
		if (extra != nil && own_method(extra, sel))
			printf(" [%s has it]", class_getName(extra));
		printf("\n");
	}
}

static void case_classes_spi(id<MTLDevice> d)
{
	Class iogpu = objc_getClass("IOGPUMetalDevice");
	printf("classes.spi device=%s", cls(d));
	for (Class c = class_getSuperclass(object_getClass(d)); c; c = class_getSuperclass(c))
		printf(" < %s", class_getName(c));
	printf(" IOGPUMetalDevice=%s\n", iogpu ? "present" : "absent");
	const char *deviceSels[] = {"copyIOSurfaceSharedTextureProperties:", "newBufferWithIOSurface:",
	                            "newArgumentEncoderWithBufferBinding:",
	                            "newLibraryWithFunctionArray:error:",
	                            "newLibraryWithMetalPackageURL:error:",
	                            "newLibraryWithMPSGraphPackageURL:name:error:"};
	print_owners("classes.spi device", d, iogpu, deviceSels, 6);
	id<MTLSharedEvent> se = [d newSharedEvent];
	Class iogpuEvent = objc_getClass("IOGPUMetalSharedEvent");
	printf("classes.spi event=%s IOGPUMetalSharedEvent=%s\n", cls(se), iogpuEvent ? "present" : "absent");
	const char *eventSels[] = {"eventPort", "encodeWithCoder:", "initWithCoder:"};
	print_owners("classes.spi event", se, iogpuEvent, eventSels, 3);
	id<MTLBuffer> b = [d newBufferWithLength:64 options:MTLResourceStorageModeShared];
	const char *bufferSels[] = {"iosurface"};
	print_owners("classes.spi buffer", b, objc_getClass("IOGPUMetalBuffer"), bufferSels, 1);
}

// --- the cases ----------------------------------------------------------------------------

static const char *const kCases[] = {
    "buffers",          "heapbuffers",      "nocopy.page",   "nocopy.3pages", "nocopy.16k",
    "nocopy.len100",    "nocopy.off16",     "nocopy.managed", "events",       "archive.system",
    "archive.made",     "failmiss",         "functypes",     "libtype",       "minlod",
    "classes",          "heapsize",         "libforms",
    "view.class",       "view.levels",      "view.slices",   "view.type",     "view.slicecount",
    "view.noformat",    "view.ok",          "buftex.bc1",    "buftex.depth",  "buftex.shortrow",
    "linear.shortrow",  "functypes.dyn",    "failmiss2",
    "devtex.depth.shared", "devtex.depth.managed", "devtex.stencil.managed",
    "devtex.depth.private",
    "argbuf.ds.id",     "argbuf.ds.encoder", "argbuf.ds.set", "argbuf.ds.mixed",
    "argbuf.ds.function", "argbuf.binding", "argbuf.binding.plain", "argbuf.binding.lib",
    "argbuf.reflection", "iosurf.props",    "iosurf.buffer", "iosurf.buffer.420v", "event.port",
    "archive.slices",   "libforms2",        "libforms.kernel", "libforms.vertex", "classes.spi",
};

static int run_case(id<MTLDevice> d, const char *c)
{
	const size_t page = (size_t)getpagesize();
	const MTLResourceOptions shared = MTLResourceStorageModeShared;
	if (!strcmp(c, "buffers"))
		case_buffers(d);
	else if (!strcmp(c, "heapbuffers"))
		case_heapbuffers(d);
	else if (!strcmp(c, "nocopy.page"))
		nocopy(d, c, 0, page, shared);
	else if (!strcmp(c, "nocopy.3pages"))
		nocopy(d, c, 0, 3 * page, shared);
	else if (!strcmp(c, "nocopy.16k"))
		nocopy(d, c, 0, 16384, shared);
	else if (!strcmp(c, "nocopy.len100"))
		nocopy(d, c, 0, 100, shared);
	else if (!strcmp(c, "nocopy.off16"))
		nocopy(d, c, 16, page, shared);
	else if (!strcmp(c, "nocopy.managed"))
		nocopy(d, c, 0, page, MTLResourceStorageModeManaged);
	else if (!strcmp(c, "events"))
		case_events(d);
	else if (!strcmp(c, "archive.system"))
		case_archive_system(d);
	else if (!strcmp(c, "archive.made"))
		case_archive_made(d);
	else if (!strcmp(c, "failmiss"))
		case_failmiss(d);
	else if (!strcmp(c, "functypes"))
		case_functypes(d);
	else if (!strcmp(c, "libtype"))
		case_libtype(d);
	else if (!strcmp(c, "minlod"))
		case_minlod();
	else if (!strcmp(c, "classes"))
		case_classes(d);
	else if (!strcmp(c, "heapsize"))
		case_heapsize(d);
	else if (!strcmp(c, "libforms"))
		case_libforms(d);
	else if (!strcmp(c, "functypes.dyn"))
		case_functypes_dyn(d);
	else if (!strcmp(c, "failmiss2"))
		case_failmiss2(d);
	else if (!strcmp(c, "view.class"))
		view_case(d, c, MTLPixelFormatRGBA16Float, MTLTextureType2D, 1, 1, MTLPixelFormatRGBA8Unorm,
		          MTLTextureType2D, NSMakeRange(0, 1), NSMakeRange(0, 1));
	else if (!strcmp(c, "view.levels"))
		view_case(d, c, MTLPixelFormatRGBA8Unorm, MTLTextureType2D, 3, 1, MTLPixelFormatRGBA8Unorm,
		          MTLTextureType2D, NSMakeRange(2, 3), NSMakeRange(0, 1));
	else if (!strcmp(c, "view.slices"))
		view_case(d, c, MTLPixelFormatRGBA8Unorm, MTLTextureType2DArray, 1, 4,
		          MTLPixelFormatRGBA8Unorm, MTLTextureType2DArray, NSMakeRange(0, 1),
		          NSMakeRange(3, 2));
	else if (!strcmp(c, "view.type"))
		view_case(d, c, MTLPixelFormatRGBA8Unorm, MTLTextureType2D, 1, 1, MTLPixelFormatRGBA8Unorm,
		          MTLTextureType3D, NSMakeRange(0, 1), NSMakeRange(0, 1));
	else if (!strcmp(c, "view.slicecount"))
		view_case(d, c, MTLPixelFormatRGBA8Unorm, MTLTextureType2DArray, 1, 4,
		          MTLPixelFormatRGBA8Unorm, MTLTextureTypeCube, NSMakeRange(0, 1),
		          NSMakeRange(0, 4));
	else if (!strcmp(c, "view.noformat"))
		view_case(d, c, MTLPixelFormatRGBA8Unorm, MTLTextureType2D, 1, 1, MTLPixelFormatInvalid,
		          MTLTextureType2D, NSMakeRange(0, 1), NSMakeRange(0, 1));
	else if (!strcmp(c, "view.ok"))
		view_case(d, c, MTLPixelFormatRGBA8Unorm, MTLTextureType2D, 1, 1, MTLPixelFormatR32Float,
		          MTLTextureType2D, NSMakeRange(0, 1), NSMakeRange(0, 1));
	else if (!strcmp(c, "buftex.bc1"))
		buffer_texture(d, c, MTLPixelFormatBC1_RGBA, 64, 64, 256, NO);
	else if (!strcmp(c, "buftex.depth"))
		buffer_texture(d, c, MTLPixelFormatDepth32Float, 64, 64, 256, NO);
	else if (!strcmp(c, "buftex.shortrow"))
		buffer_texture(d, c, MTLPixelFormatBGRA8Unorm, 468, 36, 1536, NO);
	else if (!strcmp(c, "linear.shortrow"))
		buffer_texture(d, c, MTLPixelFormatBGRA8Unorm, 468, 36, 1536, YES);
	else if (!strcmp(c, "devtex.depth.shared"))
		device_texture(d, c, MTLPixelFormatDepth32Float, MTLStorageModeShared);
	else if (!strcmp(c, "devtex.depth.managed"))
		device_texture(d, c, MTLPixelFormatDepth32Float, MTLStorageModeManaged);
	else if (!strcmp(c, "devtex.stencil.managed"))
		device_texture(d, c, MTLPixelFormatStencil8, MTLStorageModeManaged);
	else if (!strcmp(c, "devtex.depth.private"))
		device_texture(d, c, MTLPixelFormatDepth32Float, MTLStorageModePrivate);
	else if (!strcmp(c, "texaddr.heap"))
		case_texaddr_heap(d, c);
	else if (!strcmp(c, "texaddr.buffer"))
		case_texaddr_buffer(d, c);
	else if (!strcmp(c, "texaddr.view"))
		case_texaddr_view(d, c);
	else if (!strcmp(c, "texaddr.iosurface"))
		case_texaddr_iosurface(d, c);
	else if (!strncmp(c, "texaddr.", 8) || !strncmp(c, "texmem.", 7)) {
		const char *rest = strchr(c, '.') + 1;
		int mode = -1, shape = -1;
		for (int m = 0; m < 3; m++) {
			const size_t n = strlen(kTexModeNames[m]);
			if (!strncmp(rest, kTexModeNames[m], n) && rest[n] == '.') {
				mode = m;
				for (size_t k = 0; k < sizeof kTexShapes / sizeof kTexShapes[0]; k++)
					if (!strcmp(rest + n + 1, kTexShapes[k].name))
						shape = (int)k;
			}
		}
		if (mode < 0 || shape < 0) {
			fprintf(stderr, "no case %s\n", c);
			return 2;
		}
		if (c[3] == 'a')
			case_texaddr(d, c, mode, shape);
		else
			case_texmem(d, c, mode, shape);
	} else if (!strcmp(c, "nocopy.ro.anon.shared"))
		case_nocopy_ro(d, c, 0, MTLResourceStorageModeShared);
	else if (!strcmp(c, "nocopy.ro.anon.managed"))
		case_nocopy_ro(d, c, 0, MTLResourceStorageModeManaged);
	else if (!strcmp(c, "nocopy.ro.file.shared"))
		case_nocopy_ro(d, c, 1, MTLResourceStorageModeShared);
	else if (!strcmp(c, "nocopy.ro.file.managed"))
		case_nocopy_ro(d, c, 1, MTLResourceStorageModeManaged);
	else if (!strcmp(c, "nocopy.copies"))
		case_nocopy_copies(d, c);
	else if (!strncmp(c, "linalign.", 9)) {
		const unsigned long f = strtoul(c + 9, NULL, 10);
		const NSUInteger a = [d minimumLinearTextureAlignmentForPixelFormat:(MTLPixelFormat)f];
		printf("linalign %lu=%lu\n", f, (unsigned long)a);
	} else if (!strcmp(c, "argbuf.ds.id"))
		case_argbuf_ds_id(d);
	else if (!strcmp(c, "argbuf.ds.encoder"))
		argbuf_ds_encoder(d, c, @[ arg_desc(MTLDataTypeDepthStencilState, 0) ], 0, NO);
	else if (!strcmp(c, "argbuf.ds.set"))
		argbuf_ds_encoder(d, c, @[ arg_desc(MTLDataTypeDepthStencilState, 0) ], 0, YES);
	else if (!strcmp(c, "argbuf.ds.mixed"))
		argbuf_ds_encoder(d, c,
		                  @[ arg_desc(MTLDataTypeTexture, 0), arg_desc(MTLDataTypeDepthStencilState, 1),
		                     arg_desc(MTLDataTypePointer, 2) ],
		                  1, YES);
	else if (!strcmp(c, "argbuf.ds.function"))
		case_argbuf_ds_function(d);
	else if (!strcmp(c, "argbuf.binding"))
		case_argbuf_binding(d, NO, NO);
	else if (!strcmp(c, "argbuf.binding.plain"))
		case_argbuf_binding(d, YES, NO);
	else if (!strcmp(c, "argbuf.binding.lib"))
		case_argbuf_binding(d, NO, YES);
	else if (!strcmp(c, "argbuf.reflection"))
		case_argbuf_reflection(d);
	else if (!strcmp(c, "iosurf.props"))
		case_iosurf_props(d);
	else if (!strcmp(c, "iosurf.buffer"))
		case_iosurf_buffer(d, 0);
	else if (!strcmp(c, "iosurf.buffer.420v"))
		case_iosurf_buffer(d, 1);
	else if (!strcmp(c, "event.port"))
		case_event_port(d);
	else if (!strcmp(c, "archive.slices"))
		case_archive_slices(d);
	else if (!strcmp(c, "libforms2"))
		case_libforms2(d, 0);
	else if (!strcmp(c, "libforms.kernel"))
		case_libforms2(d, 1);
	else if (!strcmp(c, "libforms.vertex"))
		case_libforms2(d, 2);
	else if (!strcmp(c, "classes.spi"))
		case_classes_spi(d);
	else {
		fprintf(stderr, "no case %s\n", c);
		return 2;
	}
	return 0;
}

static void spawn_case(const char *dev, const char *c)
{
	printf("== %s\n", c);
	fflush(stdout);
	char *args[] = {(char *)g_argv0, (char *)dev, "--one", (char *)c, NULL};
	pid_t pid;
	if (posix_spawn(&pid, g_argv0, NULL, NULL, args, environ) != 0) {
		printf("== %s spawn failed\n", c);
		return;
	}
	int st = 0;
	waitpid(pid, &st, 0);
	if (WIFSIGNALED(st))
		printf("== %s died(%d)\n", c, WTERMSIG(st));
	else
		printf("== %s exit %d\n", c, WEXITSTATUS(st));
	fflush(stdout);
}

// A case, or a group of them: `texaddr` (every storage mode and shape, then .heap, .buffer,
// .view and .iosurface), `texmem` (the Shared and Managed RGBA8 shapes 0, 2 and 3), `nocopy.ro`
// (the four kinds), `linalign` (raw formats 0 to 799 and four past them).
static void spawn_group(const char *dev, const char *c)
{
	char name[96];
	if (!strcmp(c, "texaddr")) {
		for (int m = 0; m < 3; m++)
			for (size_t k = 0; k < sizeof kTexShapes / sizeof kTexShapes[0]; k++) {
				snprintf(name, sizeof name, "texaddr.%s.%s", kTexModeNames[m], kTexShapes[k].name);
				spawn_case(dev, name);
			}
		spawn_case(dev, "texaddr.heap");
		spawn_case(dev, "texaddr.buffer");
		spawn_case(dev, "texaddr.view");
		spawn_case(dev, "texaddr.iosurface");
	} else if (!strcmp(c, "texmem")) {
		const int shapes[] = {0, 2, 3};
		for (int m = 1; m < 3; m++)
			for (int k = 0; k < 3; k++) {
				snprintf(name, sizeof name, "texmem.%s.%s", kTexModeNames[m],
				         kTexShapes[shapes[k]].name);
				spawn_case(dev, name);
			}
	} else if (!strcmp(c, "nocopy.ro")) {
		spawn_case(dev, "nocopy.ro.anon.shared");
		spawn_case(dev, "nocopy.ro.anon.managed");
		spawn_case(dev, "nocopy.ro.file.shared");
		spawn_case(dev, "nocopy.ro.file.managed");
	} else if (!strcmp(c, "linalign")) {
		const unsigned long past[] = {1000, 1023, 4096, 65535};
		for (unsigned long f = 0; f < 800 + 4; f++) {
			snprintf(name, sizeof name, "linalign.%lu", f < 800 ? f : past[f - 800]);
			spawn_case(dev, name);
		}
	} else
		spawn_case(dev, c);
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IOLBF, 0);
	g_argv0 = argv[0];
	if (argc < 2) {
		fprintf(stderr, "usage: %s DEVICE [CASE...]\n", argv[0]);
		return 2;
	}
	if (argc == 4 && !strcmp(argv[2], "--one")) {
		id<MTLDevice> d = oracle_device(argv[1]);
		int rc;
		@autoreleasepool {
			rc = run_case(d, argv[3]);
		}
		fflush(stdout);
		return rc;
	}
	@autoreleasepool {
		oracle_header("oracle_bundle", oracle_device(argv[1]));
	}
	const char *vars[] = {"MTL_DEBUG_LAYER", "METAL_DEVICE_WRAPPER_TYPE", "MTL_SHADER_VALIDATION",
	                      "METAL_ERROR_MODE"};
	for (int i = 0; i < 4; i++)
		printf("# %s=%s\n", vars[i], getenv(vars[i]) ? getenv(vars[i]) : "(unset)");
	if (argc == 2)
		for (size_t i = 0; i < sizeof kCases / sizeof kCases[0]; i++)
			spawn_case(argv[1], kCases[i]);
	else
		for (int i = 2; i < argc; i++)
			spawn_group(argv[1], argv[i]);
	return 0;
}
