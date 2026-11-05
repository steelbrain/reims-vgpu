// oracle_bundle: what a Mac's Metal device answers where the guest bundle's device answered
// otherwise (the bundle's divergences from Metal). Each case runs in a child process of its own,
// since some end the process at Metal's assertions; the parent prints how each child ended.
//   oracle_bundle DEVICE [CASE...]   the cases named, or every case
// The cases, each line `CASE key=value ...`:
//   buffers      allocatedSize of buffers of each storage mode by newBufferWithLength: and
//                newBufferWithBytes:, of lengths 1 to 1 MiB + 1
//   heapbuffers  allocatedSize of buffers in a 4 MiB heap of each storage mode, and the heap's
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
// Build: see README.md.
#import "oracle_common.h"
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
	const MTLStorageMode modes[] = {MTLStorageModeShared, MTLStorageModePrivate};
	const char *names[] = {"shared", "private"};
	for (int m = 0; m < 2; m++) {
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
	printf("%s archive=%s error=%s", tag, cls(a), err_text(e));
	SEL keys = sel_registerName("keys");
	if (a != nil && [(id)a respondsToSelector:keys]) {
		NSArray *k = ((id(*)(id, SEL))objc_msgSend)(a, keys);
		printf(" keys=%lu", (unsigned long)k.count);
	}
	printf("\n");
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
		NSArray *sets[] = {@[], visible ? @[ visible ] : @[],
		                   stitchable ? @[ stitchable ] : @[]};
		const char *names[] = {"empty", "visible", "stitchable"};
		for (int i = 0; i < 3; i++) {
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

// --- the cases ----------------------------------------------------------------------------

static const char *const kCases[] = {
    "buffers",          "heapbuffers",      "nocopy.page",   "nocopy.3pages", "nocopy.16k",
    "nocopy.len100",    "nocopy.off16",     "nocopy.managed", "events",       "archive.system",
    "archive.made",     "failmiss",         "functypes",     "libtype",       "minlod",
    "classes",          "heapsize",         "libforms",
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
			spawn_case(argv[1], argv[i]);
	return 0;
}
