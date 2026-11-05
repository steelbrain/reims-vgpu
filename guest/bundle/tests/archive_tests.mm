/*
 * archive_tests: the shell check of RgfxBinaryArchive (built into shell_tests, which calls
 * rgfxArchiveTests()). Each answer and error checked here is Metal's on the M3 Max (macOS 27),
 * as guest/bundle/tools/archive_probe.m measures it: making and loading an archive, adding to
 * it, writing it, and the hits and misses of a pipeline create with
 * MTLPipelineOptionFailOnBinaryArchiveMiss.
 *
 * Our own ObjC++; built on the Mac's native arch with the NSObject-based wrappers.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/message.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include <unistd.h>

#import "RgfxBinaryArchive.h"
#import "RgfxFunction.h"

#include "RgfxBinaryArchiveFile.hpp"
#include "reims_graphics_wire.h"
#include "rgfx_test_ring.hpp" // CHECK, g_checks

void rgfxArchiveTests();

namespace {

NSString *g_dir;

NSURL *at(NSString *name)
{
	return [NSURL fileURLWithPath:[g_dir stringByAppendingPathComponent:name]];
}

RgfxFunction *function(NSString *name, uint32_t constant = 0)
{
	RgfxFunctionConstant c = {};
	c.index = 0;
	c.data_type = static_cast<uint32_t>(MTLDataTypeUInt);
	std::memcpy(c.value, &constant, sizeof(constant));
	return [RgfxFunction functionWithLibrary:nil
	                                    name:name
	                               constants:constant == 0 ? nullptr : &c
	                                   count:constant == 0 ? 0 : 1
	                                  device:nil
	                              serializer:nullptr];
}

RgfxBinaryArchive *archive(NSURL *url, NSString *architecture, NSError **error)
{
	MTLBinaryArchiveDescriptor *d = [MTLBinaryArchiveDescriptor new];
	d.url = url;
	return [RgfxBinaryArchive archiveWithDescriptor:d
	                                         device:nil
	                                   architecture:architecture
	                                          error:error];
}

MTLComputePipelineDescriptor *compute(id<MTLFunction> f, NSArray *archives)
{
	MTLComputePipelineDescriptor *c = [MTLComputePipelineDescriptor new];
	if (f != nil)
		c.computeFunction = f;
	c.binaryArchives = archives;
	return c;
}

RgfxFunction *g_vs, *g_fs, *g_vs2;

MTLRenderPipelineDescriptor *render(NSArray *archives)
{
	MTLRenderPipelineDescriptor *r = [MTLRenderPipelineDescriptor new];
	r.vertexFunction = g_vs;
	r.fragmentFunction = g_fs;
	r.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
	r.binaryArchives = archives;
	return r;
}

MTLRenderPipelineDescriptor *withVertexDescriptor(NSArray *archives, NSUInteger stride)
{
	MTLRenderPipelineDescriptor *r = render(archives);
	r.vertexFunction = g_vs2;
	MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
	vd.attributes[0].format = MTLVertexFormatFloat2;
	vd.attributes[1].format = MTLVertexFormatFloat4;
	vd.attributes[1].offset = 8;
	vd.layouts[0].stride = stride;
	r.vertexDescriptor = vd;
	return r;
}

bool isMiss(NSError *e, NSString *function, NSString *pipeline)
{
	NSString *head = [NSString stringWithFormat:@"Unable to find function \"%@\" (%@ pipeline) in "
	                                            @"binary archives. ",
	                                            function, pipeline];
	return e != nil && [e.domain isEqualToString:RgfxBinaryArchiveMissDomain] && e.code == 3 &&
	       [e.localizedDescription hasPrefix:head];
}

bool isArchiveError(NSError *e, MTLBinaryArchiveError code, NSString *what)
{
	return e != nil && [e.domain isEqualToString:MTLBinaryArchiveDomain] &&
	       e.code == static_cast<NSInteger>(code) && [e.localizedDescription isEqualToString:what];
}

// Making one, from no URL and from what a URL names.
void test_archive_making()
{
	NSError *e = nil;
	RgfxBinaryArchive *a = archive(nil, @"arch_a", &e);
	CHECK(a != nil && e == nil);
	CHECK(a.label == nil);
	a.label = @"probe";
	CHECK([a.label isEqualToString:@"probe"]);
	CHECK(a.device == nil);

	CHECK(archive(at(@"missing.bin"), @"arch_a", &e) == nil);
	CHECK(isArchiveError(e, MTLBinaryArchiveErrorInvalidFile, @"Invalid URL"));
	CHECK(archive([NSURL URLWithString:@"https://example.invalid/a.bin"], @"arch_a", &e) == nil);
	CHECK(isArchiveError(e, MTLBinaryArchiveErrorInvalidFile, @"Invalid URL"));

	[[NSData data] writeToURL:at(@"empty.bin") atomically:NO];
	CHECK(archive(at(@"empty.bin"), @"arch_a", &e) == nil);
	NSString *invalid =
	    [NSString stringWithFormat:@"The file %@ has an invalid format.", at(@"empty.bin").absoluteString];
	CHECK(isArchiveError(e, MTLBinaryArchiveErrorInvalidFile, invalid));
	[[@"not an archive at all" dataUsingEncoding:NSUTF8StringEncoding] writeToURL:at(@"garbage.bin")
	                                                                    atomically:NO];
	CHECK(archive(at(@"garbage.bin"), @"arch_a", &e) == nil);
	CHECK(e.code == MTLBinaryArchiveErrorInvalidFile);
	[[NSFileManager defaultManager] createDirectoryAtURL:at(@"subdir")
	                         withIntermediateDirectories:YES
	                                          attributes:nil
	                                               error:nil];
	NSURL *dir = at(@"subdir");
	CHECK(archive(dir, @"arch_a", &e) == nil);
	CHECK(isArchiveError(
	    e, MTLBinaryArchiveErrorInvalidFile,
	    [NSString stringWithFormat:@"The file %@ has an invalid format.", dir.absoluteString]));

	// Nothing added: Metal writes nothing, but answers a URL it cannot write to first.
	CHECK(![a serializeToURL:at(@"empty-archive.bin") error:&e]);
	CHECK(isArchiveError(e, MTLBinaryArchiveErrorCompilationFailure, @"Nothing to serialize"));
	CHECK(![[NSFileManager defaultManager] fileExistsAtPath:at(@"empty-archive.bin").path]);
	CHECK(![a serializeToURL:at(@"no/such/dir/a.bin") error:&e]);
	CHECK(isArchiveError(e, MTLBinaryArchiveErrorInvalidFile, @"Invalid URL"));
	CHECK(![a serializeToURL:[NSURL URLWithString:@"https://example.invalid/a.bin"] error:&e]);
	CHECK(isArchiveError(e, MTLBinaryArchiveErrorInvalidFile, @"Invalid URL"));
}

// Adding: what Metal asserts on is refused with its words; a fragment function may be absent.
void test_archive_adding()
{
	NSError *e = nil;
	RgfxBinaryArchive *a = archive(nil, @"arch_a", &e);
	CHECK([a addComputePipelineFunctionsWithDescriptor:compute(function(@"a"), nil) error:&e]);
	CHECK([a addComputePipelineFunctionsWithDescriptor:compute(function(@"a"), nil) error:&e]);
	CHECK(![a addComputePipelineFunctionsWithDescriptor:compute(nil, nil) error:&e]);
	CHECK([e.localizedDescription
	    isEqualToString:@"Compute Pipeline Descriptor Validation\ncomputeFunction must not be nil.\n"]);

	MTLRenderPipelineDescriptor *r = render(nil);
	r.vertexFunction = nil;
	CHECK(![a addRenderPipelineFunctionsWithDescriptor:r error:&e]);
	CHECK([e.localizedDescription
	    isEqualToString:@"Render Pipeline Descriptor Validation\nvertexFunction must not be nil.\n"]);
	r = render(nil);
	r.fragmentFunction = nil;
	CHECK([a addRenderPipelineFunctionsWithDescriptor:r error:&e]);

	MTLTileRenderPipelineDescriptor *t = [MTLTileRenderPipelineDescriptor new];
	t.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
	CHECK(![a addTileRenderPipelineFunctionsWithDescriptor:t error:&e]);
	CHECK([e.localizedDescription hasPrefix:@"Tile Render Pipeline Descriptor Validation\n"]);
	t.tileFunction = function(@"tile_k");
	CHECK([a addTileRenderPipelineFunctionsWithDescriptor:t error:&e]);

	// A mesh pipeline, with no function or with one: refused, as an x86 Mac's archive refuses it.
	MTLMeshRenderPipelineDescriptor *m = [MTLMeshRenderPipelineDescriptor new];
	m.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
	CHECK(![a addMeshRenderPipelineFunctionsWithDescriptor:m error:&e]);
	CHECK(isArchiveError(e, MTLBinaryArchiveErrorCompilationFailure,
	                     @"Compiling mesh pipelines for binary archives is not supported!"));
	m.meshFunction = function(@"a");
	e = nil;
	CHECK(![a addMeshRenderPipelineFunctionsWithDescriptor:m error:&e]);
	CHECK(isArchiveError(e, MTLBinaryArchiveErrorCompilationFailure,
	                     @"Compiling mesh pipelines for binary archives is not supported!"));
}

// Looking up: what Metal keys each compile by hits or misses as Metal's does.
void test_archive_lookups()
{
	NSError *e = nil;
	RgfxBinaryArchive *a = archive(nil, @"arch_a", &e);
	RgfxBinaryArchive *empty = archive(nil, @"arch_a", &e);
	NSArray *in = @[ a ];
	CHECK([a addComputePipelineFunctionsWithDescriptor:compute(function(@"a"), nil) error:&e]);
	CHECK([a addComputePipelineFunctionsWithDescriptor:compute(function(@"fc_kernel", 1), nil)
	                                              error:&e]);
	CHECK([a addRenderPipelineFunctionsWithDescriptor:render(nil) error:&e]);
	CHECK([a addRenderPipelineFunctionsWithDescriptor:withVertexDescriptor(nil, 24) error:&e]);

	CHECK([RgfxBinaryArchive rgfxMissOfComputePipeline:compute(function(@"a"), in)] == nil);
	CHECK(isMiss([RgfxBinaryArchive rgfxMissOfComputePipeline:compute(function(@"b"), in)], @"b",
	             @"compute"));
	CHECK([RgfxBinaryArchive rgfxMissOfComputePipeline:compute(function(@"fc_kernel", 1), in)] ==
	      nil);
	CHECK(isMiss([RgfxBinaryArchive rgfxMissOfComputePipeline:compute(function(@"fc_kernel", 2), in)],
	             @"fc_kernel", @"compute"));
	// No archive named: Metal compiles; an archive with nothing in it misses; any one may hold it.
	CHECK([RgfxBinaryArchive rgfxMissOfComputePipeline:compute(function(@"b"), nil)] == nil);
	CHECK([RgfxBinaryArchive rgfxMissOfComputePipeline:compute(function(@"b"), @[])] == nil);
	CHECK(isMiss([RgfxBinaryArchive rgfxMissOfComputePipeline:compute(function(@"a"), @[ empty ])],
	             @"a", @"compute"));
	CHECK([RgfxBinaryArchive rgfxMissOfComputePipeline:compute(function(@"a"), @[ empty, a ])] ==
	      nil);
	MTLComputePipelineDescriptor *c = compute(function(@"a"), in);
	c.label = @"another label";
	c.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
	c.maxCallStackDepth = 4;
	CHECK([RgfxBinaryArchive rgfxMissOfComputePipeline:c] == nil);
	c = compute(function(@"a"), in);
	c.maxTotalThreadsPerThreadgroup = 64;
	CHECK(isMiss([RgfxBinaryArchive rgfxMissOfComputePipeline:c], @"a", @"compute"));
	c = compute(function(@"a"), in);
	c.supportIndirectCommandBuffers = YES;
	CHECK(isMiss([RgfxBinaryArchive rgfxMissOfComputePipeline:c], @"a", @"compute"));

	CHECK([RgfxBinaryArchive rgfxMissOfRenderPipeline:render(in)] == nil);
	MTLRenderPipelineDescriptor *r = render(in);
	r.label = @"another label";
	r.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
	r.stencilAttachmentPixelFormat = MTLPixelFormatStencil8;
	r.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorZero; // blending is off
	CHECK([RgfxBinaryArchive rgfxMissOfRenderPipeline:r] == nil);
	r = render(in);
	r.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
	CHECK(isMiss([RgfxBinaryArchive rgfxMissOfRenderPipeline:r], @"tri_frag", @"render"));
	r = render(in);
	r.colorAttachments[0].blendingEnabled = YES;
	CHECK(isMiss([RgfxBinaryArchive rgfxMissOfRenderPipeline:r], @"tri_frag", @"render"));
	r = render(in);
	r.rasterSampleCount = 4;
	CHECK(isMiss([RgfxBinaryArchive rgfxMissOfRenderPipeline:r], @"tri_frag", @"render"));
	r = render(in);
	r.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
	CHECK(isMiss([RgfxBinaryArchive rgfxMissOfRenderPipeline:r], @"tri_vert", @"render"));
	// Both stages missing: the fragment function is named, as Metal looks it up first.
	r.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
	CHECK(isMiss([RgfxBinaryArchive rgfxMissOfRenderPipeline:r], @"tri_frag", @"render"));
	// A pipeline with no fragment function looks up its vertex function alone.
	r = render(in);
	r.fragmentFunction = nil;
	CHECK([RgfxBinaryArchive rgfxMissOfRenderPipeline:r] == nil);
	// The vertex descriptor is the fragment function's.
	CHECK([RgfxBinaryArchive rgfxMissOfRenderPipeline:withVertexDescriptor(in, 24)] == nil);
	CHECK(isMiss([RgfxBinaryArchive rgfxMissOfRenderPipeline:withVertexDescriptor(in, 32)],
	             @"tri_frag", @"render"));
}

// Writing it and loading it back, on this architecture and on another.
void test_archive_files()
{
	NSError *e = nil;
	RgfxBinaryArchive *a = archive(nil, @"arch_a", &e);
	CHECK([a addComputePipelineFunctionsWithDescriptor:compute(function(@"a"), nil) error:&e]);
	CHECK([a addRenderPipelineFunctionsWithDescriptor:render(nil) error:&e]);
	CHECK([a serializeToURL:at(@"full.bin") error:&e] && e == nil);
	CHECK([a serializeToURL:at(@"full.bin") error:&e]);
	CHECK(![a serializeToURL:at(@"no/such/dir/a.bin") error:&e]);
	CHECK(isArchiveError(e, MTLBinaryArchiveErrorInvalidFile, @"Invalid URL"));
	CHECK(![a serializeToURL:[NSURL URLWithString:@"https://example.invalid/a.bin"] error:&e]);
	CHECK(isArchiveError(e, MTLBinaryArchiveErrorInvalidFile, @"Invalid URL"));

	RgfxBinaryArchive *loaded = archive(at(@"full.bin"), @"arch_a", &e);
	CHECK(loaded != nil && e == nil);
	CHECK([RgfxBinaryArchive rgfxMissOfComputePipeline:compute(function(@"a"), @[ loaded ])] == nil);
	CHECK([RgfxBinaryArchive rgfxMissOfRenderPipeline:render(@[ loaded ])] == nil);
	CHECK(isMiss([RgfxBinaryArchive rgfxMissOfComputePipeline:compute(function(@"b"), @[ loaded ])],
	             @"b", @"compute"));

	// Another GPU's archive loads, and misses every lookup; what it adds goes beside the other
	// GPU's entries, which it writes back.
	RgfxBinaryArchive *other = archive(at(@"full.bin"), @"arch_b", &e);
	CHECK(other != nil && e == nil);
	CHECK(isMiss([RgfxBinaryArchive rgfxMissOfComputePipeline:compute(function(@"a"), @[ other ])],
	             @"a", @"compute"));
	CHECK([other addComputePipelineFunctionsWithDescriptor:compute(function(@"b"), nil) error:&e]);
	CHECK([other serializeToURL:at(@"both.bin") error:&e]);
	NSData *both = [NSData dataWithContentsOfURL:at(@"both.bin")];
	std::vector<RgfxArchiveSection> sections;
	CHECK(rgfxReadArchiveFile(static_cast<const uint8_t *>(both.bytes), both.length, sections) ==
	      RgfxArchiveFileKind::Ours);
	CHECK(sections.size() == 2);
	CHECK(sections[0].architecture == "arch_a" && sections[0].entries.size() == 3);
	CHECK(sections[1].architecture == "arch_b" && sections[1].entries.size() == 1);
	RgfxBinaryArchive *backA = archive(at(@"both.bin"), @"arch_a", &e);
	CHECK([RgfxBinaryArchive rgfxMissOfComputePipeline:compute(function(@"a"), @[ backA ])] == nil);

	// A Metal archive (its fat header, every slice inside the file) is another GPU's.
	std::vector<uint8_t> fat = {0xcb, 0xfe, 0xba, 0xbe, 0, 0, 0, 1, 1, 0, 0, 0x17, 0, 0, 0, 0xd,
	                            0,    0,    0,    28,   0, 0, 0, 4, 0, 0, 0, 0,    1, 2, 3, 4};
	[[NSData dataWithBytes:fat.data() length:fat.size()] writeToURL:at(@"metal.bin") atomically:NO];
	RgfxBinaryArchive *metal = archive(at(@"metal.bin"), @"arch_a", &e);
	CHECK(metal != nil && e == nil);
	CHECK(isMiss([RgfxBinaryArchive rgfxMissOfComputePipeline:compute(function(@"a"), @[ metal ])],
	             @"a", @"compute"));
	CHECK(![metal serializeToURL:at(@"metal-out.bin") error:&e]);
	CHECK(isArchiveError(e, MTLBinaryArchiveErrorCompilationFailure, @"Nothing to serialize"));
	// A fat file of the magic 0xcafebabe (a framework's binary.metallib) is one too.
	std::vector<uint8_t> cafe = fat;
	cafe[0] = 0xca;
	[[NSData dataWithBytes:cafe.data() length:cafe.size()] writeToURL:at(@"cafe.bin") atomically:NO];
	RgfxBinaryArchive *universal = archive(at(@"cafe.bin"), @"arch_a", &e);
	CHECK(universal != nil && e == nil);
	CHECK(universal.keys.count == 0);
	CHECK(isMiss([RgfxBinaryArchive rgfxMissOfComputePipeline:compute(function(@"a"), @[ universal ])],
	             @"a", @"compute"));
	// Cut short, it is no archive.
	[[NSData dataWithBytes:fat.data() length:fat.size() - 2] writeToURL:at(@"cut.bin") atomically:NO];
	CHECK(archive(at(@"cut.bin"), @"arch_a", &e) == nil);
	CHECK(e.code == MTLBinaryArchiveErrorInvalidFile);
}

// Its keys, one per entry of this architecture, and its options, the descriptor's as Metal's archive
// answers them (measured: macOS 27, M3 Max).
void test_archive_keys_and_options()
{
	NSError *e = nil;
	RgfxBinaryArchive *a = archive(nil, @"arch_a", &e);
	CHECK(a.keys.count == 0);
	CHECK(a.options == 10); // the descriptor's default, 2
	CHECK([a addComputePipelineFunctionsWithDescriptor:compute(function(@"a"), nil) error:&e]);
	CHECK(a.keys.count == 1);
	CHECK([a addComputePipelineFunctionsWithDescriptor:compute(function(@"a"), nil) error:&e]);
	CHECK(a.keys.count == 1);
	CHECK([a addRenderPipelineFunctionsWithDescriptor:render(nil) error:&e]);
	CHECK(a.keys.count == 3);
	MTLRenderPipelineDescriptor *r = render(nil);
	r.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA16Float; // a fragment function's key
	CHECK([a addRenderPipelineFunctionsWithDescriptor:r error:&e]);
	CHECK(a.keys.count == 4);
	NSArray *keys = a.keys, *again = a.keys;
	CHECK(keys != again && [keys isEqualToArray:again]);
	CHECK([keys[0] hash] == [again[0] hash] && ![keys[0] isEqual:keys[1]]);
	id copied = [(id<NSCopying>)keys[0] copyWithZone:nil];
	CHECK(copied != keys[0] && [copied isEqual:keys[0]] && [copied hash] == [keys[0] hash]);
	CHECK([[copied description] containsString:@" Key: "]);
	CHECK([[NSSet setWithArray:keys] count] == 4);

	// Loaded back, it answers the same keys; another architecture's archive answers none of them.
	CHECK([a serializeToURL:at(@"keys.bin") error:&e]);
	RgfxBinaryArchive *loaded = archive(at(@"keys.bin"), @"arch_a", &e);
	CHECK([[NSSet setWithArray:loaded.keys] isEqualToSet:[NSSet setWithArray:keys]]);
	CHECK(loaded.options == 10);
	CHECK(archive(at(@"keys.bin"), @"arch_b", &e).keys.count == 0);

	// The descriptor's options, with 8 added where it has any.
	SEL setOptions = NSSelectorFromString(@"setOptions:");
	MTLBinaryArchiveDescriptor *d = [MTLBinaryArchiveDescriptor new];
	CHECK([d respondsToSelector:setOptions]);
	for (NSUInteger o : {0ul, 1ul, 2ul, 5ul, 0x10ul, 0xfful}) {
		((void (*)(id, SEL, NSUInteger))objc_msgSend)(d, setOptions, o);
		RgfxBinaryArchive *b = [RgfxBinaryArchive archiveWithDescriptor:d
		                                                         device:nil
		                                                   architecture:@"arch_a"
		                                                          error:&e];
		CHECK(b.options == (o == 0 ? 0 : (o | 8)));
	}
}

// A function made with MTLFunctionOptionFailOnBinaryArchiveMiss and an archive misses as on the x86
// Macs (oracle_bundle failmiss and failmiss2, AMD 5300M and Intel 630, macOS 26.7.1): a kernel the
// archives do not hold by its name and constants' values, however it was added; a visible function
// asked with MTLFunctionOptionCompileToBinary that was not added with it; no other function.
void test_function_misses()
{
	NSError *e = nil;
	RgfxBinaryArchive *a = archive(nil, @"arch_a", &e);
	RgfxBinaryArchive *empty = archive(nil, @"arch_a", &e);
	auto typed = [](RgfxFunction *f, MTLFunctionType type) {
		[f rgfxSetFunctionType:type];
		return f;
	};
	auto miss = [](RgfxFunction *f, MTLFunctionOptions options, NSArray *archives) {
		return [RgfxBinaryArchive rgfxMissOfFunction:f options:options archives:archives];
	};
	RgfxFunction *kernel = typed(function(@"div_kernel", 2), MTLFunctionTypeKernel);
	RgfxFunction *other = typed(function(@"div_kernel", 3), MTLFunctionTypeKernel);
	RgfxFunction *visible = typed(function(@"div_visible"), MTLFunctionTypeVisible);
	RgfxFunction *vertex = typed(function(@"div_vertex"), MTLFunctionTypeVertex);
	[a rgfxAddFunction:kernel options:MTLFunctionOptionCompileToBinary];
	[a rgfxAddFunction:visible options:MTLFunctionOptionNone];
	const MTLFunctionOptions fail = MTLFunctionOptionFailOnBinaryArchiveMiss;
	const MTLFunctionOptions binary = fail | MTLFunctionOptionCompileToBinary;

	CHECK(miss(kernel, fail, @[ a ]) == nil && miss(kernel, binary, @[ a ]) == nil);
	CHECK(miss(kernel, fail, @[ empty, a ]) == nil);
	NSError *m = miss(other, fail, @[ a ]);
	CHECK([m.domain isEqualToString:MTLLibraryErrorDomain] &&
	      m.code == MTLLibraryErrorFunctionNotFound &&
	      [m.localizedDescription
	          isEqualToString:@"Failed to find specialized function \"div_kernel\" in binary archive"]);
	CHECK(miss(kernel, binary, @[ empty ]).code == MTLLibraryErrorFunctionNotFound);
	// No archive named, or no failing on a miss: made.
	CHECK(miss(other, fail, @[]) == nil && miss(other, fail, nil) == nil);
	CHECK(miss(other, MTLFunctionOptionCompileToBinary, @[ a ]) == nil);

	CHECK(miss(visible, fail, @[ empty ]) == nil);
	m = miss(visible, binary, @[ a ]);
	CHECK([m.domain isEqualToString:MTLLibraryErrorDomain] &&
	      m.code == MTLLibraryErrorCompileFailure &&
	      [m.localizedDescription isEqualToString:@"Compilation failed: \n\nFailed to find compiled "
	                                              @"function in binary archives."]);
	CHECK(a.keys.count == 2);
	[a rgfxAddFunction:visible options:MTLFunctionOptionCompileToBinary];
	CHECK(a.keys.count == 3 && miss(visible, binary, @[ a ]) == nil);

	CHECK(miss(vertex, binary, @[ empty ]) == nil && miss(nil, binary, @[ empty ]) == nil);
}

} // namespace

void rgfxArchiveTests()
{
	g_dir = [NSTemporaryDirectory()
	    stringByAppendingPathComponent:[NSString stringWithFormat:@"rgfx-archive-tests-%d", getpid()]];
	[[NSFileManager defaultManager] removeItemAtPath:g_dir error:nil];
	[[NSFileManager defaultManager] createDirectoryAtPath:g_dir
	                          withIntermediateDirectories:YES
	                                           attributes:nil
	                                                error:nil];
	g_vs = function(@"tri_vert");
	g_fs = function(@"tri_frag");
	g_vs2 = function(@"render_vdesc");
	test_archive_making();
	test_archive_adding();
	test_archive_lookups();
	test_archive_files();
	test_archive_keys_and_options();
	test_function_misses();
	[[NSFileManager defaultManager] removeItemAtPath:g_dir error:nil];
}
