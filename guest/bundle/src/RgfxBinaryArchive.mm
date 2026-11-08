/*
 * RgfxBinaryArchive: see RgfxBinaryArchive.h. Our own ObjC++; the one Apple private member it
 * reads is the descriptor's options, where the descriptor answers it.
 *
 * The keys, as Metal's miss and hit answers show what it keys each compile by (measured on the M3
 * Max, macOS 27, archive_probe.m), every function by its bitcode's hash, its name and the values
 * of its function constants:
 * - a compute function: with maxTotalThreadsPerThreadgroup and supportIndirectCommandBuffers;
 * - a render pipeline's vertex function: with inputPrimitiveTopology, rasterizationEnabled and
 *   maxVertexAmplificationCount;
 * - its fragment function: with each color attachment's pixel format, write mask and blending
 *   (its factors and operations where blending is on), rasterSampleCount, alpha to coverage and
 *   to one, supportIndirectCommandBuffers and the vertex descriptor.
 * None holds the label, the depth and stencil formats, the buffers' mutability, shader
 * validation, threadGroupSizeIsMultipleOfThreadExecutionWidth or maxCallStackDepth: Metal hits
 * whatever they are. A render pipeline looks its fragment function up first, then its vertex
 * function, and its miss names the first function not found; one with no fragment function looks
 * up its vertex function alone.
 */

#import "RgfxBinaryArchive.h"

#import <objc/message.h>
#import <objc/runtime.h>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <set>
#include <vector>

#import "RgfxFunction.h"
#import "RgfxInterop.h"
#import "RgfxLibrary.h"
#import "RgfxMetalError.h"
#import "RgfxUnsupported.h"

#include "RgfxBinaryArchiveFile.hpp"

NSErrorDomain const RgfxBinaryArchiveMissDomain = @"reims-graphics";

namespace {

// MTLVertexDescriptor addresses buffer and attribute indices 0..30; the color attachments are 8.
constexpr NSUInteger kVertexIndices = 31;
constexpr NSUInteger kColorAttachments = 8;

NSError *archiveError(MTLBinaryArchiveError code, NSString *what)
{
	return [NSError errorWithDomain:MTLBinaryArchiveDomain
	                           code:static_cast<NSInteger>(code)
	                       userInfo:@{NSLocalizedDescriptionKey : what}];
}

NSError *invalidURL()
{
	return archiveError(MTLBinaryArchiveErrorInvalidFile, @"Invalid URL");
}

NSError *invalidFormat(NSURL *url)
{
	return archiveError(
	    MTLBinaryArchiveErrorInvalidFile,
	    [NSString stringWithFormat:@"The file %@ has an invalid format.", url.absoluteString]);
}

// The options an archive of `d` answers: the descriptor's, with 8 added where it has any (Metal's
// answer, measured on macOS 27, M3 Max: 0 answers 0, the default 2 answers 10, 0x10 answers
// 0x18, 0xff answers 0xff; what the bit means nothing states). The member is private: 0 where
// the descriptor does not answer it.
NSUInteger optionsOf(MTLBinaryArchiveDescriptor *d)
{
	SEL options = NSSelectorFromString(@"options");
	if (![d respondsToSelector:options])
		return 0;
	const NSUInteger o = ((NSUInteger (*)(id, SEL))objc_msgSend)(d, options);
	return o != 0 ? (o | 8) : 0;
}

const char *kindName(uint8_t kind)
{
	switch (kind) {
	case kRgfxArchiveCompute:
		return "compute";
	case kRgfxArchiveVertex:
		return "vertex";
	case kRgfxArchiveFragment:
		return "fragment";
	case kRgfxArchiveTile:
		return "tile";
	case kRgfxArchiveObject:
		return "object";
	case kRgfxArchiveMesh:
		return "mesh";
	case kRgfxArchiveFunction:
		return "function";
	default:
		return "unknown";
	}
}

// The refusal of a function another device's library made, as a pipeline create refuses it.
NSError *foreignFunction(id<MTLFunction> fn)
{
	return RgfxMakeError(
	    RgfxMetalErrorInvalidDescriptor,
	    [NSString stringWithFormat:@"binary archive: function %@ (%@) is not from a library of "
	                               @"this device",
	                               fn.name, NSStringFromClass([(id)fn class])]);
}

// Metal's words where its descriptor validation asserts on the descriptor (measured).
NSError *validation(NSString *what)
{
	return RgfxMakeError(RgfxMetalErrorInvalidDescriptor, what);
}

// The function's identity: its bitcode's hash, its entry point and its constants in index order.
void addFunction(RgfxArchiveKey &k, RgfxFunction *fn)
{
	k.bytes([fn bitCodeHash]->bytes, sizeof(RgfxBitCodeHash));
	k.text([fn rgfxEntryPointUTF8]);
	const RgfxFunctionConstant *c = [fn rgfxConstants];
	std::vector<RgfxFunctionConstant> constants(c, c + (c == nullptr ? 0 : [fn rgfxConstantCount]));
	std::sort(constants.begin(), constants.end(),
	          [](const RgfxFunctionConstant &a, const RgfxFunctionConstant &b) {
		          return a.index < b.index;
	          });
	k.u32(static_cast<uint32_t>(constants.size()));
	for (const RgfxFunctionConstant &constant : constants) {
		k.u32(constant.index);
		k.u32(constant.data_type);
		k.bytes(constant.value, sizeof(constant.value));
	}
}

RgfxArchiveEntry computeKey(MTLComputePipelineDescriptor *d)
{
	RgfxArchiveKey k(kRgfxArchiveCompute);
	addFunction(k, (RgfxFunction *)d.computeFunction);
	k.u64(d.maxTotalThreadsPerThreadgroup);
	k.u32(d.supportIndirectCommandBuffers);
	return k.finish();
}

RgfxArchiveEntry vertexKey(MTLRenderPipelineDescriptor *d)
{
	RgfxArchiveKey k(kRgfxArchiveVertex);
	addFunction(k, (RgfxFunction *)d.vertexFunction);
	k.u64(static_cast<uint64_t>(d.inputPrimitiveTopology));
	k.u32(d.isRasterizationEnabled);
	k.u64(d.maxVertexAmplificationCount);
	return k.finish();
}

// The color attachments as a fragment function's compile keys them.
template <typename Attachments> void addColors(RgfxArchiveKey &k, Attachments *colors)
{
	for (NSUInteger i = 0; i < kColorAttachments; i++) {
		MTLRenderPipelineColorAttachmentDescriptor *c = colors[i];
		k.u64(static_cast<uint64_t>(c.pixelFormat));
		if (c.pixelFormat == MTLPixelFormatInvalid)
			continue;
		k.u64(c.writeMask);
		k.u32(c.isBlendingEnabled);
		if (!c.isBlendingEnabled)
			continue;
		for (NSUInteger v : {static_cast<NSUInteger>(c.sourceRGBBlendFactor),
		                     static_cast<NSUInteger>(c.destinationRGBBlendFactor),
		                     static_cast<NSUInteger>(c.rgbBlendOperation),
		                     static_cast<NSUInteger>(c.sourceAlphaBlendFactor),
		                     static_cast<NSUInteger>(c.destinationAlphaBlendFactor),
		                     static_cast<NSUInteger>(c.alphaBlendOperation)})
			k.u64(v);
	}
}

RgfxArchiveEntry fragmentKey(MTLRenderPipelineDescriptor *d)
{
	RgfxArchiveKey k(kRgfxArchiveFragment);
	addFunction(k, (RgfxFunction *)d.fragmentFunction);
	addColors(k, d.colorAttachments);
	k.u64(d.rasterSampleCount);
	k.u32(d.isAlphaToCoverageEnabled);
	k.u32(d.isAlphaToOneEnabled);
	k.u32(d.supportIndirectCommandBuffers);
	MTLVertexDescriptor *vd = d.vertexDescriptor;
	for (NSUInteger i = 0; vd != nil && i < kVertexIndices; i++) {
		MTLVertexBufferLayoutDescriptor *l = vd.layouts[i];
		if (l.stride == 0)
			continue;
		k.u64(i);
		k.u64(l.stride);
		k.u64(static_cast<uint64_t>(l.stepFunction));
		k.u64(l.stepRate);
	}
	k.u64(kVertexIndices);
	for (NSUInteger i = 0; vd != nil && i < kVertexIndices; i++) {
		MTLVertexAttributeDescriptor *a = vd.attributes[i];
		if (a.format == MTLVertexFormatInvalid)
			continue;
		k.u64(i);
		k.u64(static_cast<uint64_t>(a.format));
		k.u64(a.offset);
		k.u64(a.bufferIndex);
	}
	return k.finish();
}

// A stitching graph's node, as a stitched library's key holds it: an input node's argument index,
// or a function node's name, arguments and control dependencies, each in turn. A graph deeper
// than any a framework builds keys only its first levels.
void addStitchingNode(RgfxArchiveKey &k, id node, int depth)
{
	if ([node isKindOfClass:[MTLFunctionStitchingInputNode class]]) {
		k.u32(1);
		k.u64(((MTLFunctionStitchingInputNode *)node).argumentIndex);
		return;
	}
	if (![node isKindOfClass:[MTLFunctionStitchingFunctionNode class]] || depth > 64) {
		k.u32(0);
		return;
	}
	MTLFunctionStitchingFunctionNode *f = node;
	k.u32(2);
	k.text(f.name.UTF8String);
	k.u32(static_cast<uint32_t>(f.arguments.count));
	for (id a in f.arguments)
		addStitchingNode(k, a, depth + 1);
	k.u32(static_cast<uint32_t>(f.controlDependencies.count));
	for (id c in f.controlDependencies)
		addStitchingNode(k, c, depth + 1);
}

// What Metal's archive finds a stitched library by (measured: macOS 27, M3 Max): each graph's
// function name, nodes and attributes, and the functions it is made of, in any order and from any
// library over the same bytes; not its options. False where a function is not one of this
// device's, which no archive of it holds.
bool RgfxStitchedKey(MTLStitchedLibraryDescriptor *descriptor, RgfxArchiveEntry &out)
{
	std::vector<RgfxArchiveEntry> functions;
	for (id<MTLFunction> fn in descriptor.functions) {
		if (![(id)fn isKindOfClass:[RgfxFunction class]])
			return false;
		RgfxArchiveKey f(kRgfxArchiveFunction);
		addFunction(f, (RgfxFunction *)fn);
		functions.push_back(f.finish());
	}
	std::sort(functions.begin(), functions.end());
	RgfxArchiveKey k(kRgfxArchiveStitched);
	k.u32(static_cast<uint32_t>(functions.size()));
	for (const RgfxArchiveEntry &f : functions)
		k.bytes(f.key, sizeof(f.key));
	k.u32(static_cast<uint32_t>(descriptor.functionGraphs.count));
	for (MTLFunctionStitchingGraph *g in descriptor.functionGraphs) {
		k.text(g.functionName.UTF8String);
		k.u32(static_cast<uint32_t>(g.attributes.count));
		for (id attribute in g.attributes)
			k.text(class_getName([attribute class]));
		k.u32(static_cast<uint32_t>(g.nodes.count));
		for (MTLFunctionStitchingFunctionNode *n in g.nodes)
			addStitchingNode(k, n, 0);
		addStitchingNode(k, g.outputNode, 0);
	}
	out = k.finish();
	return true;
}

// A function no pipeline create looks up (a tile function, or one added alone):
// its identity, and the formats and sample count it draws at, which tell entries apart.
RgfxArchiveEntry recordedKey(uint8_t kind, RgfxFunction *fn, const std::vector<uint64_t> &formats,
                             NSUInteger samples)
{
	RgfxArchiveKey k(kind);
	addFunction(k, fn);
	k.u32(static_cast<uint32_t>(formats.size()));
	for (uint64_t f : formats)
		k.u64(f);
	k.u64(samples);
	return k.finish();
}

// The key of a function added alone (addFunctionWithDescriptor:library:error:), which a function
// made with MTLFunctionOptionFailOnBinaryArchiveMiss looks up: recordedKey's, and, for a visible
// function added with MTLFunctionOptionCompileToBinary, that it was, as a visible function added
// without it is not found by one made with it, where a kernel is found however it was added
// (measured on the AMD 5300M and the Intel 630 under macOS 26.7.1: oracle_bundle failmiss2).
RgfxArchiveEntry functionKey(RgfxFunction *fn, MTLFunctionOptions options)
{
	RgfxArchiveKey k(kRgfxArchiveFunction);
	addFunction(k, fn);
	k.u32(0); // recordedKey's formats, none
	k.u64(0); // and sample count
	if (fn.functionType == MTLFunctionTypeVisible &&
	    (options & MTLFunctionOptionCompileToBinary) != 0)
		k.u32(1);
	return k.finish();
}

template <typename Attachments> std::vector<uint64_t> formatsOf(Attachments *colors)
{
	std::vector<uint64_t> formats;
	for (NSUInteger i = 0; i < kColorAttachments; i++)
		formats.push_back(static_cast<uint64_t>(colors[i].pixelFormat));
	return formats;
}

// Whether `fn` is one of this device's, or nil where the caller allows none.
bool ours(id<MTLFunction> fn)
{
	return fn == nil || [(id)fn isKindOfClass:[RgfxFunction class]];
}

// The refusal of `fn` in `role` that a pipeline create would answer too: another device's
// function, or one made with no values for its constants.
NSError *refusalOf(id<MTLFunction> fn, NSString *role)
{
	if (!ours(fn))
		return foreignFunction(fn);
	return [RgfxFunction rgfxUnspecialisedError:fn role:role];
}

NSString *hex(const RgfxArchiveEntry &e)
{
	NSMutableString *s = [NSMutableString stringWithCapacity:2 * sizeof(e.key)];
	for (uint8_t b : e.key)
		[s appendFormat:@"%02X", b];
	return s;
}

NSError *miss(id<MTLFunction> fn, const char *pipeline, const RgfxArchiveEntry &e)
{
	// Metal's words, then its expected key, here the device's own, on lines of their own.
	NSString *what = [NSString stringWithFormat:@"Unable to find function \"%@\" (%s pipeline) in "
	                                            @"binary archives. \n Expected key: \n %@ \n ",
	                                            fn.name, pipeline, hex(e)];
	return [NSError errorWithDomain:RgfxBinaryArchiveMissDomain
	                           code:3
	                       userInfo:@{NSLocalizedDescriptionKey : what}];
}

} // namespace

// One entry of an archive, as its keys answer it: equal, with the same hash, to a key of the same
// entry, and a copy is another object equal to it, as Metal's binary keys are (measured).
@interface RgfxBinaryKey : NSObject <NSCopying>
- (instancetype)initWithEntry:(const RgfxArchiveEntry &)entry;
@end

@implementation RgfxBinaryKey {
	RgfxArchiveEntry _entry;
}

- (instancetype)initWithEntry:(const RgfxArchiveEntry &)entry
{
	self = [super init];
	if (self)
		_entry = entry;
	return self;
}

- (BOOL)isEqual:(id)object
{
	if (object == self)
		return YES;
	return [object isKindOfClass:[RgfxBinaryKey class]] && ((RgfxBinaryKey *)object)->_entry == _entry;
}

- (NSUInteger)hash
{
	NSUInteger h = 0;
	std::memcpy(&h, _entry.key, sizeof(h));
	return h ^ _entry.kind;
}

- (id)copyWithZone:(NSZone *)zone
{
	return [[RgfxBinaryKey allocWithZone:zone] initWithEntry:_entry];
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<RgfxBinaryKey: %p> Key: \n %@ %s", self, hex(_entry),
	                                  kindName(_entry.kind)];
}

@end

@interface RgfxBinaryArchive ()
- (BOOL)rgfxHolds:(const RgfxArchiveEntry &)entry;
@end

@implementation RgfxBinaryArchive {
	id<MTLDevice> _device;
	std::string _architecture;
	// Every architecture's entries, as read and added; this device's are the section named
	// `_architecture`, made at the first add.
	std::vector<RgfxArchiveSection> _sections;
	std::mutex _lock;
	NSUInteger _options; // the descriptor's, as Metal's archive answers them (optionsOf)
	std::set<RgfxArchiveEntry> _stitched; // stitched libraries, in memory only (kRgfxArchiveStitched)
}

@synthesize label = _label;

RGFX_LOG_UNRECOGNIZED

+ (instancetype)archiveWithDescriptor:(MTLBinaryArchiveDescriptor *)descriptor
                               device:(id<MTLDevice>)device
                         architecture:(NSString *)architecture
                                error:(__autoreleasing NSError **)error
{
	std::vector<RgfxArchiveSection> sections;
	NSURL *url = descriptor.url;
	if (url != nil) {
		BOOL directory = NO;
		if (!url.isFileURL ||
		    ![[NSFileManager defaultManager] fileExistsAtPath:url.path isDirectory:&directory]) {
			if (error)
				*error = invalidURL();
			return nil;
		}
		NSData *data = directory ? nil : [NSData dataWithContentsOfURL:url];
		if (!directory && data == nil) {
			if (error)
				*error = invalidURL();
			return nil;
		}
		if (directory || rgfxReadArchiveFile(static_cast<const uint8_t *>(data.bytes), data.length,
		                                     sections) == RgfxArchiveFileKind::Invalid) {
			if (error)
				*error = invalidFormat(url);
			return nil;
		}
	}
	RgfxBinaryArchive *a = [[RgfxBinaryArchive alloc] initWithDevice:device
	                                                    architecture:architecture];
	a->_sections = std::move(sections);
	a->_options = optionsOf(descriptor);
	if (error)
		*error = nil;
	return a;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device architecture:(NSString *)architecture
{
	self = [super init];
	if (self) {
		_device = device;
		_architecture = architecture.UTF8String ?: "";
	}
	return self;
}

- (id<MTLDevice>)device
{
	return _device;
}

// This device's section, made where there is none.
- (RgfxArchiveSection &)rgfxOwnSection
{
	for (RgfxArchiveSection &s : _sections)
		if (s.architecture == _architecture)
			return s;
	_sections.push_back(RgfxArchiveSection{_architecture, {}});
	return _sections.back();
}

- (void)rgfxAdd:(const RgfxArchiveEntry &)entry
{
	std::lock_guard<std::mutex> hold(_lock);
	[self rgfxOwnSection].entries.insert(entry);
}

- (NSArray<id<NSCopying>> *)keys
{
	std::lock_guard<std::mutex> hold(_lock);
	NSMutableArray<id<NSCopying>> *keys = [NSMutableArray array];
	for (const RgfxArchiveSection &s : _sections)
		if (s.architecture == _architecture)
			for (const RgfxArchiveEntry &e : s.entries)
				[keys addObject:[[RgfxBinaryKey alloc] initWithEntry:e]];
	return [keys copy];
}

- (NSUInteger)options
{
	return _options;
}

- (BOOL)rgfxHolds:(const RgfxArchiveEntry &)entry
{
	std::lock_guard<std::mutex> hold(_lock);
	for (const RgfxArchiveSection &s : _sections)
		if (s.architecture == _architecture)
			return s.entries.count(entry) != 0;
	return NO;
}

// --- adding ------------------------------------------------------------------

- (BOOL)addComputePipelineFunctionsWithDescriptor:(MTLComputePipelineDescriptor *)descriptor
                                            error:(__autoreleasing NSError **)error
{
	NSError *refusal = nil;
	if (descriptor.computeFunction == nil)
		refusal = validation(@"Compute Pipeline Descriptor Validation\ncomputeFunction must not be "
		                     @"nil.\n");
	else
		refusal = refusalOf(descriptor.computeFunction, @"computeFunction");
	if (refusal != nil) {
		if (error)
			*error = refusal;
		return NO;
	}
	[self rgfxAdd:computeKey(descriptor)];
	return YES;
}

- (BOOL)addRenderPipelineFunctionsWithDescriptor:(MTLRenderPipelineDescriptor *)descriptor
                                           error:(__autoreleasing NSError **)error
{
	NSError *refusal = nil;
	if (descriptor.vertexFunction == nil)
		refusal = validation(@"Render Pipeline Descriptor Validation\nvertexFunction must not be "
		                     @"nil.\n");
	else
		refusal = refusalOf(descriptor.vertexFunction, @"vertexFunction")
		              ?: refusalOf(descriptor.fragmentFunction, @"fragmentFunction");
	if (refusal != nil) {
		if (error)
			*error = refusal;
		return NO;
	}
	[self rgfxAdd:vertexKey(descriptor)];
	if (descriptor.fragmentFunction != nil)
		[self rgfxAdd:fragmentKey(descriptor)];
	return YES;
}

- (BOOL)addTileRenderPipelineFunctionsWithDescriptor:(MTLTileRenderPipelineDescriptor *)descriptor
                                               error:(__autoreleasing NSError **)error
{
	NSError *refusal = nil;
	if (descriptor.tileFunction == nil)
		refusal = validation(@"Tile Render Pipeline Descriptor Validation\ntileFunction must not "
		                     @"be nil\ntileFunction is associated with a different device\n");
	else
		refusal = refusalOf(descriptor.tileFunction, @"tileFunction");
	if (refusal != nil) {
		if (error)
			*error = refusal;
		return NO;
	}
	[self rgfxAdd:recordedKey(kRgfxArchiveTile, (RgfxFunction *)descriptor.tileFunction,
	                          formatsOf(descriptor.colorAttachments), descriptor.rasterSampleCount)];
	return YES;
}

// A mesh pipeline: refused, with or without its functions, as an x86 Mac's archive refuses every
// one (measured: guest/bundle/tools/oracle/oracle_names on the AMD Radeon Pro 5300M and the Intel
// UHD Graphics 630, macOS 26.7.1). So a mesh pipeline created with
// MTLPipelineOptionFailOnBinaryArchiveMiss misses in every archive (rgfxMissOfMeshRenderPipeline:).
- (BOOL)addMeshRenderPipelineFunctionsWithDescriptor:(MTLMeshRenderPipelineDescriptor *)descriptor
                                               error:(__autoreleasing NSError **)error
{
	(void)descriptor;
	if (error)
		*error = archiveError(MTLBinaryArchiveErrorCompilationFailure,
		                      @"Compiling mesh pipelines for binary archives is not supported!");
	return NO;
}

- (BOOL)addFunctionWithDescriptor:(MTLFunctionDescriptor *)descriptor
                          library:(id<MTLLibrary>)library
                            error:(__autoreleasing NSError **)error
{
	if (![(id)library isKindOfClass:[RgfxLibrary class]]) {
		if (error)
			*error = RgfxMakeError(RgfxMetalErrorInvalidDescriptor,
			                       @"binary archive: the library is not one of this device's");
		return NO;
	}
	// The library answers a name it lacks as Metal's does (MTLLibraryErrorFunctionNotFound). The
	// function is made as it is to be kept, not looked up in the archives the descriptor names.
	MTLFunctionDescriptor *made = [descriptor copy];
	made.binaryArchives = nil;
	NSError *err = nil;
	id<MTLFunction> fn = [library newFunctionWithDescriptor:made error:&err];
	if (fn == nil) {
		if (error)
			*error = err;
		return NO;
	}
	[self rgfxAddFunction:(RgfxFunction *)fn options:descriptor.options];
	return YES;
}

- (void)rgfxAddFunction:(RgfxFunction *)fn options:(MTLFunctionOptions)options
{
	[self rgfxAdd:functionKey(fn, options)];
}

// Metal's archive adds a stitched library by stitching it on its device into the archive
// (newLibraryWithStitchedDescriptor:destinationBinaryArchive:error:, measured: macOS 27), so it
// answers what that stitch answers: NO and Metal's error for a graph Metal cannot stitch, YES for
// one it stitched, which a create that fails on a miss then finds.
- (BOOL)addLibraryWithDescriptor:(MTLStitchedLibraryDescriptor *)descriptor
                           error:(__autoreleasing NSError **)error
{
	if (![(id)_device conformsToProtocol:@protocol(RgfxStitchingDevice)]) {
		RGFX_UNSUPPORTED_ERROR(error, "stitched libraries with no device to stitch them");
		return NO;
	}
	NSError *err = nil;
	id<MTLLibrary> lib = [(id<RgfxStitchingDevice>)_device newLibraryWithStitchedDescriptor:descriptor
	                                                                 destinationBinaryArchive:self
	                                                                                    error:&err];
	if (error)
		*error = lib != nil ? nil : err;
	return lib != nil;
}

- (void)rgfxAddStitched:(MTLStitchedLibraryDescriptor *)descriptor
{
	RgfxArchiveEntry key;
	if (!RgfxStitchedKey(descriptor, key))
		return;
	std::lock_guard<std::mutex> hold(_lock);
	_stitched.insert(key);
}

+ (BOOL)rgfxStitched:(MTLStitchedLibraryDescriptor *)descriptor heldBy:(NSArray *)archives
{
	RgfxArchiveEntry key;
	if (!RgfxStitchedKey(descriptor, key))
		return NO;
	for (id a in archives) {
		if (![a isKindOfClass:[RgfxBinaryArchive class]])
			continue;
		RgfxBinaryArchive *archive = a;
		std::lock_guard<std::mutex> hold(archive->_lock);
		if (archive->_stitched.count(key) != 0)
			return YES;
	}
	return NO;
}

// --- writing -----------------------------------------------------------------

- (BOOL)serializeToURL:(NSURL *)url error:(__autoreleasing NSError **)error
{
	// Metal answers a URL it cannot write to before it looks at what there is to write
	// (measured: an empty archive to a missing directory answers Invalid URL).
	BOOL directory = NO;
	NSString *parent = url.URLByDeletingLastPathComponent.path;
	if (url == nil || !url.isFileURL || parent == nil ||
	    ![[NSFileManager defaultManager] fileExistsAtPath:parent isDirectory:&directory] ||
	    !directory) {
		if (error)
			*error = invalidURL();
		return NO;
	}
	std::vector<uint8_t> bytes;
	{
		std::lock_guard<std::mutex> hold(_lock);
		std::vector<RgfxArchiveSection> kept;
		for (const RgfxArchiveSection &s : _sections)
			if (!s.entries.empty())
				kept.push_back(s);
		if (kept.empty()) {
			if (error)
				*error = archiveError(MTLBinaryArchiveErrorCompilationFailure,
				                      @"Nothing to serialize");
			return NO;
		}
		bytes = rgfxWriteArchiveFile(kept);
	}
	NSData *data = [NSData dataWithBytes:bytes.data() length:bytes.size()];
	if (![data writeToURL:url atomically:YES]) {
		if (error)
			*error = invalidURL();
		return NO;
	}
	if (error)
		*error = nil;
	return YES;
}

// --- looking up --------------------------------------------------------------

// Whether one of `archives` holds `entry`.
static BOOL rgfxAnyHolds(NSArray *archives, const RgfxArchiveEntry &entry)
{
	for (id a in archives)
		if ([a isKindOfClass:[RgfxBinaryArchive class]] && [(RgfxBinaryArchive *)a rgfxHolds:entry])
			return YES;
	return NO;
}

+ (NSError *)rgfxMissOfRenderPipeline:(MTLRenderPipelineDescriptor *)descriptor
{
	NSArray *archives = descriptor.binaryArchives;
	if (archives.count == 0 || !ours(descriptor.vertexFunction) ||
	    !ours(descriptor.fragmentFunction) || descriptor.vertexFunction == nil)
		return nil;
	if (descriptor.fragmentFunction != nil) {
		RgfxArchiveEntry fragment = fragmentKey(descriptor);
		if (!rgfxAnyHolds(archives, fragment))
			return miss(descriptor.fragmentFunction, "render", fragment);
	}
	RgfxArchiveEntry vertex = vertexKey(descriptor);
	if (!rgfxAnyHolds(archives, vertex))
		return miss(descriptor.vertexFunction, "render", vertex);
	return nil;
}

+ (NSError *)rgfxMissOfFunction:(RgfxFunction *)fn
                         options:(MTLFunctionOptions)options
                        archives:(NSArray *)archives
{
	if ((options & MTLFunctionOptionFailOnBinaryArchiveMiss) == 0 || archives.count == 0 ||
	    fn == nil)
		return nil;
	switch (fn.functionType) {
	case MTLFunctionTypeKernel:
		if (rgfxAnyHolds(archives, functionKey(fn, MTLFunctionOptionNone)))
			return nil;
		return [NSError
		    errorWithDomain:MTLLibraryErrorDomain
		               code:MTLLibraryErrorFunctionNotFound
		           userInfo:@{
			           NSLocalizedDescriptionKey : [NSString
			               stringWithFormat:@"Failed to find specialized function \"%@\" in binary "
			                                @"archive",
			                                fn.name]
		           }];
	case MTLFunctionTypeVisible:
		if ((options & MTLFunctionOptionCompileToBinary) == 0 ||
		    rgfxAnyHolds(archives, functionKey(fn, MTLFunctionOptionCompileToBinary)))
			return nil;
		return [NSError errorWithDomain:MTLLibraryErrorDomain
		                           code:MTLLibraryErrorCompileFailure
		                       userInfo:@{
			                       NSLocalizedDescriptionKey :
			                           @"Compilation failed: \n\nFailed to find compiled function in "
			                           @"binary archives."
		                       }];
	default:
		return nil;
	}
}

// The x86 Macs' miss, in their compiler's domain and words, without the key they name: an
// `MTLBinaryKey` of zeros, which no archive of theirs holds (measured: oracle_mesh archive-miss,
// archive-miss-norast, archive-miss-object and archive-miss-object-norast on the 5300M and the
// 630, macOS 26.7.1: CompilerError 2, `Unable to find function "f_col" in binary archives. \n
// Expected <MTLBinaryKey: 0x…> Legacy key:\n<64 zeros> Single SHA256 \n `). With no archive
// named, the 5300M and the 630 end the process with SIGSEGV and the M3 Max makes the pipeline;
// the device makes it.
+ (NSError *)rgfxMissOfMeshRenderPipeline:(MTLMeshRenderPipelineDescriptor *)descriptor
{
	if (descriptor.binaryArchives.count == 0 || descriptor.meshFunction == nil ||
	    !ours(descriptor.objectFunction) || !ours(descriptor.meshFunction) ||
	    !ours(descriptor.fragmentFunction))
		return nil;
	id<MTLFunction> named = descriptor.fragmentFunction ?: descriptor.meshFunction;
	NSString *what = [NSString
	    stringWithFormat:@"Unable to find function \"%@\" in binary archives. \n ", named.name];
	return [NSError errorWithDomain:@"CompilerError"
	                           code:2
	                       userInfo:@{NSLocalizedDescriptionKey : what}];
}

+ (NSError *)rgfxMissOfComputePipeline:(MTLComputePipelineDescriptor *)descriptor
{
	NSArray *archives = descriptor.binaryArchives;
	if (archives.count == 0 || descriptor.computeFunction == nil || !ours(descriptor.computeFunction))
		return nil;
	RgfxArchiveEntry compute = computeKey(descriptor);
	if (!rgfxAnyHolds(archives, compute))
		return miss(descriptor.computeFunction, "compute", compute);
	return nil;
}

// --- MTLBinaryArchiveSPI -----------------------------------------------------------------
// The members of Metal's binary-archive SPI that the AMD 5300M's and the Intel 630's archives both
// answer (measured with respondsToSelector:, macOS 26.7.1, tools/oracle/oracle_spi_members.m):
// both are Metal's own archive, _MTLBinaryArchive (measured: their class), whose members these
// follow where the archive's entries allow (read from code, _MTLBinaryArchive, macOS 27.0.1).

// The description Metal's archive gives: the object's own description, then its label and its
// count of entries, each on a line of its own, indented by `indent` + 4 spaces (measured: both x86
// Macs' fresh archive, "label = <none>" and "count = 0").
- (NSString *)formattedDescription:(NSUInteger)indent
{
	NSString *pad = [@"\n" stringByPaddingToLength:indent + 4 withString:@" " startingAtIndex:0];
	NSString *label = self.label;
	NSArray *fields = @[ pad, @"label =", label ?: @"<none>", pad, @"count =", @(self.keys.count) ];
	return [NSString stringWithFormat:@"%@%@", [super description], [fields componentsJoinedByString:@" "]];
}

// Metal's archive answers the hash keys of its functions, as pointers into its own tables: an
// empty list for a fresh one (measured: both x86 Macs). The archive's entries are keyed as the
// host compiles them, with none of Metal's keys, so it answers an empty list, logged.
- (NSArray *)archiveFunctionIds
{
	RGFX_UNSUPPORTED("an archive's function IDs");
	return [NSMutableArray array];
}

// Metal's pipeline collection of the archive: none here (null, as a fresh archive of both x86 Macs
// answers; measured).
- (void *)pipelineCollection
{
	return nullptr;
}

// Metal walks the archives of its own backing file or pipeline collection; the archive has
// neither, so these walk nothing.
- (BOOL)enumerateArchivesFromBackingFile:(id)enumerator
{
	(void)enumerator;
	RGFX_UNSUPPORTED("Metal's archive backing files");
	return NO;
}

- (BOOL)enumerateArchivesFromPipelineCollection:(id)enumerator
{
	(void)enumerator;
	RGFX_UNSUPPORTED("Metal's archive backing files");
	return NO;
}

// A copy of the archive recompiled for another target: the host compiles for its own GPU alone.
- (BOOL)recompileToArchiveWithURL:(NSURL *)url error:(__autoreleasing NSError **)error
{
	(void)url;
	RGFX_UNSUPPORTED_ERROR(error, "recompiling an archive for another target");
	return NO;
}

// A pipeline's descriptor kept in Metal's pipeline collection, without its functions: none here
// (the add…FunctionsWithDescriptor: members keep a pipeline's functions).
- (BOOL)storeComputePipelineDescriptor:(MTLComputePipelineDescriptor *)descriptor
{
	(void)descriptor;
	RGFX_UNSUPPORTED("Metal's archive pipeline collections");
	return NO;
}

- (BOOL)storeRenderPipelineDescriptor:(MTLRenderPipelineDescriptor *)descriptor
{
	(void)descriptor;
	RGFX_UNSUPPORTED("Metal's archive pipeline collections");
	return NO;
}

- (BOOL)storeTileRenderPipelineDescriptor:(MTLTileRenderPipelineDescriptor *)descriptor
{
	(void)descriptor;
	RGFX_UNSUPPORTED("Metal's archive pipeline collections");
	return NO;
}

- (BOOL)storeMeshRenderPipelineDescriptor:(MTLMeshRenderPipelineDescriptor *)descriptor
{
	(void)descriptor;
	RGFX_UNSUPPORTED("Metal's archive pipeline collections");
	return NO;
}

@end
