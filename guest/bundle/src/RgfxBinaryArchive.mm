/*
 * RgfxBinaryArchive: see RgfxBinaryArchive.h. Our own ObjC++; no Apple private SPI.
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

#include <algorithm>
#include <mutex>
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

// A function no pipeline create looks up (a tile, object or mesh function, or one added alone):
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

- (BOOL)addMeshRenderPipelineFunctionsWithDescriptor:(MTLMeshRenderPipelineDescriptor *)descriptor
                                               error:(__autoreleasing NSError **)error
{
	NSError *refusal = nil;
	if (descriptor.meshFunction == nil)
		refusal = RgfxMakeError(RgfxMetalErrorInvalidDescriptor,
		                        @"binary archive: a mesh pipeline descriptor with no mesh function");
	else
		refusal = refusalOf(descriptor.objectFunction, @"objectFunction")
		              ?: refusalOf(descriptor.meshFunction, @"meshFunction")
		                     ?: refusalOf(descriptor.fragmentFunction, @"fragmentFunction");
	if (refusal != nil) {
		if (error)
			*error = refusal;
		return NO;
	}
	if (descriptor.objectFunction != nil)
		[self rgfxAdd:recordedKey(kRgfxArchiveObject, (RgfxFunction *)descriptor.objectFunction, {},
		                          0)];
	[self rgfxAdd:recordedKey(kRgfxArchiveMesh, (RgfxFunction *)descriptor.meshFunction, {}, 0)];
	// A mesh pipeline's fragment function: kept apart from a render pipeline's, which a create
	// looks up, as nothing looks a mesh pipeline up.
	if (descriptor.fragmentFunction != nil)
		[self rgfxAdd:recordedKey(kRgfxArchiveMesh, (RgfxFunction *)descriptor.fragmentFunction,
		                          formatsOf(descriptor.colorAttachments),
		                          descriptor.rasterSampleCount)];
	return YES;
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
	// The library answers a name it lacks as Metal's does (MTLLibraryErrorFunctionNotFound).
	NSError *err = nil;
	id<MTLFunction> fn = [library newFunctionWithDescriptor:descriptor error:&err];
	if (fn == nil) {
		if (error)
			*error = err;
		return NO;
	}
	[self rgfxAdd:recordedKey(kRgfxArchiveFunction, (RgfxFunction *)fn, {}, 0)];
	return YES;
}

- (BOOL)addLibraryWithDescriptor:(MTLStitchedLibraryDescriptor *)descriptor
                           error:(__autoreleasing NSError **)error
{
	(void)descriptor;
	RGFX_UNSUPPORTED_ERROR(error, "stitched libraries (function pointers; the device answers "
	                              "supportsFunctionPointers NO)");
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

@end
