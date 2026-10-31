/*
 * RgfxMetalDevice (Pipelines): see RgfxMetalDevice+Pipelines.h. Slice 2 of the device
 * shell. Our own ObjC++; the only Apple-private names are in RgfxMetalPrivate.h.
 */

#import "RgfxMetalDevice+Pipelines.h"

#import "RgfxArgumentEncoder.h"
#import "RgfxBinaryArchive.h"
#import "RgfxComputePipelineState.h"
#import "RgfxLibrary.h"
#import "RgfxMetalError.h"
#import "RgfxRenderPipelineState.h"
#import "RgfxUnsupported.h"

#include "RgfxLibraryKey.hpp"
#include "RgfxRegion.hpp"
#include "RgfxSerializer.hpp"

#include <unistd.h>

@implementation RgfxMetalDevice (Pipelines)

// --- libraries ---------------------------------------------------------------

namespace {

// How long a library create waits for room on the device ring before it reports the
// ring full: the host drains the ring as it runs, so a full ring is transient.
constexpr useconds_t kLibraryRetrySleepUs = 1000;
constexpr int kLibraryRetries = 2000; // about two seconds

// The largest library create sent inline in the ring (header and bytes): a quarter of
// the device ring, so one create never holds most of it. Larger libraries go in a
// region block (LIBRARY_FROM_REGION).
constexpr uint64_t kLibraryInlineMax = RGFX_CONTROL_RING_DATA_SIZE / 4;

// A universal .metallib's slice of generic AIR bitcode: the cputype its fat_arch entry
// carries. measured: macOS 26.6.1 x86_64 guest, QuartzCore's default.metallib (19
// slices, one of them this, the rest code compiled for particular GPUs).
constexpr uint32_t kMetallibAirCpuType = 0x0100'0017;

uint32_t rdBe32(const uint8_t *p)
{
	return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | p[3];
}

uint64_t rdBe64(const uint8_t *p)
{
	return (uint64_t(rdBe32(p)) << 32) | rdBe32(p + 4);
}

// Narrows a universal .metallib (big-endian fat header, magic CA FE BA BE, or CA FE BA
// BF with 64-bit offsets) to its generic AIR slice, the only one the host compiles:
// for QuartzCore's library that is 1.6 MB of 160 MB. Other bytes are left as they are.
void selectAirSlice(const void *&bytes, NSUInteger &length)
{
	const uint8_t *p = static_cast<const uint8_t *>(bytes);
	if (p == nullptr || length < 8)
		return;
	const uint32_t magic = rdBe32(p);
	if (magic != 0xCAFE'BABE && magic != 0xCAFE'BABF)
		return;
	const bool wide = magic == 0xCAFE'BABF;
	const uint64_t entrySize = wide ? 32 : 20;
	const uint32_t count = rdBe32(p + 4);
	if (count > (length - 8) / entrySize)
		return;
	for (uint32_t i = 0; i < count; i++) {
		const uint8_t *e = p + 8 + i * entrySize;
		if (rdBe32(e) != kMetallibAirCpuType)
			continue;
		const uint64_t offset = wide ? rdBe64(e + 8) : rdBe32(e + 8);
		const uint64_t size = wide ? rdBe64(e + 16) : rdBe32(e + 12);
		if (offset > length || size > length - offset || size == 0)
			return;
		bytes = p + offset;
		length = static_cast<NSUInteger>(size);
		return;
	}
}

} // namespace

// Every library-from-bytes entry point funnels through here. A .metallib (or a fat file's AIR
// slice) is first looked up by its content key, which finds a library the device made for any
// process with no bytes sent; otherwise the bytes go as one LIBRARY_CREATE (waiting for room
// while the ring is momentarily full) or stream through region blocks, with the key, and the
// device's answer is waited for, so a library it refuses is nil and an NSError here, as Metal's.
- (nullable id<MTLLibrary>)rgfxLibraryWithBytes:(const void *)bytes
                                         length:(NSUInteger)length
                                          error:(__autoreleasing NSError **)error
{
	selectAirSlice(bytes, length);
	RgfxSerializer *ser = [self serializer];
	if (ser == nullptr) {
		if (error)
			*error = RgfxMakeError(RgfxMetalErrorNoRoute, @"library: no serialiser route");
		return nil;
	}
	uint8_t key[32];
	const bool keyed = rgfxLibraryKey(static_cast<const uint8_t *>(bytes), length, key);
	if (keyed) {
		id<MTLLibrary> found = [RgfxLibrary libraryFoundWithKey:key
		                                                  bytes:bytes
		                                                 length:length
		                                                 device:self
		                                             serializer:ser];
		if (found != nil)
			return found;
	}
	// A small library travels inline in its create; a larger one streams through region
	// blocks that its chunks name, so no library is bounded by the ring's or region's size.
	const bool inlineCreate =
	    RGFX_LIBRARY_BYTES_OFFSET + static_cast<uint64_t>(length) + 8 <= kLibraryInlineMax;
	RgfxRegion *region = [self region];
	if (!inlineCreate && region == nullptr) {
		if (error)
			*error = RgfxMakeError(RgfxMetalErrorNoRoute,
			                       @"library: no device region for a large library");
		return nil;
	}
	id<MTLLibrary> lib = nil;
	uint32_t status = kRgfxNoAnswer;
	if (inlineCreate) {
		// Retried only while the create did not post; an answered refusal is final.
		for (int i = 0; i < kLibraryRetries && lib == nil && status == kRgfxNoAnswer; i++) {
			lib = [RgfxLibrary libraryWithData:bytes
			                            length:length
			                               key:keyed ? key : nullptr
			                            device:self
			                        serializer:ser
			                            status:&status];
			if (lib == nil && status == kRgfxNoAnswer)
				usleep(kLibraryRetrySleepUs);
		}
	} else {
		// Streamed through the process's double-buffered staging window.
		lib = [RgfxLibrary libraryStreamingBytes:bytes
		                                  length:length
		                                     key:keyed ? key : nullptr
		                                  region:region
		                                  device:self
		                              serializer:ser
		                                  status:&status];
	}
	if (lib == nil && error) {
		if (status == kRgfxNoAnswer)
			*error = RgfxMakeError(RgfxMetalErrorRingFull, @"library: the device ring stayed full");
		else
			*error = RgfxMakeError(RgfxMetalErrorRefused,
			                       [NSString stringWithFormat:@"library: the device refused the "
			                                                  @"library (status %u)",
			                                                  status]);
	}
	return lib;
}

- (id<MTLLibrary>)newLibraryWithData:(dispatch_data_t)data
                               error:(__autoreleasing NSError **)error
{
	if (data == nil) {
		if (error)
			*error = RgfxMakeError(RgfxMetalErrorInvalidDescriptor, @"newLibraryWithData: nil data");
		return nil;
	}
	// Map the (possibly non-contiguous) dispatch_data_t to one byte range, then
	// forward it to the host unchanged.
	const void *bytes = nullptr;
	size_t size = 0;
	dispatch_data_t map = dispatch_data_create_map(data, &bytes, &size);
	id<MTLLibrary> lib = [self rgfxLibraryWithBytes:bytes length:size error:error];
	[self rgfxKeepSource:map ofLibrary:lib];
	return lib;
}

// Keeps `source` with `lib` for its shadow (RgfxLibrary): Metal's own library over the same
// bytes, made by the base device class, whose extern functions Core Image's link reads.
- (void)rgfxKeepSource:(dispatch_data_t)source ofLibrary:(id<MTLLibrary>)lib
{
	if (![(id)lib isKindOfClass:[RgfxLibrary class]])
		return;
	__weak RgfxMetalDevice *weakSelf = self;
	[(RgfxLibrary *)lib rgfxSetSource:source
	                      shadowMaker:^id<MTLLibrary>(dispatch_data_t bytes) {
		                      return [weakSelf rgfxMetalLibraryWithData:bytes];
	                      }];
}

// Metal's own library over `data` (measured: macOS 27 and 26.6.1, the base device class's
// newLibraryWithData:error: makes it on our device: an _MTLLibrary answering a Core Image kernel
// library's extern functions); nil where Metal refuses the bytes.
- (id<MTLLibrary>)rgfxMetalLibraryWithData:(dispatch_data_t)data
{
	NSError *error = nil;
	id<MTLLibrary> lib = [super newLibraryWithData:data error:&error];
	if (lib == nil)
		NSLog(@"RGFX: Metal's own library over %zu bytes: %@", dispatch_data_get_size(data),
		      error.localizedDescription);
	return lib;
}

// Core Image's links (measured: macOS 27 and 26.6.1, the base device class's own, which link
// Core Image's extern functions on the CPU): Metal's own link over each function's shadow, its
// library made ours (RgfxLinkImageFilterFunctions). Metal's first link calls the second on
// `self`, which makes the library ours there.
- (id<MTLLibrary>)newLibraryWithImageFilterFunctionsSPI:(NSArray<id<MTLFunction>> *)functions
                                imageFilterFunctionInfo:(const void *)info
                                                  error:(__autoreleasing NSError **)error
{
	return RgfxLinkImageFilterFunctions(
	    functions,
	    ^id<MTLLibrary>(NSArray<id<MTLFunction>> *shadows, NSError *__autoreleasing *e) {
		    return [self rgfxMetalLinkImageFilterFunctions:shadows info:info error:e];
	    },
	    ^id<MTLLibrary>(dispatch_data_t data, NSError *__autoreleasing *e) {
		    return [self newLibraryWithData:data error:e];
	    },
	    error);
}

- (id<MTLLibrary>)newLibraryWithCIFilters:(NSArray<id<MTLFunction>> *)functions
                  imageFilterFunctionInfo:(const void *)info
                                    error:(__autoreleasing NSError **)error
{
	return RgfxLinkImageFilterFunctions(
	    functions,
	    ^id<MTLLibrary>(NSArray<id<MTLFunction>> *shadows, NSError *__autoreleasing *e) {
		    return [self rgfxMetalLinkCIFilters:shadows info:info error:e];
	    },
	    ^id<MTLLibrary>(dispatch_data_t data, NSError *__autoreleasing *e) {
		    return [self newLibraryWithData:data error:e];
	    },
	    error);
}

// The base class's links, which a block cannot reach through `super`.
- (id<MTLLibrary>)rgfxMetalLinkImageFilterFunctions:(NSArray<id<MTLFunction>> *)shadows
                                               info:(const void *)info
                                              error:(__autoreleasing NSError **)error
{
	return [super newLibraryWithImageFilterFunctionsSPI:shadows imageFilterFunctionInfo:info error:error];
}

- (id<MTLLibrary>)rgfxMetalLinkCIFilters:(NSArray<id<MTLFunction>> *)shadows
                                    info:(const void *)info
                                   error:(__autoreleasing NSError **)error
{
	return [super newLibraryWithCIFilters:shadows imageFilterFunctionInfo:info error:error];
}

- (id<MTLLibrary>)newLibraryWithURL:(NSURL *)url error:(__autoreleasing NSError **)error
{
	NSError *readError = nil;
	NSData *data = url ? [NSData dataWithContentsOfURL:url options:NSDataReadingMappedIfSafe
	                                             error:&readError]
	                   : nil;
	if (data == nil) {
		if (error)
			*error = readError ?: RgfxMakeError(RgfxMetalErrorInvalidDescriptor,
			                                    @"newLibraryWithURL: could not read URL");
		return nil;
	}
	id<MTLLibrary> lib = [self rgfxLibraryWithBytes:data.bytes length:data.length error:error];
	dispatch_data_t source = dispatch_data_create(data.bytes, data.length, nil, ^{
		(void)data;
	});
	[self rgfxKeepSource:source ofLibrary:lib];
	return lib;
}

- (id<MTLLibrary>)newLibraryWithFile:(NSString *)filepath error:(__autoreleasing NSError **)error
{
	if (filepath == nil) {
		if (error)
			*error = RgfxMakeError(RgfxMetalErrorInvalidDescriptor, @"newLibraryWithFile: nil path");
		return nil;
	}
	return [self newLibraryWithURL:[NSURL fileURLWithPath:filepath] error:error];
}

- (id<MTLLibrary>)newDefaultLibraryWithBundle:(NSBundle *)bundle
                                        error:(__autoreleasing NSError **)error
{
	// A bundle's default library is its default.metallib resource, as Metal finds it.
	NSURL *url = [bundle URLForResource:@"default" withExtension:@"metallib"];
	if (url == nil) {
		if (error)
			*error = RgfxMakeError(RgfxMetalErrorInvalidDescriptor,
			                       [NSString stringWithFormat:@"no default.metallib in %@",
			                                                  bundle.bundlePath]);
		return nil;
	}
	return [self newLibraryWithURL:url error:error];
}

- (id<MTLLibrary>)newDefaultLibrary
{
	return [self newDefaultLibraryWithBundle:[NSBundle mainBundle] error:nil];
}

- (id<MTLLibrary>)newLibraryWithSource:(NSString *)source
                               options:(MTLCompileOptions *)options
                                 error:(__autoreleasing NSError **)error
{
	// DOCUMENTED UNSUPPORTED. Source→AIR needs the Metal compiler, which runs on the
	// host; the guest shell carries no compiler. We refuse rather than fabricate an
	// empty library — the way the host device-info answers an unsupported query. A
	// host-side compile path would land its own wire message.
	(void)source;
	(void)options;
	if (error)
		*error = RgfxMakeError(RgfxMetalErrorUnsupported,
		                       @"newLibraryWithSource: runtime source compilation is not supported; "
		                       @"provide a prebuilt .metallib via newLibraryWithData:/WithURL:");
	return nil;
}

- (void)newLibraryWithSource:(NSString *)source
                     options:(MTLCompileOptions *)options
           completionHandler:(MTLNewLibraryCompletionHandler)completionHandler
{
	NSError *err = nil;
	id<MTLLibrary> lib = [self newLibraryWithSource:source options:options error:&err];
	if (completionHandler)
		completionHandler(lib, err);
}

// --- render pipelines --------------------------------------------------------

// `state`, or nil and an NSError where the host refused its create: the kext hands the create's
// reply back before the create returns, so a refusal is known here, as Metal's creators know
// theirs.
template <typename State>
static State rgfxUnlessRefused(State state, NSError *__autoreleasing *error)
{
	if (state == nil || ![state rgfxRefused])
		return state;
	if (error)
		*error = RgfxMakeError(RgfxMetalErrorRefused, @"pipeline: the device refused the pipeline");
	return nil;
}

// Creates a render pipeline state and registers it with the device's reply reader, so
// its reflection fills when the host answers this create's seqno. Every render-pipeline
// entry point funnels through here.
- (nullable id<MTLRenderPipelineState>)
    rgfxRenderPipelineForDescriptor:(MTLRenderPipelineDescriptor *)descriptor
                            options:(MTLPipelineOption)options
                              error:(__autoreleasing NSError **)error
{
	// MTLPipelineOptionFailOnBinaryArchiveMiss answers Metal's miss where the descriptor's
	// archives lack one of its functions' compiles; without it a miss compiles as Metal's does.
	if ((options & MTLPipelineOptionFailOnBinaryArchiveMiss) != 0) {
		NSError *miss = [RgfxBinaryArchive rgfxMissOfRenderPipeline:descriptor];
		if (miss != nil) {
			if (error)
				*error = miss;
			return nil;
		}
	}
	RgfxRenderPipelineState *state =
	    [RgfxRenderPipelineState renderPipelineStateWithDescriptor:descriptor
	                                                        device:self
	                                                    serializer:[self serializer]
	                                                         error:error];
	[state rgfxRegisterWithReplyReader:[self replyReader]];
	return rgfxUnlessRefused(state, error);
}

- (id<MTLRenderPipelineState>)
    newRenderPipelineStateWithDescriptor:(MTLRenderPipelineDescriptor *)descriptor
                                   error:(__autoreleasing NSError **)error
{
	return [self rgfxRenderPipelineForDescriptor:descriptor options:MTLPipelineOptionNone error:error];
}

- (id<MTLRenderPipelineState>)
    newRenderPipelineStateWithDescriptor:(MTLRenderPipelineDescriptor *)descriptor
                                 options:(MTLPipelineOption)options
                              reflection:(MTLAutoreleasedRenderPipelineReflection *)reflection
                                   error:(__autoreleasing NSError **)error
{
	// The reflection is answered by the host over the completion ring (reader is a
	// later transport piece), so there is none to hand back synchronously.
	if (reflection)
		*reflection = nil;
	return [self rgfxRenderPipelineForDescriptor:descriptor options:options error:error];
}

- (void)newRenderPipelineStateWithDescriptor:(MTLRenderPipelineDescriptor *)descriptor
                           completionHandler:
                               (MTLNewRenderPipelineStateCompletionHandler)completionHandler
{
	NSError *err = nil;
	id<MTLRenderPipelineState> state = [self newRenderPipelineStateWithDescriptor:descriptor
	                                                                       error:&err];
	if (completionHandler)
		completionHandler(state, err);
}

- (void)newRenderPipelineStateWithDescriptor:(MTLRenderPipelineDescriptor *)descriptor
                                     options:(MTLPipelineOption)options
                           completionHandler:
                               (MTLNewRenderPipelineStateWithReflectionCompletionHandler)
                                   completionHandler
{
	NSError *err = nil;
	id<MTLRenderPipelineState> state = [self rgfxRenderPipelineForDescriptor:descriptor
	                                                                 options:options
	                                                                   error:&err];
	if (completionHandler)
		completionHandler(state, nil, err); // reflection: see the error: variant above
}

// --- compute pipelines -------------------------------------------------------

// Creates a compute pipeline state (from a function or a descriptor) and registers it
// with the device's reply reader so its reflection fills on the host's answer. Every
// compute-pipeline entry point funnels through one of these two.
- (nullable id<MTLComputePipelineState>)rgfxComputePipelineForFunction:(id<MTLFunction>)function
                                                               error:(__autoreleasing NSError **)error
{
	RgfxComputePipelineState *state =
	    [RgfxComputePipelineState computePipelineStateWithFunction:function
	                                                        device:self
	                                                    serializer:[self serializer]
	                                                         error:error];
	[state rgfxRegisterWithReplyReader:[self replyReader]];
	return rgfxUnlessRefused(state, error);
}

- (nullable id<MTLComputePipelineState>)
    rgfxComputePipelineForDescriptor:(MTLComputePipelineDescriptor *)descriptor
                             options:(MTLPipelineOption)options
                               error:(__autoreleasing NSError **)error
{
	// As a render pipeline's (above). A pipeline of a function alone names no archive, so it
	// compiles whatever its options, as Metal's does.
	if ((options & MTLPipelineOptionFailOnBinaryArchiveMiss) != 0) {
		NSError *miss = [RgfxBinaryArchive rgfxMissOfComputePipeline:descriptor];
		if (miss != nil) {
			if (error)
				*error = miss;
			return nil;
		}
	}
	RgfxComputePipelineState *state =
	    [RgfxComputePipelineState computePipelineStateWithDescriptor:descriptor
	                                                          device:self
	                                                      serializer:[self serializer]
	                                                           error:error];
	[state rgfxRegisterWithReplyReader:[self replyReader]];
	return rgfxUnlessRefused(state, error);
}

- (id<MTLComputePipelineState>)newComputePipelineStateWithFunction:(id<MTLFunction>)computeFunction
                                                            error:(__autoreleasing NSError **)error
{
	return [self rgfxComputePipelineForFunction:computeFunction error:error];
}

- (id<MTLComputePipelineState>)
    newComputePipelineStateWithFunction:(id<MTLFunction>)computeFunction
                                options:(MTLPipelineOption)options
                             reflection:(MTLAutoreleasedComputePipelineReflection *)reflection
                                  error:(__autoreleasing NSError **)error
{
	(void)options;
	if (reflection)
		*reflection = nil;
	return [self rgfxComputePipelineForFunction:computeFunction error:error];
}

- (void)newComputePipelineStateWithFunction:(id<MTLFunction>)computeFunction
                          completionHandler:
                              (MTLNewComputePipelineStateCompletionHandler)completionHandler
{
	NSError *err = nil;
	id<MTLComputePipelineState> state = [self newComputePipelineStateWithFunction:computeFunction
	                                                                       error:&err];
	if (completionHandler)
		completionHandler(state, err);
}

- (void)newComputePipelineStateWithFunction:(id<MTLFunction>)computeFunction
                                    options:(MTLPipelineOption)options
                          completionHandler:
                              (MTLNewComputePipelineStateWithReflectionCompletionHandler)
                                  completionHandler
{
	(void)options;
	NSError *err = nil;
	id<MTLComputePipelineState> state = [self newComputePipelineStateWithFunction:computeFunction
	                                                                       error:&err];
	if (completionHandler)
		completionHandler(state, nil, err);
}

- (id<MTLComputePipelineState>)
    newComputePipelineStateWithDescriptor:(MTLComputePipelineDescriptor *)descriptor
                                  options:(MTLPipelineOption)options
                               reflection:(MTLAutoreleasedComputePipelineReflection *)reflection
                                    error:(__autoreleasing NSError **)error
{
	if (reflection)
		*reflection = nil;
	return [self rgfxComputePipelineForDescriptor:descriptor options:options error:error];
}

- (void)newComputePipelineStateWithDescriptor:(MTLComputePipelineDescriptor *)descriptor
                                      options:(MTLPipelineOption)options
                            completionHandler:
                                (MTLNewComputePipelineStateWithReflectionCompletionHandler)
                                    completionHandler
{
	NSError *err = nil;
	id<MTLComputePipelineState> state = [self rgfxComputePipelineForDescriptor:descriptor
	                                                                  options:options
	                                                                    error:&err];
	if (completionHandler)
		completionHandler(state, nil, err);
}

// --- binary archives ---------------------------------------------------------

// The device's archives are its own, keyed by its architecture's name (RgfxBinaryArchive.h).
- (id<MTLBinaryArchive>)newBinaryArchiveWithDescriptor:(MTLBinaryArchiveDescriptor *)descriptor
                                                 error:(__autoreleasing NSError **)error
{
	return [RgfxBinaryArchive archiveWithDescriptor:descriptor
	                                         device:self
	                                   architecture:[self architecture].name ?: @""
	                                          error:error];
}

// --- argument encoders -------------------------------------------------------

// The host lays the arguments out as its platform's encoder does (ARGUMENT_LAYOUT,
// RgfxArgumentEncoder.h); nil where it answers none, as Metal answers descriptors it refuses.
- (id<MTLArgumentEncoder>)
    newArgumentEncoderWithArguments:(NSArray<MTLArgumentDescriptor *> *)arguments
{
	return [RgfxArgumentEncoder encoderOfArguments:arguments
	                                        region:[self region]
	                                        device:self
	                                    serializer:[self serializer]];
}

@end
