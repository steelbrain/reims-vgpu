/*
 * RgfxMetalDevice (Pipelines): see RgfxMetalDevice+Pipelines.h. Slice 2 of the device
 * shell. Our own ObjC++; the only Apple-private names are in RgfxMetalPrivate.h.
 */

#import "RgfxMetalDevice+Pipelines.h"

#import "RgfxArgumentEncoder.h"
#import "RgfxBinaryArchive.h"
#import "RgfxBlockCall.h"
#import "RgfxComputePipelineState.h"
#import "RgfxLibrary.h"
#import "RgfxMetalError.h"
#import "RgfxPipelineReflection.h"
#import "RgfxRenderPipelineState.h"
#import "RgfxUnsupported.h"

#include "RgfxAirArguments.hpp"
#include "RgfxLibraryKey.hpp"
#include "RgfxRegion.hpp"
#include "RgfxSerializer.hpp"

#include <sys/stat.h>
#include <unistd.h>

@implementation RgfxMetalDevice (Pipelines)

// --- libraries ---------------------------------------------------------------

namespace {

// How long a library create waits for the kext to post it on the control ring before it
// reports the ring full: the host drains the ring as it runs, so a full ring is transient.
constexpr useconds_t kLibraryRetrySleepUs = 1000;
constexpr int kLibraryRetries = 2000; // about two seconds

// The largest library create sent inline in the control ring (header and bytes): a
// quarter of the ring's data, so one create never holds most of it. Larger libraries go
// in a region block (LIBRARY_FROM_REGION).
constexpr uint64_t kLibraryInlineMax = RGFX_CONTROL_RING_DATA_SIZE / 4;

// Narrows a universal .metallib to its generic AIR slice, the only one the host compiles: for
// QuartzCore's library that is 1.6 MB of 160 MB. Other bytes are left as they are.
void selectAirSlice(const void *&bytes, NSUInteger &length)
{
	const uint8_t *p = static_cast<const uint8_t *>(bytes);
	size_t size = length;
	rgfxMetallibAirSlice(p, size);
	bytes = p;
	length = static_cast<NSUInteger>(size);
}

// The identity of the file at `path` (RgfxFileIdentity); false where it has none.
bool fileIdentity(const char *path, RgfxFileIdentity &out)
{
	struct stat st = {};
	if (path == nullptr || stat(path, &st) != 0)
		return false;
	out = RgfxFileIdentity{ static_cast<uint64_t>(st.st_dev),     static_cast<uint64_t>(st.st_ino),
	                        static_cast<uint64_t>(st.st_size),    st.st_mtimespec.tv_sec,
	                        st.st_mtimespec.tv_nsec,              st.st_ctimespec.tv_sec,
	                        st.st_ctimespec.tv_nsec };
	return true;
}

} // namespace

// Every library-from-bytes entry point funnels through here. A .metallib (or a fat file's AIR
// slice) is first looked up by its content key, which finds a library the device made for any
// process with no bytes sent; otherwise the bytes go as one LIBRARY_CREATE (waiting for room
// while the ring is momentarily full) or stream through region blocks, with the key, and the
// device's answer is waited for, so a library it refuses is nil and an NSError here, as Metal's.
// Bytes read from a file whose identity is `file` (null for none) take the key the process found
// for that identity before, if any (rgfxLibraryKeyOfFile).
- (nullable id<MTLLibrary>)rgfxLibraryWithBytes:(const void *)bytes
                                         length:(NSUInteger)length
                                           file:(const RgfxFileIdentity *)file
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
	const uint8_t *p = static_cast<const uint8_t *>(bytes);
	const bool keyed = file != nullptr ? rgfxLibraryKeyOfFile(*file, p, length, key)
	                                   : rgfxLibraryKey(p, length, key);
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
	id<MTLLibrary> lib = [self rgfxLibraryWithBytes:bytes length:size file:nullptr error:error];
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
// library made ours (RgfxLinkOverShadows). Metal's first link calls the second on `self`, which
// makes the library ours there.
- (id<MTLLibrary>)newLibraryWithImageFilterFunctionsSPI:(NSArray<id<MTLFunction>> *)functions
                                imageFilterFunctionInfo:(const void *)info
                                                  error:(__autoreleasing NSError **)error
{
	return RgfxLinkOverShadows(
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
	return RgfxLinkOverShadows(
	    functions,
	    ^id<MTLLibrary>(NSArray<id<MTLFunction>> *shadows, NSError *__autoreleasing *e) {
		    return [self rgfxMetalLinkCIFilters:shadows info:info error:e];
	    },
	    ^id<MTLLibrary>(dispatch_data_t data, NSError *__autoreleasing *e) {
		    return [self newLibraryWithData:data error:e];
	    },
	    error);
}

// A stitched library, by each of Metal's entry points for one (measured: the base device class's
// on macOS 27; macOS 26.6.1's has them all): the public one and its completion-handler form; the
// SPI form SkyLight calls, which runs the destination form on the device; the destination form,
// which also keeps the library in an archive, and through which Metal's archive adds one; and the
// DAG form SkyLight and MPS call. Each is Metal's own stitch over the functions' shadows, made ours
// (RgfxStitchOverShadows), so Metal's base path never reads our functions, which it cannot
// (-[RgfxFunction libraryData]).
- (id<MTLLibrary>)newLibraryWithStitchedDescriptor:(MTLStitchedLibraryDescriptor *)descriptor
                                             error:(__autoreleasing NSError **)error
{
	return [self newLibraryWithStitchedDescriptor:descriptor destinationBinaryArchive:nil error:error];
}

- (void)newLibraryWithStitchedDescriptor:(MTLStitchedLibraryDescriptor *)descriptor
                       completionHandler:(MTLNewLibraryCompletionHandler)completionHandler
{
	descriptor = [descriptor copy];
	completionHandler = [completionHandler copy];
	RgfxRunCreation(^{
		NSError *error = nil;
		id<MTLLibrary> lib = [self newLibraryWithStitchedDescriptor:descriptor error:&error];
		if (completionHandler)
			completionHandler(lib, error);
	});
}

- (id<MTLLibrary>)newLibraryWithStitchedDescriptorSPI:(MTLStitchedLibraryDescriptor *)descriptor
                                                error:(__autoreleasing NSError **)error
{
	return [self newLibraryWithStitchedDescriptor:descriptor destinationBinaryArchive:nil error:error];
}

- (id<MTLLibrary>)newLibraryWithStitchedDescriptor:(MTLStitchedLibraryDescriptor *)descriptor
                          destinationBinaryArchive:(id<MTLBinaryArchive>)archive
                                             error:(__autoreleasing NSError **)error
{
	return RgfxStitchOverShadows(
	    descriptor, archive,
	    ^id<MTLLibrary>(MTLStitchedLibraryDescriptor *shadowed, NSError *__autoreleasing *e) {
		    return [self rgfxMetalStitch:shadowed error:e];
	    },
	    ^id<MTLLibrary>(dispatch_data_t data, NSError *__autoreleasing *e) {
		    return [self newLibraryWithData:data error:e];
	    },
	    error);
}

// A library of a descriptor, by _MTLDevice's entry points for one (measured: macOS 27's x86-64
// slice, `@32@0:8@16^@24` for the first two, `@40@0:8@16@24^@32` and `v32@0:8@16@?24`). On the
// M3 Max a stitched library's descriptor answers its stitched library, and any other descriptor
// MTLLibraryErrorDomain's MTLLibraryErrorInternal, "descriptor is not a
// MTLStitchedLibraryDescriptor." Ours is the same over the stitched route above.
- (nullable id<MTLLibrary>)rgfxLibraryWithDescriptor:(id)descriptor
                                             archive:(nullable id<MTLBinaryArchive>)archive
                                               error:(__autoreleasing NSError **)error
{
	if ([descriptor isKindOfClass:[MTLStitchedLibraryDescriptor class]])
		return [self newLibraryWithStitchedDescriptor:descriptor
		                     destinationBinaryArchive:archive
		                                        error:error];
	RGFX_UNSUPPORTED_ERROR_IN(error, "libraries of a descriptor other than a stitched library's",
	                          MTLLibraryErrorDomain, MTLLibraryErrorInternal);
	return nil;
}

- (id<MTLLibrary>)newLibraryWithDescriptor:(id)descriptor error:(__autoreleasing NSError **)error
{
	return [self rgfxLibraryWithDescriptor:descriptor archive:nil error:error];
}

- (id<MTLLibrary>)newLibraryWithDescriptorSPI:(id)descriptor error:(__autoreleasing NSError **)error
{
	return [self rgfxLibraryWithDescriptor:descriptor archive:nil error:error];
}

- (id<MTLLibrary>)newLibraryWithDescriptor:(id)descriptor
                  destinationBinaryArchive:(id<MTLBinaryArchive>)archive
                                     error:(__autoreleasing NSError **)error
{
	return [self rgfxLibraryWithDescriptor:descriptor archive:archive error:error];
}

- (void)newLibraryWithDescriptor:(id)descriptor completionHandler:(id)completionHandler
{
	descriptor = RgfxCopyOf(descriptor);
	completionHandler = [completionHandler copy];
	RgfxRunCreation(^{
		NSError *error = nil;
		id<MTLLibrary> lib = [self rgfxLibraryWithDescriptor:descriptor archive:nil error:&error];
		RgfxCallCreationHandler(completionHandler, lib, error, _cmd);
	});
}

// The DAG's text is Metal's to read (its parser answers its errors, MTLLibraryErrorDomain 3).
- (id<MTLLibrary>)newLibraryWithDAG:(id)dag
                          functions:(NSArray<id<MTLFunction>> *)functions
                              error:(__autoreleasing NSError **)error
{
	return RgfxLinkOverShadows(
	    functions,
	    ^id<MTLLibrary>(NSArray<id<MTLFunction>> *shadows, NSError *__autoreleasing *e) {
		    return [self rgfxMetalStitchDAG:dag functions:shadows error:e];
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

- (id<MTLLibrary>)rgfxMetalStitch:(MTLStitchedLibraryDescriptor *)descriptor
                            error:(__autoreleasing NSError **)error
{
	return [super newLibraryWithStitchedDescriptor:descriptor error:error];
}

- (id<MTLLibrary>)rgfxMetalStitchDAG:(id)dag
                           functions:(NSArray<id<MTLFunction>> *)shadows
                               error:(__autoreleasing NSError **)error
{
	return [super newLibraryWithDAG:dag functions:shadows error:error];
}

// A file's library is keyed by the file's identity where it is the same before and after the
// read (rgfxLibraryKeyOfFile), so a framework's library loaded again is not hashed again.
- (id<MTLLibrary>)newLibraryWithURL:(NSURL *)url error:(__autoreleasing NSError **)error
{
	const char *path = url.isFileURL ? url.fileSystemRepresentation : nullptr;
	RgfxFileIdentity before = {}, after = {};
	const bool identified = fileIdentity(path, before);
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
	const bool same = identified && fileIdentity(path, after) && after == before &&
	                  after.size == data.length;
	id<MTLLibrary> lib = [self rgfxLibraryWithBytes:data.bytes
	                                         length:data.length
	                                           file:same ? &before : nullptr
	                                          error:error];
	dispatch_data_t source = dispatch_data_create(data.bytes, data.length, nil, ^{
		(void)data;
	});
	[self rgfxKeepSource:source ofLibrary:lib];
	if ([(id)lib isKindOfClass:[RgfxLibrary class]])
		[(RgfxLibrary *)lib rgfxSetMadeFromURL:url];
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

// A library of source compiled at run time: the form with a compiler task, with none.
- (id<MTLLibrary>)newLibraryWithSource:(NSString *)source
                               options:(MTLCompileOptions *)options
                                 error:(__autoreleasing NSError **)error
{
	return [self newLibraryWithSource:source options:options compilerTask:nil error:error];
}

- (void)newLibraryWithSource:(NSString *)source
                     options:(MTLCompileOptions *)options
           completionHandler:(MTLNewLibraryCompletionHandler)completionHandler
{
	source = [source copy];
	options = [options copy];
	completionHandler = [completionHandler copy];
	RgfxRunCreation(^{
		NSError *err = nil;
		id<MTLLibrary> lib = [self newLibraryWithSource:source options:options error:&err];
		if (completionHandler)
			completionHandler(lib, err);
	});
}

// The forms with a compiler task (measured: _MTLDevice's on macOS 27's x86-64 slice,
// `@48@0:8@16@24@32^@40` and `v48@0:8@16@24@32@?40`). The first is Metal's own frontend over the
// device (RgfxCompileOverMetal): the base class's, which _MTLDevice's form without a task calls on
// the device, and which reaches Metal's compiler service through the base class alone, never the
// device's `compiler`, reading of the device only its feature profile, its families, and Metal's
// own settings for the compiler service and for shader capture (measured: macOS 26.7.1 on an AMD
// Radeon Pro 5300M and an Intel UHD 630, macOS 27 on the M3 Max). Its library's bytes are AIR, the
// same on the AMD and the Intel device.
// Where the base class lacks the form, its form without a task, which then calls none of ours.
- (id<MTLLibrary>)newLibraryWithSource:(NSString *)source
                               options:(MTLCompileOptions *)options
                          compilerTask:(id)compilerTask
                                 error:(__autoreleasing NSError **)error
{
	BOOL tasks = [IOGPUMetalDevice instancesRespondToSelector:_cmd];
	return RgfxCompileOverMetal(
	    ^id<MTLLibrary>(NSError *__autoreleasing *e) {
		    if (!tasks)
			    return [super newLibraryWithSource:source options:options error:e];
		    return [super newLibraryWithSource:source options:options compilerTask:compilerTask error:e];
	    },
	    ^id<MTLLibrary>(dispatch_data_t data, NSError *__autoreleasing *e) {
		    return [self newLibraryWithData:data error:e];
	    },
	    error);
}

- (void)newLibraryWithSource:(NSString *)source
                     options:(MTLCompileOptions *)options
                compilerTask:(id)compilerTask
           completionHandler:(id)completionHandler
{
	(void)compilerTask;
	source = [source copy];
	options = [options copy];
	completionHandler = [completionHandler copy];
	RgfxRunCreation(^{
		NSError *err = nil;
		id<MTLLibrary> lib = [self newLibraryWithSource:source options:options error:&err];
		RgfxCallCreationHandler(completionHandler, lib, err, _cmd);
	});
}

// --- libraries of a function array and of packages ---------------------------

// A library of the functions given (newLibraryWithFunctionArray:error:, SPI), which holds them
// themselves, as both x86 Macs make one (RgfxFunctionArrayLibrary).
- (id<MTLLibrary>)newLibraryWithFunctionArray:(NSArray<id<MTLFunction>> *)functions
                                        error:(__autoreleasing NSError **)error
{
	return [RgfxFunctionArrayLibrary libraryOfFunctions:functions device:self error:error];
}

// The library of a Metal package and of an MPS graph package: Metal's own reading of the package,
// the base class's, made ours over the bytes of the library it answers (RgfxLibraryOverMetal), so
// a refusal is Metal's own, words and log line included. Both x86 Macs answer nil and
// MTLLibraryErrorDomain 1 "Invalid metal package" for a missing path and a .metallib, and nil and
// MTLLibraryErrorDomain 6 "failed to open MPSGraph package" for a missing package, logging "Error:
// did not find file at url: ..." (oracle_bundle libforms). Where the base class lacks the form, its
// words for a missing package.
- (id<MTLLibrary>)newLibraryWithMetalPackageURL:(NSURL *)url error:(__autoreleasing NSError **)error
{
	if (![IOGPUMetalDevice instancesRespondToSelector:_cmd]) {
		if (error)
			*error = [NSError errorWithDomain:MTLLibraryErrorDomain
			                             code:MTLLibraryErrorUnsupported
			                         userInfo:@{NSLocalizedDescriptionKey : @"Invalid metal package"}];
		return nil;
	}
	return RgfxLibraryOverMetal(
	    ^id<MTLLibrary>(NSError *__autoreleasing *e) {
		    return [self rgfxMetalLibraryOfPackage:url error:e];
	    },
	    ^id<MTLLibrary>(dispatch_data_t data, NSError *__autoreleasing *e) {
		    return [self newLibraryWithData:data error:e];
	    },
	    @"newLibraryWithMetalPackageURL:", error);
}

- (id<MTLLibrary>)rgfxMetalLibraryOfPackage:(NSURL *)url error:(__autoreleasing NSError **)error
{
	return [super newLibraryWithMetalPackageURL:url error:error];
}

- (id<MTLLibrary>)newLibraryWithMPSGraphPackageURL:(NSURL *)url
                                              name:(NSString *)name
                                             error:(__autoreleasing NSError **)error
{
	if (![IOGPUMetalDevice instancesRespondToSelector:_cmd]) {
		if (error)
			*error = [NSError errorWithDomain:MTLLibraryErrorDomain
			                             code:MTLLibraryErrorFileNotFound
			                         userInfo:@{NSLocalizedDescriptionKey : @"failed to open MPSGraph package"}];
		return nil;
	}
	return RgfxLibraryOverMetal(
	    ^id<MTLLibrary>(NSError *__autoreleasing *e) {
		    return [self rgfxMetalLibraryOfGraphPackage:url name:name error:e];
	    },
	    ^id<MTLLibrary>(dispatch_data_t data, NSError *__autoreleasing *e) {
		    return [self newLibraryWithData:data error:e];
	    },
	    @"newLibraryWithMPSGraphPackageURL:", error);
}

- (id<MTLLibrary>)rgfxMetalLibraryOfGraphPackage:(NSURL *)url
                                            name:(NSString *)name
                                           error:(__autoreleasing NSError **)error
{
	return [super newLibraryWithMPSGraphPackageURL:url name:name error:error];
}

// --- render pipelines --------------------------------------------------------

// `state`, or nil and an NSError where the host refused its create: the kext hands the create's
// reply back before the create returns, so a refusal is known here, as Metal's creators know
// theirs. A create whose descriptor equals one the device accepted for this process is posted
// without waiting (RgfxKnownPipelines.hpp): the device refuses it only where the platform cannot
// make the new state's own pipeline, which shows at its gpuResourceID and in the command buffers
// that name it (docs/graphics-kext/design.md, "Known pipeline creates").
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
	                                                         known:[self knownPipelines]
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
	id<MTLRenderPipelineState> state = [self rgfxRenderPipelineForDescriptor:descriptor options:options error:error];
	if (reflection)
		*reflection = state != nil ? RgfxRenderPipelineReflectionOf(descriptor.vertexFunction, descriptor.vertexDescriptor,
		                                                            descriptor.fragmentFunction, options)
		                           : nil;
	return state;
}

- (void)newRenderPipelineStateWithDescriptor:(MTLRenderPipelineDescriptor *)descriptor
                           completionHandler:
                               (MTLNewRenderPipelineStateCompletionHandler)completionHandler
{
	descriptor = [descriptor copy];
	completionHandler = [completionHandler copy];
	RgfxRunCreation(^{
		NSError *err = nil;
		id<MTLRenderPipelineState> state = [self newRenderPipelineStateWithDescriptor:descriptor
		                                                                       error:&err];
		if (completionHandler)
			completionHandler(state, err);
	});
}

- (void)newRenderPipelineStateWithDescriptor:(MTLRenderPipelineDescriptor *)descriptor
                                     options:(MTLPipelineOption)options
                           completionHandler:
                               (MTLNewRenderPipelineStateWithReflectionCompletionHandler)
                                   completionHandler
{
	descriptor = [descriptor copy];
	completionHandler = [completionHandler copy];
	RgfxRunCreation(^{
		NSError *err = nil;
		id<MTLRenderPipelineState> state = [self rgfxRenderPipelineForDescriptor:descriptor
		                                                                 options:options
		                                                                   error:&err];
		if (completionHandler)
			completionHandler(state,
			                  state != nil ? RgfxRenderPipelineReflectionOf(descriptor.vertexFunction,
			                                                                descriptor.vertexDescriptor,
			                                                                descriptor.fragmentFunction, options)
			                               : nil,
			                  err);
	});
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
	                                                         known:[self knownPipelines]
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
	                                                           known:[self knownPipelines]
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
	id<MTLComputePipelineState> state = [self rgfxComputePipelineForFunction:computeFunction error:error];
	if (reflection)
		*reflection = state != nil ? RgfxComputePipelineReflectionOf(computeFunction, nil, options) : nil;
	return state;
}

- (void)newComputePipelineStateWithFunction:(id<MTLFunction>)computeFunction
                          completionHandler:
                              (MTLNewComputePipelineStateCompletionHandler)completionHandler
{
	completionHandler = [completionHandler copy];
	RgfxRunCreation(^{
		NSError *err = nil;
		id<MTLComputePipelineState> state =
		    [self newComputePipelineStateWithFunction:computeFunction error:&err];
		if (completionHandler)
			completionHandler(state, err);
	});
}

- (void)newComputePipelineStateWithFunction:(id<MTLFunction>)computeFunction
                                    options:(MTLPipelineOption)options
                          completionHandler:
                              (MTLNewComputePipelineStateWithReflectionCompletionHandler)
                                  completionHandler
{
	completionHandler = [completionHandler copy];
	RgfxRunCreation(^{
		NSError *err = nil;
		id<MTLComputePipelineState> state =
		    [self newComputePipelineStateWithFunction:computeFunction error:&err];
		if (completionHandler)
			completionHandler(state,
			                  state != nil ? RgfxComputePipelineReflectionOf(computeFunction, nil, options) : nil,
			                  err);
	});
}

- (id<MTLComputePipelineState>)
    newComputePipelineStateWithDescriptor:(MTLComputePipelineDescriptor *)descriptor
                                  options:(MTLPipelineOption)options
                               reflection:(MTLAutoreleasedComputePipelineReflection *)reflection
                                    error:(__autoreleasing NSError **)error
{
	id<MTLComputePipelineState> state = [self rgfxComputePipelineForDescriptor:descriptor
	                                                                 options:options
	                                                                   error:error];
	if (reflection)
		*reflection = state != nil ? RgfxComputePipelineReflectionOf(descriptor.computeFunction,
		                                                             descriptor.stageInputDescriptor, options)
		                           : nil;
	return state;
}

- (void)newComputePipelineStateWithDescriptor:(MTLComputePipelineDescriptor *)descriptor
                                      options:(MTLPipelineOption)options
                            completionHandler:
                                (MTLNewComputePipelineStateWithReflectionCompletionHandler)
                                    completionHandler
{
	descriptor = [descriptor copy];
	completionHandler = [completionHandler copy];
	RgfxRunCreation(^{
		NSError *err = nil;
		id<MTLComputePipelineState> state = [self rgfxComputePipelineForDescriptor:descriptor
		                                                                  options:options
		                                                                    error:&err];
		if (completionHandler)
			completionHandler(state,
			                  state != nil ? RgfxComputePipelineReflectionOf(descriptor.computeFunction,
			                                                                 descriptor.stageInputDescriptor, options)
			                               : nil,
			                  err);
	});
}

// The descriptor's forms without options, which the MTLDevice protocol does not declare and
// _MTLDevice implements (measured: macOS 27 Metal.framework's x86-64 slice, `@32@0:8@16^@24` and
// `v32@0:8@16@?24`). WindowServer's Core Animation makes its compute_average_luma pipeline through
// the first (measured: the x86-64 guest, macOS 26, on opening Calculator): inherited, it sent the
// create on to the base device's `compiler`, which is nil on our device, so it answered nil with
// no error and Core Animation ended WindowServer. Both answer as the options form does with no
// options. No header gives the second's handler type: it is called by its own signature
// (RgfxBlockCall.h), as the other creators below without a public form call theirs.
- (id<MTLComputePipelineState>)newComputePipelineStateWithDescriptor:
                                   (MTLComputePipelineDescriptor *)descriptor
                                                               error:(__autoreleasing NSError **)error
{
	return [self rgfxComputePipelineForDescriptor:descriptor
	                                      options:MTLPipelineOptionNone
	                                        error:error];
}

- (void)newComputePipelineStateWithDescriptor:(MTLComputePipelineDescriptor *)descriptor
                            completionHandler:
                                (MTLNewComputePipelineStateCompletionHandler)completionHandler
{
	descriptor = [descriptor copy];
	id handler = [completionHandler copy];
	RgfxRunCreation(^{
		NSError *err = nil;
		id<MTLComputePipelineState> state =
		    [self rgfxComputePipelineForDescriptor:descriptor
		                                   options:MTLPipelineOptionNone
		                                     error:&err];
		RgfxCallCreationHandler(handler, state, err, _cmd);
	});
}

// The precompiled forms (measured: _MTLDevice's on macOS 27's x86-64 slice, `v48@0:8@16Q24@32@?40`
// each): the pipeline cache they take is Metal's own, which the device keeps none of, so each
// makes the pipeline as the options form does, and refuses a descriptor of any other kind.
- (void)newPrecompiledComputePipelineStateWithDescriptor:(id)descriptor
                                                 options:(MTLPipelineOption)options
                                           pipelineCache:(id)pipelineCache
                                       completionHandler:(id)completionHandler
{
	(void)pipelineCache;
	descriptor = RgfxCopyOf(descriptor);
	completionHandler = [completionHandler copy];
	RgfxRunCreation(^{
		NSError *err = nil;
		id<MTLComputePipelineState> state = nil;
		id reflection = nil;
		if ([descriptor isKindOfClass:[MTLComputePipelineDescriptor class]]) {
			MTLComputePipelineDescriptor *d = descriptor;
			state = [self rgfxComputePipelineForDescriptor:d options:options error:&err];
			if (state != nil)
				reflection = RgfxComputePipelineReflectionOf(d.computeFunction, d.stageInputDescriptor, options);
		} else {
			RGFX_UNSUPPORTED_ERROR(&err, "precompiled compute pipelines of another descriptor");
		}
		RgfxCallCreationHandlerWithReflection(completionHandler, state, reflection, err, _cmd);
	});
}

- (void)newPrecompiledRenderPipelineStateWithDescriptor:(id)descriptor
                                                options:(MTLPipelineOption)options
                                          pipelineCache:(id)pipelineCache
                                      completionHandler:(id)completionHandler
{
	(void)pipelineCache;
	descriptor = RgfxCopyOf(descriptor);
	completionHandler = [completionHandler copy];
	RgfxRunCreation(^{
		NSError *err = nil;
		id<MTLRenderPipelineState> state = nil;
		id reflection = nil;
		if ([descriptor isKindOfClass:[MTLRenderPipelineDescriptor class]]) {
			MTLRenderPipelineDescriptor *d = descriptor;
			state = [self rgfxRenderPipelineForDescriptor:d options:options error:&err];
			if (state != nil)
				reflection = RgfxRenderPipelineReflectionOf(d.vertexFunction, d.vertexDescriptor, d.fragmentFunction,
				                                            options);
		} else {
			RGFX_UNSUPPORTED_ERROR(&err, "precompiled render pipelines of another descriptor");
		}
		RgfxCallCreationHandlerWithReflection(completionHandler, state, reflection, err, _cmd);
	});
}

// Core Image's pipeline of a link (measured: _MTLDevice's on macOS 27's x86-64 slice,
// `@40@0:8@16r^{?=BQ^{?}}24^@32`). On the M3 Max, given the functions and info Core Image links,
// it answers the pipeline of the one kernel their link makes (`ciKernelMain`, of functionType 3;
// the two pipelines' limits match), and the link's error where the link fails. Ours is the same:
// our link (above), then our pipeline of its one function.
- (id<MTLComputePipelineState>)
    newComputePipelineStateWithImageFilterFunctionsSPI:(NSArray<id<MTLFunction>> *)functions
                               imageFilterFunctionInfo:(const void *)info
                                                 error:(__autoreleasing NSError **)error
{
	id<MTLLibrary> linked = [self newLibraryWithImageFilterFunctionsSPI:functions
	                                            imageFilterFunctionInfo:info
	                                                              error:error];
	if (linked == nil)
		return nil;
	NSArray<NSString *> *names = linked.functionNames;
	if (names.count != 1) {
		RGFX_UNSUPPORTED_ERROR(error, "a Core Image link of other than one kernel");
		return nil;
	}
	return [self rgfxComputePipelineForFunction:[linked newFunctionWithName:names[0]] error:error];
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

// The encoder of the argument buffer a buffer binding describes, as both x86 Macs make one (the
// AMD 5300M and the Intel 630 under macOS 26.7.1: oracle_bundle argbuf.binding.lib, the same
// encoder as the function's own of that buffer); nil for a binding of no argument buffer, as the
// AMD answers (argbuf.binding.plain). RgfxArgumentEncoder.h says how it is laid out.
- (id<MTLArgumentEncoder>)newArgumentEncoderWithBufferBinding:(id<MTLBufferBinding>)bufferBinding
{
	return [RgfxArgumentEncoder encoderOfBinding:bufferBinding
	                                      region:[self region]
	                                      device:self
	                                  serializer:[self serializer]];
}

// The same with the encoder's struct type answered through `structType` (measured: _MTLDevice's on
// macOS 27's x86-64 slice, `@32@0:8@16^@24`; the M3 Max answers its encoder and an
// MTLStructTypeInternal). The device has no argument reflection, so it answers nil there.
- (id<MTLArgumentEncoder>)
    newArgumentEncoderWithArguments:(NSArray<MTLArgumentDescriptor *> *)arguments
                         structType:(__autoreleasing id *)structType
{
	if (structType)
		*structType = nil;
	return [self newArgumentEncoderWithArguments:arguments];
}

@end
