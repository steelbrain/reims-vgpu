/*
 * RgfxMetalDevice (Pipelines): see RgfxMetalDevice+Pipelines.h. Slice 2 of the device
 * shell. Our own ObjC++; the only Apple-private names are in RgfxMetalPrivate.h.
 */

#import "RgfxMetalDevice+Pipelines.h"

#import "RgfxArgumentEncoder.h"
#import "RgfxComputePipelineState.h"
#import "RgfxLibrary.h"
#import "RgfxMetalError.h"
#import "RgfxRenderPipelineState.h"

#include "RgfxRegion.hpp"
#include "RgfxSerializer.hpp"

#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

@implementation RgfxMetalDevice (Pipelines)

// --- libraries ---------------------------------------------------------------

namespace {

// RGFX DIAG (WindowServer bring-up; remove before merge): one line per library
// create to /tmp/rgfx-diag.log, so a guest without logd shows which entry point a
// client loads its shaders through and whether the create posted.
void rgfxLibraryDiag(const char *how, NSString *origin, NSUInteger length, BOOL posted)
{
	char line[600];
	int n = snprintf(line, sizeof(line), "RGFX: library %s %s bytes=%lu posted=%d pid=%d\n", how,
	                 origin.UTF8String ? origin.UTF8String : "-", (unsigned long)length,
	                 posted ? 1 : 0, getpid());
	if (n <= 0)
		return;
	int fd = open("/tmp/rgfx-diag.log", O_WRONLY | O_CREAT | O_APPEND, 0666);
	if (fd < 0)
		return;
	(void)!write(fd, line, static_cast<size_t>(n < (int)sizeof(line) ? n : (int)sizeof(line)));
	close(fd);
}

// RGFX DIAG (remove before merge): how long a streamed library's upload took.
void rgfxUploadDiag(NSString *origin, NSUInteger length, uint64_t ns)
{
	char line[600];
	int n = snprintf(line, sizeof(line), "RGFX: library upload %s bytes=%lu ms=%.1f pid=%d\n",
	                 origin.UTF8String ? origin.UTF8String : "-", (unsigned long)length,
	                 static_cast<double>(ns) / 1e6, getpid());
	if (n <= 0)
		return;
	int fd = open("/tmp/rgfx-diag.log", O_WRONLY | O_CREAT | O_APPEND, 0666);
	if (fd < 0)
		return;
	(void)!write(fd, line, static_cast<size_t>(n < (int)sizeof(line) ? n : (int)sizeof(line)));
	close(fd);
}

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

// Every library-from-bytes entry point funnels through here: posts the bytes as one
// LIBRARY_CREATE, waiting for room while the ring is momentarily full. A library
// larger than the ring cannot be posted in one message and is refused.
- (nullable id<MTLLibrary>)rgfxLibraryWithBytes:(const void *)bytes
                                         length:(NSUInteger)length
                                            how:(const char *)how
                                         origin:(NSString *)origin
                                          error:(__autoreleasing NSError **)error
{
	selectAirSlice(bytes, length);
	RgfxSerializer *ser = [self serializer];
	if (ser == nullptr) {
		rgfxLibraryDiag(how, origin, length, NO);
		if (error)
			*error = RgfxMakeError(RgfxMetalErrorNoRoute, @"library: no serialiser route");
		return nil;
	}
	// A small library travels inline in its create; a larger one streams through region
	// blocks that its chunks name, so no library is bounded by the ring's or region's size.
	const bool inlineCreate =
	    RGFX_LIBRARY_BYTES_OFFSET + static_cast<uint64_t>(length) + 8 <= kLibraryInlineMax;
	RgfxRegion *region = [self region];
	if (!inlineCreate && region == nullptr) {
		rgfxLibraryDiag(how, origin, length, NO);
		if (error)
			*error = RgfxMakeError(RgfxMetalErrorNoRoute,
			                       @"library: no device region for a large library");
		return nil;
	}
	id<MTLLibrary> lib = nil;
	if (inlineCreate) {
		for (int i = 0; i < kLibraryRetries && lib == nil; i++) {
			lib = [RgfxLibrary libraryWithData:bytes length:length device:self serializer:ser];
			if (lib == nil)
				usleep(kLibraryRetrySleepUs);
		}
	} else {
		// Streamed through the process's double-buffered staging window, timed for the
		// diagnostic line (RGFX DIAG).
		const uint64_t t0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
		lib = [RgfxLibrary libraryStreamingBytes:bytes
		                                  length:length
		                                  region:region
		                                  device:self
		                              serializer:ser];
		rgfxUploadDiag(origin, length, clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - t0);
	}
	rgfxLibraryDiag(how, origin, length, lib != nil);
	if (lib == nil && error)
		*error = RgfxMakeError(RgfxMetalErrorRingFull, @"library: the device ring stayed full");
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
	id<MTLLibrary> lib = [self rgfxLibraryWithBytes:bytes
	                                         length:size
	                                            how:"data"
	                                         origin:@"(data)"
	                                          error:error];
	(void)map; // released by ARC; `bytes` stayed valid for the create above
	return lib;
}

- (id<MTLLibrary>)newLibraryWithURL:(NSURL *)url error:(__autoreleasing NSError **)error
{
	NSError *readError = nil;
	NSData *data = url ? [NSData dataWithContentsOfURL:url options:NSDataReadingMappedIfSafe
	                                             error:&readError]
	                   : nil;
	if (data == nil) {
		rgfxLibraryDiag("url-unreadable", url.path, 0, NO);
		if (error)
			*error = readError ?: RgfxMakeError(RgfxMetalErrorInvalidDescriptor,
			                                    @"newLibraryWithURL: could not read URL");
		return nil;
	}
	return [self rgfxLibraryWithBytes:data.bytes
	                           length:data.length
	                              how:"url"
	                           origin:url.path
	                            error:error];
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
		rgfxLibraryDiag("default-missing", bundle.bundlePath, 0, NO);
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

// Creates a render pipeline state and registers it with the device's reply reader, so
// its reflection fills when the host answers this create's seqno. Every render-pipeline
// entry point funnels through here.
- (nullable id<MTLRenderPipelineState>)
    rgfxRenderPipelineForDescriptor:(MTLRenderPipelineDescriptor *)descriptor
                              error:(__autoreleasing NSError **)error
{
	RgfxRenderPipelineState *state =
	    [RgfxRenderPipelineState renderPipelineStateWithDescriptor:descriptor
	                                                        device:self
	                                                    serializer:[self serializer]
	                                                         error:error];
	[state rgfxRegisterWithReplyReader:[self replyReader]];
	return state;
}

- (id<MTLRenderPipelineState>)
    newRenderPipelineStateWithDescriptor:(MTLRenderPipelineDescriptor *)descriptor
                                   error:(__autoreleasing NSError **)error
{
	return [self rgfxRenderPipelineForDescriptor:descriptor error:error];
}

- (id<MTLRenderPipelineState>)
    newRenderPipelineStateWithDescriptor:(MTLRenderPipelineDescriptor *)descriptor
                                 options:(MTLPipelineOption)options
                              reflection:(MTLAutoreleasedRenderPipelineReflection *)reflection
                                   error:(__autoreleasing NSError **)error
{
	(void)options;
	// The reflection is answered by the host over the completion ring (reader is a
	// later transport piece), so there is none to hand back synchronously.
	if (reflection)
		*reflection = nil;
	return [self rgfxRenderPipelineForDescriptor:descriptor error:error];
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
	(void)options;
	NSError *err = nil;
	id<MTLRenderPipelineState> state = [self newRenderPipelineStateWithDescriptor:descriptor
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
	return state;
}

- (nullable id<MTLComputePipelineState>)
    rgfxComputePipelineForDescriptor:(MTLComputePipelineDescriptor *)descriptor
                               error:(__autoreleasing NSError **)error
{
	RgfxComputePipelineState *state =
	    [RgfxComputePipelineState computePipelineStateWithDescriptor:descriptor
	                                                          device:self
	                                                      serializer:[self serializer]
	                                                           error:error];
	[state rgfxRegisterWithReplyReader:[self replyReader]];
	return state;
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
	(void)options;
	if (reflection)
		*reflection = nil;
	return [self rgfxComputePipelineForDescriptor:descriptor error:error];
}

- (void)newComputePipelineStateWithDescriptor:(MTLComputePipelineDescriptor *)descriptor
                                      options:(MTLPipelineOption)options
                            completionHandler:
                                (MTLNewComputePipelineStateWithReflectionCompletionHandler)
                                    completionHandler
{
	(void)options;
	NSError *err = nil;
	id<MTLComputePipelineState> state = [self rgfxComputePipelineForDescriptor:descriptor
	                                                                    error:&err];
	if (completionHandler)
		completionHandler(state, nil, err);
}

// --- argument encoders -------------------------------------------------------

- (id<MTLArgumentEncoder>)
    newArgumentEncoderWithArguments:(NSArray<MTLArgumentDescriptor *> *)arguments
{
	return [RgfxArgumentEncoder argumentEncoderWithArguments:arguments
	                                                 device:self
	                                             serializer:[self serializer]];
}

@end
