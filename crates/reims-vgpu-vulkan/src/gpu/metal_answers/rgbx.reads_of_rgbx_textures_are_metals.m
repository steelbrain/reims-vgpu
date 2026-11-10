// Records Metal's answers to gpu/rgbx.rs's reads_of_rgbx_textures_are_metals on this Mac's own
// Metal, in the lines of its file of answers beside this source
// (rgbx.reads_of_rgbx_textures_are_metals.txt, the forms core's testkit::metal_answers reads).
// It runs what the test's `Gpu::reads` runs, as the deleted Metal platform sent it, with no
// part of this library in between:
// - for each RGBX format the test takes, in its order, Shared (options 0x0) then Private
//   (0x20), a 16 by 16 texture of `ShaderRead` usage holding the test's random bytes
//   (`Rng(0x9e37_79b9_7f4a_7c15 ^ seed)`, seed the format's 1-based place in `FORMATS`),
//   written by replaceRegion: where it is Shared and by a blit from a Shared buffer where it is
//   Private;
// - rgbx.metal's kernels over it, each in a command buffer of its own over the 16 by 16 grid in
//   8 by 8 threadgroups: read() in each type the format reads as, at each texel and 8 texels
//   down and across; for the formats read as floats, sample() as floats and halves through each
//   of the test's samplers and gather() of each component through each of its gathering
//   samplers, at the test's coordinates; and read() of the texture an argument buffer holds by
//   its gpuResourceID, made resident for reading.
// Each (format, options) prints its `cases` count, then each case's bytes as
// `bytes <length> <FNV-1a 128> <first 32 bytes in hex>`, keyed as the test keys them.
//
// Provenance: on the M3 Max under macOS 27.0.1 (26A434), 2026-11-10, it printed each of the 198
// answers the file held, recorded from the Metal platform on 2026-11-07 (nine formats, Shared and
// Private), byte for byte, before its RGBX32Float lines were added to the file from its output.
// It printed the same with rgbx.metal compiled by Xcode 27.2's Metal 32023.922 and with the
// metallib committed in metallibs/, whose bytes differ.
//
// Build and run on a Mac, from the repository's root, with Xcode's tools; the metallib is the
// one build.rs compiles of rgbx.metal, which the test loads:
//   T=$(mktemp -d)
//   xcrun -sdk macosx metal -std=metal4.0 -c crates/reims-vgpu-vulkan/src/gpu/rgbx.metal \
//       -o "$T/rgbx.air"
//   xcrun -sdk macosx metallib "$T/rgbx.air" -o "$T/rgbx.metallib"
//   xcrun -sdk macosx clang -O1 -ffp-contract=off -framework Foundation -framework Metal \
//       crates/reims-vgpu-vulkan/src/gpu/metal_answers/rgbx.reads_of_rgbx_textures_are_metals.m \
//       -o "$T/record"
//   "$T/record" "$T/rgbx.metallib" [FORMAT...]
// With no FORMAT it records every format, RGBX16Float's included, whose reads die on every
// device the test runs on, so the file holds none of its lines. Built without ARC: a run is
// short, and what it makes lives until it exits.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// A texture's width and height; the texels of a texture, and the threads of a dispatch.
enum { SIDE = 16, TEXELS = 256 };

// MTLResourceStorageModeShared and MTLResourceStorageModePrivate, as resource options.
static const uint64_t SHARED = 0, PRIVATE = 0x20;

// How a format's texels read in a kernel: as floats (or halves), unsigned or signed integers.
typedef enum { FLOAT, UINT, INT } ReadClass;

// Each RGBX format, in the test's order: its raw value, its name, its texel's bytes, how kernels
// read it.
static const struct {
	MTLPixelFormat raw;
	const char *name;
	size_t texel;
	ReadClass class_of;
} FORMATS[11] = {
    {300, "RGBX8Unorm", 4, FLOAT},  {301, "RGBX8Unorm_sRGB", 4, FLOAT}, {302, "RGBX8Snorm", 4, FLOAT},
    {303, "RGBX8Uint", 4, UINT},    {304, "RGBX8Sint", 4, INT},         {313, "RGBX16Uint", 8, UINT},
    {314, "RGBX16Sint", 8, INT},    {315, "RGBX16Float", 8, FLOAT},     {323, "RGBX32Uint", 16, UINT},
    {324, "RGBX32Sint", 16, INT},   {325, "RGBX32Float", 16, FLOAT},
};

// The samplers the sampling kernels read through: the filter (0 nearest, 1 linear), the address
// mode on both axes (0 clamp to edge, 2 repeat, 4 clamp to zero, 5 clamp to the border colour)
// and the border colour (0 transparent black, 1 opaque black, 2 opaque white).
static const uint64_t SAMPLERS[8][3] = {
    {0, 0, 0}, {1, 0, 0}, {1, 2, 0}, {1, 4, 0}, {1, 5, 0}, {1, 5, 1}, {1, 5, 2}, {0, 5, 2},
};

// The samplers the gathering kernel reads through, as SAMPLERS.
static const uint64_t GATHERS[3][3] = {{0, 0, 0}, {0, 5, 0}, {0, 5, 2}};

static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static id<MTLLibrary> library;

// The test's Rng (xorshift64*): the next word of the stream in `s`.
static uint64_t next_word(uint64_t *s)
{
	*s ^= *s >> 12;
	*s ^= *s << 25;
	*s ^= *s >> 27;
	return *s * 0x2545f4914f6cdd1dULL;
}

// `n` random bytes of the stream in `s`: each word's top byte.
static void random_bytes(uint64_t *s, uint8_t *out, size_t n)
{
	for (size_t i = 0; i < n; i++)
		out[i] = (uint8_t)(next_word(s) >> 56);
}

// The normalized coordinates the sampling kernels read at, two floats a thread: every fourth a
// random texel's centre, the others random in [-0.25, 1.25), past the edges.
static void coordinates(float out[2 * TEXELS])
{
	uint64_t s = 0x6a09e667f3bcc908ULL;
	for (int i = 0; i < TEXELS; i++) {
		for (int k = 0; k < 2; k++) {
			float v;
			if (i % 4 == 0) {
				v = ((float)(uint8_t)(next_word(&s) % SIDE) + 0.5f) / (float)SIDE;
			} else {
				float unit = (float)(uint16_t)(next_word(&s) >> 48) / 65536.0f;
				v = unit * 1.5f - 0.25f;
			}
			out[2 * i + k] = v;
		}
	}
}

// Prints `key`'s line of a byte buffer: its length, FNV-1a 128 digest and first 32 bytes.
static void print_bytes(NSString *key, NSData *data)
{
	const uint8_t *b = data.bytes;
	unsigned __int128 h = ((unsigned __int128)0x6c62272e07bb0142ULL << 64) | 0x62b821756295c58dULL;
	const unsigned __int128 prime = ((unsigned __int128)0x0000000001000000ULL << 64) | 0x13bULL;
	for (NSUInteger i = 0; i < data.length; i++) {
		h ^= b[i];
		h *= prime;
	}
	printf("%s\tbytes %lu %016llx%016llx ", key.UTF8String, (unsigned long)data.length,
	       (unsigned long long)(h >> 64), (unsigned long long)h);
	for (NSUInteger i = 0; i < data.length && i < 32; i++)
		printf("%02x", b[i]);
	printf("\n");
}

static void fail(const char *what, NSError *error)
{
	fprintf(stderr, "record: %s%s%s\n", what, error ? ": " : "",
	        error ? error.description.UTF8String : "");
	exit(1);
}

// A Shared buffer holding `n` bytes of `bytes` (zeroes where `bytes` is NULL), of at least 16.
static id<MTLBuffer> buffer(const void *bytes, size_t n)
{
	id<MTLBuffer> b = [device newBufferWithLength:(n < 16 ? 16 : n) options:MTLResourceStorageModeShared];
	if (b == nil)
		fail("no buffer", nil);
	if (bytes != NULL)
		memcpy(b.contents, bytes, n);
	else
		memset(b.contents, 0, n);
	return b;
}

// Runs what `encode` encodes in a command buffer of its own, and waits for it.
static void run(void (^encode)(id<MTLCommandBuffer>))
{
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	encode(cb);
	[cb commit];
	[cb waitUntilCompleted];
	if (cb.error != nil)
		fail("a command buffer failed", cb.error);
}

// A 16 by 16 texture of `format`, `ShaderRead` usage and one level, Shared or Private
// (`options`), as the Metal platform's descriptor held it, holding `bytes`, a row of 16 texels
// after another: written by replaceRegion: where it is Shared, by a blit from a buffer where it
// is Private.
static id<MTLTexture> texture(MTLPixelFormat format, uint64_t options, const uint8_t *bytes, size_t n)
{
	MTLTextureDescriptor *d = [MTLTextureDescriptor new];
	d.textureType = MTLTextureType2D;
	d.width = SIDE;
	d.height = SIDE;
	d.depth = 1;
	d.mipmapLevelCount = 1;
	d.sampleCount = 1;
	d.arrayLength = 1;
	d.usage = MTLTextureUsageShaderRead;
	d.pixelFormat = format;
	d.resourceOptions = options;
	id<MTLTexture> t = [device newTextureWithDescriptor:d];
	if (t == nil)
		fail("no texture", nil);
	NSUInteger row = n / SIDE;
	if (options == SHARED) {
		[t replaceRegion:MTLRegionMake2D(0, 0, SIDE, SIDE)
		     mipmapLevel:0
		           slice:0
		       withBytes:bytes
		     bytesPerRow:row
		   bytesPerImage:0];
	} else {
		id<MTLBuffer> b = buffer(bytes, n);
		run(^(id<MTLCommandBuffer> cb) {
			id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
			[e copyFromBuffer:b
			           sourceOffset:0
			      sourceBytesPerRow:row
			    sourceBytesPerImage:row * SIDE
			             sourceSize:MTLSizeMake(SIDE, SIDE, 1)
			              toTexture:t
			       destinationSlice:0
			       destinationLevel:0
			      destinationOrigin:MTLOriginMake(0, 0, 0)];
			[e endEncoding];
		});
	}
	return t;
}

// A sampler state: `filter` (0 nearest, 1 linear) and `address` on both axes, normalized
// coordinates, and the border colour `border`; every other property a new descriptor's.
static id<MTLSamplerState> sampler(uint64_t filter, uint64_t address, uint64_t border)
{
	MTLSamplerDescriptor *d = [MTLSamplerDescriptor new];
	d.minFilter = (MTLSamplerMinMagFilter)filter;
	d.magFilter = (MTLSamplerMinMagFilter)filter;
	d.sAddressMode = (MTLSamplerAddressMode)address;
	d.tAddressMode = (MTLSamplerAddressMode)address;
	d.normalizedCoordinates = YES;
	d.borderColor = (MTLSamplerBorderColor)border;
	id<MTLSamplerState> s = [device newSamplerStateWithDescriptor:d];
	if (s == nil)
		fail("no sampler state", nil);
	return s;
}

// A dispatch of `kernel` over the 16 by 16 grid in 8 by 8 threadgroups: `t` at texture 0 where
// given, the `count` buffers from buffer 0, `s` at sampler 0 where given, and `held` made
// resident for reading where given.
static void dispatch(const char *kernel, id<MTLTexture> t, id<MTLBuffer> *buffers, NSUInteger count,
                     id<MTLSamplerState> s, id<MTLTexture> held)
{
	id<MTLFunction> f = [library newFunctionWithName:@(kernel)];
	if (f == nil)
		fail(kernel, nil);
	NSError *error = nil;
	id<MTLComputePipelineState> state = [device newComputePipelineStateWithFunction:f error:&error];
	if (state == nil)
		fail(kernel, error);
	run(^(id<MTLCommandBuffer> cb) {
		id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
		NSUInteger offsets[4] = {0, 0, 0, 0};
		[e setComputePipelineState:state];
		[e setBuffers:buffers offsets:offsets withRange:NSMakeRange(0, count)];
		if (t != nil)
			[e setTexture:t atIndex:0];
		if (s != nil) {
			id<MTLSamplerState> list[1] = {s};
			[e setSamplerStates:list withRange:NSMakeRange(0, 1)];
		}
		if (held != nil) {
			id<MTLResource> list[1] = {held};
			[e useResources:list count:1 usage:MTLResourceUsageRead];
		}
		[e dispatchThreads:MTLSizeMake(SIDE, SIDE, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
		[e endEncoding];
	});
}

// What `kernel` writes into a buffer of `bytes` bytes a thread, `input` after it, over `t`,
// through `s` where given.
static NSData *output(const char *kernel, id<MTLTexture> t, size_t bytes, id<MTLBuffer> input,
                      id<MTLSamplerState> s)
{
	id<MTLBuffer> out = buffer(NULL, TEXELS * bytes);
	id<MTLBuffer> list[2] = {out, input};
	dispatch(kernel, t, list, 2, s, nil);
	return [NSData dataWithBytes:out.contents length:TEXELS * bytes];
}

// A kernel and the bytes it writes a thread.
typedef struct {
	const char *kernel;
	size_t bytes;
} Reader;

// Each reader's case and what it reads of `t`, of `class_of` (the test's `Gpu::reads`): read() in
// each type at each texel and 8 texels down and across; for floats also sample() as floats and
// halves through each of SAMPLERS, and gather() of each component through each of GATHERS; then
// read() of `t` through an argument buffer that holds its ID.
static void reads(ReadClass class_of, id<MTLTexture> t, NSMutableArray<NSString *> *cases,
                  NSMutableArray<NSData *> *answers)
{
	static const Reader floats[] = {{"rgbx_read_float", 16}, {"rgbx_read_half", 8}},
	                    uints[] = {{"rgbx_read_uint", 16}}, ints[] = {{"rgbx_read_int", 16}};
	const Reader *readers = class_of == FLOAT ? floats : class_of == UINT ? uints : ints;
	size_t n = class_of == FLOAT ? 2 : 1;
	for (size_t r = 0; r < n; r++) {
		for (uint32_t offset = 0; offset <= 8; offset += 8) {
			uint32_t at[2] = {offset, offset};
			id<MTLBuffer> offset_buffer = buffer(at, sizeof at);
			[answers addObject:output(readers[r].kernel, t, readers[r].bytes, offset_buffer, nil)];
			[cases addObject:[NSString stringWithFormat:@"%s at +[%u, %u]", readers[r].kernel, offset, offset]];
		}
	}
	if (class_of == FLOAT) {
		float where[2 * TEXELS];
		coordinates(where);
		id<MTLBuffer> at = buffer(where, sizeof where);
		for (int i = 0; i < 8; i++) {
			uint64_t filter = SAMPLERS[i][0], address = SAMPLERS[i][1], border = SAMPLERS[i][2];
			id<MTLSamplerState> s = sampler(filter, address, border);
			static const Reader samplers[] = {{"rgbx_sample_float", 16}, {"rgbx_sample_half", 8}};
			for (int k = 0; k < 2; k++) {
				[answers addObject:output(samplers[k].kernel, t, samplers[k].bytes, at, s)];
				[cases addObject:[NSString stringWithFormat:@"%s filter %llu address %llu border %llu",
				                                            samplers[k].kernel, filter, address, border]];
			}
		}
		for (int i = 0; i < 3; i++) {
			uint64_t filter = GATHERS[i][0], address = GATHERS[i][1], border = GATHERS[i][2];
			id<MTLSamplerState> s = sampler(filter, address, border);
			[answers addObject:output("rgbx_gather", t, 64, at, s)];
			[cases addObject:[NSString stringWithFormat:@"rgbx_gather filter %llu address %llu border %llu",
			                                            filter, address, border]];
		}
	}
	const char *kernel = class_of == FLOAT ? "rgbx_read_held_float" :
	                     class_of == UINT  ? "rgbx_read_held_uint" :
	                                         "rgbx_read_held_int";
	uint64_t resource_id = t.gpuResourceID._impl;
	id<MTLBuffer> held = buffer(&resource_id, sizeof resource_id);
	id<MTLBuffer> read = buffer(NULL, TEXELS * 16);
	id<MTLBuffer> list[2] = {read, held};
	dispatch(kernel, nil, list, 2, nil, t);
	[answers addObject:[NSData dataWithBytes:read.contents length:TEXELS * 16]];
	[cases addObject:@(kernel)];
}

int main(int argc, char **argv)
{
	@autoreleasepool {
		if (argc < 2) {
			fprintf(stderr, "usage: record <rgbx.metallib> [FORMAT...]\n");
			return 2;
		}
		device = MTLCreateSystemDefaultDevice();
		if (device == nil)
			fail("no Metal device", nil);
		queue = [device newCommandQueue];
		NSError *error = nil;
		library = [device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
		if (library == nil)
			fail(argv[1], error);
		fprintf(stderr, "record: %s, %s\n", device.name.UTF8String,
		        NSProcessInfo.processInfo.operatingSystemVersionString.UTF8String);
		for (int i = 0; i < 11; i++) {
			int wanted = argc == 2;
			for (int a = 2; a < argc; a++)
				wanted |= strcmp(argv[a], FORMATS[i].name) == 0;
			if (!wanted)
				continue;
			uint64_t seed = (uint64_t)i + 1;
			for (int o = 0; o < 2; o++) {
				uint64_t options = o == 0 ? SHARED : PRIVATE;
				size_t n = TEXELS * FORMATS[i].texel;
				uint8_t bytes[TEXELS * 16];
				uint64_t s = 0x9e3779b97f4a7c15ULL ^ seed;
				random_bytes(&s, bytes, n);
				id<MTLTexture> t = texture(FORMATS[i].raw, options, bytes, n);
				NSMutableArray<NSString *> *cases = [NSMutableArray array];
				NSMutableArray<NSData *> *answers = [NSMutableArray array];
				reads(FORMATS[i].class_of, t, cases, answers);
				NSString *of = [NSString stringWithFormat:@"%s options 0x%llx", FORMATS[i].name, options];
				printf("%s: cases\t%lu\n", of.UTF8String, (unsigned long)cases.count);
				for (NSUInteger c = 0; c < cases.count; c++)
					print_bytes([NSString stringWithFormat:@"%@ %@", of, cases[c]], answers[c]);
			}
		}
	}
	return 0;
}
