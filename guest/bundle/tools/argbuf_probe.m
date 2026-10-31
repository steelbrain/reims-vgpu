// argbuf_probe: argument buffers written by MTLArgumentEncoder and read by a kernel, in one
// process, compared with what Metal gives on an M3 Max (macOS 27).
//   argbuf_probe [DIR]  loads DIR/argbuf_probe.metallib (default: the executable's directory),
//                       prints the encoders' lengths and alignments (the device's from the
//                       descriptors, the function's and its nested one's), then runs the
//                       kernel over two argument buffers, one linear and repeating sampler and
//                       one nearest and clamping, each with a 2 x 2 texture, a buffer, a constant
//                       and a nested argument buffer of a texture and a sampler, and prints the
//                       eight values it read.
// Prints "ARGBUF OK" when the lengths are Metal's and every value is within 1/64 of Metal's,
// else "ARGBUF FAIL: why" and exits 1. A completion 20 s late prints HANG and exits 3.
// Build: xcrun clang -arch x86_64 -fobjc-arc -framework Metal -framework Foundation
//        argbuf_probe.m -o argbuf_probe
//        xcrun -sdk macosx metal -c argbuf_probe.metal -o argbuf_probe.air
//        xcrun -sdk macosx metallib argbuf_probe.air -o argbuf_probe.metallib
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(const char *why)
{
	printf("ARGBUF FAIL: %s\n", why);
	return 1;
}

static id<MTLSamplerState> sampler(id<MTLDevice> d, MTLSamplerMinMagFilter filter,
                                   MTLSamplerAddressMode mode)
{
	MTLSamplerDescriptor *s = [MTLSamplerDescriptor new];
	s.minFilter = filter;
	s.magFilter = filter;
	s.sAddressMode = mode;
	s.tAddressMode = mode;
	s.supportArgumentBuffers = YES;
	return [d newSamplerStateWithDescriptor:s];
}

int main(int argc, char **argv)
{
	@autoreleasepool {
		id<MTLDevice> d = MTLCreateSystemDefaultDevice();
		if (!d)
			return fail("no device");
		NSString *dir = argc > 1 ? @(argv[1])
		                         : [@(argv[0]) stringByDeletingLastPathComponent];
		NSURL *url = [NSURL fileURLWithPath:[dir stringByAppendingPathComponent:
		                                             @"argbuf_probe.metallib"]];
		NSError *e = nil;
		id<MTLLibrary> lib = [d newLibraryWithURL:url error:&e];
		if (!lib)
			return fail("no library");
		id<MTLFunction> fn = [lib newFunctionWithName:@"argbuf_probe"];
		id<MTLComputePipelineState> pso = [d newComputePipelineStateWithFunction:fn error:&e];
		if (!pso)
			return fail("no pipeline");

		// The device's encoder of the same arguments as the function's buffer 0.
		MTLArgumentDescriptor *(^arg)(MTLDataType, NSUInteger) =
		    ^(MTLDataType type, NSUInteger index) {
			    MTLArgumentDescriptor *a = [MTLArgumentDescriptor argumentDescriptor];
			    a.dataType = type;
			    a.index = index;
			    a.access = MTLBindingAccessReadOnly;
			    if (type == MTLDataTypeTexture)
				    a.textureType = MTLTextureType2D;
			    return a;
		    };
		id<MTLArgumentEncoder> byDescriptors = [d newArgumentEncoderWithArguments:@[
			arg(MTLDataTypeTexture, 0), arg(MTLDataTypeSampler, 1), arg(MTLDataTypePointer, 2),
			arg(MTLDataTypeFloat4, 3), arg(MTLDataTypePointer, 4)
		]];
		id<MTLArgumentEncoder> enc = [fn newArgumentEncoderWithBufferIndex:0];
		id<MTLArgumentEncoder> inner = [enc newArgumentEncoderForBufferAtIndex:4];
		if (!byDescriptors || !enc || !inner)
			return fail("no encoder");
		printf("descriptors: length %lu alignment %lu\n", (unsigned long)byDescriptors.encodedLength,
		       (unsigned long)byDescriptors.alignment);
		printf("function: length %lu alignment %lu\n", (unsigned long)enc.encodedLength,
		       (unsigned long)enc.alignment);
		printf("nested: length %lu alignment %lu\n", (unsigned long)inner.encodedLength,
		       (unsigned long)inner.alignment);
		// Metal's (M3 Max, macOS 27).
		if (byDescriptors.encodedLength != 64 || byDescriptors.alignment != 16 ||
		    enc.encodedLength != 64 || enc.alignment != 16 || inner.encodedLength != 16 ||
		    inner.alignment != 8)
			return fail("an encoder's length or alignment is not Metal's");

		// A 2 x 2 texture: red, green / blue, white.
		MTLTextureDescriptor *td =
		    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
		                                                       width:2
		                                                      height:2
		                                                   mipmapped:NO];
		td.storageMode = MTLStorageModePrivate;
		td.usage = MTLTextureUsageShaderRead;
		id<MTLTexture> tex = [d newTextureWithDescriptor:td];
		const float texels[16] = {1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1};
		id<MTLBuffer> staging = [d newBufferWithBytes:texels
		                                       length:sizeof(texels)
		                                      options:MTLResourceStorageModeShared];
		const float data[16] = {10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33, 40, 41, 42, 43};
		id<MTLBuffer> dataBuf = [d newBufferWithBytes:data
		                                       length:sizeof(data)
		                                      options:MTLResourceStorageModeShared];
		id<MTLSamplerState> linear = sampler(d, MTLSamplerMinMagFilterLinear,
		                                     MTLSamplerAddressModeRepeat);
		id<MTLSamplerState> nearest = sampler(d, MTLSamplerMinMagFilterNearest,
		                                      MTLSamplerAddressModeClampToEdge);
		if (!tex || !staging || !dataBuf || !linear || !nearest)
			return fail("no texture, buffer or sampler");

		const NSUInteger stride = enc.encodedLength;
		id<MTLBuffer> args = [d newBufferWithLength:2 * stride options:MTLResourceStorageModeShared];
		id<MTLBuffer> inners = [d newBufferWithLength:2 * inner.encodedLength
		                                      options:MTLResourceStorageModeShared];
		id<MTLSamplerState> samplers[2] = {linear, nearest};
		for (NSUInteger i = 0; i < 2; i++) {
			[enc setArgumentBuffer:args startOffset:0 arrayElement:i];
			[enc setTexture:tex atIndex:0];
			[enc setSamplerState:samplers[i] atIndex:1];
			[enc setBuffer:dataBuf offset:16 * i atIndex:2];
			float *k = (float *)[enc constantDataAtIndex:3];
			k[0] = 0.5f;
			k[1] = 0.25f;
			k[2] = 0.125f;
			k[3] = (float)i;
			[enc setBuffer:inners offset:inner.encodedLength * i atIndex:4];
			[inner setArgumentBuffer:inners offset:inner.encodedLength * i];
			[inner setTexture:tex atIndex:0];
			[inner setSamplerState:samplers[1 - i] atIndex:1];
		}

		id<MTLBuffer> out = [d newBufferWithLength:8 * 16 options:MTLResourceStorageModeShared];
		memset(out.contents, 0xff, 8 * 16);
		id<MTLCommandQueue> q = [d newCommandQueue];
		id<MTLCommandBuffer> cb = [q commandBuffer];
		id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
		[blit copyFromBuffer:staging
		           sourceOffset:0
		      sourceBytesPerRow:32
		    sourceBytesPerImage:64
		             sourceSize:MTLSizeMake(2, 2, 1)
		              toTexture:tex
		       destinationSlice:0
		       destinationLevel:0
		      destinationOrigin:MTLOriginMake(0, 0, 0)];
		[blit endEncoding];
		id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
		[ce setComputePipelineState:pso];
		[ce setBuffer:args offset:0 atIndex:0];
		[ce setBuffer:out offset:0 atIndex:1];
		[ce useResource:tex usage:MTLResourceUsageRead];
		[ce useResource:dataBuf usage:MTLResourceUsageRead];
		[ce useResource:inners usage:MTLResourceUsageRead];
		[ce dispatchThreads:MTLSizeMake(8, 1, 1) threadsPerThreadgroup:MTLSizeMake(8, 1, 1)];
		[ce endEncoding];
		dispatch_semaphore_t done = dispatch_semaphore_create(0);
		[cb addCompletedHandler:^(id<MTLCommandBuffer> b) {
			dispatch_semaphore_signal(done);
		}];
		[cb commit];
		if (dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 20 * NSEC_PER_SEC))) {
			printf("HANG\n");
			return 3;
		}
		if (cb.status != MTLCommandBufferStatusCompleted)
			return fail("the command buffer failed");

		// Metal's values (M3 Max, macOS 27), each within 1/64.
		const float want[8][4] = {
		    {0.30078125f, 0.69921875f, 0, 1}, // linear at x 0.6: red to green
		    {1, 0, 0, 1},                     // linear, repeating: x 1.25 is red's edge
		    {20.5f, 21.25f, 22.125f, 23},     // data[1] + k
		    {0, 0, 1, 1},                     // nested, nearest: blue
		    {0, 1, 0, 1},                     // nearest at x 0.6: green
		    {0, 1, 0, 1},                     // nearest, clamping: green's edge
		    {30.5f, 31.25f, 32.125f, 34},     // data[2] + k
		    {0, 0, 1, 1},                     // nested, linear at a texel's centre: blue
		};
		const float *got = (const float *)out.contents;
		int bad = 0;
		for (int i = 0; i < 8; i++) {
			printf("out[%d] = %.8f %.8f %.8f %.8f\n", i, got[4 * i], got[4 * i + 1],
			       got[4 * i + 2], got[4 * i + 3]);
			for (int c = 0; c < 4; c++)
				if (!(fabsf(got[4 * i + c] - want[i][c]) <= 1.0f / 64))
					bad++;
		}
		if (bad)
			return fail("a value is not Metal's");
		printf("ARGBUF OK\n");
	}
	return 0;
}
