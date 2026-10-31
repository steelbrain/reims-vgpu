// ci_probe: Core Image color kernels on the default Metal device, whose pixels must come back as
// the kernels compute them. Core Image makes each kernel's library from the kernel library's
// bytes, links the kernel with its own helpers on the CPU into a library of one kernel, and runs
// that kernel's compute pipeline; on the reims-graphics device the link takes the libraries'
// shadows (RgfxLibrary) and the linked library is made the device's own.
//   ci_probe KERNELS    renders a 16 x 16 image of one color through each kernel of KERNELS (the
//                       shell tests' kernel library, tests/ci_kernel.metal: `make
//                       build/ci_kernel.metallib`) with no color management, and checks every
//                       pixel: swap_red_blue swaps red and blue, scale_red_green(0.5) halves red
//                       and green. Prints each kernel's first pixel and exits 0 when all match.
//   ci_probe stitch STITCHABLE
//                       stitches `fused` = (a + b) * c from the stitchable functions of STITCHABLE
//                       (tests/stitchable.metal: `make build/stitchable.metallib`), links it as a
//                       private function of its stitch_caller kernel, runs one thread on 2, 3 and 4,
//                       and checks the 20 it writes. Exits 0 when it matches.
//
// Build: xcrun clang -arch x86_64 -fobjc-arc -framework Metal -framework Foundation
//        -framework CoreImage tools/ci_probe.m -o ci_probe

#import <CoreImage/CoreImage.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { kSide = 16 };

// Renders `image` (kSide x kSide) through `context` into RGBA8 bytes and checks each pixel against
// `want` within one step; prints the first pixel. Answers whether all match.
static BOOL renderMatches(CIContext *context, CIImage *image, const uint8_t want[4], const char *what)
{
	uint8_t px[kSide * kSide * 4] = {0};
	[context render:image
	       toBitmap:px
	       rowBytes:kSide * 4
	         bounds:CGRectMake(0, 0, kSide, kSide)
	         format:kCIFormatRGBA8
	     colorSpace:nil];
	printf("%s: first pixel %u %u %u %u, want %u %u %u %u\n", what, px[0], px[1], px[2], px[3], want[0],
	       want[1], want[2], want[3]);
	for (int i = 0; i < kSide * kSide; i++) {
		for (int c = 0; c < 4; c++) {
			if (abs((int)px[i * 4 + c] - (int)want[c]) > 1) {
				printf("%s: pixel %d channel %d is %u\n", what, i, c, px[i * 4 + c]);
				return NO;
			}
		}
	}
	return YES;
}

// The stitch mode: answers the process's exit status.
static int stitchMatches(id<MTLDevice> device, NSData *stitchable)
{
	NSError *error = nil;
	dispatch_data_t data = dispatch_data_create(stitchable.bytes, stitchable.length, nil,
	                                            DISPATCH_DATA_DESTRUCTOR_DEFAULT);
	id<MTLLibrary> lib = [device newLibraryWithData:data error:&error];
	if (lib == nil) {
		printf("library: %s\n", error.description.UTF8String);
		return 1;
	}
	MTLFunctionStitchingInputNode *a = [[MTLFunctionStitchingInputNode alloc] initWithArgumentIndex:0];
	MTLFunctionStitchingInputNode *b = [[MTLFunctionStitchingInputNode alloc] initWithArgumentIndex:1];
	MTLFunctionStitchingInputNode *c = [[MTLFunctionStitchingInputNode alloc] initWithArgumentIndex:2];
	MTLFunctionStitchingFunctionNode *sum = [[MTLFunctionStitchingFunctionNode alloc] initWithName:@"stitch_add"
	                                                                                      arguments:@[ a, b ]
	                                                                            controlDependencies:@[]];
	MTLFunctionStitchingFunctionNode *product =
	    [[MTLFunctionStitchingFunctionNode alloc] initWithName:@"stitch_mul"
	                                                 arguments:@[ sum, c ]
	                                       controlDependencies:@[]];
	MTLStitchedLibraryDescriptor *sd = [[MTLStitchedLibraryDescriptor alloc] init];
	sd.functionGraphs = @[ [[MTLFunctionStitchingGraph alloc] initWithFunctionName:@"fused"
	                                                                         nodes:@[ sum ]
	                                                                    outputNode:product
	                                                                    attributes:@[]] ];
	sd.functions = @[ [lib newFunctionWithName:@"stitch_add"], [lib newFunctionWithName:@"stitch_mul"] ];
	id<MTLLibrary> stitched = [device newLibraryWithStitchedDescriptor:sd error:&error];
	if (stitched == nil) {
		printf("stitched library: %s\n", error.description.UTF8String);
		return 1;
	}
	printf("stitched library: functions %s\n", [[stitched.functionNames componentsJoinedByString:@","] UTF8String]);
	MTLComputePipelineDescriptor *pd = [[MTLComputePipelineDescriptor alloc] init];
	pd.computeFunction = [lib newFunctionWithName:@"stitch_caller"];
	MTLLinkedFunctions *linked = [MTLLinkedFunctions linkedFunctions];
	linked.privateFunctions = @[ [stitched newFunctionWithName:@"fused"] ];
	pd.linkedFunctions = linked;
	id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithDescriptor:pd
	                                                                             options:MTLPipelineOptionNone
	                                                                          reflection:nil
	                                                                               error:&error];
	if (pipeline == nil) {
		printf("pipeline: %s\n", error.description.UTF8String);
		return 1;
	}
	const float in[3] = {2, 3, 4};
	id<MTLBuffer> input = [device newBufferWithBytes:in length:sizeof(in) options:MTLResourceStorageModeShared];
	id<MTLBuffer> output = [device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
	id<MTLCommandQueue> queue = [device newCommandQueue];
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
	[enc setComputePipelineState:pipeline];
	[enc setBuffer:output offset:0 atIndex:0];
	[enc setBuffer:input offset:0 atIndex:1];
	[enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
	[enc endEncoding];
	[cb commit];
	[cb waitUntilCompleted];
	const float got = *(const float *)output.contents;
	printf("stitch: (2 + 3) * 4 = %g, command buffer status %ld\n", got, (long)cb.status);
	printf("%s\n", got == 20 ? "ci_probe: OK" : "ci_probe: FAILED");
	return got == 20 ? 0 : 1;
}

int main(int argc, char **argv)
{
	@autoreleasepool {
		if (argc == 3 && strcmp(argv[1], "stitch") == 0) {
			NSData *stitchable = [NSData dataWithContentsOfFile:@(argv[2])];
			id<MTLDevice> device = MTLCreateSystemDefaultDevice();
			printf("device %s\n", device ? device.name.UTF8String : "(none)");
			if (stitchable == nil || device == nil)
				return stitchable == nil ? 2 : 1;
			return stitchMatches(device, stitchable);
		}
		if (argc != 2) {
			fprintf(stderr, "usage: ci_probe KERNELS.metallib | ci_probe stitch STITCHABLE.metallib\n");
			return 2;
		}
		NSData *kernels = [NSData dataWithContentsOfFile:@(argv[1])];
		if (kernels == nil) {
			fprintf(stderr, "ci_probe: cannot read %s\n", argv[1]);
			return 2;
		}
		id<MTLDevice> device = MTLCreateSystemDefaultDevice();
		printf("device %s\n", device ? device.name.UTF8String : "(none)");
		if (device == nil)
			return 1;
		CIContext *context = [CIContext contextWithMTLDevice:device
		                                             options:@{
			                                             kCIContextWorkingColorSpace : [NSNull null],
			                                             kCIContextOutputColorSpace : [NSNull null],
			                                             kCIContextCacheIntermediates : @NO,
		                                             }];
		CIImage *source = [[CIImage imageWithColor:[CIColor colorWithRed:1 green:0.5 blue:0.25 alpha:1]]
		    imageByCroppingToRect:CGRectMake(0, 0, kSide, kSide)];

		NSError *error = nil;
		CIColorKernel *swap = [CIColorKernel kernelWithFunctionName:@"swap_red_blue"
		                                       fromMetalLibraryData:kernels
		                                                      error:&error];
		CIColorKernel *scale = [CIColorKernel kernelWithFunctionName:@"scale_red_green"
		                                        fromMetalLibraryData:kernels
		                                                       error:&error];
		if (swap == nil || scale == nil) {
			printf("kernels: %s\n", error.description.UTF8String);
			return 1;
		}
		const uint8_t swapped[4] = {64, 128, 255, 255};
		const uint8_t scaled[4] = {128, 64, 64, 255};
		BOOL ok = renderMatches(context, [swap applyWithExtent:source.extent arguments:@[ source ]], swapped,
		                        "swap_red_blue");
		ok = renderMatches(context, [scale applyWithExtent:source.extent arguments:@[ source, @0.5 ]], scaled,
		                   "scale_red_green") &&
		     ok;
		printf("%s\n", ok ? "ci_probe: OK" : "ci_probe: FAILED");
		return ok ? 0 : 1;
	}
}
