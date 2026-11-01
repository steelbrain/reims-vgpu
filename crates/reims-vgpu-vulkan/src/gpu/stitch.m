// The build's stitcher (build.rs): writes the library Metal's stitcher makes of the graph `fused`
// over the functions of the metallib it is given, the shell tests' stitchable.metal, as the bytes
// Metal answers for it, which the guest's route sends (the bundle's RgfxStitchOverShadows).
// `fused` is `stitch_mul(stitch_add(a, b), c)`, the graph the shell tests stitch.
//
// Usage: stitch <stitchable.metallib> <stitched.metallib>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// Metal's library answers its bytes to this private selector (measured: macOS 27).
@interface NSObject (StitchLibraryData)
- (NSData *)libraryDataContents;
@end

int main(int argc, char **argv)
{
	@autoreleasepool {
		if (argc != 3) {
			fprintf(stderr, "usage: stitch <stitchable.metallib> <stitched.metallib>\n");
			return 2;
		}
		id<MTLDevice> device = MTLCreateSystemDefaultDevice();
		if (device == nil) {
			fprintf(stderr, "stitch: no Metal device\n");
			return 1;
		}
		NSError *error = nil;
		id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
		if (library == nil) {
			fprintf(stderr, "stitch: %s: %s\n", argv[1], error.description.UTF8String);
			return 1;
		}
		MTLFunctionStitchingInputNode *a = [[MTLFunctionStitchingInputNode alloc] initWithArgumentIndex:0];
		MTLFunctionStitchingInputNode *b = [[MTLFunctionStitchingInputNode alloc] initWithArgumentIndex:1];
		MTLFunctionStitchingInputNode *c = [[MTLFunctionStitchingInputNode alloc] initWithArgumentIndex:2];
		MTLFunctionStitchingFunctionNode *sum =
		    [[MTLFunctionStitchingFunctionNode alloc] initWithName:@"stitch_add"
		                                                 arguments:@[ a, b ]
		                                       controlDependencies:@[]];
		MTLFunctionStitchingFunctionNode *product =
		    [[MTLFunctionStitchingFunctionNode alloc] initWithName:@"stitch_mul"
		                                                 arguments:@[ sum, c ]
		                                       controlDependencies:@[]];
		MTLStitchedLibraryDescriptor *descriptor = [MTLStitchedLibraryDescriptor new];
		descriptor.functionGraphs = @[ [[MTLFunctionStitchingGraph alloc] initWithFunctionName:@"fused"
		                                                                                 nodes:@[ sum ]
		                                                                            outputNode:product
		                                                                            attributes:@[]] ];
		descriptor.functions = @[
			[library newFunctionWithName:@"stitch_add"], [library newFunctionWithName:@"stitch_mul"]
		];
		id<MTLLibrary> stitched = [device newLibraryWithStitchedDescriptor:descriptor error:&error];
		if (stitched == nil) {
			fprintf(stderr, "stitch: %s\n", error.description.UTF8String);
			return 1;
		}
		NSData *bytes = [(NSObject *)stitched libraryDataContents];
		if (bytes.length == 0 || ![bytes writeToFile:@(argv[2]) atomically:YES]) {
			fprintf(stderr, "stitch: no bytes written to %s\n", argv[2]);
			return 1;
		}
	}
	return 0;
}
