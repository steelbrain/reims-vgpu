// Records Metal's answers to gpu/resource/tests/metal_tests.rs's
// compute_state_answers_are_metals_for_every_kernel_here on this Mac's own Metal, in the lines of
// its file of answers beside this source
// (metal_tests.compute_state_answers_are_metals_for_every_kernel_here.txt, the forms core's
// testkit::metal_answers reads). It runs what the test's `compute_answers` runs, as the deleted
// Metal platform sent it, with no part of this library in between: for each metallib it is
// given, `newLibraryWithData:error:`, then, for each of the library's `functionNames`,
// `newFunctionWithName:` and `newComputePipelineStateWithFunction:error:`. It prints one line a
// function, `<library> <function>\tSome((maxTotalThreadsPerThreadgroup, threadExecutionWidth,
// staticThreadgroupMemoryLength, supportIndirectCommandBuffers))`, or `None` where Metal makes no
// state; and `<library> the library\tNone` where it makes no library of the bytes.
//
// A function that is no kernel (a vertex, fragment or tile function, `functionType`) is answered
// `None` without asking for a state: Metal makes none of one, and its validation may stop the
// process at the ask.
//
// Provenance: on the M3 Max under macOS 27.0.1 (26A434), 2026-11-13, given the 50 metallibs the
// test reads (the build script's from its OUT_DIR), it printed each of the 197 answers the file
// held, recorded from the Metal platform on 2026-11-07, and one more, the line of the display's
// kernel `displayMirrorTile`, which was then added to the file from its output.
// On 2026-11-14, on the same Mac and macOS, given the 52 metallibs the test then read (the build
// script's 13 as their committed copies in crates/reims-vgpu-vulkan/metallibs), it printed each of
// the file's 198 answers and two more, those of the test shaders' vertex function `render_small`
// and fragment function `render_flat`, which were then added to the file from its output.
// Later on 2026-11-14, on the same Mac and macOS, given the same 52 metallibs (the display's
// shaders.metallib as committed with its kernel `displayMirrorTileList`), it printed each of the
// file's 200 answers and one more, that kernel's, which was then added to the file from its
// output.
// On 2026-11-18, on the same Mac and macOS, given the 53 metallibs the test then read (the 52 and
// the test shaders' render_tess_bump.metallib), it printed each of the file's 201 answers and
// three more, `None` for that library's two vertex functions `render_tess_bump` and
// `render_tess_bump_cp` and its fragment function `render_tess_facing`, which were then added
// to the file from its output.
//
// Build and run on a Mac, from the repository's root, with Xcode's tools, giving each metallib
// as `<key>=<path>`, the key the test names it by: a fixture of tests/shaders by its file name
// (fc.metallib left out, as the test leaves it), a metallib the build script writes to the
// crate's OUT_DIR by its stem:
//   T=$(mktemp -d)
//   xcrun -sdk macosx clang -O1 -framework Foundation -framework Metal \
//       crates/reims-vgpu-vulkan/src/gpu/resource/tests/metal_answers/metal_tests.compute_state_answers_are_metals_for_every_kernel_here.m \
//       -o "$T/record"
//   "$T/record" compute_add.metallib=tests/shaders/compute_add.metallib ... \
//       shaders="$OUT_DIR/shaders.metallib" ...
// Built without ARC: a run is short, and what it makes lives until it exits.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>

// Function names in byte order, as the test sorts them.
static NSInteger by_bytes(NSString *a, NSString *b, void *unused) {
	(void)unused;
	int c = strcmp([a UTF8String], [b UTF8String]);
	return c < 0 ? NSOrderedAscending : (c > 0 ? NSOrderedDescending : NSOrderedSame);
}

int main(int argc, const char **argv) {
	@autoreleasepool {
		id<MTLDevice> device = MTLCreateSystemDefaultDevice();
		if (!device) {
			fprintf(stderr, "no Metal device\n");
			return 1;
		}
		for (int i = 1; i < argc; i++) {
			const char *eq = strchr(argv[i], '=');
			if (!eq) {
				fprintf(stderr, "not <key>=<path>: %s\n", argv[i]);
				return 2;
			}
			NSString *key = [[NSString alloc] initWithBytes:argv[i]
			                                         length:(NSUInteger)(eq - argv[i])
			                                       encoding:NSUTF8StringEncoding];
			NSData *bytes = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:eq + 1]];
			if (!bytes) {
				fprintf(stderr, "cannot read %s\n", eq + 1);
				return 3;
			}
			dispatch_data_t data = dispatch_data_create(bytes.bytes, bytes.length, NULL,
			                                            DISPATCH_DATA_DESTRUCTOR_DEFAULT);
			NSError *error = nil;
			id<MTLLibrary> library = [device newLibraryWithData:data error:&error];
			if (!library) {
				printf("%s the library\tNone\n", [key UTF8String]);
				continue;
			}
			NSArray<NSString *> *names =
			    [library.functionNames sortedArrayUsingFunction:by_bytes context:NULL];
			for (NSString *name in names) {
				id<MTLFunction> function = [library newFunctionWithName:name];
				id<MTLComputePipelineState> state = nil;
				if (function && function.functionType == MTLFunctionTypeKernel) {
					state = [device newComputePipelineStateWithFunction:function error:&error];
				}
				if (state) {
					printf("%s %s\tSome((%lu, %lu, %lu, %s))\n", [key UTF8String],
					       [name UTF8String], (unsigned long)state.maxTotalThreadsPerThreadgroup,
					       (unsigned long)state.threadExecutionWidth,
					       (unsigned long)state.staticThreadgroupMemoryLength,
					       state.supportIndirectCommandBuffers ? "true" : "false");
				} else {
					printf("%s %s\tNone\n", [key UTF8String], [name UTF8String]);
				}
			}
		}
	}
	return 0;
}
