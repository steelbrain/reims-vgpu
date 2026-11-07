// stitch_probe: a stitched library over functions made with constant values, as MPS makes its
// uber shaders' visible functions (MPSLibrary::CreateUberShaderKey: newFunctionWithDescriptor:
// with constant values and a specialized name, then newLibraryWithStitchedDescriptor:), on the
// default Metal device, and the stitched function run.
//   stitch_probe          compiles a library from source with two stitchable functions, one
//                         reading a function constant; makes `scaled` with the constant set (2.0)
//                         under the specialized name `scaled_k2`, by a descriptor, and `bias` with
//                         constant values by name (5.0), and `add1` plain; stitches
//                         `fused(x) = add1(bias(scaled_k2(x)))`; links `fused` into a kernel through
//                         a visible function table, runs it over 64 elements and checks each
//                         result, 2 * i + 5 + 1. Prints each step's object or error. Exits 0 when
//                         every step made its object and every result matched.
//   stitch_probe --metal  the same steps, with no run, on Metal's own classes: the functions made
//                         by the library's shadow (Metal's own library over the same bytes, which
//                         the device keeps beside its own) and stitched by the device's base
//                         class: what the device's stitch rests on.
//
// Build: xcrun clang -arch x86_64 -fobjc-arc -mmacosx-version-min=26.0 -framework Metal
//        -framework Foundation tools/stitch_probe.m -o stitch_probe

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <objc/message.h>
#include <objc/runtime.h>
#include <stdio.h>

static NSString *const kSource = @"#include <metal_stdlib>\n"
                                  "using namespace metal;\n"
                                  "constant float k [[function_constant(0)]];\n"
                                  "constant float b [[function_constant(1)]];\n"
                                  "[[stitchable]] float scaled(float x) { return x * k; }\n"
                                  "[[stitchable]] float bias(float x) { return x + b; }\n"
                                  "[[stitchable]] float add1(float x) { return x + 1.0f; }\n"
                                  "kernel void run(device float *out [[buffer(0)]],\n"
                                  "                visible_function_table<float(float)> t [[buffer(1)]],\n"
                                  "                uint i [[thread_position_in_grid]]) {\n"
                                  "  out[i] = t[0](float(i));\n"
                                  "}\n";

static int g_failed = 0;

static void step(const char *what, id made, NSError *error)
{
	if (made == nil)
		g_failed = 1;
	printf("%-10s %s%s%s\n", what, made ? "made " : "FAILED ",
	       made ? [NSStringFromClass([made class]) UTF8String] : "",
	       error ? [[NSString stringWithFormat:@" (%@)", error.localizedDescription] UTF8String]
	             : "");
}

// The base class's implementation of `sel` on `device`: Metal's own, not the device's override.
static IMP baseImp(id<MTLDevice> device, SEL sel)
{
	return class_getMethodImplementation(class_getSuperclass(object_getClass(device)), sel);
}

int main(int argc, char **argv)
{
	@autoreleasepool {
		setvbuf(stdout, NULL, _IONBF, 0);
		const BOOL metal = argc > 1 && strcmp(argv[1], "--metal") == 0;
		id<MTLDevice> device = MTLCreateSystemDefaultDevice();
		printf("device     %s (%s)\n", device.name.UTF8String,
		       NSStringFromClass(object_getClass(device)).UTF8String);
		NSError *error = nil;
		id<MTLLibrary> lib = [device newLibraryWithSource:kSource options:nil error:&error];
		step("library", lib, error);
		if (lib != nil && metal) {
			SEL shadow = sel_registerName("rgfxShadow");
			lib = [(id)lib respondsToSelector:shadow] ? ((id (*)(id, SEL))objc_msgSend)(lib, shadow) : nil;
			step("shadow", lib, nil);
		}
		if (lib == nil)
			return 1;

		MTLFunctionConstantValues *k = [MTLFunctionConstantValues new];
		float two = 2.0f, five = 5.0f;
		[k setConstantValue:&two type:MTLDataTypeFloat atIndex:0];
		MTLFunctionDescriptor *d = [MTLFunctionDescriptor functionDescriptor];
		d.name = @"scaled";
		d.specializedName = @"scaled_k2";
		d.constantValues = k;
		error = nil;
		id<MTLFunction> scaled = [lib newFunctionWithDescriptor:d error:&error];
		step("scaled_k2", scaled, error);
		MTLFunctionConstantValues *bv = [MTLFunctionConstantValues new];
		[bv setConstantValue:&five type:MTLDataTypeFloat withName:@"b"];
		error = nil;
		id<MTLFunction> bias = [lib newFunctionWithName:@"bias" constantValues:bv error:&error];
		step("bias", bias, error);
		id<MTLFunction> add1 = [lib newFunctionWithName:@"add1"];
		step("add1", add1, nil);
		if (scaled == nil || bias == nil || add1 == nil)
			return 1;
		printf("names      %s %s %s\n", scaled.name.UTF8String, bias.name.UTF8String,
		       add1.name.UTF8String);

		MTLFunctionStitchingInputNode *x = [[MTLFunctionStitchingInputNode alloc] initWithArgumentIndex:0];
		MTLFunctionStitchingFunctionNode *s =
		    [[MTLFunctionStitchingFunctionNode alloc] initWithName:scaled.name arguments:@[ x ] controlDependencies:@[]];
		MTLFunctionStitchingFunctionNode *b =
		    [[MTLFunctionStitchingFunctionNode alloc] initWithName:bias.name arguments:@[ s ] controlDependencies:@[]];
		MTLFunctionStitchingFunctionNode *a =
		    [[MTLFunctionStitchingFunctionNode alloc] initWithName:@"add1" arguments:@[ b ] controlDependencies:@[]];
		MTLStitchedLibraryDescriptor *sd = [MTLStitchedLibraryDescriptor new];
		sd.functionGraphs = @[ [[MTLFunctionStitchingGraph alloc] initWithFunctionName:@"fused"
		                                                                         nodes:@[ s, b ]
		                                                                    outputNode:a
		                                                                    attributes:@[]] ];
		sd.functions = @[ scaled, bias, add1 ];
		error = nil;
		id<MTLLibrary> stitched =
		    metal ? ((id (*)(id, SEL, id, NSError **))baseImp(
		                device, @selector(newLibraryWithStitchedDescriptor:error:)))(
		                device, @selector(newLibraryWithStitchedDescriptor:error:), sd, &error)
		          : [device newLibraryWithStitchedDescriptor:sd error:&error];
		step("stitched", stitched, error);
		id<MTLFunction> fused = [stitched newFunctionWithName:@"fused"];
		step("fused", fused, nil);
		if (fused == nil || metal)
			return g_failed;

		// The stitched function run: out[i] = fused(i) = 2 * i + 5 + 1.
		id<MTLFunction> kernel = [lib newFunctionWithName:@"run"];
		MTLComputePipelineDescriptor *pd = [MTLComputePipelineDescriptor new];
		pd.computeFunction = kernel;
		MTLLinkedFunctions *linked = [MTLLinkedFunctions linkedFunctions];
		linked.functions = @[ fused ];
		pd.linkedFunctions = linked;
		error = nil;
		id<MTLComputePipelineState> pso = [device newComputePipelineStateWithDescriptor:pd
		                                                                       options:0
		                                                                    reflection:nil
		                                                                         error:&error];
		step("pipeline", pso, error);
		if (pso == nil)
			return 1;
		MTLVisibleFunctionTableDescriptor *td = [MTLVisibleFunctionTableDescriptor new];
		td.functionCount = 1;
		id<MTLVisibleFunctionTable> table = [pso newVisibleFunctionTableWithDescriptor:td];
		[table setFunction:[pso functionHandleWithFunction:fused] atIndex:0];
		step("table", table, nil);
		enum { kCount = 64 };
		id<MTLBuffer> out = [device newBufferWithLength:kCount * sizeof(float)
		                                        options:MTLResourceStorageModeShared];
		id<MTLCommandQueue> queue = [device newCommandQueue];
		id<MTLCommandBuffer> cb = [queue commandBuffer];
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
		[enc setComputePipelineState:pso];
		[enc setBuffer:out offset:0 atIndex:0];
		[enc setVisibleFunctionTable:table atBufferIndex:1];
		[enc dispatchThreads:MTLSizeMake(kCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(kCount, 1, 1)];
		[enc endEncoding];
		[cb commit];
		[cb waitUntilCompleted];
		const float *v = (const float *)out.contents;
		int wrong = 0;
		for (int i = 0; i < kCount; i++)
			if (v[i] != 2.0f * i + 6.0f)
				wrong++;
		printf("run        %s: %d of %d wrong (out[3] = %g, want %g)\n",
		       cb.error ? cb.error.localizedDescription.UTF8String : "completed", wrong, kCount,
		       v[3], 2.0f * 3 + 6.0f);
		return g_failed || wrong || cb.error ? 1 : 0;
	}
}
