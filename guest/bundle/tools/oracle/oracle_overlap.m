// oracle_overlap: which function constant values Metal refuses for a function whose arguments of
// one argument table may take overlapping locations, each present under function constants, at
// which call and with which error. Shaders: oracle_overlap.metallib (oracle_overlap.metal), next
// to the probe.
//   oracle_overlap DEVICE
// For each function, each of useA and useB false and true (and, for k_idx, kIdx 0 and 1; kW 3),
// each way of making the specialised function:
//   values      newFunctionWithName:constantValues:error:, the values set by index
//   names       the same, set by name
//   range       the same, useA and useB set by setConstantValues:type:withRange:
//   copy        the same, of a copy of the values
//   descriptor  newFunctionWithDescriptor:error:
//   desc-copy   the same, of a copy of the descriptor
//   specialized the same, the descriptor naming a specializedName
//   async       newFunctionWithName:constantValues:completionHandler:
//   async-desc  newFunctionWithDescriptor:completionHandler:
// one line:
//   FUNCTION a=A b=B idx=I WAY: fn yes|nil [err DOMAIN CODE "DESCRIPTION"] | pipeline yes|nil|- [err ...]
// the description's newlines written \n, the pipeline a kernel's compute pipeline of the function
// made (- for a vertex or fragment function, or none made).
// Build: see README.md.
#import "oracle_common.h"

static id<MTLDevice> gDevice;

static void errorText(NSError *e)
{
	if (e == nil)
		return;
	NSString *d = [e.localizedDescription stringByReplacingOccurrencesOfString:@"\n" withString:@"\\n"];
	printf(" err %s %ld \"%s\"", e.domain.UTF8String, (long)e.code, d.UTF8String);
}

static void say(NSString *tag, id<MTLFunction> f, NSError *e)
{
	printf("%s: fn %s", tag.UTF8String, f ? "yes" : "nil");
	errorText(e);
	printf(" | pipeline ");
	if (f == nil || f.functionType != MTLFunctionTypeKernel) {
		printf("-\n");
		return;
	}
	NSError *pe = nil;
	id<MTLComputePipelineState> p = [gDevice newComputePipelineStateWithFunction:f error:&pe];
	printf("%s", p ? "yes" : "nil");
	errorText(pe);
	printf("\n");
}

static MTLFunctionConstantValues *values(bool a, bool b, uint32_t idx, int how)
{
	MTLFunctionConstantValues *v = [MTLFunctionConstantValues new];
	short w = 3;
	if (how == 1) {
		[v setConstantValue:&a type:MTLDataTypeBool withName:@"useA"];
		[v setConstantValue:&b type:MTLDataTypeBool withName:@"useB"];
		[v setConstantValue:&w type:MTLDataTypeShort withName:@"kW"];
		[v setConstantValue:&idx type:MTLDataTypeUInt withName:@"kIdx"];
		return v;
	}
	if (how == 2) {
		const bool both[2] = {a, b};
		[v setConstantValues:both type:MTLDataTypeBool withRange:NSMakeRange(0, 2)];
	} else {
		[v setConstantValue:&a type:MTLDataTypeBool atIndex:0];
		[v setConstantValue:&b type:MTLDataTypeBool atIndex:1];
	}
	[v setConstantValue:&w type:MTLDataTypeShort atIndex:2];
	[v setConstantValue:&idx type:MTLDataTypeUInt atIndex:3];
	return v;
}

int main(int argc, char **argv)
{
	@autoreleasepool {
		if (argc < 2) {
			fprintf(stderr, "usage: oracle_overlap DEVICE\n");
			return 2;
		}
		gDevice = oracle_device(argv[1]);
		oracle_header("oracle_overlap", gDevice);
		id<MTLLibrary> lib = oracle_library(gDevice, argv[0], "oracle_overlap");
		if (lib == nil)
			return 1;
		NSArray *functions = @[
			@"k_overlap", @"k_tex", @"k_texarr", @"k_smp", @"k_tg", @"k_derived", @"k_idx", @"k_bufconst",
			@"k_unused", @"k_dead", @"k_argbuf", @"k_three", @"v_overlap", @"f_overlap"
		];
		const bool cases[4][2] = {{false, false}, {true, false}, {false, true}, {true, true}};
		for (NSString *name in functions) {
			const int idxCount = [name isEqual:@"k_idx"] ? 2 : 1;
			for (int c = 0; c < 4; c++) {
				for (uint32_t idx = 0; idx < (uint32_t)idxCount; idx++) {
					const bool a = cases[c][0], b = cases[c][1];
					NSString *tag = [NSString stringWithFormat:@"%@ a=%d b=%d idx=%u", name, a, b, idx];
					NSError *e = nil;
					say([tag stringByAppendingString:@" values"],
					    [lib newFunctionWithName:name constantValues:values(a, b, idx, 0) error:&e], e);
					e = nil;
					say([tag stringByAppendingString:@" names"],
					    [lib newFunctionWithName:name constantValues:values(a, b, idx, 1) error:&e], e);
					e = nil;
					say([tag stringByAppendingString:@" range"],
					    [lib newFunctionWithName:name constantValues:values(a, b, idx, 2) error:&e], e);
					e = nil;
					say([tag stringByAppendingString:@" copy"],
					    [lib newFunctionWithName:name constantValues:[values(a, b, idx, 0) copy] error:&e], e);
					MTLFunctionDescriptor *fd = [MTLFunctionDescriptor functionDescriptor];
					fd.name = name;
					fd.constantValues = values(a, b, idx, 0);
					e = nil;
					say([tag stringByAppendingString:@" descriptor"], [lib newFunctionWithDescriptor:fd error:&e], e);
					e = nil;
					say([tag stringByAppendingString:@" desc-copy"], [lib newFunctionWithDescriptor:[fd copy] error:&e],
					    e);
					MTLFunctionDescriptor *sd = [fd copy];
					sd.specializedName = [name stringByAppendingString:@"_spec"];
					e = nil;
					say([tag stringByAppendingString:@" specialized"], [lib newFunctionWithDescriptor:sd error:&e], e);
					dispatch_semaphore_t done = dispatch_semaphore_create(0);
					__block id<MTLFunction> got = nil;
					__block NSError *gotError = nil;
					[lib newFunctionWithName:name
					          constantValues:values(a, b, idx, 0)
					       completionHandler:^(id<MTLFunction> f, NSError *err) {
						       got = f;
						       gotError = err;
						       dispatch_semaphore_signal(done);
					       }];
					dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
					say([tag stringByAppendingString:@" async"], got, gotError);
					got = nil;
					gotError = nil;
					[lib newFunctionWithDescriptor:fd
					             completionHandler:^(id<MTLFunction> f, NSError *err) {
						             got = f;
						             gotError = err;
						             dispatch_semaphore_signal(done);
					             }];
					dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
					say([tag stringByAppendingString:@" async-desc"], got, gotError);
				}
			}
		}
	}
	return 0;
}
