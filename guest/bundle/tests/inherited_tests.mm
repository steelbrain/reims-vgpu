// The device's base classes' creators: every one is answered by the device itself or is in this
// file's list of those left to the base, each with why, so a creator a new macOS adds to
// IOGPUMetalDevice or _MTLDevice shows here rather than answering nil on the guest: inherited, a
// creator builds on the base device's compiler or IOGPU objects, which the device does not have,
// and answers nil with no error (as WindowServer's luma pipeline did, which ended it).
//   inherited_tests BUNDLE   loads BUNDLE, the device's x86-64 bundle (run under Rosetta, as the
//                            guest runs it), and enumerates both base classes at run time.
// It fails on:
//   - a base-class creator the device inherits that the list does not name: a selector of either
//     base class starting with new, compile, load, alloc, functionHandle or resolve;
//   - a listed selector the device answers itself (the list would be stale);
//   - a selector the device answers that a base class also has, with another number of arguments
//     or an argument or return value of another size.
// A listed selector this macOS's base classes lack is reported, not failed: the guest's macOS can
// differ.
#import <Foundation/Foundation.h>

#include <dlfcn.h>
#include <objc/runtime.h>
#include <stdio.h>
#include <string.h>

namespace {

// Why a base-class creator is left to the base.
constexpr const char *kCompiler =
    "the base's compiler and its thread priority, which Metal's own library routes ask on the "
    "device (the link and stitch the device takes through super); nil there";
constexpr const char *kVariants =
    "the compiler's back half: pipelines and variants of the base compiler's output, which only "
    "the base's own creators have to give";
constexpr const char *kSerialized =
    "descriptors and struct types rebuilt from serialized bytes: plain objects, of no device state";
constexpr const char *kValues =
    "plain values Metal's compiler routes make: translated compiler options, a DAG's text, an "
    "error";
constexpr const char *kDynamicLoads =
    "the dynamic libraries of a pipeline's functions, which Metal's own pipeline routes load "
    "(measured on the M3 Max: Core Image's and a source library's pipelines); the device's "
    "functions name none";
constexpr const char *kSubAllocation =
    "sub-allocations of Metal's own command buffers' memory (measured on the M3 Max: Core Image "
    "asks the device); the device's command buffers are its own";

struct Left {
	const char *selector;
	const char *why;
};

constexpr Left kLeftToTheBase[] = {
    {"compiler", kCompiler},
    {"compilerPropagatesThreadPriority:", kCompiler},
    {"newComputePipelineWithDescriptor:variant:", kVariants},
    {"newComputePipelineWithDescriptor:variant:errorMessage:", kVariants},
    {"newRenderPipelineWithDescriptor:vertexVariant:fragmentVariant:", kVariants},
    {"newRenderPipelineWithDescriptor:vertexVariant:fragmentVariant:errorMessage:", kVariants},
    {"newRenderPipelineWithDescriptor:objectVariant:meshVariant:fragmentVariant:errorMessage:",
     kVariants},
    {"newRenderPipelineWithTileDescriptor:tileVariant:errorMessage:", kVariants},
    {"newComputeVariantWithCompilerOutput:pipelineStatisticsOutput:", kVariants},
    {"newVertexVariantWithCompilerOutput:pipelineStatisticsOutput:", kVariants},
    {"newFragmentVariantWithCompilerOutput:pipelineStatisticsOutput:", kVariants},
    {"newObjectVariantWithCompilerOutput:pipelineStatisticsOutput:", kVariants},
    {"newMeshVariantWithCompilerOutput:pipelineStatisticsOutput:", kVariants},
    {"newTileVariantWithCompilerOutput:pipelineStatisticsOutput:functionType:", kVariants},
    {"newComputePipelineDescriptorWithSerializedData:deserializationContext:", kSerialized},
    {"newRenderPipelineDescriptorWithSerializedData:deserializationContext:", kSerialized},
    {"newTileRenderPipelineDescriptorWithSerializedData:deserializationContext:", kSerialized},
    {"newMeshRenderPipelineDescriptorWithSerializedData:deserializationContext:", kSerialized},
    {"newStructTypeWithSerializedData:", kSerialized},
    {"newTranslatedDriverCompilerOptions:compilerOptionsSize:", kValues},
    {"newDagStringWithGraphs:", kValues},
    {"newDynamicLibraryError:message:", kValues},
    {"loadDynamicLibrariesForComputeDescriptor:error:", kDynamicLoads},
    {"loadDynamicLibrariesForComputeDescriptor:options:error:", kDynamicLoads},
    {"loadDynamicLibrariesForFunction:insertLibraries:error:", kDynamicLoads},
    {"loadDynamicLibrariesForFunction:insertLibraries:options:error:", kDynamicLoads},
    {"loadDynamicLibrariesForFunctionDescriptor:insertLibraries:error:", kDynamicLoads},
    {"loadDynamicLibrariesForFunctionDescriptor:insertLibraries:options:error:", kDynamicLoads},
    {"loadLibrariesRecursive:dylibs:insertLibraries:options:error:", kDynamicLoads},
    {"allocBufferSubDataWithLength:options:alignment:heapIndex:bufferIndex:bufferOffset:",
     kSubAllocation},
    {"allocBufferSubDataWithLength:options:alignment:heapIndex:bufferIndex:bufferOffset:"
     "parentAddress:parentLength:",
     kSubAllocation},
};

constexpr const char *kCreatorPrefixes[] = {"new", "compile", "load", "alloc", "functionHandle",
                                            "resolve"};

int g_failures;

void fail(NSString *message)
{
	printf("FAIL: %s\n", message.UTF8String);
	g_failures++;
}

BOOL isCreator(NSString *name)
{
	for (const char *prefix : kCreatorPrefixes) {
		if ([name hasPrefix:@(prefix)])
			return YES;
	}
	return NO;
}

const Left *leftToTheBase(NSString *name)
{
	for (const Left &left : kLeftToTheBase) {
		if ([name isEqualToString:@(left.selector)])
			return &left;
	}
	return nullptr;
}

// The class's own instance methods (its categories' included), by selector name.
NSDictionary<NSString *, NSValue *> *ownMethods(Class cls)
{
	NSMutableDictionary<NSString *, NSValue *> *out = [NSMutableDictionary dictionary];
	unsigned n = 0;
	Method *ms = class_copyMethodList(cls, &n);
	for (unsigned i = 0; i < n; i++)
		out[@(sel_getName(method_getName(ms[i])))] = [NSValue valueWithPointer:ms[i]];
	free(ms);
	return out;
}

NSUInteger sizeOfType(const char *type)
{
	NSUInteger size = 0, align = 0;
	NSGetSizeAndAlignment(type, &size, &align);
	return size;
}

// Where `ours` and `base`'s signatures differ in arity or in an argument's or the return's size, a
// description of it; else nil.
NSString *signatureDifference(Method ours, Method base)
{
	NSMethodSignature *a = [NSMethodSignature signatureWithObjCTypes:method_getTypeEncoding(ours)];
	NSMethodSignature *b = [NSMethodSignature signatureWithObjCTypes:method_getTypeEncoding(base)];
	if (a.numberOfArguments != b.numberOfArguments)
		return [NSString stringWithFormat:@"%lu arguments, the base's %lu",
		                                  (unsigned long)a.numberOfArguments,
		                                  (unsigned long)b.numberOfArguments];
	if (a.methodReturnLength != b.methodReturnLength)
		return [NSString stringWithFormat:@"a return of %lu bytes, the base's %lu",
		                                  (unsigned long)a.methodReturnLength,
		                                  (unsigned long)b.methodReturnLength];
	for (NSUInteger i = 2; i < a.numberOfArguments; i++) {
		const NSUInteger sa = sizeOfType([a getArgumentTypeAtIndex:i]);
		const NSUInteger sb = sizeOfType([b getArgumentTypeAtIndex:i]);
		if (sa != sb)
			return [NSString
			    stringWithFormat:@"argument %lu of %lu bytes (%s), the base's %lu (%s)",
			                     (unsigned long)(i - 2), (unsigned long)sa,
			                     [a getArgumentTypeAtIndex:i], (unsigned long)sb,
			                     [b getArgumentTypeAtIndex:i]];
	}
	return nil;
}

} // namespace

int main(int argc, char **argv)
{
	@autoreleasepool {
		if (argc != 2) {
			fprintf(stderr, "usage: inherited_tests BUNDLE\n");
			return 2;
		}
		if (dlopen(argv[1], RTLD_NOW) == nullptr) {
			printf("FAIL: cannot load %s: %s\n", argv[1], dlerror());
			return 1;
		}
		Class device = objc_getClass("RgfxMetalDevice");
		Class bases[] = {objc_getClass("IOGPUMetalDevice"), objc_getClass("_MTLDevice")};
		if (device == Nil || bases[0] == Nil || bases[1] == Nil ||
		    class_getSuperclass(device) != bases[0] || class_getSuperclass(bases[0]) != bases[1]) {
			printf("FAIL: RgfxMetalDevice is not a subclass of IOGPUMetalDevice, of _MTLDevice\n");
			return 1;
		}
		NSDictionary<NSString *, NSValue *> *ours = ownMethods(device);
		NSMutableSet<NSString *> *inBase = [NSMutableSet set];
		NSUInteger creators = 0, answered = 0, left = 0;
		for (Class base : bases) {
			NSDictionary<NSString *, NSValue *> *theirs = ownMethods(base);
			for (NSString *name in [theirs.allKeys sortedArrayUsingSelector:@selector(compare:)]) {
				[inBase addObject:name];
				NSValue *mine = ours[name];
				if (mine != nil) {
					NSString *difference =
					    signatureDifference(static_cast<Method>(mine.pointerValue),
					                        static_cast<Method>(theirs[name].pointerValue));
					if (difference != nil)
						fail([NSString stringWithFormat:@"-[RgfxMetalDevice %@] takes %@", name,
						                                difference]);
				}
				if (!isCreator(name))
					continue;
				creators++;
				if (mine != nil) {
					answered++;
					if (leftToTheBase(name) != nullptr)
						fail([NSString stringWithFormat:@"%@ is listed as left to the base, but "
						                                @"the device answers it",
						                                name]);
				} else if (leftToTheBase(name) != nullptr) {
					left++;
				} else {
					fail([NSString stringWithFormat:@"the device inherits -[%s %@] (%s): answer "
					                                @"it, or list it with why",
					                                class_getName(base), name,
					                                method_getTypeEncoding(static_cast<Method>(
					                                    theirs[name].pointerValue))]);
				}
			}
		}
		for (const Left &l : kLeftToTheBase) {
			if (![inBase containsObject:@(l.selector)])
				printf("note: %s, listed, is not in this macOS's base classes\n", l.selector);
		}
		printf("inherited_tests: %lu base-class creators: %lu answered by the device, %lu left "
		       "to the base; %d failures\n",
		       (unsigned long)creators, (unsigned long)answered, (unsigned long)left, g_failures);
		return g_failures == 0 ? 0 : 1;
	}
}
