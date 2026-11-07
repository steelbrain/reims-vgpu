// gl_inject: a library inserted into one process (DYLD_INSERT_LIBRARIES) that gives the
// reims-graphics device installed in a guest the bundle's GL members of this tree before the kext
// carries them: the device's functions of GL IR (src/RgfxMetalDevice+GL.mm, src/RgfxGLFunction.mm
// and the lowering they run, compiled into this library) and the GL variants' plugin return data
// on its render and compute pipelines' reflection (src/RgfxPipelineReflection.mm's). So a GL
// renderer can be tried on a guest's device with a bundle built from this tree's sources but
// installed from an older kext.
//
// It interposes MTLCopyAllDevices and MTLCreateSystemDefaultDevice, the device lists the GL
// renderer and Metal's callers read, and on the first answer that holds a device of class
// GL_INJECT_CLASS (default RgfxMetalDevice) gives that class:
//   - the six newFunctionWith{GL,GLCore,GLES}IR: members, each making an RgfxGLFunction;
//   - newRenderPipelineStateWithDescriptor:options:reflection:error: and
//     newComputePipelineStateWithFunction:options:reflection:error:, which answer as the class
//     did, then set the reflection's plugin return data from the descriptor's GL variants.
// Each change is written to stderr once. Nothing outside the process changes. Use it with
// gl_redirect (tools/gl_redirect.c), which points the accelerator's IOGLBundleName at the
// renderer.
// Our own code.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>

#import "RgfxGLFunction.h"

#include <mutex>

#define DYLD_INTERPOSE(_replacement, _replacee)                                              \
	__attribute__((used)) static struct {                                                    \
		const void *replacement;                                                             \
		const void *replacee;                                                                \
	} _interpose_##_replacee __attribute__((section("__DATA,__interpose"))) = {              \
	    (const void *)(unsigned long)&_replacement, (const void *)(unsigned long)&_replacee};

namespace {

id makeFunction(id self, void *ir, id inputs, NSUInteger type)
{
	return [[RgfxGLFunction alloc] initWithIR:ir
	                             functionType:static_cast<MTLFunctionType>(type)
	                        inputsDescription:inputs
	                                   device:self];
}

id glIR2(id self, SEL, void *ir, NSUInteger type)
{
	return makeFunction(self, ir, nil, type);
}

id glIR3(id self, SEL, void *ir, id inputs, NSUInteger type)
{
	return makeFunction(self, ir, inputs, type);
}

void setIvar(id object, const char *name, id value)
{
	if (object == nil || value == nil)
		return;
	Ivar v = class_getInstanceVariable(object_getClass(object), name);
	if (v != nullptr)
		object_setIvarWithStrongDefault(object, v, value);
}

typedef id (*RenderImp)(id, SEL, id, NSUInteger, __autoreleasing id *, NSError *__autoreleasing *);
typedef id (*ComputeImp)(id, SEL, id, NSUInteger, __autoreleasing id *, NSError *__autoreleasing *);
RenderImp gRender;
ComputeImp gCompute;

id render(id self, SEL cmd, MTLRenderPipelineDescriptor *d, NSUInteger options, __autoreleasing id *reflection,
          NSError *__autoreleasing *error)
{
	id state = gRender(self, cmd, d, options, reflection, error);
	if (reflection != nullptr && *reflection != nil) {
		setIvar(*reflection, "_vertexPluginReturnData", RgfxGLPluginReturnData(d.vertexFunction));
		setIvar(*reflection, "_fragmentPluginReturnData", RgfxGLPluginReturnData(d.fragmentFunction));
	}
	return state;
}

id compute(id self, SEL cmd, id<MTLFunction> f, NSUInteger options, __autoreleasing id *reflection,
           NSError *__autoreleasing *error)
{
	id state = gCompute(self, cmd, f, options, reflection, error);
	if (reflection != nullptr && *reflection != nil)
		setIvar(*reflection, "_pluginReturnData", RgfxGLPluginReturnData(f));
	return state;
}

void put(Class cls, const char *name, IMP imp, const char *types)
{
	SEL sel = sel_registerName(name);
	Method m = class_getInstanceMethod(cls, sel);
	if (m != nullptr && class_addMethod(cls, sel, imp, method_getTypeEncoding(m)) == NO)
		method_setImplementation(class_getInstanceMethod(cls, sel), imp);
	else if (m == nullptr)
		class_addMethod(cls, sel, imp, types);
	fprintf(stderr, "gl_inject: -[%s %s]\n", class_getName(cls), name);
}

IMP wrap(Class cls, const char *name, IMP imp)
{
	SEL sel = sel_registerName(name);
	Method m = class_getInstanceMethod(cls, sel);
	if (m == nullptr)
		return nullptr;
	IMP old = method_getImplementation(m);
	if (!class_addMethod(cls, sel, imp, method_getTypeEncoding(m)))
		method_setImplementation(m, imp);
	fprintf(stderr, "gl_inject: wrapped -[%s %s]\n", class_getName(cls), name);
	return old;
}

void patch(NSArray *devices)
{
	static std::once_flag once;
	const char *want = getenv("GL_INJECT_CLASS");
	Class cls = objc_getClass(want != nullptr && want[0] ? want : "RgfxMetalDevice");
	if (cls == nil)
		return;
	BOOL found = NO;
	for (id d in devices)
		found = found || [d isKindOfClass:cls];
	if (!found)
		return;
	std::call_once(once, [cls] {
		put(cls, "newFunctionWithGLIR:functionType:", (IMP)glIR2, "@32@0:8^v16Q24");
		put(cls, "newFunctionWithGLIR:inputsDescription:functionType:", (IMP)glIR3, "@40@0:8^v16@24Q32");
		put(cls, "newFunctionWithGLCoreIR:functionType:", (IMP)glIR2, "@32@0:8^v16Q24");
		put(cls, "newFunctionWithGLCoreIR:inputsDescription:functionType:", (IMP)glIR3, "@40@0:8^v16@24Q32");
		put(cls, "newFunctionWithGLESIR:functionType:", (IMP)glIR2, "@32@0:8^v16Q24");
		put(cls, "newFunctionWithGLESIR:inputsDescription:functionType:", (IMP)glIR3, "@40@0:8^v16@24Q32");
		gRender = (RenderImp)wrap(cls, "newRenderPipelineStateWithDescriptor:options:reflection:error:", (IMP)render);
		gCompute = (ComputeImp)wrap(cls, "newComputePipelineStateWithFunction:options:reflection:error:", (IMP)compute);
	});
}

// Both answer retained, as the functions they stand for do.
NSArray *myCopyAll(void) NS_RETURNS_RETAINED;
id<MTLDevice> myDefault(void) NS_RETURNS_RETAINED;

NSArray *myCopyAll(void)
{
	NSArray *devices = MTLCopyAllDevices();
	patch(devices);
	return devices;
}

id<MTLDevice> myDefault(void)
{
	id<MTLDevice> d = MTLCreateSystemDefaultDevice();
	if (d != nil)
		patch(@[ d ]);
	return d;
}

} // namespace

DYLD_INTERPOSE(myCopyAll, MTLCopyAllDevices)
DYLD_INTERPOSE(myDefault, MTLCreateSystemDefaultDevice)
