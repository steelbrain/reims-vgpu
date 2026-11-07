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
// Each change is written to stderr once; with GL_INJECT_LOG=1, so is each render pipeline the
// class makes, with its functions and its error; with GL_INJECT_DUMP=DIR, each library it makes of
// bytes is written to DIR/library-N.metallib (the variants' among them), and each render pipeline
// to DIR/pipeline-N.txt: whether it was made, the libraries of its two functions, and its
// descriptor's state (vertex attributes and layouts, color attachments, depth and stencil formats,
// sample count, alpha to coverage, rasterization, topology), one line each; with
// GL_INJECT_VENDOR=NAME, the class answers vendorName with NAME (the GL renderer picks its
// per-vendor settings by it, and stops the process on a vendor it does not know). Nothing outside
// the process changes. Use it with gl_redirect (tools/gl_redirect.c), which points the
// accelerator's IOGLBundleName at the renderer.
// Our own code.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>

#import "RgfxGLFunction.h"

#include <atomic>
#include <cstring>
#include <map>
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

bool logging()
{
	static const bool on = [] {
		const char *v = getenv("GL_INJECT_LOG");
		return v != nullptr && std::strcmp(v, "1") == 0;
	}();
	return on;
}

// GL_INJECT_DUMP's bookkeeping: each library made of bytes holds its number, and each function
// made of it by name the library's.
char kLibraryNumber, kFunctionLibrary;
typedef id (*ByNameImp)(id, SEL, NSString *);
std::mutex gByNameLock;
std::map<Class, ByNameImp> gByName;

id byName(id self, SEL cmd, NSString *name)
{
	ByNameImp imp = nullptr;
	{
		std::lock_guard<std::mutex> hold(gByNameLock);
		for (Class c = object_getClass(self); c != nil && imp == nullptr; c = class_getSuperclass(c)) {
			auto it = gByName.find(c);
			if (it != gByName.end())
				imp = it->second;
		}
	}
	if (imp == nullptr)
		return nil;
	id f = imp(self, cmd, name);
	id number = objc_getAssociatedObject(self, &kLibraryNumber);
	if (f != nil && number != nil)
		objc_setAssociatedObject(f, &kFunctionLibrary, number, OBJC_ASSOCIATION_RETAIN);
	return f;
}

IMP wrap(Class cls, const char *name, IMP imp);

void numberLibrary(id library, int number)
{
	if (library == nil)
		return;
	objc_setAssociatedObject(library, &kLibraryNumber, @(number), OBJC_ASSOCIATION_RETAIN);
	Class cls = object_getClass(library);
	std::lock_guard<std::mutex> hold(gByNameLock);
	if (gByName.count(cls) == 0)
		gByName[cls] = (ByNameImp)wrap(cls, "newFunctionWithName:", (IMP)byName);
}

int libraryOf(id function)
{
	NSNumber *n = function != nil ? objc_getAssociatedObject(function, &kFunctionLibrary) : nil;
	return n != nil ? n.intValue : -1;
}

// DIR/pipeline-N.txt: the pipeline's two libraries, then its descriptor's state, one line each.
void dumpPipeline(MTLRenderPipelineDescriptor *d, bool made)
{
	static std::atomic<int> count{0};
	const char *dir = getenv("GL_INJECT_DUMP");
	if (dir == nullptr || !dir[0])
		return;
	NSMutableString *t = [NSMutableString string];
	[t appendFormat:@"made %d\nvertex %d\nfragment %d\n", made ? 1 : 0, libraryOf(d.vertexFunction),
	                libraryOf(d.fragmentFunction)];
	MTLVertexDescriptor *v = d.vertexDescriptor;
	for (NSUInteger i = 0; i < 31; i++) {
		MTLVertexAttributeDescriptor *a = v.attributes[i];
		if (a.format != MTLVertexFormatInvalid)
			[t appendFormat:@"attribute %lu %lu %lu %lu\n", (unsigned long)i, (unsigned long)a.format,
			                (unsigned long)a.offset, (unsigned long)a.bufferIndex];
		MTLVertexBufferLayoutDescriptor *l = v.layouts[i];
		if (l.stride != 0)
			[t appendFormat:@"layout %lu %lu %lu %lu\n", (unsigned long)i, (unsigned long)l.stride,
			                (unsigned long)l.stepFunction, (unsigned long)l.stepRate];
	}
	for (NSUInteger i = 0; i < 8; i++) {
		MTLRenderPipelineColorAttachmentDescriptor *c = d.colorAttachments[i];
		if (c.pixelFormat != MTLPixelFormatInvalid)
			[t appendFormat:@"color %lu %lu %d %lu %lu %lu %lu %lu %lu %lu\n", (unsigned long)i,
			                (unsigned long)c.pixelFormat, c.blendingEnabled ? 1 : 0,
			                (unsigned long)c.sourceRGBBlendFactor, (unsigned long)c.destinationRGBBlendFactor,
			                (unsigned long)c.rgbBlendOperation, (unsigned long)c.sourceAlphaBlendFactor,
			                (unsigned long)c.destinationAlphaBlendFactor, (unsigned long)c.alphaBlendOperation,
			                (unsigned long)c.writeMask];
	}
	[t appendFormat:@"depth %lu\nstencil %lu\nsamples %lu\nalpha_to_coverage %d\nrasterization %d\ntopology %lu\n",
	                (unsigned long)d.depthAttachmentPixelFormat, (unsigned long)d.stencilAttachmentPixelFormat,
	                (unsigned long)d.rasterSampleCount, d.alphaToCoverageEnabled ? 1 : 0,
	                d.rasterizationEnabled ? 1 : 0, (unsigned long)d.inputPrimitiveTopology];
	NSString *path = [NSString stringWithFormat:@"%s/pipeline-%d.txt", dir, count++];
	[t writeToFile:path atomically:NO encoding:NSUTF8StringEncoding error:nil];
}

id render(id self, SEL cmd, MTLRenderPipelineDescriptor *d, NSUInteger options, __autoreleasing id *reflection,
          NSError *__autoreleasing *error)
{
	__autoreleasing NSError *local = nil;
	NSError *__autoreleasing *out = error != nullptr ? error : &local;
	id state = gRender(self, cmd, d, options, reflection, out);
	if (logging())
		fprintf(stderr, "gl_inject: render pipeline of %s%s and %s%s: %s%s\n", class_getName([d.vertexFunction class]),
		        RgfxGLPluginReturnData(d.vertexFunction) != nil ? " (GL)" : "",
		        d.fragmentFunction != nil ? class_getName([d.fragmentFunction class]) : "no fragment function",
		        RgfxGLPluginReturnData(d.fragmentFunction) != nil ? " (GL)" : "", state != nil ? "made" : "nil",
		        *out != nil ? [NSString stringWithFormat:@", %@", *out].UTF8String : "");
	if (reflection != nullptr && *reflection != nil) {
		setIvar(*reflection, "_vertexPluginReturnData", RgfxGLPluginReturnData(d.vertexFunction));
		setIvar(*reflection, "_fragmentPluginReturnData", RgfxGLPluginReturnData(d.fragmentFunction));
	}
	dumpPipeline(d, state != nil);
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

typedef id (*LibraryImp)(id, SEL, dispatch_data_t, NSError *__autoreleasing *);
LibraryImp gLibrary;

// GL_INJECT_DUMP=DIR writes each library the class makes of bytes to DIR/library-N.metallib.
id library(id self, SEL cmd, dispatch_data_t data, NSError *__autoreleasing *error)
{
	static std::atomic<int> count{0};
	const char *dir = getenv("GL_INJECT_DUMP");
	int number = -1;
	if (dir != nullptr && dir[0] && data != nil) {
		number = count++;
		NSString *path = [NSString stringWithFormat:@"%s/library-%d.metallib", dir, number];
		[(NSData *)data writeToFile:path atomically:NO];
		if (logging())
			fprintf(stderr, "gl_inject: wrote %s\n", path.UTF8String);
	}
	id library = gLibrary(self, cmd, data, error);
	if (number >= 0)
		numberLibrary(library, number);
	return library;
}

// GL_INJECT_VENDOR's name, as vendorName answers it.
id vendor(id self, SEL cmd)
{
	return @(getenv("GL_INJECT_VENDOR"));
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
		if (getenv("GL_INJECT_DUMP") != nullptr)
			gLibrary = (LibraryImp)wrap(cls, "newLibraryWithData:error:", (IMP)library);
		const char *v = getenv("GL_INJECT_VENDOR");
		if (v != nullptr && v[0])
			put(cls, "vendorName", (IMP)vendor, "@16@0:8");
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
