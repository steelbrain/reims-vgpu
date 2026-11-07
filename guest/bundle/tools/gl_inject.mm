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
// per-vendor settings by it, and stops the process on a vendor it does not know); with
// GL_INJECT_TEXTURES=1, the first calls of each kind that make, fill, view, copy into and bind the
// device's textures and samplers are written to stderr with their arguments (RgfxTexture,
// RgfxBuffer, RgfxBlitCommandEncoder and RgfxRenderCommandEncoder's, where the process has them),
// each texture and sampler numbered at its create. Nothing outside
// the process changes. Use it with gl_redirect (tools/gl_redirect.c), which points the
// accelerator's IOGLBundleName at the renderer.
// Our own code.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>

#import "RgfxGLFunction.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>

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

bool texturesTraced();
void traceTextures(Class device);

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
		if (texturesTraced())
			traceTextures(cls);
	});
}

// --- GL_INJECT_TEXTURES ---------------------------------------------------------------------

bool texturesTraced()
{
	static const bool on = [] {
		const char *v = getenv("GL_INJECT_TEXTURES");
		return v != nullptr && std::strcmp(v, "1") == 0;
	}();
	return on;
}

char kObjectNumber;
std::atomic<int> gObjects{0};

// The number `o` was given at its create ("t3", "s1"), or "?" for one made elsewhere.
std::string tag(id o)
{
	if (o == nil)
		return "nil";
	NSString *n = objc_getAssociatedObject(o, &kObjectNumber);
	return n != nil ? n.UTF8String : std::string("?") + class_getName(object_getClass(o));
}

void number(id o, char kind)
{
	if (o != nil)
		objc_setAssociatedObject(o, &kObjectNumber, [NSString stringWithFormat:@"%c%d", kind, gObjects++],
		                         OBJC_ASSOCIATION_RETAIN);
}

// Whether a call of kind `what` is still written (the first 80 of each).
bool say(const char *what)
{
	static std::mutex lock;
	static std::map<std::string, int> counts;
	std::lock_guard<std::mutex> hold(lock);
	return counts[what]++ < 80;
}

std::string bytesAt(const void *p, size_t n)
{
	if (p == nullptr)
		return "(no bytes)";
	char b[64];
	std::string out;
	for (size_t i = 0; i < n; i++) {
		snprintf(b, sizeof b, "%02x", static_cast<const uint8_t *>(p)[i]);
		out += b;
	}
	return out;
}

std::string texDesc(MTLTextureDescriptor *d)
{
	MTLTextureSwizzleChannels sw = d.swizzle;
	char b[400];
	snprintf(b, sizeof b,
	         "type %lu format %lu %lux%lux%lu mips %lu array %lu samples %lu storage %lu cpu %lu usage 0x%lx "
	         "options 0x%lx swizzle %u%u%u%u",
	         (unsigned long)d.textureType, (unsigned long)d.pixelFormat, (unsigned long)d.width,
	         (unsigned long)d.height, (unsigned long)d.depth, (unsigned long)d.mipmapLevelCount,
	         (unsigned long)d.arrayLength, (unsigned long)d.sampleCount, (unsigned long)d.storageMode,
	         (unsigned long)d.cpuCacheMode, (unsigned long)d.usage, (unsigned long)d.resourceOptions,
	         (unsigned)sw.red, (unsigned)sw.green, (unsigned)sw.blue, (unsigned)sw.alpha);
	return b;
}

typedef id (*NewTexImp)(id, SEL, MTLTextureDescriptor *);
NewTexImp gNewTex;
id newTex(id self, SEL cmd, MTLTextureDescriptor *d)
{
	id t = gNewTex(self, cmd, d);
	number(t, 't');
	if (say("newTexture"))
		fprintf(stderr, "gl_tex: newTextureWithDescriptor %s -> %s\n", texDesc(d).c_str(), tag(t).c_str());
	return t;
}

typedef id (*NewSamplerImp)(id, SEL, MTLSamplerDescriptor *);
NewSamplerImp gNewSampler;
id newSampler(id self, SEL cmd, MTLSamplerDescriptor *d)
{
	id s = gNewSampler(self, cmd, d);
	number(s, 's');
	if (say("newSampler"))
		fprintf(stderr,
		        "gl_tex: newSamplerState min %lu mag %lu mip %lu address %lu/%lu/%lu normalized %d compare %lu "
		        "lod %g..%g aniso %lu border %lu -> %s\n",
		        (unsigned long)d.minFilter, (unsigned long)d.magFilter, (unsigned long)d.mipFilter,
		        (unsigned long)d.sAddressMode, (unsigned long)d.tAddressMode, (unsigned long)d.rAddressMode,
		        d.normalizedCoordinates ? 1 : 0, (unsigned long)d.compareFunction, d.lodMinClamp, d.lodMaxClamp,
		        (unsigned long)d.maxAnisotropy, (unsigned long)d.borderColor, tag(s).c_str());
	return s;
}

typedef id (*BufTexImp)(id, SEL, MTLTextureDescriptor *, NSUInteger, NSUInteger);
BufTexImp gBufTex;
id bufTex(id self, SEL cmd, MTLTextureDescriptor *d, NSUInteger offset, NSUInteger bytesPerRow)
{
	id t = gBufTex(self, cmd, d, offset, bytesPerRow);
	number(t, 't');
	if (say("bufferTexture")) {
		const uint8_t *c = static_cast<const uint8_t *>([(id<MTLBuffer>)self contents]);
		fprintf(stderr, "gl_tex: buffer %lu bytes, options 0x%lx: newTexture offset %lu row %lu %s -> %s; bytes %s\n",
		        (unsigned long)[(id<MTLBuffer>)self length], (unsigned long)[(id<MTLBuffer>)self resourceOptions],
		        (unsigned long)offset, (unsigned long)bytesPerRow, texDesc(d).c_str(), tag(t).c_str(),
		        bytesAt(c != nullptr ? c + offset : nullptr, 8).c_str());
	}
	return t;
}

typedef void (*Replace6Imp)(id, SEL, MTLRegion, NSUInteger, NSUInteger, const void *, NSUInteger, NSUInteger);
Replace6Imp gReplace6;
void replace6(id self, SEL cmd, MTLRegion r, NSUInteger level, NSUInteger slice, const void *bytes, NSUInteger row,
              NSUInteger image)
{
	if (say("replaceRegion6"))
		fprintf(stderr, "gl_tex: %s replaceRegion %lu,%lu %lux%lux%lu level %lu slice %lu row %lu image %lu bytes %s\n",
		        tag(self).c_str(), (unsigned long)r.origin.x, (unsigned long)r.origin.y, (unsigned long)r.size.width,
		        (unsigned long)r.size.height, (unsigned long)r.size.depth, (unsigned long)level, (unsigned long)slice,
		        (unsigned long)row, (unsigned long)image, bytesAt(bytes, 8).c_str());
	gReplace6(self, cmd, r, level, slice, bytes, row, image);
}

typedef void (*Replace4Imp)(id, SEL, MTLRegion, NSUInteger, const void *, NSUInteger);
Replace4Imp gReplace4;
void replace4(id self, SEL cmd, MTLRegion r, NSUInteger level, const void *bytes, NSUInteger row)
{
	if (say("replaceRegion4"))
		fprintf(stderr, "gl_tex: %s replaceRegion %lu,%lu %lux%lu level %lu row %lu bytes %s\n", tag(self).c_str(),
		        (unsigned long)r.origin.x, (unsigned long)r.origin.y, (unsigned long)r.size.width,
		        (unsigned long)r.size.height, (unsigned long)level, (unsigned long)row, bytesAt(bytes, 8).c_str());
	gReplace4(self, cmd, r, level, bytes, row);
}

typedef id (*View5Imp)(id, SEL, MTLPixelFormat, MTLTextureType, NSRange, NSRange, MTLTextureSwizzleChannels);
View5Imp gView5;
id view5(id self, SEL cmd, MTLPixelFormat f, MTLTextureType type, NSRange levels, NSRange slices,
         MTLTextureSwizzleChannels sw)
{
	id v = gView5(self, cmd, f, type, levels, slices, sw);
	number(v, 't');
	if (say("view5"))
		fprintf(stderr, "gl_tex: %s view format %lu type %lu levels %lu+%lu slices %lu+%lu swizzle %u%u%u%u -> %s\n",
		        tag(self).c_str(), (unsigned long)f, (unsigned long)type, (unsigned long)levels.location,
		        (unsigned long)levels.length, (unsigned long)slices.location, (unsigned long)slices.length,
		        (unsigned)sw.red, (unsigned)sw.green, (unsigned)sw.blue, (unsigned)sw.alpha, tag(v).c_str());
	return v;
}

typedef id (*View4Imp)(id, SEL, MTLPixelFormat, MTLTextureType, NSRange, NSRange);
View4Imp gView4;
id view4(id self, SEL cmd, MTLPixelFormat f, MTLTextureType type, NSRange levels, NSRange slices)
{
	id v = gView4(self, cmd, f, type, levels, slices);
	number(v, 't');
	if (say("view4"))
		fprintf(stderr, "gl_tex: %s view format %lu type %lu levels %lu+%lu slices %lu+%lu -> %s\n", tag(self).c_str(),
		        (unsigned long)f, (unsigned long)type, (unsigned long)levels.location, (unsigned long)levels.length,
		        (unsigned long)slices.location, (unsigned long)slices.length, tag(v).c_str());
	return v;
}

typedef id (*View1Imp)(id, SEL, MTLPixelFormat);
View1Imp gView1;
id view1(id self, SEL cmd, MTLPixelFormat f)
{
	id v = gView1(self, cmd, f);
	number(v, 't');
	if (say("view1"))
		fprintf(stderr, "gl_tex: %s view format %lu -> %s\n", tag(self).c_str(), (unsigned long)f, tag(v).c_str());
	return v;
}

typedef void (*CopyB2TImp)(id, SEL, id, NSUInteger, NSUInteger, NSUInteger, MTLSize, id, NSUInteger, NSUInteger,
                           MTLOrigin);
CopyB2TImp gCopyB2T;
void copyB2T(id self, SEL cmd, id buffer, NSUInteger offset, NSUInteger row, NSUInteger image, MTLSize size, id texture,
             NSUInteger slice, NSUInteger level, MTLOrigin origin)
{
	if (say("copyBufferToTexture")) {
		const uint8_t *c = static_cast<const uint8_t *>([(id<MTLBuffer>)buffer contents]);
		fprintf(stderr,
		        "gl_tex: blit buffer (%lu bytes, options 0x%lx) offset %lu row %lu image %lu %lux%lux%lu -> %s slice "
		        "%lu level %lu at %lu,%lu,%lu; bytes %s\n",
		        (unsigned long)[(id<MTLBuffer>)buffer length], (unsigned long)[(id<MTLBuffer>)buffer resourceOptions],
		        (unsigned long)offset, (unsigned long)row, (unsigned long)image, (unsigned long)size.width,
		        (unsigned long)size.height, (unsigned long)size.depth, tag(texture).c_str(), (unsigned long)slice,
		        (unsigned long)level, (unsigned long)origin.x, (unsigned long)origin.y, (unsigned long)origin.z,
		        bytesAt(c != nullptr ? c + offset : nullptr, 8).c_str());
	}
	gCopyB2T(self, cmd, buffer, offset, row, image, size, texture, slice, level, origin);
}

typedef void (*CopyT2TImp)(id, SEL, id, NSUInteger, NSUInteger, MTLOrigin, MTLSize, id, NSUInteger, NSUInteger,
                           MTLOrigin);
CopyT2TImp gCopyT2T;
void copyT2T(id self, SEL cmd, id from, NSUInteger slice, NSUInteger level, MTLOrigin origin, MTLSize size, id to,
             NSUInteger toSlice, NSUInteger toLevel, MTLOrigin toOrigin)
{
	if (say("copyTextureToTexture"))
		fprintf(stderr, "gl_tex: blit %s slice %lu level %lu %lux%lux%lu -> %s slice %lu level %lu\n", tag(from).c_str(),
		        (unsigned long)slice, (unsigned long)level, (unsigned long)size.width, (unsigned long)size.height,
		        (unsigned long)size.depth, tag(to).c_str(), (unsigned long)toSlice, (unsigned long)toLevel);
	gCopyT2T(self, cmd, from, slice, level, origin, size, to, toSlice, toLevel, toOrigin);
}

typedef void (*BindImp)(id, SEL, id, NSUInteger);
BindImp gFragTex, gFragSampler, gVertTex;
void fragTex(id self, SEL cmd, id t, NSUInteger index)
{
	if (say("setFragmentTexture"))
		fprintf(stderr, "gl_tex: setFragmentTexture %s at %lu\n", tag(t).c_str(), (unsigned long)index);
	gFragTex(self, cmd, t, index);
}
void fragSampler(id self, SEL cmd, id s, NSUInteger index)
{
	if (say("setFragmentSamplerState"))
		fprintf(stderr, "gl_tex: setFragmentSamplerState %s at %lu\n", tag(s).c_str(), (unsigned long)index);
	gFragSampler(self, cmd, s, index);
}
void vertTex(id self, SEL cmd, id t, NSUInteger index)
{
	if (say("setVertexTexture"))
		fprintf(stderr, "gl_tex: setVertexTexture %s at %lu\n", tag(t).c_str(), (unsigned long)index);
	gVertTex(self, cmd, t, index);
}

typedef void (*BindClampImp)(id, SEL, id, float, float, NSUInteger);
BindClampImp gFragSamplerClamp;
void fragSamplerClamp(id self, SEL cmd, id s, float lo, float hi, NSUInteger index)
{
	if (say("setFragmentSamplerStateClamp"))
		fprintf(stderr, "gl_tex: setFragmentSamplerState %s lod %g..%g at %lu\n", tag(s).c_str(), lo, hi,
		        (unsigned long)index);
	gFragSamplerClamp(self, cmd, s, lo, hi, index);
}

typedef void (*BindRangeImp)(id, SEL, const __unsafe_unretained id *, NSRange);
BindRangeImp gFragTexs, gFragSamplers;
void fragTexs(id self, SEL cmd, const __unsafe_unretained id *ts, NSRange r)
{
	if (say("setFragmentTextures")) {
		std::string list;
		for (NSUInteger i = 0; i < r.length; i++)
			list += " " + tag(ts[i]);
		fprintf(stderr, "gl_tex: setFragmentTextures at %lu:%s\n", (unsigned long)r.location, list.c_str());
	}
	gFragTexs(self, cmd, ts, r);
}
void fragSamplers(id self, SEL cmd, const __unsafe_unretained id *ss, NSRange r)
{
	if (say("setFragmentSamplerStates")) {
		std::string list;
		for (NSUInteger i = 0; i < r.length; i++)
			list += " " + tag(ss[i]);
		fprintf(stderr, "gl_tex: setFragmentSamplerStates at %lu:%s\n", (unsigned long)r.location, list.c_str());
	}
	gFragSamplers(self, cmd, ss, r);
}

void traceTextures(Class device)
{
	gNewTex = (NewTexImp)wrap(device, "newTextureWithDescriptor:", (IMP)newTex);
	gNewSampler = (NewSamplerImp)wrap(device, "newSamplerStateWithDescriptor:", (IMP)newSampler);
	if (Class c = objc_getClass("RgfxBuffer"))
		gBufTex = (BufTexImp)wrap(c, "newTextureWithDescriptor:offset:bytesPerRow:", (IMP)bufTex);
	if (Class c = objc_getClass("RgfxTexture")) {
		gReplace6 = (Replace6Imp)wrap(c, "replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:",
		                              (IMP)replace6);
		gReplace4 = (Replace4Imp)wrap(c, "replaceRegion:mipmapLevel:withBytes:bytesPerRow:", (IMP)replace4);
		gView5 = (View5Imp)wrap(c, "newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:", (IMP)view5);
		gView4 = (View4Imp)wrap(c, "newTextureViewWithPixelFormat:textureType:levels:slices:", (IMP)view4);
		gView1 = (View1Imp)wrap(c, "newTextureViewWithPixelFormat:", (IMP)view1);
	}
	if (Class c = objc_getClass("RgfxBlitCommandEncoder")) {
		gCopyB2T = (CopyB2TImp)wrap(c,
		                            "copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:"
		                            "toTexture:destinationSlice:destinationLevel:destinationOrigin:",
		                            (IMP)copyB2T);
		gCopyT2T = (CopyT2TImp)wrap(c,
		                            "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:"
		                            "destinationSlice:destinationLevel:destinationOrigin:",
		                            (IMP)copyT2T);
	}
	if (Class c = objc_getClass("RgfxRenderCommandEncoder")) {
		gFragTex = (BindImp)wrap(c, "setFragmentTexture:atIndex:", (IMP)fragTex);
		gFragSampler = (BindImp)wrap(c, "setFragmentSamplerState:atIndex:", (IMP)fragSampler);
		gVertTex = (BindImp)wrap(c, "setVertexTexture:atIndex:", (IMP)vertTex);
		gFragSamplerClamp =
		    (BindClampImp)wrap(c, "setFragmentSamplerState:lodMinClamp:lodMaxClamp:atIndex:", (IMP)fragSamplerClamp);
		gFragTexs = (BindRangeImp)wrap(c, "setFragmentTextures:withRange:", (IMP)fragTexs);
		gFragSamplers = (BindRangeImp)wrap(c, "setFragmentSamplerStates:withRange:", (IMP)fragSamplers);
	}
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
