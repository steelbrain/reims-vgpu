// gl_probe: what OpenGL gives a guest. No window: every context is offscreen and draws into a
// framebuffer object of its own, which the probe reads back.
//   gl_probe [-x DIR] [-X DIR] [-c CLASS] [-n FRAMES]
//     -x DIR   before any context is made, wraps the device's newFunctionWith{GL,GLCore,GLES}IR:
//              members so each writes the IR it is given (DIR/ir-N.bin, the bytes from the u32
//              length that heads them; DIR/in-N.bin, the inputs description) and then answers as
//              the device does; the class of the functions it answers so each variant's plugin
//              data is written (DIR/plugin-N-M.bin); and the device's
//              newRenderPipelineStateWithDescriptor:options:reflection:error: so each pipeline's
//              GL properties, its reflection's arguments and plugin return data are printed
//              and the return data written (DIR/render-K-*.bin).
//     -X DIR   as -x, but the wrapped members answer a function object of the probe's, whose
//              newFunctionWithPluginData:bitcodeType: writes DIR/plugin-N-M.bin and answers nil.
//     -c CLASS the device class -x and -X wrap (default RgfxMetalDevice), such as another Mac's
//              Metal device class under gl_redirect.
//     -n N     frames the timed loop of each context draws (default 200).
// It prints:
//   - each IOAccelerator service: its class, registry ID, IOGLBundleName and MetalPluginName;
//   - each Metal device: its name and registry ID;
//   - each renderer CGL lists (display mask 0xffffffff), with its ID, whether it is online and
//     accelerated, its display mask, video memory and the highest GL version it offers;
//   - for each of three profiles (legacy, 3.2 core, 4.1 core), asking an accelerated renderer
//     and then any: the pixel format's renderer, the context's GL_VENDOR, GL_RENDERER,
//     GL_VERSION and GLSL version, then three cases, each with its pixel at the centre and its
//     glGetError:
//       clear     a 64x64 RGBA8 framebuffer cleared to (64, 128, 191, 255);
//       triangle  the same framebuffer cleared, then covered by one triangle whose fragment
//                 shader writes a uniform's colour (0, 255, 0, 255), with the program's link
//                 status and log;
//       loop      N frames of clear and triangle, then glFinish: the first frame's time and the
//                 mean of the rest;
//     and, on a context whose renderer can, iosurface: a 64x64 BGRA IOSurface bound to a
//     rectangle texture (CGLTexImageIOSurface2D) as the framebuffer, cleared, then read by the
//     CPU through IOSurfaceLock.
// Our own code.
#define GL_SILENCE_DEPRECATION 1
#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <OpenGL/CGLIOSurface.h>
#import <OpenGL/OpenGL.h>
#import <OpenGL/gl3.h>
#import <mach/mach_time.h>
#import <objc/runtime.h>
#include <getopt.h>

static double now_ms(void) {
	static mach_timebase_info_data_t tb;
	if (tb.denom == 0) mach_timebase_info(&tb);
	return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6;
}

// --- the IR dump -------------------------------------------------------------------------------

static NSString *g_dumpDir;
static const char *g_dumpClass;
static BOOL g_stubFunctions;
static int g_irCount;

static void write_bytes(NSString *name, const void *bytes, size_t length) {
	NSString *path = [g_dumpDir stringByAppendingPathComponent:name];
	BOOL ok = [[NSData dataWithBytes:bytes length:length] writeToFile:path atomically:NO];
	printf("  dump %s: %zu bytes%s\n", path.UTF8String, length, ok ? "" : " (write failed)");
}

static void dump_ir(int n, const char *which, const void *ir, id inputs, NSUInteger type) {
	uint32_t length = 0;
	if (ir) memcpy(&length, ir, 4);
	printf("  %s IR %d: functionType %lu, %u bytes after the length\n", which, n,
	       (unsigned long)type, length);
	if (ir) write_bytes([NSString stringWithFormat:@"ir-%d.bin", n], ir, (size_t)length + 4);
	if (inputs) {
		NSData *data = (NSData *)inputs; // a dispatch_data_t is an NSData
		write_bytes([NSString stringWithFormat:@"in-%d.bin", n], data.bytes, data.length);
	}
}

// The function object -X answers: enough of MTLFunction for the renderer to ask a variant.
@interface GLProbeIRFunction : NSObject
@property(nonatomic) int index;
@property(nonatomic) NSUInteger functionType;
@property(nonatomic, strong) id device;
@property(nonatomic, copy) NSString *label;
@property(nonatomic) int variants;
@end

@implementation GLProbeIRFunction
- (NSString *)name { return [NSString stringWithFormat:@"glprobe_ir_%d", self.index]; }
- (id)newFunctionWithPluginData:(NSData *)data bitcodeType:(NSUInteger)bitcodeType {
	int m = self.variants++;
	printf("  IR %d: newFunctionWithPluginData: %lu bytes, bitcodeType %lu\n", self.index,
	       (unsigned long)data.length, (unsigned long)bitcodeType);
	write_bytes([NSString stringWithFormat:@"plugin-%d-%d.bin", self.index, m], data.bytes,
	            data.length);
	return nil;
}
- (id)forwardingTargetForSelector:(SEL)sel {
	printf("  IR %d: asked %s\n", self.index, sel_getName(sel));
	return nil;
}
@end

// -x: the functions the device answers keep their IR's number, and their class's
// newFunctionWithPluginData:bitcodeType: writes each variant's plugin data before it answers.
static IMP wrap_method(Class cls, SEL sel, IMP wrapper);
static char kIrKey;
typedef id (*PluginImp)(id, SEL, NSData *, NSUInteger);
static PluginImp g_origPlugin;
static int g_pluginCount;

static id wrap_plugin(id self, SEL cmd, NSData *data, NSUInteger bitcodeType) {
	NSNumber *n = objc_getAssociatedObject(self, &kIrKey);
	int m = g_pluginCount++;
	printf("  IR %d: newFunctionWithPluginData: %lu bytes, bitcodeType %lu (variant %d)\n",
	       n ? n.intValue : -1, (unsigned long)data.length, (unsigned long)bitcodeType, m);
	write_bytes([NSString stringWithFormat:@"plugin-%d-%d.bin", n ? n.intValue : -1, m], data.bytes,
	            data.length);
	id v = g_origPlugin(self, cmd, data, bitcodeType);
	if (v) objc_setAssociatedObject(v, &kIrKey, @(n ? n.intValue : -1), OBJC_ASSOCIATION_RETAIN);
	printf("    variant: %s\n", v ? class_getName([v class]) : "nil");
	return v;
}

static void note_function(id f, int n) {
	printf("    function: %s\n", f ? class_getName([f class]) : "nil");
	if (!f) return;
	objc_setAssociatedObject(f, &kIrKey, @(n), OBJC_ASSOCIATION_RETAIN);
	SEL sel = sel_registerName("newFunctionWithPluginData:bitcodeType:");
	if (!g_origPlugin && [f respondsToSelector:sel]) {
		g_origPlugin = (PluginImp)wrap_method([f class], sel, (IMP)wrap_plugin);
		printf("    wrapped -[%s newFunctionWithPluginData:bitcodeType:]\n", class_getName([f class]));
	}
}

// -x also writes what each render pipeline the renderer makes asks: the descriptor's GL
// properties, and the reflection's plugin return data.
typedef id (*RenderImp)(id, SEL, id, NSUInteger, __autoreleasing id *, NSError *__autoreleasing *);
static RenderImp g_origRender;
static int g_renderCount;

static void print_key(id object, NSString *key) {
	@try {
		id v = [object valueForKey:key];
		printf(" %s=%s", key.UTF8String, v ? [[v description] UTF8String] : "nil");
	} @catch (NSException *e) {
		printf(" %s=(none)", key.UTF8String);
	}
}

static id wrap_render(id self, SEL cmd, id desc, NSUInteger options, __autoreleasing id *reflection,
                      NSError *__autoreleasing *error) {
	int n = g_renderCount++;
	printf("  render pipeline %d: options 0x%lx\n   ", n, (unsigned long)options);
	for (NSString *k in @[ @"openGLModeEnabled", @"logicOperationEnabled", @"logicOperation",
	                       @"sampleCoverage", @"sampleMask", @"pointCoordLowerLeft",
	                       @"pointSizeOutputVS", @"twoSideEnabled", @"depthStencilWriteDisabled",
	                       @"alphaToCoverageEnabled", @"alphaToOneEnabled", @"clipDistanceEnableMask",
	                       @"rasterSampleCount", @"inputPrimitiveTopology", @"rasterizationEnabled",
	                       @"alphaTestEnabled", @"pointSmoothEnabled", @"needsCustomBorderColorSamplers",
	                       @"pluginData" ])
		print_key(desc, k);
	printf("\n");
	NSNumber *vs = objc_getAssociatedObject([desc valueForKey:@"vertexFunction"], &kIrKey);
	NSNumber *fs = objc_getAssociatedObject([desc valueForKey:@"fragmentFunction"], &kIrKey);
	printf("    vertex IR %d, fragment IR %d\n", vs ? vs.intValue : -1, fs ? fs.intValue : -1);
	__autoreleasing id local = nil;
	__autoreleasing id *out = reflection ? reflection : &local;
	id state = g_origRender(self, cmd, desc, options, out, error);
	id r = *out;
	printf("    state %s, reflection %s%s\n", state ? class_getName([state class]) : "nil",
	       r ? class_getName([r class]) : "nil",
	       error && *error ? [NSString stringWithFormat:@", %@", *error].UTF8String : "");
	for (NSString *k in @[ @"vertexPluginReturnData", @"fragmentPluginReturnData", @"pluginReturnData" ]) {
		if (![r respondsToSelector:NSSelectorFromString(k)]) continue;
		NSData *d = [r valueForKey:k];
		printf("    %s: %lu bytes\n", k.UTF8String, (unsigned long)d.length);
		if (d.length) write_bytes([NSString stringWithFormat:@"render-%d-%@.bin", n, k], d.bytes, d.length);
	}
	if ([r respondsToSelector:NSSelectorFromString(@"performanceStatistics")])
		printf("    performanceStatistics: %s\n",
		       [[[r valueForKey:@"performanceStatistics"] description] UTF8String]);
	for (NSString *k in @[ @"vertexArguments", @"fragmentArguments" ]) {
		if (![r respondsToSelector:NSSelectorFromString(k)]) continue;
		for (id a in [r valueForKey:k])
			printf("    %s: %s\n", k.UTF8String,
			       [[[a description] stringByReplacingOccurrencesOfString:@"\n" withString:@" "] UTF8String]);
	}
	return state;
}

typedef id (*IrImp3)(id, SEL, void *, id, NSUInteger);
typedef id (*IrImp2)(id, SEL, void *, NSUInteger);
static IrImp3 g_orig3[3];
static IrImp2 g_orig2[3];
static const char *const kIrNames[3] = {"GL", "GLCore", "GLES"};

static id answer_ir(int which, id device, SEL cmd, void *ir, id inputs, NSUInteger type) {
	int n = g_irCount++;
	dump_ir(n, kIrNames[which], ir, inputs, type);
	if (g_stubFunctions) {
		GLProbeIRFunction *f = [GLProbeIRFunction new];
		f.index = n;
		f.functionType = type;
		f.device = device;
		return f;
	}
	id f = g_orig3[which](device, cmd, ir, inputs, type);
	note_function(f, n);
	return f;
}

#define IR_WRAPPERS(i)                                                                          \
	static id wrap3_##i(id self, SEL cmd, void *ir, id inputs, NSUInteger type) {            \
		return answer_ir(i, self, cmd, ir, inputs, type);                                     \
	}                                                                                         \
	static id wrap2_##i(id self, SEL cmd, void *ir, NSUInteger type) {                       \
		int n = g_irCount++;                                                                  \
		dump_ir(n, kIrNames[i], ir, nil, type);                                               \
		if (g_stubFunctions) {                                                                \
			GLProbeIRFunction *f = [GLProbeIRFunction new];                                   \
			f.index = n;                                                                      \
			f.functionType = type;                                                            \
			f.device = self;                                                                  \
			return f;                                                                         \
		}                                                                                     \
		id f = g_orig2[i](self, cmd, ir, type);                                               \
		note_function(f, n);                                                                  \
		return f;                                                                             \
	}
IR_WRAPPERS(0)
IR_WRAPPERS(1)
IR_WRAPPERS(2)

// Wraps `sel` on `cls` with `wrapper`, answering the implementation it had: a method `cls`
// inherits is added to `cls` itself, so its base and the base's other subclasses keep theirs.
static IMP wrap_method(Class cls, SEL sel, IMP wrapper) {
	Method m = class_getInstanceMethod(cls, sel);
	if (!m) return NULL;
	IMP old = method_getImplementation(m);
	if (!class_addMethod(cls, sel, wrapper, method_getTypeEncoding(m))) method_setImplementation(m, wrapper);
	return old;
}

static void wrap_device_class(Class cls) {
	const char *three[3] = {"newFunctionWithGLIR:inputsDescription:functionType:",
	                        "newFunctionWithGLCoreIR:inputsDescription:functionType:",
	                        "newFunctionWithGLESIR:inputsDescription:functionType:"};
	const char *two[3] = {"newFunctionWithGLIR:functionType:", "newFunctionWithGLCoreIR:functionType:",
	                      "newFunctionWithGLESIR:functionType:"};
	IMP w3[3] = {(IMP)wrap3_0, (IMP)wrap3_1, (IMP)wrap3_2};
	IMP w2[3] = {(IMP)wrap2_0, (IMP)wrap2_1, (IMP)wrap2_2};
	for (int i = 0; i < 3; i++) {
		g_orig3[i] = (IrImp3)wrap_method(cls, sel_registerName(three[i]), w3[i]);
		g_orig2[i] = (IrImp2)wrap_method(cls, sel_registerName(two[i]), w2[i]);
		printf("  wrapped -[%s %s]%s, -[%s %s]%s\n", class_getName(cls), three[i],
		       g_orig3[i] ? "" : " (none)", class_getName(cls), two[i], g_orig2[i] ? "" : " (none)");
	}
	if (!g_stubFunctions)
		g_origRender = (RenderImp)wrap_method(
		    cls, sel_registerName("newRenderPipelineStateWithDescriptor:options:reflection:error:"),
		    (IMP)wrap_render);
}

// --- inventory ---------------------------------------------------------------------------------

static void print_cf(const char *key, CFTypeRef v) {
	if (!v) {
		printf("    %s: -\n", key);
		return;
	}
	NSString *d = [(__bridge id)v description];
	printf("    %s: %s\n", key, d.UTF8String);
}

static void list_accelerators(void) {
	printf("accelerators:\n");
	io_iterator_t it = 0;
	if (IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("IOAccelerator"), &it) !=
	    KERN_SUCCESS)
		return;
	io_service_t s;
	while ((s = IOIteratorNext(it))) {
		io_name_t cls;
		IOObjectGetClass(s, cls);
		uint64_t id = 0;
		IORegistryEntryGetRegistryEntryID(s, &id);
		printf("  %s, registry ID 0x%llx\n", cls, (unsigned long long)id);
		const char *keys[] = {"IOGLBundleName", "IOGLESBundleName", "MetalPluginName",
		                      "MetalPluginClassName", "IOOCDBundleName"};
		for (size_t i = 0; i < sizeof keys / sizeof *keys; i++) {
			CFStringRef k = CFStringCreateWithCString(NULL, keys[i], kCFStringEncodingUTF8);
			CFTypeRef v = IORegistryEntryCreateCFProperty(s, k, NULL, 0);
			print_cf(keys[i], v);
			if (v) CFRelease(v);
			CFRelease(k);
		}
		IOObjectRelease(s);
	}
	IOObjectRelease(it);
}

static void list_metal_devices(void) {
	printf("metal devices:\n");
	for (id<MTLDevice> d in MTLCopyAllDevices())
		printf("  %s, registry ID 0x%llx, class %s\n", d.name.UTF8String, d.registryID,
		       class_getName([(id)d class]));
}

static void list_renderers(void) {
	printf("renderers:\n");
	CGLRendererInfoObj info = NULL;
	GLint n = 0;
	CGLError e = CGLQueryRendererInfo(0xffffffff, &info, &n);
	if (e != kCGLNoError) {
		printf("  CGLQueryRendererInfo: %s\n", CGLErrorString(e));
		return;
	}
	for (GLint i = 0; i < n; i++) {
		GLint id = 0, online = 0, accel = 0, mask = 0, vram = 0, major = 0, rlo = 0, rhi = 0;
		CGLDescribeRenderer(info, i, kCGLRPRendererID, &id);
		CGLDescribeRenderer(info, i, kCGLRPOnline, &online);
		CGLDescribeRenderer(info, i, kCGLRPAccelerated, &accel);
		CGLDescribeRenderer(info, i, kCGLRPDisplayMask, &mask);
		CGLDescribeRenderer(info, i, kCGLRPVideoMemoryMegabytes, &vram);
		CGLDescribeRenderer(info, i, kCGLRPMajorGLVersion, &major);
		CGLDescribeRenderer(info, i, kCGLRPRegistryIDLow, &rlo);
		CGLDescribeRenderer(info, i, kCGLRPRegistryIDHigh, &rhi);
		printf("  renderer 0x%08x: online %d, accelerated %d, display mask 0x%x, %d MB, GL %d, "
		       "registry ID 0x%llx\n",
		       id, online, accel, mask, vram, major,
		       ((unsigned long long)(uint32_t)rhi << 32) | (uint32_t)rlo);
	}
	CGLDestroyRendererInfo(info);
}

// --- the cases ---------------------------------------------------------------------------------

static void read_centre(const char *what) {
	uint8_t px[4] = {0};
	glReadPixels(32, 32, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
	printf("    %s: centre (%u, %u, %u, %u), glGetError 0x%x\n", what, px[0], px[1], px[2], px[3],
	       glGetError());
}

static GLuint make_fbo(void) {
	GLuint fbo = 0, rb = 0;
	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	glGenRenderbuffers(1, &rb);
	glBindRenderbuffer(GL_RENDERBUFFER, rb);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, 64, 64);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb);
	GLenum st = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	printf("    framebuffer: status 0x%x%s\n", st, st == GL_FRAMEBUFFER_COMPLETE ? " (complete)" : "");
	glViewport(0, 0, 64, 64);
	return fbo;
}

static GLuint compile(GLenum type, const char *src) {
	GLuint s = glCreateShader(type);
	glShaderSource(s, 1, &src, NULL);
	glCompileShader(s);
	GLint ok = 0;
	glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
	if (!ok) {
		char log[1024] = {0};
		glGetShaderInfoLog(s, sizeof log, NULL, log);
		printf("    compile 0x%x failed: %s\n", type, log);
	}
	return s;
}

static GLuint make_program(BOOL core) {
	const char *vs = core ? "#version 150\nin vec2 pos;\nvoid main() { gl_Position = vec4(pos, 0.0, 1.0); }\n"
	                      : "#version 120\nattribute vec2 pos;\nvoid main() { gl_Position = vec4(pos, 0.0, 1.0); }\n";
	const char *fs = core ? "#version 150\nuniform vec4 colour;\nout vec4 frag;\nvoid main() { frag = colour; }\n"
	                      : "#version 120\nuniform vec4 colour;\nvoid main() { gl_FragColor = colour; }\n";
	GLuint p = glCreateProgram();
	glAttachShader(p, compile(GL_VERTEX_SHADER, vs));
	glAttachShader(p, compile(GL_FRAGMENT_SHADER, fs));
	glBindAttribLocation(p, 0, "pos");
	glLinkProgram(p);
	GLint ok = 0;
	glGetProgramiv(p, GL_LINK_STATUS, &ok);
	char log[1024] = {0};
	glGetProgramInfoLog(p, sizeof log, NULL, log);
	printf("    program: link status %d%s%s\n", ok, log[0] ? ", log: " : "", log);
	return p;
}

static void draw_frame(GLuint program, GLint colour) {
	glClearColor(0.25f, 0.5f, 0.75f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);
	glUseProgram(program);
	glUniform4f(colour, 0.0f, 1.0f, 0.0f, 1.0f);
	glDrawArrays(GL_TRIANGLES, 0, 3);
}

static void run_cases(BOOL core, int frames) {
	GLuint fbo = make_fbo();
	glClearColor(0.25f, 0.5f, 0.75f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);
	read_centre("clear");

	static const float tri[] = {-1, -1, 3, -1, -1, 3};
	GLuint vao = 0, vbo = 0;
	if (core) {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
	}
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof tri, tri, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
	double t0 = now_ms();
	GLuint program = make_program(core);
	GLint colour = glGetUniformLocation(program, "colour");
	draw_frame(program, colour);
	read_centre("triangle");
	double first = now_ms() - t0;

	t0 = now_ms();
	for (int i = 0; i < frames; i++) draw_frame(program, colour);
	glFinish();
	double rest = now_ms() - t0;
	printf("    loop: first frame (program, draw, read) %.2f ms; %d more frames %.3f ms each, "
	       "glGetError 0x%x\n",
	       first, frames, frames ? rest / frames : 0.0, glGetError());
	read_centre("after loop");

	glDeleteProgram(program);
	glDeleteBuffers(1, &vbo);
	if (vao) glDeleteVertexArrays(1, &vao);
	glDeleteFramebuffers(1, &fbo);
}

static void run_iosurface(CGLContextObj ctx) {
	NSDictionary *props = @{
		(id)kIOSurfaceWidth : @64,
		(id)kIOSurfaceHeight : @64,
		(id)kIOSurfaceBytesPerElement : @4,
		(id)kIOSurfacePixelFormat : @((uint32_t)'BGRA'),
	};
	IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)props);
	if (!surface) {
		printf("    iosurface: IOSurfaceCreate failed\n");
		return;
	}
	GLuint tex = 0, fbo = 0;
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_RECTANGLE, tex);
	CGLError e = CGLTexImageIOSurface2D(ctx, GL_TEXTURE_RECTANGLE, GL_RGBA, 64, 64, GL_BGRA,
	                                    GL_UNSIGNED_INT_8_8_8_8_REV, surface, 0);
	printf("    iosurface: CGLTexImageIOSurface2D %s\n", CGLErrorString(e));
	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE, tex, 0);
	printf("    iosurface: framebuffer status 0x%x\n", glCheckFramebufferStatus(GL_FRAMEBUFFER));
	glViewport(0, 0, 64, 64);
	glClearColor(0.25f, 0.5f, 0.75f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);
	glFinish();
	IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);
	const uint8_t *p = (const uint8_t *)IOSurfaceGetBaseAddress(surface) +
	                   32 * IOSurfaceGetBytesPerRow(surface) + 32 * 4;
	printf("    iosurface: CPU reads B %u G %u R %u A %u, glGetError 0x%x\n", p[0], p[1], p[2], p[3],
	       glGetError());
	IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
	glDeleteFramebuffers(1, &fbo);
	glDeleteTextures(1, &tex);
	CFRelease(surface);
}

static void run_profile(const char *name, CGLOpenGLProfile profile, BOOL accelerated, int frames) {
	CGLPixelFormatAttribute attrs[16];
	int a = 0;
	attrs[a++] = kCGLPFAOpenGLProfile;
	attrs[a++] = (CGLPixelFormatAttribute)profile;
	attrs[a++] = kCGLPFAColorSize;
	attrs[a++] = (CGLPixelFormatAttribute)24;
	attrs[a++] = kCGLPFAAlphaSize;
	attrs[a++] = (CGLPixelFormatAttribute)8;
	attrs[a++] = kCGLPFAAllowOfflineRenderers;
	if (accelerated) attrs[a++] = kCGLPFAAccelerated;
	attrs[a] = (CGLPixelFormatAttribute)0;
	printf("context %s, %s:\n", name, accelerated ? "accelerated asked" : "any renderer");
	CGLPixelFormatObj pix = NULL;
	GLint npix = 0;
	CGLError e = CGLChoosePixelFormat(attrs, &pix, &npix);
	if (e != kCGLNoError || !pix) {
		printf("    CGLChoosePixelFormat: %s, %d formats\n", CGLErrorString(e), npix);
		return;
	}
	GLint rid = 0, accel = 0;
	CGLDescribePixelFormat(pix, 0, kCGLPFARendererID, &rid);
	CGLDescribePixelFormat(pix, 0, kCGLPFAAccelerated, &accel);
	printf("    pixel format: %d virtual screens, renderer 0x%08x, accelerated %d\n", npix, rid, accel);
	CGLContextObj ctx = NULL;
	double t0 = now_ms();
	e = CGLCreateContext(pix, NULL, &ctx);
	double made = now_ms() - t0;
	CGLReleasePixelFormat(pix);
	if (e != kCGLNoError || !ctx) {
		printf("    CGLCreateContext: %s\n", CGLErrorString(e));
		return;
	}
	CGLSetCurrentContext(ctx);
	printf("    made in %.2f ms\n    GL_VENDOR %s\n    GL_RENDERER %s\n    GL_VERSION %s\n"
	       "    GLSL %s\n",
	       made, glGetString(GL_VENDOR), glGetString(GL_RENDERER), glGetString(GL_VERSION),
	       glGetString(GL_SHADING_LANGUAGE_VERSION));
	run_cases(profile != kCGLOGLPVersion_Legacy, frames);
	run_iosurface(ctx);
	CGLSetCurrentContext(NULL);
	CGLDestroyContext(ctx);
}

int main(int argc, char **argv) {
	@autoreleasepool {
		int frames = 200, c;
		while ((c = getopt(argc, argv, "x:X:n:c:")) != -1) {
			if (c == 'x' || c == 'X') {
				g_dumpDir = @(optarg);
				g_stubFunctions = c == 'X';
			} else if (c == 'c') {
				g_dumpClass = optarg;
			} else if (c == 'n') {
				frames = atoi(optarg);
			} else {
				fprintf(stderr, "usage: gl_probe [-x DIR | -X DIR] [-c CLASS] [-n FRAMES]\n");
				return 2;
			}
		}
		setvbuf(stdout, NULL, _IOLBF, 0);
		list_accelerators();
		list_metal_devices();
		if (g_dumpDir) {
			[[NSFileManager defaultManager] createDirectoryAtPath:g_dumpDir
			                          withIntermediateDirectories:YES
			                                           attributes:nil
			                                                error:NULL];
			const char *name = g_dumpClass ? g_dumpClass : "RgfxMetalDevice";
			Class cls = objc_getClass(name);
			printf("IR dump into %s (%s%s)\n", g_dumpDir.UTF8String, cls ? "" : "no ", name);
			if (cls) wrap_device_class(cls);
		}
		list_renderers();
		run_profile("legacy", kCGLOGLPVersion_Legacy, YES, frames);
		run_profile("legacy", kCGLOGLPVersion_Legacy, NO, frames);
		run_profile("3.2 core", kCGLOGLPVersion_GL3_Core, YES, frames);
		run_profile("4.1 core", kCGLOGLPVersion_GL4_Core, YES, frames);
		run_profile("4.1 core", kCGLOGLPVersion_GL4_Core, NO, frames);
		printf("done\n");
	}
	return 0;
}
