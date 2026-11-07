// gld_lower: lowers the GL IR AppleMetalOpenGLRenderer hands a Metal device to AIR, through the
// GL code generator the system ships beside the renderer (libGLDPlugin32023.dylib, whose
// oglCodeGenService* functions Metal's compiler calls for a function of GL IR), and wraps the AIR
// in a .metallib of one function that a Metal device's newLibraryWithData: takes.
//   gld_lower lower IR [PLUGIN] OUT      IR as gl_probe -x writes it (a u32 length, then the IR),
//                                        PLUGIN a variant's plugin data (gl_probe -X); writes
//                                        OUT.ll (the AIR module's text), OUT.metallib, OUT.ret
//                                        (the plugin's return data), and loads OUT.metallib. It
//                                        verifies the module and reads its bitcode back first.
//   gld_lower pipeline VS [VSP] -- FS [FSP] OUT   lowers both, as `lower` does, then makes a render
//                                        pipeline (one RGBA8 colour attachment) of the two;
//                                        GLD_DRAW=1 then draws gl_probe's triangle with it.
//   gld_lower libs x                     loads the two libraries and initializes the generator.
//   gld_lower wrap IN.metallib OUT       rewraps IN's first function as this tool's writer
//                                        writes a .metallib, and loads it (the writer's check);
//                                        a vertex function also makes a depth-only pipeline.
//   gld_lower assemble IN.ll OUT         IN's text through the system's LLVM to bitcode, then
//                                        wrapped and loaded as `wrap` does.
// -d NAME picks the first of MTLCopyAllDevices whose name holds NAME.
// Our own code. It calls the system's own libraries in place; nothing of them is copied.
#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dlfcn.h>

// --- the system's LLVM, by its C API -----------------------------------------------------------

static const char *const kLLVM =
    "/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/libLLVM.dylib";
static const char *const kGLD =
    "/System/Library/Extensions/AppleMetalOpenGLRenderer.bundle/Contents/PlugIns/libGLDPlugin32023.dylib";

typedef void *Ref;
static Ref (*LLVMContextCreate)(void);
static Ref (*LLVMWriteBitcodeToMemoryBuffer)(Ref module);
static const char *(*LLVMGetBufferStart)(Ref buf);
static size_t (*LLVMGetBufferSize)(Ref buf);
static char *(*LLVMPrintModuleToString)(Ref module);
static unsigned (*LLVMGetNamedMetadataNumOperands)(Ref module, const char *name);
static void (*LLVMGetNamedMetadataOperands)(Ref module, const char *name, Ref *dest);
static unsigned (*LLVMGetMDNodeNumOperands)(Ref node);
static void (*LLVMGetMDNodeOperands)(Ref node, Ref *dest);
static const char *(*LLVMGetValueName2)(Ref value, size_t *length);
static const char *(*LLVMGetTarget)(Ref module);
static Ref (*LLVMGetFirstFunction)(Ref module);
static Ref (*LLVMGetNextFunction)(Ref fn);
static int (*LLVMIsDeclaration)(Ref global);
// Optional: the bitcode read back in a fresh context, as Metal's compiler reads it, and verified.
static Ref (*LLVMCreateMemoryBufferWithMemoryRangeCopy)(const char *data, size_t length, const char *name);
static int (*LLVMParseBitcodeInContext2)(Ref context, Ref buf, Ref *module);
static void (*LLVMContextSetDiagnosticHandler)(Ref context, void (*handler)(Ref info, void *), void *arg);
static char *(*LLVMGetDiagInfoDescription)(Ref info);
static int (*LLVMVerifyModule)(Ref module, int action, char **message);
static int (*LLVMParseIRInContext)(Ref context, Ref buf, Ref *module, char **message);

static void (*gldInitialize)(void);
typedef Ref (*GenerateFn)(Ref context, const void *ir, size_t length, const void *plugin,
                          size_t pluginLength, void **ret, size_t *retLength);
static GenerateFn gldGenerate;

#define SYM(h, name)                                                                   \
	do {                                                                               \
		*(void **)&name = dlsym(h, #name);                                             \
		if (!name) {                                                                   \
			fprintf(stderr, "gld_lower: no %s: %s\n", #name, dlerror());               \
			exit(1);                                                                   \
		}                                                                              \
	} while (0)

static void load_libraries(void) {
	void *llvm = dlopen(kLLVM, RTLD_NOW | RTLD_GLOBAL);
	if (!llvm) {
		fprintf(stderr, "gld_lower: %s\n", dlerror());
		exit(1);
	}
	SYM(llvm, LLVMContextCreate);
	SYM(llvm, LLVMWriteBitcodeToMemoryBuffer);
	SYM(llvm, LLVMGetBufferStart);
	SYM(llvm, LLVMGetBufferSize);
	SYM(llvm, LLVMPrintModuleToString);
	SYM(llvm, LLVMGetNamedMetadataNumOperands);
	SYM(llvm, LLVMGetNamedMetadataOperands);
	SYM(llvm, LLVMGetMDNodeNumOperands);
	SYM(llvm, LLVMGetMDNodeOperands);
	SYM(llvm, LLVMGetValueName2);
	SYM(llvm, LLVMGetTarget);
	SYM(llvm, LLVMGetFirstFunction);
	SYM(llvm, LLVMGetNextFunction);
	SYM(llvm, LLVMIsDeclaration);
	*(void **)&LLVMCreateMemoryBufferWithMemoryRangeCopy = dlsym(llvm, "LLVMCreateMemoryBufferWithMemoryRangeCopy");
	*(void **)&LLVMParseBitcodeInContext2 = dlsym(llvm, "LLVMParseBitcodeInContext2");
	*(void **)&LLVMContextSetDiagnosticHandler = dlsym(llvm, "LLVMContextSetDiagnosticHandler");
	*(void **)&LLVMGetDiagInfoDescription = dlsym(llvm, "LLVMGetDiagInfoDescription");
	*(void **)&LLVMVerifyModule = dlsym(llvm, "LLVMVerifyModule");
	*(void **)&LLVMParseIRInContext = dlsym(llvm, "LLVMParseIRInContext");
	void *gld = dlopen(kGLD, RTLD_NOW);
	if (!gld) {
		fprintf(stderr, "gld_lower: %s\n", dlerror());
		exit(1);
	}
	*(void **)&gldInitialize = dlsym(gld, "oglCodeGenServiceInitialize");
	*(void **)&gldGenerate = dlsym(gld, "oglCodeGenServiceGenerateIRWithPluginDataAndPluginReturnData");
	if (!gldInitialize || !gldGenerate) {
		fprintf(stderr, "gld_lower: the plugin lacks its entry points\n");
		exit(1);
	}
	gldInitialize();
}

// --- the .metallib writer ----------------------------------------------------------------------
//
// The layout `metal -mmacosx-version-min=11.0` writes for one function (read from such a file):
// an 88-byte header, the function list (a u32 count, then the function's tag group: a u32 size
// counting itself, then NAME, TYPE, HASH, OFFT, VERS and MDSZ, each a 4-byte name, a u16 size and
// the data, and ENDT), the header's extension (UUID, ENDT), empty public and private metadata (a
// u32 size of 8 and ENDT each), then the bitcode: a wrapper (0x0B17C0DE, 0, 20, the bitcode's
// size, 0xffffffff), the bitcode, zeros to a multiple of 16. HASH is the SHA-256 of that section.

static void put(NSMutableData *d, const void *p, size_t n) { [d appendBytes:p length:n]; }
static void put16(NSMutableData *d, uint16_t v) { put(d, &v, 2); }
static void put32(NSMutableData *d, uint32_t v) { put(d, &v, 4); }
static void put64(NSMutableData *d, uint64_t v) { put(d, &v, 8); }
static void tag(NSMutableData *d, const char *name, const void *p, uint16_t n) {
	put(d, name, 4);
	put16(d, n);
	put(d, p, n);
}

static NSData *write_metallib(const char *name, uint8_t type, const uint8_t vers[8], const void *bc,
                              size_t bcLength) {
	NSMutableData *section = [NSMutableData data];
	put32(section, 0x0B17C0DE);
	put32(section, 0);
	put32(section, 20);
	put32(section, (uint32_t)bcLength);
	put32(section, 0xffffffff);
	put(section, bc, bcLength);
	while (section.length % 16) put(section, "\0", 1);
	uint8_t hash[32];
	CC_SHA256(section.bytes, (CC_LONG)section.length, hash);

	NSMutableData *tags = [NSMutableData data];
	tag(tags, "NAME", name, (uint16_t)(strlen(name) + 1));
	tag(tags, "TYPE", &type, 1);
	tag(tags, "HASH", hash, 32);
	uint64_t offt[3] = {0, 0, 0};
	tag(tags, "OFFT", offt, 24);
	tag(tags, "VERS", vers, 8);
	uint64_t mdsz = section.length;
	tag(tags, "MDSZ", &mdsz, 8);
	put(tags, "ENDT", 4);
	NSMutableData *group = [NSMutableData data];
	put32(group, (uint32_t)(4 + tags.length));
	[group appendData:tags];

	NSMutableData *ext = [NSMutableData data];
	tag(ext, "UUID", hash, 16); // the library's UUID: the first 16 bytes of the bitcode's hash
	put(ext, "ENDT", 4);

	const uint64_t listOffset = 0x58, listSize = group.length;
	const uint64_t pubOffset = listOffset + 4 + group.length + ext.length;
	const uint64_t privOffset = pubOffset + 8, bcOffset = privOffset + 8;
	const uint64_t size = bcOffset + section.length;
	NSMutableData *out = [NSMutableData data];
	put(out, "MTLB", 4);
	put16(out, 0x8001);
	put16(out, 2);
	put16(out, 5);
	put(out, "\0\0", 2);
	put16(out, 0);
	put16(out, 0);
	put64(out, size);
	put64(out, listOffset);
	put64(out, listSize);
	put64(out, pubOffset);
	put64(out, 8);
	put64(out, privOffset);
	put64(out, 8);
	put64(out, bcOffset);
	put64(out, section.length);
	put32(out, 1);
	[out appendData:group];
	[out appendData:ext];
	put32(out, 8);
	put(out, "ENDT", 4);
	put32(out, 8);
	put(out, "ENDT", 4);
	[out appendData:section];
	return out;
}

// The system's LLVM writes bitcode for an Apple triple inside the same wrapper the .metallib's
// section has (0x0B17C0DE, 0, the offset, the size, 0xffffffff), and Metal's compiler refuses a
// section wrapped twice (measured: its service aborts reading the module, and Metal reports an
// interrupted connection). This returns the bitcode inside such a wrapper, or the buffer as it is.
static const char *unwrap(const char *bc, size_t *length) {
	uint32_t w[5];
	if (*length < sizeof w) return bc;
	memcpy(w, bc, sizeof w);
	if (w[0] != 0x0B17C0DE || (size_t)w[2] + w[3] > *length) return bc;
	*length = w[3];
	return bc + w[2];
}

// --- lowering ----------------------------------------------------------------------------------

typedef struct {
	NSData *metallib;
	NSString *name;
	uint8_t type;
} Lowered;

static NSData *read_file(const char *path) {
	NSData *d = [NSData dataWithContentsOfFile:@(path)];
	if (!d) {
		fprintf(stderr, "gld_lower: cannot read %s\n", path);
		exit(1);
	}
	return d;
}

static BOOL entry_of(Ref module, const char *md, const char **name) {
	unsigned n = LLVMGetNamedMetadataNumOperands(module, md);
	if (n == 0) return NO;
	Ref nodes[n];
	LLVMGetNamedMetadataOperands(module, md, nodes);
	unsigned k = LLVMGetMDNodeNumOperands(nodes[0]);
	if (k == 0) return NO;
	Ref ops[k];
	LLVMGetMDNodeOperands(nodes[0], ops);
	size_t len = 0;
	*name = ops[0] ? LLVMGetValueName2(ops[0], &len) : NULL;
	printf("  %s: %u entries, the first %s\n", md, n, *name ? *name : "(unnamed)");
	return YES;
}

static void diagnose(Ref info, void *arg) {
	char *d = LLVMGetDiagInfoDescription ? LLVMGetDiagInfoDescription(info) : NULL;
	printf("  read back: diagnostic: %s\n", d ? d : "(no text)");
}

// Reads the bitcode back in a fresh context and verifies the module, printing what fails.
static void reparse(const char *bc, size_t length) {
	if (!LLVMCreateMemoryBufferWithMemoryRangeCopy || !LLVMParseBitcodeInContext2) {
		printf("  read back: the library lacks the reader's C API\n");
		return;
	}
	Ref context = LLVMContextCreate();
	if (LLVMContextSetDiagnosticHandler) LLVMContextSetDiagnosticHandler(context, diagnose, NULL);
	Ref buf = LLVMCreateMemoryBufferWithMemoryRangeCopy(bc, length, "bitcode");
	Ref module = NULL;
	int failed = LLVMParseBitcodeInContext2(context, buf, &module);
	printf("  read back: %s\n", failed ? "failed" : "a module");
	if (!failed && module && LLVMVerifyModule) {
		char *message = NULL;
		int bad = LLVMVerifyModule(module, 2, &message);
		printf("  verifier: %s%s\n", bad ? "invalid: " : "valid", bad && message ? message : "");
	}
}

static Lowered lower(const char *irPath, const char *pluginPath, const char *out) {
	NSData *ir = read_file(irPath);
	NSData *plugin = pluginPath ? read_file(pluginPath) : nil;
	Ref context = LLVMContextCreate();
	void *ret = NULL;
	size_t retLength = 0;
	NSDate *t0 = [NSDate date];
	// The IR as the renderer hands it: a u32 length, then that many bytes, which the generator
	// takes (they open with a u32 kind or a bitcode wrapper).
	uint32_t irLength = 0;
	if (ir.length >= 4) memcpy(&irLength, ir.bytes, 4);
	if (irLength > ir.length - 4) {
		fprintf(stderr, "gld_lower: %s: a length of %u past its %lu bytes\n", irPath, irLength,
		        (unsigned long)ir.length);
		exit(1);
	}
	const uint8_t *payload = (const uint8_t *)ir.bytes + 4;
	printf("lower %s: %u bytes, opening %02x%02x%02x%02x %02x%02x%02x%02x\n", irPath, irLength, payload[0],
	       payload[1], payload[2], payload[3], payload[4], payload[5], payload[6], payload[7]);
	Ref module = gldGenerate(context, payload, irLength, plugin.bytes, plugin.length, &ret, &retLength);
	double ms = -[t0 timeIntervalSinceNow] * 1e3;
	printf("lower %s%s%s: %s in %.2f ms, return data %zu bytes\n", irPath, plugin ? " with " : "",
	       plugin ? pluginPath : "", module ? "a module" : "no module", ms, retLength);
	if (!module) exit(1);
	NSString *base = @(out);
	if (ret && retLength)
		[[NSData dataWithBytes:ret length:retLength] writeToFile:[base stringByAppendingString:@".ret"]
		                                               atomically:NO];
	char *text = LLVMPrintModuleToString(module);
	[@(text) writeToFile:[base stringByAppendingString:@".ll"] atomically:NO encoding:NSUTF8StringEncoding error:NULL];
	// GLD_REPLACE holds lines of FROM<tab>TO: each FROM in the module's text becomes TO, and the
	// text is parsed again in place of the module (to find which part of it a reader refuses).
	const char *replace = getenv("GLD_REPLACE");
	if (replace && replace[0] && LLVMParseIRInContext) {
		NSMutableString *edited = [@(text) mutableCopy];
		for (NSString *line in [@(replace) componentsSeparatedByString:@"\n"]) {
			NSArray<NSString *> *pair = [line componentsSeparatedByString:@"\t"];
			if (pair.count != 2) continue;
			NSUInteger n = [edited replaceOccurrencesOfString:pair[0] withString:pair[1] options:0
			                                            range:NSMakeRange(0, edited.length)];
			printf("  replace \"%s\": %lu\n", pair[0].UTF8String, (unsigned long)n);
		}
		const char *et = edited.UTF8String;
		Ref eb = LLVMCreateMemoryBufferWithMemoryRangeCopy(et, strlen(et), "edited");
		Ref em = NULL;
		char *message = NULL;
		Ref ec = LLVMContextCreate();
		if (LLVMParseIRInContext(ec, eb, &em, &message) || !em) {
			fprintf(stderr, "gld_lower: the edited text does not parse: %s\n", message ? message : "");
			exit(1);
		}
		module = em;
		text = LLVMPrintModuleToString(module);
		[@(text) writeToFile:[base stringByAppendingString:@".edited.ll"] atomically:NO
		            encoding:NSUTF8StringEncoding error:NULL];
	}
	printf("  target %s, %zu bytes of text\n", LLVMGetTarget(module), strlen(text));
	unsigned undefined = 0;
	for (Ref f = LLVMGetFirstFunction(module); f; f = LLVMGetNextFunction(f)) {
		size_t len = 0;
		const char *n = LLVMGetValueName2(f, &len);
		if (LLVMIsDeclaration(f) && strncmp(n, "air.", 4) != 0 && strncmp(n, "llvm.", 5) != 0) {
			if (undefined++ < 12) printf("  declared, not defined: %s\n", n);
		}
	}
	printf("  %u functions declared outside air.* and llvm.*\n", undefined);
	Lowered l = {nil, nil, 0};
	const char *name = NULL;
	if (entry_of(module, "air.vertex", &name)) l.type = 0;
	else if (entry_of(module, "air.fragment", &name)) l.type = 1;
	else if (entry_of(module, "air.kernel", &name)) l.type = 2;
	if (!name) {
		fprintf(stderr, "gld_lower: no entry point in the module\n");
		exit(1);
	}
	l.name = @(name);
	unsigned vmaj = 2, vmin = 3;
	const char *triple = LLVMGetTarget(module);
	const char *v = strstr(triple, "_v");
	if (v && v[2] && v[3]) {
		vmaj = (unsigned)(v[2] - '0');
		vmin = (unsigned)(v[3] - '0');
	}
	uint8_t vers[8] = {(uint8_t)vmaj, 0, (uint8_t)vmin, 0, (uint8_t)vmaj, 0, (uint8_t)vmin, 0};
	if (LLVMVerifyModule) {
		char *message = NULL;
		int bad = LLVMVerifyModule(module, 2, &message);
		printf("  verifier on the generated module: %s%s\n", bad ? "invalid: " : "valid",
		       bad && message ? message : "");
	}
	Ref buf = LLVMWriteBitcodeToMemoryBuffer(module);
	reparse(LLVMGetBufferStart(buf), LLVMGetBufferSize(buf));
	size_t bcLength = LLVMGetBufferSize(buf);
	const char *bc = unwrap(LLVMGetBufferStart(buf), &bcLength);
	l.metallib = write_metallib(name, l.type, vers, bc, bcLength);
	[l.metallib writeToFile:[base stringByAppendingString:@".metallib"] atomically:NO];
	printf("  %s, type %u, AIR %u.%u: %zu bytes of bitcode, %lu bytes of .metallib\n", name, l.type,
	       vmaj, vmin, bcLength, (unsigned long)l.metallib.length);
	return l;
}

static id<MTLFunction> load(id<MTLDevice> device, NSData *metallib, NSString *name) {
	NSError *error = nil;
	dispatch_data_t data = dispatch_data_create(metallib.bytes, metallib.length, NULL,
	                                            DISPATCH_DATA_DESTRUCTOR_DEFAULT);
	id<MTLLibrary> lib = [device newLibraryWithData:data error:&error];
	printf("  newLibraryWithData on %s: %s%s\n", device.name.UTF8String, lib ? "a library" : "nil",
	       error ? [NSString stringWithFormat:@", %@", error].UTF8String : "");
	if (!lib) return nil;
	printf("  functionNames %s\n", [lib.functionNames componentsJoinedByString:@" "].UTF8String);
	id<MTLFunction> f = [lib newFunctionWithName:name];
	printf("  newFunctionWithName %s: %s, functionType %lu\n", name.UTF8String, f ? "a function" : "nil",
	       (unsigned long)f.functionType);
	return f;
}

static id<MTLDevice> pick(const char *want) {
	if (!want) return MTLCreateSystemDefaultDevice();
	for (id<MTLDevice> d in MTLCopyAllDevices())
		if (strstr(d.name.UTF8String, want)) return d;
	return nil;
}

static id<MTLRenderPipelineState> make_pipeline(id<MTLDevice> device, id<MTLFunction> vs,
                                                id<MTLFunction> fs) {
	MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
	d.vertexFunction = vs;
	d.fragmentFunction = fs;
	// With no fragment function the pipeline is depth-only, so the vertex function's position
	// still reaches the rasterizer.
	if (fs) d.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
	else d.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
	// GLD_VERTEX=BUFFER:STRIDE gives attribute 0 a float2 from that buffer (the renderer's
	// vertexBuffer.0, at 2 with stride 8 for gl_probe's triangle); GLD_GL_MODE=1 sets the
	// descriptor's openGLModeEnabled, as the renderer does.
	const char *vertex = getenv("GLD_VERTEX");
	unsigned buffer = 0, stride = 0;
	if (vertex && sscanf(vertex, "%u:%u", &buffer, &stride) == 2) {
		MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
		vd.attributes[0].format = MTLVertexFormatFloat2;
		vd.attributes[0].bufferIndex = buffer;
		vd.layouts[buffer].stride = stride;
		d.vertexDescriptor = vd;
		printf("  vertex descriptor: attribute 0 float2, buffer %u, stride %u\n", buffer, stride);
	}
	const char *glMode = getenv("GLD_GL_MODE");
	if (glMode && strcmp(glMode, "1") == 0) {
		@try {
			[d setValue:@YES forKey:@"openGLModeEnabled"];
			printf("  openGLModeEnabled set\n");
		} @catch (NSException *e) {
			printf("  openGLModeEnabled: %s\n", e.reason.UTF8String);
		}
	}
	NSError *error = nil;
	NSDate *t0 = [NSDate date];
	id<MTLRenderPipelineState> ps = [device newRenderPipelineStateWithDescriptor:d error:&error];
	printf("  render pipeline: %s in %.1f ms%s\n", ps ? "made" : "nil", -[t0 timeIntervalSinceNow] * 1e3,
	       error ? [NSString stringWithFormat:@", %@", error].UTF8String : "");
	return ps;
}

// Draws gl_probe's triangle case with the pipeline: a 64x64 RGBA8 target cleared to
// (0.25, 0.5, 0.75, 1), the triangle (-0.5,-0.5) (0.5,-0.5) (0,0.5) from vertex buffer BUFFER, and
// fragment buffer 0 holding the uniform colour (0, 1, 0, 1). Returns whether the centre reads
// (0,255,0,255) and a corner (64,128,191,255).
static BOOL draw(id<MTLDevice> device, id<MTLRenderPipelineState> ps, unsigned buffer) {
	MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                                                                              width:64
	                                                                             height:64
	                                                                          mipmapped:NO];
	td.usage = MTLTextureUsageRenderTarget;
	td.storageMode = MTLStorageModeManaged;
	id<MTLTexture> target = [device newTextureWithDescriptor:td];
	id<MTLCommandQueue> queue = [device newCommandQueue];
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
	rp.colorAttachments[0].texture = target;
	rp.colorAttachments[0].loadAction = MTLLoadActionClear;
	rp.colorAttachments[0].storeAction = MTLStoreActionStore;
	rp.colorAttachments[0].clearColor = MTLClearColorMake(0.25, 0.5, 0.75, 1);
	id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
	[e setRenderPipelineState:ps];
	const float vertices[6] = {-0.5f, -0.5f, 0.5f, -0.5f, 0.0f, 0.5f};
	const float colour[4] = {0, 1, 0, 1};
	[e setVertexBytes:vertices length:sizeof vertices atIndex:buffer];
	[e setFragmentBytes:colour length:sizeof colour atIndex:0];
	[e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
	[e endEncoding];
	id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
	[b synchronizeResource:target];
	[b endEncoding];
	[cb commit];
	[cb waitUntilCompleted];
	uint8_t centre[4], corner[4];
	[target getBytes:centre bytesPerRow:256 fromRegion:MTLRegionMake2D(32, 32, 1, 1) mipmapLevel:0];
	[target getBytes:corner bytesPerRow:256 fromRegion:MTLRegionMake2D(2, 2, 1, 1) mipmapLevel:0];
	BOOL ok = centre[0] == 0 && centre[1] == 255 && centre[2] == 0 && centre[3] == 255 && corner[0] == 64 &&
	          corner[1] == 128 && corner[2] == 191 && corner[3] == 255;
	printf("  draw: %s, centre %u,%u,%u,%u corner %u,%u,%u,%u%s\n", ok ? "pass" : "FAIL", centre[0], centre[1],
	       centre[2], centre[3], corner[0], corner[1], corner[2], corner[3],
	       cb.error ? [NSString stringWithFormat:@", %@", cb.error].UTF8String : "");
	return ok;
}

int main(int argc, char **argv) {
	@autoreleasepool {
		setvbuf(stdout, NULL, _IOLBF, 0);
		const char *want = NULL;
		int a = 1;
		if (argc > 2 && strcmp(argv[1], "-d") == 0) {
			want = argv[2];
			a = 3;
		}
		if (argc - a < 2) {
			fprintf(stderr, "usage: gld_lower [-d NAME] lower IR [PLUGIN] OUT | pipeline VS [VSP] -- FS "
			                "[FSP] OUT | wrap IN.metallib OUT\n");
			return 2;
		}
		id<MTLDevice> device = pick(want);
		printf("device %s\n", device ? device.name.UTF8String : "(none)");
		const char *mode = argv[a];
		char **rest = argv + a + 1;
		int n = argc - a - 1;
		if (strcmp(mode, "wrap") == 0 && n == 2) {
			NSData *in = read_file(rest[0]);
			const uint8_t *p = in.bytes;
			uint64_t bcOff, bcSize;
			memcpy(&bcOff, p + 0x48, 8);
			memcpy(&bcSize, p + 0x50, 8);
			uint32_t inner, innerOff;
			memcpy(&innerOff, p + bcOff + 8, 4);
			memcpy(&inner, p + bcOff + 12, 4);
			// The first function's name and type from its group (NAME then TYPE lead it).
			const char *name = (const char *)p + 0x58 + 4 + 4 + 6;
			uint8_t type = p[0x58 + 4 + 4 + 6 + strlen(name) + 1 + 6];
			uint8_t vers[8] = {2, 0, 3, 0, 2, 0, 3, 0};
			NSData *out = write_metallib(name, type, vers, p + bcOff + innerOff, inner);
			[out writeToFile:@(rest[1]) atomically:NO];
			printf("wrap %s: %s, type %u, %u bytes of bitcode, %lu bytes out\n", rest[0], name, type, inner,
			       (unsigned long)out.length);
			id<MTLFunction> f = load(device, out, @(name));
			if (f && type == 0) make_pipeline(device, f, nil);
			return f ? 0 : 1;
		}
		load_libraries();
		if (strcmp(mode, "libs") == 0) {
			printf("libraries: the system's LLVM and GL code generator loaded and initialized\n");
			return 0;
		}
		if (strcmp(mode, "lower") == 0 && (n == 2 || n == 3)) {
			Lowered l = lower(rest[0], n == 3 ? rest[1] : NULL, rest[n - 1]);
			id<MTLFunction> f = load(device, l.metallib, l.name);
			if (f && l.type == 0) make_pipeline(device, f, nil);
			return f ? 0 : 1;
		}
		if (strcmp(mode, "assemble") == 0 && n == 2) {
			// IN.ll's text through the system's LLVM to bitcode, then as `lower` goes on: the
			// check of this LLVM's bitcode writer apart from the GL code generator.
			NSData *text = read_file(rest[0]);
			NSMutableData *z = [text mutableCopy];
			[z appendBytes:"" length:1];
			Ref context = LLVMContextCreate();
			Ref buf = LLVMCreateMemoryBufferWithMemoryRangeCopy(z.bytes, text.length, "text");
			Ref module = NULL;
			char *message = NULL;
			if (LLVMParseIRInContext(context, buf, &module, &message) || !module) {
				fprintf(stderr, "gld_lower: %s does not parse: %s\n", rest[0], message ? message : "");
				return 1;
			}
			const char *name = NULL;
			uint8_t type = 0;
			if (entry_of(module, "air.vertex", &name)) type = 0;
			else if (entry_of(module, "air.fragment", &name)) type = 1;
			else if (entry_of(module, "air.kernel", &name)) type = 2;
			if (!name) return 1;
			uint8_t vers[8] = {2, 0, 3, 0, 2, 0, 3, 0};
			Ref bc = LLVMWriteBitcodeToMemoryBuffer(module);
			reparse(LLVMGetBufferStart(bc), LLVMGetBufferSize(bc));
			size_t bcLength = LLVMGetBufferSize(bc);
			const char *raw = unwrap(LLVMGetBufferStart(bc), &bcLength);
			NSData *lib = write_metallib(name, type, vers, raw, bcLength);
			[lib writeToFile:@(rest[1]) atomically:NO];
			id<MTLFunction> f = load(device, lib, @(name));
			if (f && type == 0) make_pipeline(device, f, nil);
			return f ? 0 : 1;
		}
		if (strcmp(mode, "pipeline") == 0) {
			int sep = -1;
			for (int i = 0; i < n; i++)
				if (strcmp(rest[i], "--") == 0) sep = i;
			if (sep < 1 || n - sep < 3) return 2;
			NSString *out = @(rest[n - 1]);
			Lowered v = lower(rest[0], sep == 2 ? rest[1] : NULL,
			                  [out stringByAppendingString:@".vs"].UTF8String);
			Lowered f = lower(rest[sep + 1], n - sep - 1 == 3 ? rest[sep + 2] : NULL,
			                  [out stringByAppendingString:@".fs"].UTF8String);
			id<MTLFunction> vf = load(device, v.metallib, v.name), ff = load(device, f.metallib, f.name);
			id<MTLRenderPipelineState> ps = vf && ff ? make_pipeline(device, vf, ff) : nil;
			if (!ps) return 1;
			// GLD_DRAW=1 draws with it (the vertex buffer is GLD_VERTEX's).
			const char *drawIt = getenv("GLD_DRAW"), *vertex = getenv("GLD_VERTEX");
			unsigned buffer = 0, stride = 0;
			if (drawIt && strcmp(drawIt, "1") == 0 && vertex && sscanf(vertex, "%u:%u", &buffer, &stride) == 2)
				return draw(device, ps, buffer) ? 0 : 1;
			return 0;
		}
		fprintf(stderr, "gld_lower: unknown mode or arguments\n");
		return 2;
	}
}
