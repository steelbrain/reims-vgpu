// oracle_nilbind: what a command buffer does with nil where an object belongs: a draw or a
// dispatch whose shader reads a texture, sampler or buffer bound nil (or never bound), one that
// writes a texture bound nil, a render pass whose attachment is nil, a pipeline state set nil, a
// blit from or to nil, and a use of nil. No Metal validation layer is asked for: what a guest's
// process meets. Shaders: oracle_nilbind.metallib, next to the probe.
//   oracle_nilbind DEVICE              every case, each in a child process of its own
//   oracle_nilbind DEVICE --case N     case N alone (what the children run)
//   oracle_nilbind DEVICE --list       the cases, one a line, without running them
// The target is a 4 x 4 RGBA8Unorm texture cleared to (64, 64, 64, 64); a draw covers it whole
// and writes red one half plus half of what it read, so a texture reading zero draws
// (128, 0, 0, 255) and a draw that did not run leaves (64, 64, 64, 64). A dispatch writes what
// it read and the marker 42 into out[0..4], which starts at -1. The bound texture is 1 x 1 of
// (255, 255, 255, 255), the bound buffer a float4 of 1s. Each case prints:
//   case N NAME => encoder made|nil, status S error 'E' | rt R G B A | out a b c d marker
// and, where the case ends the process, Metal's words (its stderr is the child's stdout), then
//   => case N died on signal X | exited X
// Build: see README.md.
#import "oracle_common.h"
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

enum Kind { kDraw, kDispatch, kPass, kBlit };

struct Case {
	const char *name;
	enum Kind kind;
	const char *function; // the fragment function of a draw, the kernel of a dispatch
	int texture;          // 0 nil, 1 bound, 2 never set
	int sampler;          // 0 nil, 1 bound, 2 never set
	int buffer;           // 0 nil, 1 bound, 2 never set
	int variant;          // a pass's or a blit's own: see runCase
};

static const struct Case kCases[] = {
    {"frag.tex.bound", kDraw, "nil_tex_fs", 1, 2, 2, 0},
    {"frag.tex.nil", kDraw, "nil_tex_fs", 0, 2, 2, 0},
    {"frag.tex.unset", kDraw, "nil_tex_fs", 2, 2, 2, 0},
    {"frag.sample.bound", kDraw, "nil_sample_fs", 1, 1, 2, 0},
    {"frag.sample.tex.nil", kDraw, "nil_sample_fs", 0, 1, 2, 0},
    {"frag.sample.sampler.nil", kDraw, "nil_sample_fs", 1, 0, 2, 0},
    {"frag.buf.bound", kDraw, "nil_buf_fs", 2, 2, 1, 0},
    {"frag.buf.nil", kDraw, "nil_buf_fs", 2, 2, 0, 0},
    {"frag.buf.unset", kDraw, "nil_buf_fs", 2, 2, 2, 0},
    {"kern.tex.bound", kDispatch, "nil_tex_k", 1, 2, 2, 0},
    {"kern.tex.nil", kDispatch, "nil_tex_k", 0, 2, 2, 0},
    {"kern.tex.unset", kDispatch, "nil_tex_k", 2, 2, 2, 0},
    {"kern.buf.bound", kDispatch, "nil_buf_k", 2, 2, 1, 0},
    {"kern.buf.nil", kDispatch, "nil_buf_k", 2, 2, 0, 0},
    {"kern.texwrite.nil", kDispatch, "nil_texwrite_k", 0, 2, 2, 0},
    // variant 0: color 0 nil, the target's size set, a pipeline of no color format;
    // 1: color 0 nil, a pipeline of the target's format; 2: color 0 nil and color 1 the target;
    // 3: the target's depth attachment nil beside color 0; 4: color 0 nil, no size set.
    {"pass.color.nil.sized", kPass, "nil_solid_fs", 2, 2, 2, 0},
    {"pass.color.nil.format", kPass, "nil_solid_fs", 2, 2, 2, 1},
    {"pass.color0.nil.color1", kPass, "nil_solid_fs", 2, 2, 2, 2},
    {"pass.depth.nil", kPass, "nil_solid_fs", 2, 2, 2, 3},
    {"pass.color.nil.unsized", kPass, "nil_solid_fs", 2, 2, 2, 4},
    {"pass.pipeline.nil", kPass, "nil_solid_fs", 2, 2, 2, 5},
    {"pass.use.nil", kPass, "nil_solid_fs", 2, 2, 2, 6},
    // variant 0: a copy from a nil texture into a buffer; 1: from the target into a nil buffer;
    // 2: a fill of a nil buffer; 3: a copy from a nil texture into the target; 4: a nil
    // texture's mipmaps generated.
    {"blit.copy.from.nil", kBlit, NULL, 2, 2, 2, 0},
    {"blit.copy.to.nil", kBlit, NULL, 2, 2, 2, 1},
    {"blit.fill.nil", kBlit, NULL, 2, 2, 2, 2},
    {"blit.copytex.from.nil", kBlit, NULL, 2, 2, 2, 3},
    {"blit.mipmaps.nil", kBlit, NULL, 2, 2, 2, 4},
};
enum { kCaseCount = sizeof kCases / sizeof kCases[0] };

static id<MTLTexture> targetTexture(id<MTLDevice> d)
{
	MTLTextureDescriptor *td =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                                                       width:4
	                                                      height:4
	                                                   mipmapped:NO];
	td.storageMode = MTLStorageModePrivate;
	td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
	return [d newTextureWithDescriptor:td];
}

// A 1 x 1 Private RGBA8 texture of 255s, filled by a blit from a buffer.
static id<MTLTexture> whiteTexture(id<MTLDevice> d, id<MTLCommandQueue> q)
{
	MTLTextureDescriptor *td =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                                                       width:1
	                                                      height:1
	                                                   mipmapped:NO];
	td.storageMode = MTLStorageModePrivate;
	td.usage = MTLTextureUsageShaderRead;
	id<MTLTexture> t = [d newTextureWithDescriptor:td];
	uint8_t white[256];
	memset(white, 0xff, sizeof white);
	id<MTLBuffer> b = [d newBufferWithBytes:white length:sizeof white
	                                options:MTLResourceStorageModeShared];
	id<MTLCommandBuffer> cb = [q commandBuffer];
	id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
	[blit copyFromBuffer:b
	           sourceOffset:0
	      sourceBytesPerRow:256
	    sourceBytesPerImage:256
	             sourceSize:MTLSizeMake(1, 1, 1)
	              toTexture:t
	       destinationSlice:0
	       destinationLevel:0
	      destinationOrigin:MTLOriginMake(0, 0, 0)];
	[blit endEncoding];
	[cb commit];
	[cb waitUntilCompleted];
	return t;
}

// Clears `target` to 64s.
static void clearTarget(id<MTLCommandQueue> q, id<MTLTexture> target)
{
	MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
	rp.colorAttachments[0].texture = target;
	rp.colorAttachments[0].loadAction = MTLLoadActionClear;
	rp.colorAttachments[0].storeAction = MTLStoreActionStore;
	rp.colorAttachments[0].clearColor = MTLClearColorMake(64 / 255.0, 64 / 255.0, 64 / 255.0,
	                                                      64 / 255.0);
	id<MTLCommandBuffer> cb = [q commandBuffer];
	[[cb renderCommandEncoderWithDescriptor:rp] endEncoding];
	[cb commit];
	[cb waitUntilCompleted];
}

// The target's first pixel, by a blit into a Shared buffer.
static uint32_t targetPixel(id<MTLDevice> d, id<MTLCommandQueue> q, id<MTLTexture> target)
{
	id<MTLBuffer> b = [d newBufferWithLength:256 options:MTLResourceStorageModeShared];
	id<MTLCommandBuffer> cb = [q commandBuffer];
	id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
	[blit copyFromTexture:target
	                 sourceSlice:0
	                 sourceLevel:0
	                sourceOrigin:MTLOriginMake(0, 0, 0)
	                  sourceSize:MTLSizeMake(1, 1, 1)
	                    toBuffer:b
	           destinationOffset:0
	      destinationBytesPerRow:256
	    destinationBytesPerImage:256];
	[blit endEncoding];
	[cb commit];
	[cb waitUntilCompleted];
	return *(const uint32_t *)b.contents;
}

static void printEnd(int n, const struct Case *c, BOOL encoder, id<MTLCommandBuffer> cb,
                     uint32_t pixel, const float *out)
{
	printf("case %d %s => encoder %s, status %ld error '%s' | rt %u %u %u %u | out %g %g %g %g "
	       "%g\n",
	       n, c->name, encoder ? "made" : "nil", (long)cb.status,
	       cb.error ? cb.error.localizedDescription.UTF8String : "", pixel & 0xff,
	       (pixel >> 8) & 0xff, (pixel >> 16) & 0xff, pixel >> 24, out ? out[0] : 0,
	       out ? out[1] : 0, out ? out[2] : 0, out ? out[3] : 0, out ? out[4] : 0);
	fflush(stdout);
}

static int runCase(id<MTLDevice> d, const char *argv0, int n)
{
	const struct Case *c = &kCases[n];
	printf("case %d %s\n", n, c->name);
	fflush(stdout);
	id<MTLLibrary> lib = oracle_library(d, argv0, "oracle_nilbind");
	if (lib == nil)
		return 1;
	id<MTLCommandQueue> q = [d newCommandQueue];
	id<MTLTexture> target = targetTexture(d);
	id<MTLTexture> white = whiteTexture(d, q);
	MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
	id<MTLSamplerState> sampler = [d newSamplerStateWithDescriptor:sd];
	const float ones[4] = {1, 1, 1, 1};
	id<MTLBuffer> buffer = [d newBufferWithBytes:ones length:16 options:MTLResourceStorageModeShared];
	id<MTLBuffer> out = [d newBufferWithLength:64 options:MTLResourceStorageModeShared];
	for (int i = 0; i < 16; i++)
		((float *)out.contents)[i] = -1;
	clearTarget(q, target);
	NSError *err = nil;
	id<MTLCommandBuffer> cb = [q commandBuffer];
	BOOL encoder = YES;
	switch (c->kind) {
	case kDraw:
	case kPass: {
		MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
		pd.vertexFunction = [lib newFunctionWithName:@"nil_vs"];
		pd.fragmentFunction = [lib newFunctionWithName:@(c->function)];
		MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
		int v = c->kind == kPass ? c->variant : -1;
		if (v == 0 || v == 4) {
			pd.fragmentFunction = nil; // no color target to write
			if (v == 0) {
				rp.renderTargetWidth = 4;
				rp.renderTargetHeight = 4;
			}
		} else if (v == 1) {
			pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
			rp.renderTargetWidth = 4;
			rp.renderTargetHeight = 4;
		} else if (v == 2) {
			pd.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA8Unorm;
			rp.colorAttachments[1].texture = target;
			rp.colorAttachments[1].loadAction = MTLLoadActionLoad;
			rp.colorAttachments[1].storeAction = MTLStoreActionStore;
		} else {
			pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
			rp.colorAttachments[0].texture = target;
			rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
			rp.colorAttachments[0].storeAction = MTLStoreActionStore;
			if (v == 3) {
				pd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
				rp.depthAttachment.texture = nil;
				rp.depthAttachment.loadAction = MTLLoadActionClear;
				rp.depthAttachment.storeAction = MTLStoreActionDontCare;
			}
		}
		id<MTLRenderPipelineState> ps = [d newRenderPipelineStateWithDescriptor:pd error:&err];
		if (ps == nil) {
			printf("case %d %s => no pipeline: %s\n", n, c->name,
			       err ? err.localizedDescription.UTF8String : "nil");
			return 1;
		}
		id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
		if (enc == nil) {
			encoder = NO;
			break;
		}
		id<MTLRenderPipelineState> noPipeline = nil;
		[enc setRenderPipelineState:v == 5 ? noPipeline : ps];
		id<MTLResource> noResource = nil;
		if (v == 6)
			[enc useResource:noResource usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
		if (c->texture != 2)
			[enc setFragmentTexture:c->texture ? white : nil atIndex:0];
		if (c->sampler != 2)
			[enc setFragmentSamplerState:c->sampler ? sampler : nil atIndex:0];
		if (c->buffer != 2)
			[enc setFragmentBuffer:c->buffer ? buffer : nil offset:0 atIndex:0];
		[enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
		[enc endEncoding];
		break;
	}
	case kDispatch: {
		id<MTLComputePipelineState> ps =
		    [d newComputePipelineStateWithFunction:[lib newFunctionWithName:@(c->function)]
		                                     error:&err];
		if (ps == nil) {
			printf("case %d %s => no pipeline: %s\n", n, c->name,
			       err ? err.localizedDescription.UTF8String : "nil");
			return 1;
		}
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
		[enc setComputePipelineState:ps];
		if (c->texture != 2)
			[enc setTexture:c->texture ? white : nil atIndex:0];
		if (c->buffer != 2)
			[enc setBuffer:c->buffer ? buffer : nil offset:0 atIndex:0];
		[enc setBuffer:out offset:0 atIndex:1];
		[enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
		[enc endEncoding];
		break;
	}
	case kBlit: {
		id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
		id<MTLTexture> none = nil;
		id<MTLBuffer> noBuffer = nil;
		switch (c->variant) {
		case 0:
		case 1:
			[blit copyFromTexture:c->variant == 0 ? none : target
			                 sourceSlice:0
			                 sourceLevel:0
			                sourceOrigin:MTLOriginMake(0, 0, 0)
			                  sourceSize:MTLSizeMake(1, 1, 1)
			                    toBuffer:c->variant == 0 ? out : noBuffer
			           destinationOffset:0
			      destinationBytesPerRow:256
			    destinationBytesPerImage:256];
			break;
		case 2:
			[blit fillBuffer:noBuffer range:NSMakeRange(0, 16) value:7];
			break;
		case 3:
			[blit copyFromTexture:none toTexture:target];
			break;
		case 4:
			[blit generateMipmapsForTexture:none];
			break;
		}
		[blit endEncoding];
		break;
	}
	}
	[cb commit];
	[cb waitUntilCompleted];
	printEnd(n, c, encoder, cb, targetPixel(d, q, target), (const float *)out.contents);
	return 0;
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IOLBF, 0);
	if (argc < 2) {
		fprintf(stderr, "usage: %s DEVICE [--case N | --list]\n", argv[0]);
		return 2;
	}
	if (argc > 2 && !strcmp(argv[2], "--list")) {
		for (int n = 0; n < kCaseCount; n++)
			printf("%d %s\n", n, kCases[n].name);
		return 0;
	}
	if (argc > 3 && !strcmp(argv[2], "--case")) {
		// Metal's words where it ends the process go to stderr: put them in the case's lines.
		dup2(STDOUT_FILENO, STDERR_FILENO);
		int n = atoi(argv[3]);
		if (n < 0 || n >= kCaseCount)
			return 2;
		return runCase(oracle_device(argv[1]), argv[0], n);
	}
	oracle_header("oracle_nilbind", oracle_device(argv[1]));
	fflush(stdout);
	for (int n = 0; n < kCaseCount; n++) {
		char index[16];
		snprintf(index, sizeof index, "%d", n);
		char *args[] = {argv[0], argv[1], "--case", index, NULL};
		pid_t pid;
		if (posix_spawn(&pid, argv[0], NULL, NULL, args, environ))
			return 1;
		int st;
		waitpid(pid, &st, 0);
		if (WIFSIGNALED(st))
			printf("=> case %d died on signal %d\n", n, WTERMSIG(st));
		else if (WEXITSTATUS(st))
			printf("=> case %d exited %d\n", n, WEXITSTATUS(st));
		fflush(stdout);
	}
	return 0;
}
