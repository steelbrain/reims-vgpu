// oracle_sampler: what a sampler's lodBias and reductionMode do to its samples, and what the
// device answers about them. Headless: compute dispatches and one offscreen render pass.
//   oracle_sampler DEVICE
// Lines:
//   key=value   the device's families, and every unary SPI getter of its classes whose name
//               mentions reduction or bias (n/r where it does not respond)
//   bias B mip M  level|gradient|fragment L->R
//               the R a sample with sampler lodBias B and mip filter M (linear, nearest) returns
//               at LOD L, from an explicit level(L), a gradient2d() of LOD L, or a fragment
//               shader's implicit LOD L. The texture is 256 x 256 RGBA16Float of 9 levels, level
//               i holding i everywhere, so with a linear mip filter R is the LOD the sampler used.
//   reduce MODE mip M  AT R,G
//               the R and G a sample with reductionMode MODE (weighted, min, max), linear min and
//               mag filters and mip filter M returns from a 2 x 2 RGBA16Float texture of 2 levels:
//               level 0 R = [0.1 0.9; 0.3 0.7] and G = [0.8 0.2; 0.6 0.4], level 1 R = 0.05 and
//               G = 0.95. AT is the coordinate and LOD: centre@0 (four texels at 0.25 each:
//               weighted 0.5,0.5, min 0.1,0.2, max 0.9,0.8), centre@0.5 (both levels, half each:
//               weighted 0.275,0.725, min 0.05,0.2, max 0.9,0.95), texel00@0 (one texel at weight
//               1: 0.1,0.8 in every mode) and x0.375@0 (x weights 0.75 and 0.25: weighted
//               0.35,0.6).
// A min or max sampler on a device without the feature ends the process under Metal's
// validation layer (MTL_DEBUG_LAYER=1); without it, the line shows what the device samples.
// Build: see README.md.
#import "oracle_common.h"
#include <ctype.h>
#include <objc/message.h>
#include <objc/runtime.h>

// An IEEE half of `f`, rounded to nearest even (normal numbers and zero).
static uint16_t half_of(float f)
{
	if (f == 0.0f)
		return 0;
	uint32_t b;
	memcpy(&b, &f, 4);
	uint32_t sign = (b >> 16) & 0x8000;
	int32_t exp = (int32_t)((b >> 23) & 0xff) - 127 + 15;
	uint32_t man = b & 0x7fffff;
	uint32_t m = man >> 13;
	uint32_t rest = man & 0x1fff;
	if (rest > 0x1000 || (rest == 0x1000 && (m & 1))) {
		m++;
		if (m == 0x400) {
			m = 0;
			exp++;
		}
	}
	return (uint16_t)(sign | ((uint32_t)exp << 10) | m);
}

// A size x size RGBA16Float texture of `levels` levels, each level's texels from `fill`.
static id<MTLTexture> make_texture(id<MTLDevice> d, NSUInteger size, NSUInteger levels,
                                   void (^fill)(NSUInteger level, NSUInteger i, uint16_t px[4]))
{
	MTLTextureDescriptor *td =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
	                                                       width:size
	                                                      height:size
	                                                   mipmapped:YES];
	td.mipmapLevelCount = levels;
	td.storageMode = d.hasUnifiedMemory ? MTLStorageModeShared : MTLStorageModeManaged;
	td.usage = MTLTextureUsageShaderRead;
	id<MTLTexture> t = [d newTextureWithDescriptor:td];
	for (NSUInteger l = 0; l < levels; l++) {
		NSUInteger w = MAX(size >> l, 1u);
		uint16_t *px = calloc(w * w * 4, 2);
		for (NSUInteger i = 0; i < w * w; i++)
			fill(l, i, px + i * 4);
		[t replaceRegion:MTLRegionMake2D(0, 0, w, w) mipmapLevel:l withBytes:px bytesPerRow:w * 8];
		free(px);
	}
	return t;
}

static id<MTLSamplerState> make_sampler(id<MTLDevice> d, float bias, MTLSamplerReductionMode mode,
                                        MTLSamplerMipFilter mip)
{
	MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
	sd.minFilter = MTLSamplerMinMagFilterLinear;
	sd.magFilter = MTLSamplerMinMagFilterLinear;
	sd.mipFilter = mip;
	sd.sAddressMode = MTLSamplerAddressModeClampToEdge;
	sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
	sd.lodBias = bias;
	sd.reductionMode = mode;
	return [d newSamplerStateWithDescriptor:sd];
}

// out[i] = the compute kernel's sample at params[i], for n params.
static void run_compute(id<MTLDevice> d, id<MTLCommandQueue> q, id<MTLComputePipelineState> ps,
                        id<MTLTexture> t, id<MTLSamplerState> s, const float *params, NSUInteger n,
                        float *out)
{
	id<MTLBuffer> pb = [d newBufferWithBytes:params length:n * 16 options:MTLResourceStorageModeShared];
	id<MTLBuffer> ob = [d newBufferWithLength:n * 16 options:MTLResourceStorageModeShared];
	id<MTLCommandBuffer> cb = [q commandBuffer];
	id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
	[e setComputePipelineState:ps];
	[e setTexture:t atIndex:0];
	[e setSamplerState:s atIndex:0];
	[e setBuffer:pb offset:0 atIndex:0];
	[e setBuffer:ob offset:0 atIndex:1];
	[e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
	[e endEncoding];
	oracle_commit_wait(cb, NULL, NULL);
	memcpy(out, ob.contents, n * 16);
}

// R of pixel (3, 3) of an 8 x 8 target the fragment shader samples into at implicit LOD `lod`.
static float run_fragment(id<MTLDevice> d, id<MTLCommandQueue> q, id<MTLRenderPipelineState> ps,
                          id<MTLTexture> t, id<MTLSamplerState> s, float lod)
{
	MTLTextureDescriptor *td =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
	                                                       width:8
	                                                      height:8
	                                                   mipmapped:NO];
	td.storageMode = MTLStorageModePrivate;
	td.usage = MTLTextureUsageRenderTarget;
	id<MTLTexture> rt = [d newTextureWithDescriptor:td];
	id<MTLBuffer> ob = [d newBufferWithLength:8 * 8 * 16 options:MTLResourceStorageModeShared];
	id<MTLCommandBuffer> cb = [q commandBuffer];
	MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
	rp.colorAttachments[0].texture = rt;
	rp.colorAttachments[0].loadAction = MTLLoadActionClear;
	rp.colorAttachments[0].storeAction = MTLStoreActionStore;
	id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
	[e setRenderPipelineState:ps];
	[e setFragmentTexture:t atIndex:0];
	[e setFragmentSamplerState:s atIndex:0];
	[e setFragmentBytes:&lod length:4 atIndex:0];
	[e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
	[e endEncoding];
	id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
	[b copyFromTexture:rt
	             sourceSlice:0
	             sourceLevel:0
	            sourceOrigin:MTLOriginMake(0, 0, 0)
	              sourceSize:MTLSizeMake(8, 8, 1)
	                toBuffer:ob
	       destinationOffset:0
	  destinationBytesPerRow:8 * 16
	destinationBytesPerImage:8 * 8 * 16];
	[b endEncoding];
	oracle_commit_wait(cb, NULL, NULL);
	return ((const float *)ob.contents)[(3 * 8 + 3) * 4];
}

static void print_answers(id<MTLDevice> d)
{
	const struct {
		const char *name;
		MTLGPUFamily family;
	} fams[] = {
	    {"Apple7", MTLGPUFamilyApple7}, {"Apple8", MTLGPUFamilyApple8},
	    {"Apple9", MTLGPUFamilyApple9}, {"Apple10", MTLGPUFamilyApple10},
	    {"Mac2", MTLGPUFamilyMac2},     {"Metal3", MTLGPUFamilyMetal3},
	    {"Metal4", MTLGPUFamilyMetal4},
	};
	for (size_t i = 0; i < sizeof fams / sizeof fams[0]; i++)
		printf("supportsFamily(%s)=%d\n", fams[i].name, (int)[d supportsFamily:fams[i].family]);
	const char *getters[] = {
	    "supportsSamplerReductionMode",
	    "samplerReductionModeSupport",
	    "supportsExtendedSamplerLODBiasRange",
	};
	for (size_t i = 0; i < sizeof getters / sizeof getters[0]; i++) {
		SEL sel = sel_registerName(getters[i]);
		Method m = class_getInstanceMethod(object_getClass(d), sel);
		if (!m || ![d respondsToSelector:sel]) {
			printf("%s=n/r\n", getters[i]);
			continue;
		}
		char ret[16];
		method_getReturnType(m, ret, sizeof ret);
		unsigned long long v = ((unsigned long long (*)(id, SEL))objc_msgSend)(d, sel);
		if (ret[0] == 'c' || ret[0] == 'B' || ret[0] == 'C')
			v &= 0xff;
		printf("%s=%llu\n", getters[i], v);
	}
	// Any other unary getter of the device's classes that names reduction or bias.
	for (Class c = object_getClass(d); c && c != [NSObject class]; c = class_getSuperclass(c)) {
		unsigned int n = 0;
		Method *ms = class_copyMethodList(c, &n);
		for (unsigned int i = 0; i < n; i++) {
			const char *name = sel_getName(method_getName(ms[i]));
			char lower[256];
			size_t k = 0;
			for (; name[k] && k < sizeof lower - 1; k++)
				lower[k] = (char)tolower((unsigned char)name[k]);
			lower[k] = 0;
			bool known = false;
			for (size_t g = 0; g < sizeof getters / sizeof getters[0]; g++)
				known |= strcmp(name, getters[g]) == 0;
			if (known || method_getNumberOfArguments(ms[i]) != 2 ||
			    (!strstr(lower, "reduction") && !strstr(lower, "bias")))
				continue;
			printf("# also %s.%s\n", class_getName(c), name);
		}
		free(ms);
	}
}

int main(int argc, char **argv)
{
	@autoreleasepool {
		if (argc < 2) {
			fprintf(stderr, "usage: oracle_sampler DEVICE\n");
			return 2;
		}
		id<MTLDevice> d = oracle_device(argv[1]);
		oracle_header("oracle_sampler", d);
		print_answers(d);
		id<MTLLibrary> lib = oracle_library(d, argv[0], "oracle");
		if (!lib)
			return 1;
		NSError *err = nil;
		id<MTLComputePipelineState> cps =
		    [d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"sampler_probe"]
		                                     error:&err];
		MTLRenderPipelineDescriptor *rpd = [MTLRenderPipelineDescriptor new];
		rpd.vertexFunction = [lib newFunctionWithName:@"sampler_vs"];
		rpd.fragmentFunction = [lib newFunctionWithName:@"sampler_fs"];
		rpd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
		id<MTLRenderPipelineState> rps = [d newRenderPipelineStateWithDescriptor:rpd error:&err];
		if (!cps || !rps) {
			printf("  ! pipeline: %s\n", err.description.UTF8String);
			return 1;
		}
		id<MTLCommandQueue> q = [d newCommandQueue];
		id<MTLTexture> mips = make_texture(d, 256, 9, ^(NSUInteger l, NSUInteger i, uint16_t px[4]) {
		  (void)i;
		  px[0] = px[1] = px[2] = half_of((float)l);
		  px[3] = half_of(1.0f);
		});
		static const float r0[4] = {0.1f, 0.9f, 0.3f, 0.7f};
		static const float g0[4] = {0.8f, 0.2f, 0.6f, 0.4f};
		id<MTLTexture> red = make_texture(d, 2, 2, ^(NSUInteger l, NSUInteger i, uint16_t px[4]) {
		  px[0] = half_of(l == 0 ? r0[i] : 0.05f);
		  px[1] = half_of(l == 0 ? g0[i] : 0.95f);
		  px[2] = half_of(0.0f);
		  px[3] = half_of(1.0f);
		});

		const float biases[] = {0.0f, 1.0f, -0.5f, 2.5f};
		const float lods[] = {0.0f, 2.0f, 3.25f};
		for (int mip = 1; mip >= 0; mip--)
			for (size_t b = 0; b < 4; b++) {
				id<MTLSamplerState> s =
				    make_sampler(d, biases[b], MTLSamplerReductionModeWeightedAverage,
				                 mip ? MTLSamplerMipFilterLinear : MTLSamplerMipFilterNearest);
				float params[6 * 4], out[6 * 4];
				for (size_t n = 0; n < 6; n++) {
					params[n * 4 + 0] = (float)(n / 3);
					params[n * 4 + 1] = lods[n % 3];
					params[n * 4 + 2] = 0.5f;
					params[n * 4 + 3] = 0.5f;
				}
				run_compute(d, q, cps, mips, s, params, 6, out);
				printf("bias %+.2f mip %-7s level", biases[b], mip ? "linear" : "nearest");
				for (size_t i = 0; i < 3; i++)
					printf(" %.2f->%.4f", lods[i], out[i * 4]);
				printf("  gradient");
				for (size_t i = 0; i < 3; i++)
					printf(" %.2f->%.4f", lods[i], out[(3 + i) * 4]);
				printf("  fragment");
				for (size_t i = 0; i < 3; i++)
					printf(" %.2f->%.4f", lods[i], run_fragment(d, q, rps, mips, s, lods[i]));
				printf("\n");
			}

		const char *modes[] = {"weighted", "min", "max"};
		const char *ats[] = {"centre@0", "centre@0.5", "texel00@0", "x0.375@0"};
		const float params[] = {
		    0, 0.0f, 0.5f, 0.5f, 0, 0.5f, 0.5f, 0.5f, 0, 0.0f, 0.25f, 0.25f, 0, 0.0f, 0.375f, 0.5f,
		};
		for (int mode = 0; mode < 3; mode++)
			for (int mip = 1; mip >= 0; mip--) {
				fflush(stdout);
				id<MTLSamplerState> s =
				    make_sampler(d, 0.0f, (MTLSamplerReductionMode)mode,
				                 mip ? MTLSamplerMipFilterLinear : MTLSamplerMipFilterNearest);
				float out[16];
				run_compute(d, q, cps, red, s, params, 4, out);
				printf("reduce %-8s mip %-7s", modes[mode], mip ? "linear" : "nearest");
				for (size_t i = 0; i < 4; i++)
					printf("  %s %.4f,%.4f", ats[i], out[i * 4], out[i * 4 + 1]);
				printf("\n");
			}
	}
	return 0;
}
