// oracle_allocsize: what the device's currentAllocatedSize counts. Each step makes one object, reads
// the count, frees the object and reads it again (at once and after 50 ms), in one process:
//   oracle_allocsize DEVICE
// Each line: `STEP made=+D freed=+F late=+L size=S` where D is the count's change from before the
// make to after it, F from before the make to right after the free, L the same 50 ms later, and S
// the object's own allocatedSize (or heap size, `-` where it has none). `base` lines give the count
// itself after the device, a queue and a compute pipeline are made.
#include "oracle_common.h"
#import <IOSurface/IOSurface.h>
#include <mach/mach.h>
#include <objc/message.h>

static id<MTLDevice> dev;

static long long count(void) { return (long long)dev.currentAllocatedSize; }

static void line(const char *step, long long before, long long made, long long freed, long long late,
                 long long size)
{
	char s[32] = "-";
	if (size >= 0)
		snprintf(s, sizeof s, "%lld", size);
	printf("%-28s made=%+lld freed=%+lld late=%+lld size=%s\n", step, made - before, freed - before,
	       late - before, s);
	fflush(stdout);
}

// Makes an object with `make` (which answers its size, or -1), frees it, and prints the line.
static void step(const char *name, long long (^make)(id *keep))
{
	long long before, made, freed, late, size;
	@autoreleasepool {
		before = count();
		id keep = nil;
		size = make(&keep);
		if (keep == nil) {
			printf("%-28s not made\n", name);
			return;
		}
		made = count();
		keep = nil;
	}
	freed = count();
	usleep(50 * 1000);
	late = count();
	line(name, before, made, freed, late, size);
}

static MTLTextureDescriptor *tex2d(NSUInteger w, NSUInteger h, MTLPixelFormat f, MTLStorageMode m)
{
	MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:f
	                                                                             width:w
	                                                                            height:h
	                                                                         mipmapped:NO];
	d.storageMode = m;
	d.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
	return d;
}

int main(int argc, char **argv)
{
	@autoreleasepool {
		dev = oracle_device(argc > 1 ? argv[1] : "AMD");
		oracle_header("oracle_allocsize", dev);
		printf("%-28s count=%lld\n", "base.device", count());
		id<MTLCommandQueue> q = [dev newCommandQueue];
		printf("%-28s count=%lld\n", "base.queue", count());
		NSError *err = nil;
		id<MTLLibrary> lib = [dev newLibraryWithSource:@"kernel void k(device uint *o [[buffer(0)]]) { o[0] = 1; }"
		                                       options:nil
		                                         error:&err];
		id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k"]
		                                                                    error:&err];
		printf("%-28s count=%lld\n", "base.pipeline", count());

		const struct { const char *name; NSUInteger len; MTLResourceOptions opt; } bufs[] = {
		    {"buffer.shared.1", 1, MTLResourceStorageModeShared},
		    {"buffer.shared.4096", 4096, MTLResourceStorageModeShared},
		    {"buffer.shared.4097", 4097, MTLResourceStorageModeShared},
		    {"buffer.shared.1m", 1 << 20, MTLResourceStorageModeShared},
		    {"buffer.shared.16m", 16 << 20, MTLResourceStorageModeShared},
		    {"buffer.managed.1m", 1 << 20, MTLResourceStorageModeManaged},
		    {"buffer.private.1", 1, MTLResourceStorageModePrivate},
		    {"buffer.private.1m", 1 << 20, MTLResourceStorageModePrivate},
		    {"buffer.private.16m", 16 << 20, MTLResourceStorageModePrivate},
		};
		for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; i++) {
			const NSUInteger len = bufs[i].len;
			const MTLResourceOptions opt = bufs[i].opt;
			step(bufs[i].name, ^long long(id *keep) {
				id<MTLBuffer> b = [dev newBufferWithLength:len options:opt];
				*keep = b;
				return (long long)b.allocatedSize;
			});
		}
		step("buffer.bytes.shared.1m", ^long long(id *keep) {
			void *p = calloc(1, 1 << 20);
			id<MTLBuffer> b = [dev newBufferWithBytes:p length:1 << 20 options:MTLResourceStorageModeShared];
			free(p);
			*keep = b;
			return (long long)b.allocatedSize;
		});
		// No-copy buffers over pages the probe maps; the pages stay mapped until the buffer goes.
		const struct { const char *name; NSUInteger len; MTLResourceOptions opt; } ncs[] = {
		    {"nocopy.shared.page", 4096, MTLResourceStorageModeShared},
		    {"nocopy.shared.1m", 1 << 20, MTLResourceStorageModeShared},
		    {"nocopy.managed.1m", 1 << 20, MTLResourceStorageModeManaged},
		};
		for (size_t i = 0; i < sizeof ncs / sizeof ncs[0]; i++) {
			const NSUInteger len = ncs[i].len;
			const MTLResourceOptions opt = ncs[i].opt;
			step(ncs[i].name, ^long long(id *keep) {
				vm_address_t a = 0;
				if (vm_allocate(mach_task_self(), &a, len, VM_FLAGS_ANYWHERE) != KERN_SUCCESS)
					return -1;
				id<MTLBuffer> b = [dev newBufferWithBytesNoCopy:(void *)a
				                                         length:len
				                                        options:opt
				                                    deallocator:^(void *p, NSUInteger l) {
					                                    vm_deallocate(mach_task_self(), (vm_address_t)p, l);
				                                    }];
				*keep = b;
				return (long long)b.allocatedSize;
			});
		}
		const struct { const char *name; NSUInteger w; MTLPixelFormat f; MTLStorageMode m; } texs[] = {
		    {"texture.private.256", 256, MTLPixelFormatRGBA8Unorm, MTLStorageModePrivate},
		    {"texture.managed.256", 256, MTLPixelFormatRGBA8Unorm, MTLStorageModeManaged},
		    {"texture.private.1024", 1024, MTLPixelFormatRGBA8Unorm, MTLStorageModePrivate},
		    {"texture.private.depth32.1024", 1024, MTLPixelFormatDepth32Float, MTLStorageModePrivate},
		    {"texture.private.r8.100", 100, MTLPixelFormatR8Unorm, MTLStorageModePrivate},
		};
		for (size_t i = 0; i < sizeof texs / sizeof texs[0]; i++) {
			MTLTextureDescriptor *d = tex2d(texs[i].w, texs[i].w, texs[i].f, texs[i].m);
			step(texs[i].name, ^long long(id *keep) {
				id<MTLTexture> t = [dev newTextureWithDescriptor:d];
				*keep = t;
				return (long long)t.allocatedSize;
			});
		}
		step("texture.private.1024.mips", ^long long(id *keep) {
			MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
			                                                                             width:1024
			                                                                            height:1024
			                                                                         mipmapped:YES];
			d.storageMode = MTLStorageModePrivate;
			id<MTLTexture> t = [dev newTextureWithDescriptor:d];
			*keep = t;
			return (long long)t.allocatedSize;
		});
		step("texture.private.msaa4.512", ^long long(id *keep) {
			MTLTextureDescriptor *d = tex2d(512, 512, MTLPixelFormatBGRA8Unorm, MTLStorageModePrivate);
			d.textureType = MTLTextureType2DMultisample;
			d.sampleCount = 4;
			d.usage = MTLTextureUsageRenderTarget;
			id<MTLTexture> t = [dev newTextureWithDescriptor:d];
			*keep = t;
			return (long long)t.allocatedSize;
		});
		// Objects over another's memory: made while the other lives, the step counts the view alone.
		id<MTLTexture> base = [dev newTextureWithDescriptor:tex2d(256, 256, MTLPixelFormatRGBA8Unorm,
		                                                          MTLStorageModePrivate)];
		step("texture.view", ^long long(id *keep) {
			id<MTLTexture> v = [base newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm_sRGB];
			*keep = v;
			return (long long)v.allocatedSize;
		});
		id<MTLBuffer> backing = [dev newBufferWithLength:1 << 20 options:MTLResourceStorageModeManaged];
		step("texture.over.buffer", ^long long(id *keep) {
			MTLTextureDescriptor *d = tex2d(256, 256, MTLPixelFormatRGBA8Unorm, MTLStorageModeManaged);
			d.usage = MTLTextureUsageShaderRead;
			id<MTLTexture> t = [backing newTextureWithDescriptor:d offset:0 bytesPerRow:1024];
			*keep = t;
			return (long long)t.allocatedSize;
		});
		step("texture.iosurface.256", ^long long(id *keep) {
			IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef) @{
				(id)kIOSurfaceWidth : @256,
				(id)kIOSurfaceHeight : @256,
				(id)kIOSurfaceBytesPerElement : @4,
				(id)kIOSurfacePixelFormat : @((uint32_t)'BGRA'),
			});
			MTLTextureDescriptor *d = tex2d(256, 256, MTLPixelFormatBGRA8Unorm, MTLStorageModeManaged);
			id<MTLTexture> t = [dev newTextureWithDescriptor:d iosurface:s plane:0];
			CFRelease(s);
			*keep = t;
			return (long long)t.allocatedSize;
		});
		// A buffer over an IOSurface (the device SPI newBufferWithIOSurface:).
		step("buffer.iosurface.256k", ^long long(id *keep) {
			if (![(id)dev respondsToSelector:sel_registerName("newBufferWithIOSurface:")])
				return -1;
			IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef) @{
				(id)kIOSurfaceWidth : @65536,
				(id)kIOSurfaceHeight : @1,
				(id)kIOSurfaceBytesPerElement : @4,
			});
			id<MTLBuffer> b = ((id(*)(id, SEL, IOSurfaceRef))objc_msgSend)((id)dev, sel_registerName("newBufferWithIOSurface:"), s);
			CFRelease(s);
			*keep = b;
			return b ? (long long)b.allocatedSize : -1;
		});
		// Heaps (Private: an x86-64 Mac's Metal refuses a Shared one at an assertion): the heap
		// alone, then a resource made in a heap that lives across the step.
		const struct { const char *name; MTLStorageMode m; MTLHeapType t; } heaps[] = {
		    {"heap.private.16m", MTLStorageModePrivate, MTLHeapTypeAutomatic},
		    {"heap.private.placement.16m", MTLStorageModePrivate, MTLHeapTypePlacement},
		};
		for (size_t i = 0; i < sizeof heaps / sizeof heaps[0]; i++) {
			MTLHeapDescriptor *hd = [MTLHeapDescriptor new];
			hd.size = 16 << 20;
			hd.storageMode = heaps[i].m;
			hd.type = heaps[i].t;
			step(heaps[i].name, ^long long(id *keep) {
				id<MTLHeap> h = [dev newHeapWithDescriptor:hd];
				*keep = h;
				return h ? (long long)h.size : -1;
			});
		}
		MTLHeapDescriptor *hd = [MTLHeapDescriptor new];
		hd.size = 16 << 20;
		hd.storageMode = MTLStorageModePrivate;
		id<MTLHeap> heap = [dev newHeapWithDescriptor:hd];
		printf("%-28s heap.currentAllocatedSize=%llu usedSize=%llu\n", "heap.held", (unsigned long long)heap.currentAllocatedSize,
		       (unsigned long long)heap.usedSize);
		step("heap.buffer.1m", ^long long(id *keep) {
			id<MTLBuffer> b = [heap newBufferWithLength:1 << 20 options:MTLResourceStorageModePrivate];
			*keep = b;
			return (long long)b.allocatedSize;
		});
		step("heap.texture.256", ^long long(id *keep) {
			id<MTLTexture> t = [heap newTextureWithDescriptor:tex2d(256, 256, MTLPixelFormatRGBA8Unorm,
			                                                        MTLStorageModePrivate)];
			*keep = t;
			return (long long)t.allocatedSize;
		});
		// An indirect command buffer, an argument encoder's buffer, states and events.
		step("icb.draw.1024", ^long long(id *keep) {
			MTLIndirectCommandBufferDescriptor *d = [MTLIndirectCommandBufferDescriptor new];
			d.commandTypes = MTLIndirectCommandTypeDraw;
			d.maxVertexBufferBindCount = 4;
			d.maxFragmentBufferBindCount = 4;
			id<MTLIndirectCommandBuffer> icb = [dev newIndirectCommandBufferWithDescriptor:d
			                                                               maxCommandCount:1024
			                                                                       options:0];
			*keep = icb;
			return icb ? (long long)icb.allocatedSize : -1;
		});
		step("icb.dispatch.64", ^long long(id *keep) {
			MTLIndirectCommandBufferDescriptor *d = [MTLIndirectCommandBufferDescriptor new];
			d.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
			d.maxKernelBufferBindCount = 4;
			id<MTLIndirectCommandBuffer> icb = [dev newIndirectCommandBufferWithDescriptor:d
			                                                               maxCommandCount:64
			                                                                       options:0];
			*keep = icb;
			return icb ? (long long)icb.allocatedSize : -1;
		});
		// A visible function table of a compute pipeline that links a visible function.
		step("vft.compute.16", ^long long(id *keep) {
			if (!dev.supportsFunctionPointers)
				return -1;
			NSError *e = nil;
			id<MTLLibrary> l = [dev newLibraryWithSource:
			                            @"#include <metal_stdlib>\nusing namespace metal;\n"
			                            @"[[visible]] uint f(uint x) { return x + 1; }\n"
			                            @"kernel void kv(device uint *o [[buffer(0)]], visible_function_table<uint(uint)> t [[buffer(1)]]) { o[0] = t[0](o[0]); }"
			                                     options:nil
			                                       error:&e];
			MTLComputePipelineDescriptor *cd = [MTLComputePipelineDescriptor new];
			cd.computeFunction = [l newFunctionWithName:@"kv"];
			MTLLinkedFunctions *lf = [MTLLinkedFunctions linkedFunctions];
			lf.functions = @[ [l newFunctionWithName:@"f"] ];
			cd.linkedFunctions = lf;
			id<MTLComputePipelineState> cps = [dev newComputePipelineStateWithDescriptor:cd
			                                                                     options:0
			                                                                  reflection:nil
			                                                                       error:&e];
			if (cps == nil) {
				printf("  ! vft pipeline: %s\n", e ? e.description.UTF8String : "nil");
				return -1;
			}
			MTLVisibleFunctionTableDescriptor *vd = [MTLVisibleFunctionTableDescriptor new];
			vd.functionCount = 16;
			long long b0 = count();
			id<MTLVisibleFunctionTable> vft = [cps newVisibleFunctionTableWithDescriptor:vd];
			printf("  vft alone: %+lld\n", count() - b0);
			*keep = vft;
			return vft ? (long long)vft.allocatedSize : -1;
		});
		step("sampler", ^long long(id *keep) {
			*keep = [dev newSamplerStateWithDescriptor:[MTLSamplerDescriptor new]];
			return -1;
		});
		step("depthstencil", ^long long(id *keep) {
			*keep = [dev newDepthStencilStateWithDescriptor:[MTLDepthStencilDescriptor new]];
			return -1;
		});
		step("sharedevent", ^long long(id *keep) {
			*keep = [dev newSharedEvent];
			return -1;
		});
		step("fence", ^long long(id *keep) {
			*keep = [dev newFence];
			return -1;
		});
		step("queue", ^long long(id *keep) {
			*keep = [dev newCommandQueue];
			return -1;
		});
		step("pipeline.compute", ^long long(id *keep) {
			NSError *e = nil;
			*keep = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k"] error:&e];
			return -1;
		});
		// A command buffer that ran a dispatch: what a submission leaves counted once it completes.
		step("commandbuffer.dispatch", ^long long(id *keep) {
			id<MTLBuffer> o = [dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
			id<MTLCommandBuffer> cb = [q commandBuffer];
			id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
			[e setComputePipelineState:ps];
			[e setBuffer:o offset:0 atIndex:0];
			[e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
			[e endEncoding];
			oracle_commit_wait(cb, NULL, NULL);
			*keep = cb;
			return -1;
		});
		// Many buffers at once: whether the count is the sum of their allocatedSizes.
		step("buffers.shared.64x64k", ^long long(id *keep) {
			NSMutableArray *a = [NSMutableArray array];
			long long sum = 0;
			for (int i = 0; i < 64; i++) {
				id<MTLBuffer> b = [dev newBufferWithLength:65536 options:MTLResourceStorageModeShared];
				sum += (long long)b.allocatedSize;
				[a addObject:b];
			}
			*keep = a;
			return sum;
		});
		printf("%-28s count=%lld\n", "end", count());
		(void)base;
		(void)backing;
	}
	return 0;
}
