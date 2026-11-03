// oracle_storage: when Managed and Shared resources' CPU and GPU views meet.
//   oracle_storage DEVICE [CASE...]   every case (or the ones named), each in a child process of
//                                     its own, so a Metal abort ends only that case
// Cases:
//   mbuf SIZE     a Managed buffer of SIZE bytes (4096, 67108864, ...): CPU writes with and without
//                 didModifyRange:, GPU writes (blit fills) with and without synchronizeResource:,
//                 a partial didModifyRange: after a GPU write, a didModifyRange: made after the
//                 command buffer that reads the range was made but before its commit
//   mtex          a 256 x 256 RGBA8 Managed texture: replaceRegion:, GPU writes with and without
//                 synchronizeTexture:, a replaceRegion: of a corner after an unsynchronized GPU
//                 write
//   shtex         newTextureWithDescriptor: of storage Shared (2D BGRA8), used as a render target
//                 and read by the CPU (getBytes:)
//   lintex MODE   a linear BGRA8 texture over a Shared (MODE shared) or Managed (managed) buffer:
//                 CPU writes to the buffer seen by the GPU's reads of the texture, and a render
//                 pass's clear of the texture seen in the buffer, and replaceRegion: of the texture
//                 (whole, then a corner after a CPU write to the buffer) seen by the GPU
//   argenc MODE   an argument encoder's setBuffer: and setTexture: into a Shared (MODE shared) or
//                 Managed (managed) argument buffer: the GPU's copy of the bytes against the CPU's,
//                 with and without the app's didModifyRange:
//   nocopy        a no-copy Managed buffer over the app's pages and a linear texture over it: CPU
//                 writes to the pages with and without didModifyRange:, seen by the GPU or not
//   didmodify MODE  a linear texture (the SPI form) over a Managed buffer, of newBufferWithLength:
//                 (MODE length) or no-copy (nocopy): CPU writes seen by the GPU with no call and
//                 after the texture's didModifyData (SPI), as CoreAnimation calls it
//   linalign      the offsets and row pitches a texture over a Shared or Managed buffer takes, by
//                 the public form, the linear SPI and the tiled SPI: made or nil for each
//   spin          a Shared buffer the CPU writes while one GPU thread polls it (atomic and plain
//                 loads), and one the GPU writes step by step while the CPU polls it
// A buffer's bytes print as runs, "A[0,4096)" or "B[0,256) A[256,4096)", each letter a byte.
// Build: see README.md.
#import "oracle_common.h"
#import <objc/message.h>
#include <mach/mach.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

static id<MTLDevice> D;
// The spin case's bounds: polls before the GPU gives up, and LCG steps between progress stores
// (each run stays near a second on the 5300M; a kernel of ~6 s or more is ended by the GPU's
// watchdog on these Macs). ORACLE_SPIN_POLLS and ORACLE_PROGRESS_SPIN override them.
static uint32_t gSpinPolls = 10000000, gProgressSpin = 50000;
static id<MTLCommandQueue> Q;
static const char *gArgv0;

// The bytes as runs of one value: "A[0,4096)" (letters for A-Z, hex otherwise), at most 6 runs.
static void runs(const uint8_t *p, size_t n, char *out, size_t cap)
{
	size_t o = 0, start = 0, nr = 0;
	out[0] = 0;
	for (size_t i = 1; i <= n; i++) {
		if (i < n && p[i] == p[start])
			continue;
		if (nr < 6) {
			uint8_t b = p[start];
			if (b >= 'A' && b <= 'Z')
				o += snprintf(out + o, cap - o, "%s%c[%zu,%zu)", nr ? " " : "", b, start, i);
			else
				o += snprintf(out + o, cap - o, "%s%02x[%zu,%zu)", nr ? " " : "", b, start, i);
		}
		nr++;
		start = i;
	}
	if (nr > 6)
		snprintf(out + o, cap - o, " ... %zu runs", nr);
}

static void line(const char *step, const char *result, double ms)
{
	printf("  %-58s | %-44s | %7.2f ms\n", step, result, ms);
}

// The GPU's view of `b`'s first `n` bytes: a blit into a Shared buffer.
static void gpu_view(id<MTLBuffer> b, size_t n, char *out, size_t cap)
{
	id<MTLBuffer> rb = [D newBufferWithLength:n options:MTLResourceStorageModeShared];
	id<MTLCommandBuffer> cb = [Q commandBuffer];
	id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
	[e copyFromBuffer:b sourceOffset:0 toBuffer:rb destinationOffset:0 size:n];
	[e endEncoding];
	oracle_commit_wait(cb, NULL, NULL);
	runs(rb.contents, n, out, cap);
}

static void gpu_fill(id<MTLBuffer> b, NSRange r, uint8_t v, BOOL sync)
{
	id<MTLCommandBuffer> cb = [Q commandBuffer];
	id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
	[e fillBuffer:b range:r value:v];
	if (sync)
		[e synchronizeResource:b];
	[e endEncoding];
	oracle_commit_wait(cb, NULL, NULL);
}

static void sync_buffer(id<MTLBuffer> b)
{
	id<MTLCommandBuffer> cb = [Q commandBuffer];
	id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
	[e synchronizeResource:b];
	[e endEncoding];
	oracle_commit_wait(cb, NULL, NULL);
}

static void case_mbuf(size_t n)
{
	char g[200], c[200];
	printf("case mbuf %zu: a Managed buffer\n", n);
	id<MTLBuffer> b = [D newBufferWithLength:n options:MTLResourceStorageModeManaged];
	uint8_t *p = b.contents;
	double t0;

	memset(p, 'A', n);
	[b didModifyRange:NSMakeRange(0, n)];
	gpu_view(b, n, g, sizeof g);
	line("CPU A, didModifyRange all; GPU sees", g, 0);

	memset(p, 'B', n);
	gpu_view(b, n, g, sizeof g);
	line("CPU B, no didModifyRange; GPU sees", g, 0);
	gpu_fill([D newBufferWithLength:256 options:MTLResourceStorageModeShared],
	         NSMakeRange(0, 256), 1, NO);
	gpu_view(b, n, g, sizeof g);
	line("  after another command buffer; GPU sees", g, 0);
	usleep(200000);
	gpu_view(b, n, g, sizeof g);
	line("  200 ms later; GPU sees", g, 0);
	[b didModifyRange:NSMakeRange(0, 256)];
	gpu_view(b, n, g, sizeof g);
	line("  didModifyRange [0,256) only; GPU sees", g, 0);
	[b didModifyRange:NSMakeRange(0, n)];
	gpu_view(b, n, g, sizeof g);
	line("  didModifyRange all; GPU sees", g, 0);

	memset(p, 'C', n);
	[b didModifyRange:NSMakeRange(0, n)];
	gpu_fill(b, NSMakeRange(0, n), 'D', NO);
	runs(p, n, c, sizeof c);
	line("GPU fill D, no synchronizeResource; CPU sees", c, 0);
	usleep(200000);
	runs(p, n, c, sizeof c);
	line("  200 ms later; CPU sees", c, 0);
	t0 = oracle_now_ms();
	sync_buffer(b);
	runs(p, n, c, sizeof c);
	line("  synchronizeResource in a later buffer; CPU sees", c, oracle_now_ms() - t0);
	gpu_fill(b, NSMakeRange(0, n), 'E', YES);
	runs(p, n, c, sizeof c);
	line("GPU fill E + synchronizeResource; CPU sees", c, 0);

	gpu_fill(b, NSMakeRange(0, n), 'F', NO);
	memset(p, 'G', 256);
	[b didModifyRange:NSMakeRange(0, 256)];
	gpu_view(b, n, g, sizeof g);
	line("GPU fill F (no sync), CPU G [0,256) + didModify it; GPU", g, 0);
	sync_buffer(b);
	runs(p, n, c, sizeof c);
	line("  then synchronizeResource; CPU sees", c, 0);

	memset(p, 'H', n);
	[b didModifyRange:NSMakeRange(0, n)];
	id<MTLBuffer> rb = [D newBufferWithLength:n options:MTLResourceStorageModeShared];
	id<MTLCommandBuffer> cb = [Q commandBuffer];
	id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
	[e copyFromBuffer:b sourceOffset:0 toBuffer:rb destinationOffset:0 size:n];
	[e endEncoding];
	memset(p, 'I', n);
	[b didModifyRange:NSMakeRange(0, n)];
	oracle_commit_wait(cb, NULL, NULL);
	runs(rb.contents, n, g, sizeof g);
	line("CPU I + didModify after the reading buffer was encoded", g, 0);

	cb = [Q commandBuffer];
	e = [cb blitCommandEncoder];
	[e copyFromBuffer:b sourceOffset:0 toBuffer:rb destinationOffset:0 size:n];
	[e endEncoding];
	[cb commit];
	memset(p, 'J', n);
	[b didModifyRange:NSMakeRange(0, n)];
	[cb waitUntilCompleted];
	runs(rb.contents, n, g, sizeof g);
	line("CPU J + didModify just after the reading buffer's commit", g, 0);

	// Cost: a didModifyRange of all of it, then a read, against a read with nothing modified.
	double tm[3];
	for (int k = 0; k < 3; k++) {
		memset(p, 'K' + k, n);
		[b didModifyRange:NSMakeRange(0, n)];
		t0 = oracle_now_ms();
		gpu_view(b, n, g, sizeof g);
		tm[k] = oracle_now_ms() - t0;
	}
	line("cost: didModify all, then a GPU read (3 runs)", g, (tm[0] + tm[1] + tm[2]) / 3);
	for (int k = 0; k < 3; k++) {
		t0 = oracle_now_ms();
		gpu_view(b, n, g, sizeof g);
		tm[k] = oracle_now_ms() - t0;
	}
	line("cost: a GPU read, nothing modified (3 runs)", g, (tm[0] + tm[1] + tm[2]) / 3);
}

static void tex_view(id<MTLTexture> t, int w, int h, char *out, size_t cap)
{
	id<MTLBuffer> rb = [D newBufferWithLength:(size_t)w * h * 4 options:MTLResourceStorageModeShared];
	id<MTLCommandBuffer> cb = [Q commandBuffer];
	id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
	[e copyFromTexture:t
	         sourceSlice:0
	         sourceLevel:0
	        sourceOrigin:MTLOriginMake(0, 0, 0)
	          sourceSize:MTLSizeMake(w, h, 1)
	            toBuffer:rb
	   destinationOffset:0
	destinationBytesPerRow:(NSUInteger)w * 4
	destinationBytesPerImage:(NSUInteger)w * h * 4];
	[e endEncoding];
	oracle_commit_wait(cb, NULL, NULL);
	runs(rb.contents, (size_t)w * h * 4, out, cap);
}

static void tex_cpu(id<MTLTexture> t, int w, int h, char *out, size_t cap)
{
	NSMutableData *d = [NSMutableData dataWithLength:(size_t)w * h * 4];
	[t getBytes:d.mutableBytes
	    bytesPerRow:(NSUInteger)w * 4
	     fromRegion:MTLRegionMake2D(0, 0, w, h)
	    mipmapLevel:0];
	runs(d.bytes, d.length, out, cap);
}

static void tex_gpu_write(id<MTLTexture> t, int w, int h, uint8_t v, BOOL sync)
{
	id<MTLBuffer> src = [D newBufferWithLength:(size_t)w * h * 4 options:MTLResourceStorageModeShared];
	memset(src.contents, v, src.length);
	id<MTLCommandBuffer> cb = [Q commandBuffer];
	id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
	[e copyFromBuffer:src
	           sourceOffset:0
	      sourceBytesPerRow:(NSUInteger)w * 4
	    sourceBytesPerImage:(NSUInteger)w * h * 4
	             sourceSize:MTLSizeMake(w, h, 1)
	              toTexture:t
	       destinationSlice:0
	       destinationLevel:0
	      destinationOrigin:MTLOriginMake(0, 0, 0)];
	if (sync)
		[e synchronizeTexture:t slice:0 level:0];
	[e endEncoding];
	oracle_commit_wait(cb, NULL, NULL);
}

static void tex_replace(id<MTLTexture> t, int x, int y, int w, int h, uint8_t v)
{
	NSMutableData *d = [NSMutableData dataWithLength:(size_t)w * h * 4];
	memset(d.mutableBytes, v, d.length);
	[t replaceRegion:MTLRegionMake2D(x, y, w, h)
	     mipmapLevel:0
	       withBytes:d.bytes
	     bytesPerRow:(NSUInteger)w * 4];
}

static void case_mtex(void)
{
	const int w = 256, h = 256;
	char g[200], c[200];
	printf("case mtex: a 256x256 RGBA8 Managed texture (rows of 1024 bytes)\n");
	MTLTextureDescriptor *td =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
	                                                       width:w
	                                                      height:h
	                                                   mipmapped:NO];
	td.storageMode = MTLStorageModeManaged;
	td.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
	id<MTLTexture> t = [D newTextureWithDescriptor:td];
	tex_replace(t, 0, 0, w, h, 'A');
	tex_view(t, w, h, g, sizeof g);
	line("replaceRegion all A; GPU sees", g, 0);
	tex_gpu_write(t, w, h, 'B', NO);
	tex_cpu(t, w, h, c, sizeof c);
	line("GPU write B, no synchronizeTexture; getBytes", c, 0);
	usleep(200000);
	tex_cpu(t, w, h, c, sizeof c);
	line("  200 ms later; getBytes", c, 0);
	tex_gpu_write(t, w, h, 'C', YES);
	tex_cpu(t, w, h, c, sizeof c);
	line("GPU write C + synchronizeTexture; getBytes", c, 0);
	tex_gpu_write(t, w, h, 'D', NO);
	tex_replace(t, 0, 0, 16, 1, 'E');
	tex_view(t, w, h, g, sizeof g);
	line("GPU D (no sync), replaceRegion 16x1 at 0,0 E; GPU sees", g, 0);
	tex_cpu(t, w, h, c, sizeof c);
	line("  getBytes", c, 0);
}

static id<MTLRenderPipelineState> gSolid;

static void clear_tex(id<MTLTexture> t, uint8_t v)
{
	double f = v / 255.0;
	MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
	rp.colorAttachments[0].texture = t;
	rp.colorAttachments[0].loadAction = MTLLoadActionClear;
	rp.colorAttachments[0].storeAction = MTLStoreActionStore;
	rp.colorAttachments[0].clearColor = MTLClearColorMake(f, f, f, f);
	id<MTLCommandBuffer> cb = [Q commandBuffer];
	[[cb renderCommandEncoderWithDescriptor:rp] endEncoding];
	oracle_commit_wait(cb, NULL, NULL);
}

static void case_shtex(void)
{
	const int w = 256, h = 256;
	char g[200], c[200];
	printf("case shtex: newTextureWithDescriptor: of storage Shared, 256x256 BGRA8\n");
	MTLTextureDescriptor *td =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:w
	                                                      height:h
	                                                   mipmapped:NO];
	td.storageMode = MTLStorageModeShared;
	td.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
	id<MTLTexture> t = [D newTextureWithDescriptor:td];
	line("made", t ? "texture" : "nil", 0);
	if (!t)
		return;
	printf("  storageMode %lu\n", (unsigned long)t.storageMode);
	tex_replace(t, 0, 0, w, h, 'A');
	tex_view(t, w, h, g, sizeof g);
	line("replaceRegion A; GPU sees", g, 0);
	clear_tex(t, 'B');
	tex_cpu(t, w, h, c, sizeof c);
	line("render clear B; getBytes", c, 0);
	tex_view(t, w, h, g, sizeof g);
	line("  GPU sees", g, 0);
	usleep(200000);
	tex_cpu(t, w, h, c, sizeof c);
	line("  200 ms later; getBytes", c, 0);
	tex_replace(t, 0, 0, 16, 1, 'C');
	tex_cpu(t, w, h, c, sizeof c);
	line("  replaceRegion 16x1 C; getBytes", c, 0);
	tex_view(t, w, h, g, sizeof g);
	line("  GPU sees", g, 0);
	tex_gpu_write(t, w, h, 'D', NO);
	tex_cpu(t, w, h, c, sizeof c);
	line("blit write D; getBytes", c, 0);
}

static void case_lintex(const char *mode)
{
	const int w = 256, h = 256;
	char g[200], c[200];
	MTLResourceOptions opt = !strcmp(mode, "managed") ? MTLResourceStorageModeManaged
	                                                   : MTLResourceStorageModeShared;
	printf("case lintex %s: a linear 256x256 BGRA8 texture over a %s buffer\n", mode, mode);
	size_t align = [D minimumLinearTextureAlignmentForPixelFormat:MTLPixelFormatBGRA8Unorm];
	size_t bpr = ((size_t)w * 4 + align - 1) / align * align;
	printf("  minimumLinearTextureAlignment(BGRA8) %zu, bytesPerRow %zu\n", align, bpr);
	id<MTLBuffer> b = [D newBufferWithLength:bpr * h options:opt];
	MTLTextureDescriptor *td =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:w
	                                                      height:h
	                                                   mipmapped:NO];
	td.storageMode = b.storageMode;
	td.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
	id<MTLTexture> t = [b newTextureWithDescriptor:td offset:0 bytesPerRow:bpr];
	line("made", t ? "texture" : "nil", 0);
	if (!t)
		return;
	memset(b.contents, 'A', b.length);
	if (opt == MTLResourceStorageModeManaged)
		[b didModifyRange:NSMakeRange(0, b.length)];
	tex_view(t, w, h, g, sizeof g);
	line(opt == MTLResourceStorageModeManaged ? "CPU A + didModify; GPU sees texture"
	                                          : "CPU A to the buffer; GPU sees texture",
	     g, 0);
	memset(b.contents, 'B', b.length);
	tex_view(t, w, h, g, sizeof g);
	line("CPU B to the buffer, nothing else; GPU sees texture", g, 0);
	clear_tex(t, 'C');
	runs(b.contents, b.length, c, sizeof c);
	line("render clear C of the texture; CPU sees buffer", c, 0);
	if (opt == MTLResourceStorageModeManaged) {
		sync_buffer(b);
		runs(b.contents, b.length, c, sizeof c);
		line("  synchronizeResource(buffer); CPU sees", c, 0);
	}
	memset(b.contents, 'D', b.length);
	if (opt == MTLResourceStorageModeManaged)
		[b didModifyRange:NSMakeRange(0, b.length)];
	tex_view(t, w, h, g, sizeof g);
	line("CPU D (+didModify if Managed); GPU sees texture", g, 0);
	NSMutableData *e = [NSMutableData dataWithLength:(size_t)w * 4 * h];
	memset(e.mutableBytes, 'E', e.length);
	[t replaceRegion:MTLRegionMake2D(0, 0, w, h)
	     mipmapLevel:0
	       withBytes:e.bytes
	     bytesPerRow:(NSUInteger)w * 4];
	tex_view(t, w, h, g, sizeof g);
	line("replaceRegion E of the texture, nothing else; GPU sees texture", g, 0);
	runs(b.contents, b.length, c, sizeof c);
	line("  CPU sees buffer", c, 0);
	memset(b.contents, 'G', b.length);
	memset(e.mutableBytes, 'F', e.length);
	[t replaceRegion:MTLRegionMake2D(0, 0, 16, 1)
	     mipmapLevel:0
	       withBytes:e.bytes
	     bytesPerRow:(NSUInteger)w * 4];
	tex_view(t, w, h, g, sizeof g);
	line("CPU G to the buffer, then replaceRegion F of a 16x1 corner; GPU sees", g, 0);
}

// The GPU's copy of `b`'s first `n` bytes against the CPU's: "same as the CPU's" or the GPU's runs.
static void gpu_against_cpu(id<MTLBuffer> b, size_t n, char *out, size_t cap)
{
	id<MTLBuffer> rb = [D newBufferWithLength:n options:MTLResourceStorageModeShared];
	id<MTLCommandBuffer> cb = [Q commandBuffer];
	id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
	[e copyFromBuffer:b sourceOffset:0 toBuffer:rb destinationOffset:0 size:n];
	[e endEncoding];
	oracle_commit_wait(cb, NULL, NULL);
	if (!memcmp(rb.contents, b.contents, n)) {
		snprintf(out, cap, "same as the CPU's");
		return;
	}
	char r[160];
	runs(rb.contents, n, r, sizeof r);
	snprintf(out, cap, "differs: %s", r);
}

static void case_argenc(const char *mode)
{
	char g[200];
	MTLResourceOptions opt = !strcmp(mode, "managed") ? MTLResourceStorageModeManaged
	                                                   : MTLResourceStorageModeShared;
	printf("case argenc %s: an argument encoder writing a %s argument buffer\n", mode, mode);
	MTLArgumentDescriptor *p = [MTLArgumentDescriptor argumentDescriptor];
	p.dataType = MTLDataTypePointer;
	p.index = 0;
	MTLArgumentDescriptor *t = [MTLArgumentDescriptor argumentDescriptor];
	t.dataType = MTLDataTypeTexture;
	t.textureType = MTLTextureType2D;
	t.index = 1;
	id<MTLArgumentEncoder> enc = [D newArgumentEncoderWithArguments:@[ p, t ]];
	if (!enc) {
		line("argument encoder", "nil", 0);
		return;
	}
	const size_t n = (enc.encodedLength + 255) / 256 * 256;
	id<MTLBuffer> ab = [D newBufferWithLength:n options:opt];
	memset(ab.contents, 0, n);
	if (opt == MTLResourceStorageModeManaged)
		[ab didModifyRange:NSMakeRange(0, n)];
	gpu_against_cpu(ab, n, g, sizeof g);
	line("zeroed (+didModify if Managed); GPU's bytes", g, 0);
	id<MTLBuffer> x = [D newBufferWithLength:4096 options:MTLResourceStorageModeShared];
	MTLTextureDescriptor *td =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:16
	                                                      height:16
	                                                   mipmapped:NO];
	td.storageMode = MTLStorageModePrivate;
	id<MTLTexture> tex = [D newTextureWithDescriptor:td];
	[enc setArgumentBuffer:ab offset:0];
	[enc setBuffer:x offset:0 atIndex:0];
	[enc setTexture:tex atIndex:1];
	gpu_against_cpu(ab, n, g, sizeof g);
	line("setBuffer: + setTexture:, nothing else; GPU's bytes", g, 0);
	if (opt == MTLResourceStorageModeManaged) {
		[ab didModifyRange:NSMakeRange(0, n)];
		gpu_against_cpu(ab, n, g, sizeof g);
		line("  then didModifyRange: of the buffer; GPU's bytes", g, 0);
	}
}

// A no-copy Managed buffer over the app's pages, and a linear texture over it as WindowServer makes
// one (BGRA8 468 x 36 at offset 0x100, rows 0x780 apart, in 0x11000 bytes): CPU writes to the
// app's memory with no call, seen by the GPU's reads of the buffer and of the texture or not.
static void case_nocopy(void)
{
	char g[200];
	const size_t n = 0x11000, offset = 0x100, bpr = 0x780;
	const int w = 468, h = 36;
	printf("case nocopy: a no-copy Managed buffer of %zu bytes and a linear texture over it\n", n);
	vm_address_t at = 0;
	if (vm_allocate(mach_task_self(), &at, n, VM_FLAGS_ANYWHERE) != KERN_SUCCESS) {
		line("vm_allocate", "failed", 0);
		return;
	}
	uint8_t *p = (uint8_t *)at;
	memset(p, 'A', n);
	id<MTLBuffer> b = [D newBufferWithBytesNoCopy:p
	                                       length:n
	                                      options:MTLResourceStorageModeManaged
	                                  deallocator:nil];
	line("made", !b ? "nil" : b.contents == p ? "buffer, contents the app's pages" : "buffer, contents elsewhere", 0);
	if (!b)
		return;
	printf("  storageMode %lu\n", (unsigned long)b.storageMode);
	MTLTextureDescriptor *td =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:w
	                                                      height:h
	                                                   mipmapped:NO];
	td.storageMode = b.storageMode;
	td.usage = MTLTextureUsageShaderRead;
	id<MTLTexture> t = [b newTextureWithDescriptor:td offset:offset bytesPerRow:bpr];
	line("linear texture", t ? "made" : "nil", 0);
	gpu_view(b, n, g, sizeof g);
	line("CPU A before the make; GPU sees buffer", g, 0);
	if (t) {
		tex_view(t, w, h, g, sizeof g);
		line("  GPU sees texture", g, 0);
	}
	memset(p, 'B', n);
	gpu_view(b, n, g, sizeof g);
	line("CPU B to the pages, no didModifyRange; GPU sees buffer", g, 0);
	if (t) {
		tex_view(t, w, h, g, sizeof g);
		line("  GPU sees texture", g, 0);
	}
	usleep(200000);
	gpu_view(b, n, g, sizeof g);
	line("  200 ms later; GPU sees buffer", g, 0);
	memset(p, 'C', n);
	[b didModifyRange:NSMakeRange(0, n)];
	gpu_view(b, n, g, sizeof g);
	line("CPU C + didModifyRange all; GPU sees buffer", g, 0);
	if (t) {
		tex_view(t, w, h, g, sizeof g);
		line("  GPU sees texture", g, 0);
	}
}

// didModifyData (SPI) of a texture over a buffer, as CoreAnimation calls it on an image texture it
// reuses once the CPU redrew the image's bytes: a linear texture (newLinearTextureWithDescriptor:
// offset:bytesPerRow:bytesPerImage:, SPI) of 468 x 36 BGRA8 at offset 0x100, rows 0x800 apart
// (the AMD's 256-byte alignment), over a Managed buffer of 0x13000 bytes, made by
// newBufferWithLength: (MODE length) or over the app's pages (nocopy); CPU writes seen by the GPU's
// reads of the texture with no call, and after didModifyData.
static void case_didmodify(const char *mode)
{
	char g[200];
	const size_t n = 0x13000, offset = 0x100, bpr = 0x800;
	const int w = 468, h = 36;
	printf("case didmodify %s: a linear texture over a Managed buffer, and didModifyData\n", mode);
	id<MTLBuffer> b;
	if (!strcmp(mode, "nocopy")) {
		vm_address_t at = 0;
		if (vm_allocate(mach_task_self(), &at, n, VM_FLAGS_ANYWHERE) != KERN_SUCCESS)
			return;
		b = [D newBufferWithBytesNoCopy:(void *)at
		                         length:n
		                        options:MTLResourceStorageModeManaged
		                    deallocator:nil];
	} else {
		b = [D newBufferWithLength:n options:MTLResourceStorageModeManaged];
	}
	SEL lin = NSSelectorFromString(@"newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:");
	if (!b || ![(id)b respondsToSelector:lin]) {
		line("buffer, or its linear texture SPI", "none", 0);
		return;
	}
	MTLTextureDescriptor *td =
	    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
	                                                       width:w
	                                                      height:h
	                                                   mipmapped:NO];
	td.storageMode = MTLStorageModeManaged;
	td.usage = MTLTextureUsageShaderRead;
	id<MTLTexture> t = ((id(*)(id, SEL, id, NSUInteger, NSUInteger, NSUInteger))objc_msgSend)(
	    b, lin, td, offset, bpr, bpr * h);
	line("linear texture", t ? "made" : "nil", 0);
	if (!t)
		return;
	SEL dmd = NSSelectorFromString(@"didModifyData");
	printf("  texture storage %lu, answers didModifyData: %s\n", (unsigned long)t.storageMode,
	       [(id)t respondsToSelector:dmd] ? "yes" : "no");
	uint8_t *p = b.contents;
	memset(p, 'A', n);
	[b didModifyRange:NSMakeRange(0, n)];
	tex_view(t, w, h, g, sizeof g);
	line("CPU A + didModifyRange; GPU sees texture", g, 0);
	memset(p, 'B', n);
	tex_view(t, w, h, g, sizeof g);
	line("CPU B, no call; GPU sees texture", g, 0);
	if ([(id)t respondsToSelector:dmd]) {
		((void (*)(id, SEL))objc_msgSend)(t, dmd);
		tex_view(t, w, h, g, sizeof g);
		line("  then didModifyData; GPU sees texture", g, 0);
		gpu_view(b, n, g, sizeof g);
		line("  and the buffer", g, 0);
	}
}

// Which offsets and row pitches a texture over a buffer takes: 468 x 36 BGRA8 over a buffer of
// 0x12000 bytes, by the public form (newTextureWithDescriptor:offset:bytesPerRow:), the linear
// SPI (newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:) and the tiled SPI
// (newTiledTextureWithDescriptor:offset:bytesPerRow:, a Managed descriptor over a Shared buffer),
// over Shared and Managed buffers: "made" or "nil" for each. Where Metal's validation asserts
// instead (the Intel's), the case ends there.
static void case_linalign(void)
{
	printf("case linalign: the offsets and pitches a texture over a buffer takes\n");
	const int w = 468, h = 36;
	const NSUInteger offsets[] = {0, 0x40, 0x100}, pitches[] = {1872, 1920, 2048};
	SEL lin = NSSelectorFromString(@"newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:");
	SEL tiled = NSSelectorFromString(@"newTiledTextureWithDescriptor:offset:bytesPerRow:");
	for (int form = 0; form < 3; form++) {
		for (int managed = 0; managed < 2; managed++) {
			if (form == 2 && managed)
				continue; // a tiled texture's buffer is Shared
			id<MTLBuffer> b = [D newBufferWithLength:0x12000
			                                 options:managed ? MTLResourceStorageModeManaged
			                                                 : MTLResourceStorageModeShared];
			MTLTextureDescriptor *td =
			    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
			                                                       width:w
			                                                      height:h
			                                                   mipmapped:NO];
			td.storageMode = form == 2 || managed ? MTLStorageModeManaged : MTLStorageModeShared;
			td.usage = MTLTextureUsageShaderRead;
			char out[200];
			size_t o = 0;
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					id t = nil;
					if (form == 0)
						t = [b newTextureWithDescriptor:td offset:offsets[i] bytesPerRow:pitches[j]];
					else if ([(id)b respondsToSelector:form == 1 ? lin : tiled])
						t = form == 1 ? ((id(*)(id, SEL, id, NSUInteger, NSUInteger, NSUInteger))
						                     objc_msgSend)(b, lin, td, offsets[i], pitches[j],
						                                   pitches[j] * h)
						              : ((id(*)(id, SEL, id, NSUInteger, NSUInteger))objc_msgSend)(
						                    b, tiled, td, offsets[i], pitches[j]);
					o += snprintf(out + o, sizeof out - o, "%s@%lx/%lu:%s", o ? " " : "",
					              (unsigned long)offsets[i], (unsigned long)pitches[j],
					              t ? "made" : "nil");
				}
			}
			char step[80];
			snprintf(step, sizeof step, "%s over %s",
			         form == 0 ? "public" : form == 1 ? "linear SPI" : "tiled SPI",
			         managed ? "Managed" : "Shared");
			printf("  %-22s %s\n", step, out);
		}
	}
}

static id<MTLComputePipelineState> pipeline(id<MTLLibrary> lib, const char *name)
{
	NSError *err = nil;
	id<MTLFunction> f = [lib newFunctionWithName:@(name)];
	id<MTLComputePipelineState> p = [D newComputePipelineStateWithFunction:f error:&err];
	if (!p)
		printf("  ! pipeline %s: %s\n", name, err.description.UTF8String);
	return p;
}

static void case_spin(void)
{
	printf("case spin: a Shared buffer written by one side while the other runs\n");
	id<MTLLibrary> lib = oracle_library(D, gArgv0, "oracle");
	if (!lib)
		return;
	const char *names[2] = {"spin_atomic", "spin_plain"};
	for (int k = 0; k < 2; k++) {
		id<MTLComputePipelineState> p = pipeline(lib, names[k]);
		if (!p)
			continue;
		id<MTLBuffer> flag = [D newBufferWithLength:64 options:MTLResourceStorageModeShared];
		id<MTLBuffer> res = [D newBufferWithLength:64 options:MTLResourceStorageModeShared];
		memset(flag.contents, 0, 64);
		memset(res.contents, 0xff, 64);
		uint32_t max = gSpinPolls;
		id<MTLCommandBuffer> cb = [Q commandBuffer];
		id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
		[e setComputePipelineState:p];
		[e setBuffer:flag offset:0 atIndex:0];
		[e setBuffer:res offset:0 atIndex:1];
		[e setBytes:&max length:4 atIndex:2];
		[e dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
		[e endEncoding];
		double t0 = oracle_now_ms();
		[cb commit];
		[cb waitUntilScheduled];
		usleep(20000);
		((volatile uint32_t *)flag.contents)[0] = 7;
		double tw = oracle_now_ms() - t0;
		[cb waitUntilCompleted];
		double tc = oracle_now_ms() - t0;
		uint32_t *r = res.contents;
		char out[200];
		snprintf(out, sizeof out, "polls %u of %u, saw %u", r[0], max, r[1]);
		line(k ? "GPU polls (plain), CPU writes 20 ms after scheduled"
		       : "GPU polls (atomic), CPU writes 20 ms after scheduled",
		     out, tc);
		printf("  CPU wrote at %.1f ms, the buffer completed at %.1f ms, GPU time %.1f ms, status %ld%s%s\n",
		       tw, tc, (cb.GPUEndTime - cb.GPUStartTime) * 1e3, (long)cb.status,
		       cb.error ? " error " : "", cb.error ? cb.error.description.UTF8String : "");
	}
	id<MTLComputePipelineState> p = pipeline(lib, "progress");
	if (!p)
		return;
	id<MTLBuffer> ctr = [D newBufferWithLength:64 options:MTLResourceStorageModeShared];
	id<MTLBuffer> sink = [D newBufferWithLength:64 options:MTLResourceStorageModeShared];
	memset(ctr.contents, 0, 64);
	uint32_t steps = 200, spin = gProgressSpin;
	id<MTLCommandBuffer> cb = [Q commandBuffer];
	id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
	[e setComputePipelineState:p];
	[e setBuffer:ctr offset:0 atIndex:0];
	[e setBytes:&steps length:4 atIndex:1];
	[e setBytes:&spin length:4 atIndex:2];
	[e setBuffer:sink offset:0 atIndex:3];
	[e dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
	[e endEncoding];
	double t0 = oracle_now_ms();
	[cb commit];
	char seen[400];
	size_t o = 0;
	uint32_t last = 0;
	int changes = 0, polls = 0;
	while (cb.status < MTLCommandBufferStatusCompleted && oracle_now_ms() - t0 < 20000) {
		uint32_t v = ((volatile uint32_t *)ctr.contents)[0];
		polls++;
		if (v != last) {
			changes++;
			if (o < sizeof seen - 32)
				o += snprintf(seen + o, sizeof seen - o, "%u@%.0fms ", v, oracle_now_ms() - t0);
			last = v;
		}
		usleep(1000);
	}
	[cb waitUntilCompleted];
	char out[200];
	snprintf(out, sizeof out, "%d changes seen in %d polls before completion", changes, polls);
	line("GPU stores 1..200 while the CPU polls every ms", out, oracle_now_ms() - t0);
	printf("  values seen: %s\n  final %u, GPU time %.1f ms, status %ld%s%s\n", changes ? seen : "(none)",
	       ((uint32_t *)ctr.contents)[0], (cb.GPUEndTime - cb.GPUStartTime) * 1e3, (long)cb.status,
	       cb.error ? " error " : "", cb.error ? cb.error.description.UTF8String : "");
}

static int run_case(int argc, char **argv)
{
	const char *c = argv[0];
	if (!strcmp(c, "mbuf"))
		case_mbuf(argc > 1 ? strtoull(argv[1], NULL, 0) : 4096);
	else if (!strcmp(c, "mtex"))
		case_mtex();
	else if (!strcmp(c, "shtex"))
		case_shtex();
	else if (!strcmp(c, "lintex"))
		case_lintex(argc > 1 ? argv[1] : "shared");
	else if (!strcmp(c, "linalign"))
		case_linalign();
	else if (!strcmp(c, "didmodify"))
		case_didmodify(argc > 1 ? argv[1] : "length");
	else if (!strcmp(c, "nocopy"))
		case_nocopy();
	else if (!strcmp(c, "argenc"))
		case_argenc(argc > 1 ? argv[1] : "shared");
	else if (!strcmp(c, "spin"))
		case_spin();
	else {
		printf("unknown case %s\n", c);
		return 2;
	}
	return 0;
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IOLBF, 0);
	gArgv0 = argv[0];
	if (argc < 2) {
		fprintf(stderr, "usage: %s DEVICE [CASE [ARG]]\n", argv[0]);
		return 2;
	}
	if (getenv("ORACLE_SPIN_POLLS"))
		gSpinPolls = (uint32_t)strtoul(getenv("ORACLE_SPIN_POLLS"), NULL, 0);
	if (getenv("ORACLE_PROGRESS_SPIN"))
		gProgressSpin = (uint32_t)strtoul(getenv("ORACLE_PROGRESS_SPIN"), NULL, 0);
	if (argc > 2 && !strcmp(argv[2], "--case")) {
		D = oracle_device(argv[1]);
		Q = [D newCommandQueue];
		return run_case(argc - 3, argv + 3);
	}
	D = oracle_device(argv[1]);
	oracle_header("oracle_storage", D);
	D = nil;
	static const char *const all[][2] = {
	    {"mbuf", "4096"}, {"mbuf", "67108864"}, {"mtex", NULL}, {"shtex", NULL},
	    {"lintex", "shared"}, {"lintex", "managed"}, {"argenc", "shared"},
	    {"argenc", "managed"}, {"nocopy", NULL}, {"didmodify", "length"},
	    {"didmodify", "nocopy"}, {"linalign", NULL}, {"spin", NULL},
	};
	size_t n = sizeof all / sizeof all[0];
	for (size_t i = 0; i < n; i++) {
		if (argc > 2 && strcmp(argv[2], all[i][0]))
			continue;
		const char *arg = argc > 3 ? argv[3] : all[i][1];
		char *args[] = {argv[0], argv[1], "--case", (char *)all[i][0], (char *)arg, NULL};
		pid_t pid;
		if (posix_spawn(&pid, argv[0], NULL, NULL, args, environ))
			return 1;
		int st;
		waitpid(pid, &st, 0);
		if (WIFSIGNALED(st))
			printf("  => case %s died on signal %d\n", all[i][0], WTERMSIG(st));
		else if (WEXITSTATUS(st))
			printf("  => case %s exited %d\n", all[i][0], WEXITSTATUS(st));
		if (argc > 3)
			break;
	}
	return 0;
}
