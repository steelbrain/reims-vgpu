// oracle_pastlen: what a texture over a buffer does with texels past the buffer's length: which
// such textures each creator makes (`limits`), and, for one it makes, what each CPU and GPU path
// reads and writes there (`paths`). No Metal validation layer is asked for: what a guest's process
// meets.
//   oracle_pastlen DEVICE paths                    every shape and path, each in a child process
//   oracle_pastlen DEVICE path SHAPE PATH          one shape and one path (what the children run)
//   oracle_pastlen DEVICE limits [KIND]            the creators' bounds, by buffer kind and length
//
// paths. Each shape is a texture of R32Uint over a buffer, whose texels run past its length:
//   2d.s4096.in    Shared 4096 bytes, 2D 64 x 16 from 0, rows 256 apart: all inside, the control
//   2d.s4096       Shared 4096 bytes, 2D 64 x 17 from 0: its last row past the length
//   2d.s4096.o2048 Shared 4096 bytes, 2D 64 x 16 from 2048: its last 8 rows past it
//   2d.s20000      Shared 20000 bytes, 2D 64 x 84 from 0: row 78 across the length, 79 to 83 past
//                  it (inside an allocation of 16 KiB pages, 79 inside one of 4 KiB pages)
//   2d.m4096, 2d.m20000   the same over a Managed buffer
//   tb.s4096       Shared 4096 bytes, a texture buffer of 128 texels from 3840: its last 64 past
//   tb.s20000      Shared 20000 bytes, a texture buffer of 64 texels from 19968: its last 56 past
//   tb.m4096       the same over a Managed buffer
//   2d.nc          a no-copy Shared buffer over one page whose next page the process maps
//                  PROT_NONE, 2D 64 x (page / 256 + 1) from 0: its last row in that page
//   tb.nc          the same, a texture buffer of 128 texels from the page's last 256 bytes
//   2d.nc4k, tb.nc4k   the same over a no-copy buffer of the page's last 4096 bytes (on a page of
//                  16 KiB; where the page is 4 KiB, as 2d.nc and tb.nc)
// Paths, one a process, as a write past the buffer may reach memory that is not the buffer's,
// which a later path would read: made (the texture alone), read (a kernel's read()), blit
// (copyFromTexture:toBuffer:), getbytes (getBytes:), and the writes kwrite (a kernel's write()),
// bwrite (copyFromBuffer:toTexture:) and replace (replaceRegion:), each of the rows (2D) or texels
// (texture buffer) with a byte past the length, after which the process reads the texture back by
// read, blit and getbytes, and, over a Managed buffer, by getbytes again after a blit's
// synchronizeResource: of the buffer.
// Every word of the target buffer is 0xA0000000 | its offset / 4, and every word of each of the
// 128 neighbour buffers made around it (4096 bytes each, of its storage) 0xB0000000 | k << 16 |
// offset / 4, so a texel names the byte it came from: the buffer's own where the texture lays it
// (`own`), another of the target's (`tgt@O`), neighbour k's (`nbK@O`, with that neighbour's
// gpuAddress and contents against the target's), zero, or other (its word). A written texel is
// 0xC0000000 | tag << 13 | row << 6 | column (2D) or 0xC0000000 | tag << 13 | index (texture
// buffer), tag the process's id's low 15 bits: `written` where the tag is the process's own,
// `stale(T)` where another process wrote it. Lines:
//   SHAPE PATH: buffer L bytes, allocatedSize A, page P; nearest neighbours by gpuAddress at
//     target-X and target+Y, by contents at target-X' and target+Y'
//   texture ... made | nil
//   PATH-OR-READBACK: row R (bytes B..E[, past the length]): RUN xN RUN xN ...
//   target words changed: N of M / neighbour K (...) words changed: N, from offset O
// A process that a path ends prints nothing past it; the parent's line says how it ended:
//   => SHAPE PATH died on signal S | exited X
//
// limits. For each buffer kind and length, a few buffers made one after another, and for each
// buffer each creator's bound, by bisection: the largest end (offset of the texture's last byte
// plus one, from the buffer's start) a texture is made at, and the first it answers nil at:
//   tb      a texture buffer of R8Uint from 0, W texels, bytesPerRow W: end W
//   tb.pad  the same, bytesPerRow W + 4096: end W (its last row's bytes), bytesPerRow past it
//   tb.off  the same from the buffer's length rounded up to 256: end offset + W
//   2d      a 2D R32Uint texture 64 wide, rows 256 apart, from 0, H rows: end 256 x H
//   2d.pad  the same, rows 512 apart: end 512 x (H - 1) + 256, and its rows' end 512 x H
//   2d.padoff  the same from 256: end 512 x H, and its rows' end 512 x H + 256
//   2d.lin  the same by the linear SPI,
//           newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:
//   2da.pad a 2D array of 2 slices by the linear SPI, rows 512 apart, H rows a slice (its slices
//           512 x H apart): end 1024 x H - 256 (its last slice's last texel), and its slices' end
//           1024 x H
//   2da.padoff  the same from 256: end 1024 x H, and its slices' end 1024 x H + 256
// Kinds: shared (newBufferWithLength:, Shared), bytes (newBufferWithBytes:, Shared), managed
// (newBufferWithLength:, Managed), private (newBufferWithLength:, Private, its textures Private),
// heap (a Shared heap of 4 MiB's newBufferWithLength:, on an Apple GPU alone: the x86 GPUs take no
// Shared heap), nocopy (newBufferWithBytesNoCopy: of the length, at the start of a page of
// anonymous memory) and nocopy.mis (the same 4096 bytes short of a page's end).
//   limit KIND L #I gpu G (mod 128 KiB M) alloc A rsize R heapOffset H | FORM made to E, nil at E'
// where E is "all" where every end bisected (up to 16 MiB for a texture buffer, 4 MiB for 2D) is
// made, and E' "-" then.
// Build: see README.md.
#import "oracle_common.h"
#include <objc/message.h>
#include <spawn.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

enum { kNeighbours = 128, kNeighbourLength = 4096, kPitch = 256, kW2D = 64 };

static const char *kSource =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void rd2(texture2d<uint, access::read> t [[texture(0)]], device uint *o [[buffer(0)]],\n"
    "                uint2 g [[thread_position_in_grid]]) {\n"
    "  if (g.x < t.get_width() && g.y < t.get_height()) o[g.y * t.get_width() + g.x] = t.read(g).x;\n"
    "}\n"
    "kernel void wr2(texture2d<uint, access::write> t [[texture(0)]], constant uint2 &a [[buffer(0)]],\n"
    "                uint2 g [[thread_position_in_grid]]) {\n"
    "  if (g.x < t.get_width() && g.y < t.get_height() && g.y >= a.x)\n"
    "    t.write(uint4(0xC0000000u | (a.y << 13) | (g.y << 6) | g.x, 0, 0, 0), g);\n"
    "}\n"
    "kernel void rdb(texture_buffer<uint, access::read> t [[texture(0)]], device uint *o [[buffer(0)]],\n"
    "                uint i [[thread_position_in_grid]]) {\n"
    "  if (i < t.get_width()) o[i] = t.read(i).x;\n"
    "}\n"
    "kernel void wrb(texture_buffer<uint, access::write> t [[texture(0)]], constant uint2 &a [[buffer(0)]],\n"
    "                uint i [[thread_position_in_grid]]) {\n"
    "  if (i < t.get_width() && i >= a.x) t.write(uint4(0xC0000000u | (a.y << 13) | i, 0, 0, 0), i);\n"
    "}\n";

struct Shape {
	const char *name;
	MTLStorageMode storage;
	NSUInteger length; // a no-copy buffer: 0 for a page, else its bytes at the page's end
	bool nocopy;
	bool tb;
	NSUInteger offset; // for a no-copy texture buffer: from the page's end
	NSUInteger width, height;
};

static const struct Shape kShapes[] = {
    {"2d.s4096.in", MTLStorageModeShared, 4096, false, false, 0, kW2D, 16},
    {"2d.s4096", MTLStorageModeShared, 4096, false, false, 0, kW2D, 17},
    {"2d.s4096.o2048", MTLStorageModeShared, 4096, false, false, 2048, kW2D, 16},
    {"2d.s20000", MTLStorageModeShared, 20000, false, false, 0, kW2D, 84},
    {"2d.m4096", MTLStorageModeManaged, 4096, false, false, 0, kW2D, 17},
    {"2d.m20000", MTLStorageModeManaged, 20000, false, false, 0, kW2D, 84},
    {"tb.s4096", MTLStorageModeShared, 4096, false, true, 3840, 128, 1},
    {"tb.s20000", MTLStorageModeShared, 20000, false, true, 19968, 64, 1},
    {"tb.m4096", MTLStorageModeManaged, 4096, false, true, 3840, 128, 1},
    {"2d.nc", MTLStorageModeShared, 0, true, false, 0, kW2D, 0},
    {"tb.nc", MTLStorageModeShared, 0, true, true, 256, 128, 1},
    {"2d.nc4k", MTLStorageModeShared, 4096, true, false, 0, kW2D, 0},
    {"tb.nc4k", MTLStorageModeShared, 4096, true, true, 256, 128, 1},
};
static const int kShapeCount = sizeof kShapes / sizeof kShapes[0];
static const char *kPaths[] = {"made", "read", "blit", "getbytes", "kwrite", "bwrite", "replace"};
static const int kPathCount = sizeof kPaths / sizeof kPaths[0];

static id<MTLDevice> dev;
static id<MTLCommandQueue> queue;
static id<MTLLibrary> lib;
static uint32_t tag;

static uint32_t targetWord(NSUInteger o) { return 0xA0000000u | (uint32_t)(o / 4); }
static uint32_t neighbourWord(unsigned k, NSUInteger o)
{
	return 0xB0000000u | (k << 16) | (uint32_t)(o / 4);
}

static void fill(id<MTLBuffer> b, uint32_t (^word)(NSUInteger))
{
	uint32_t *w = b.contents;
	for (NSUInteger o = 0; o < b.length; o += 4)
		w[o / 4] = word(o);
	if (b.storageMode == MTLStorageModeManaged)
		[b didModifyRange:NSMakeRange(0, b.length)];
}

static MTLResourceOptions optionsOf(MTLStorageMode s)
{
	return s == MTLStorageModeManaged ? MTLResourceStorageModeManaged : MTLResourceStorageModeShared;
}

static void waitFor(id<MTLCommandBuffer> cb, const char *what)
{
	[cb commit];
	[cb waitUntilCompleted];
	if (cb.status != MTLCommandBufferStatusCompleted)
		printf("  ! %s: command buffer status %ld, error %s\n", what, (long)cb.status,
		       cb.error ? cb.error.description.UTF8String : "none");
}

static void dispatchKernel(NSString *name, id<MTLTexture> t, id<MTLBuffer> b, MTLSize grid)
{
	NSError *err = nil;
	id<MTLComputePipelineState> p =
	    [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&err];
	if (!p) {
		printf("  ! pipeline %s: %s\n", name.UTF8String, err.description.UTF8String);
		exit(1);
	}
	id<MTLCommandBuffer> cb = [queue commandBuffer];
	id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
	[e setComputePipelineState:p];
	[e setTexture:t atIndex:0];
	[e setBuffer:b offset:0 atIndex:0];
	[e dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(16, 1, 1)];
	[e endEncoding];
	waitFor(cb, name.UTF8String);
}

// The words of the shape's texels, as runs: words naming consecutive bytes of one source (or all
// zero, or written in order), a line a row (2D) or a line of 64 texels (texture buffer).
static void classify(const char *what, const uint32_t *v, const struct Shape *s, NSUInteger offset,
                     NSUInteger length, NSUInteger width, NSUInteger rows,
                     NSArray<id<MTLBuffer>> *nb, id<MTLBuffer> target)
{
	printf("  %s:\n", what);
	const NSUInteger lineTexels = s->tb ? 64 : width;
	const NSUInteger lines = s->tb ? (width + 63) / 64 : rows;
	for (NSUInteger r = 0; r < lines; r++) {
		const NSUInteger first = r * lineTexels;
		const NSUInteger count = s->tb ? MIN(64, width - first) : width;
		const NSUInteger start = s->tb ? offset + first * 4 : offset + r * kPitch;
		char line[4096];
		int n = snprintf(line, sizeof line, "    %s %2lu (bytes %lu..%lu%s):", s->tb ? "texels" : "row",
		                 (unsigned long)(s->tb ? first : r), (unsigned long)start,
		                 (unsigned long)(start + count * 4),
		                 start + count * 4 > length ? ", past the length" : "");
		NSUInteger c = 0;
#define WORD(i) v[(s->tb ? first + (i) : r * width + (i))]
		while (c < count) {
			const uint32_t x = WORD(c);
			const unsigned kind = x >> 28;
			const NSUInteger c0 = c;
			char label[128];
			if (kind == 0xA) {
				const NSUInteger o = (x & 0xFFFFFF) * 4;
				const NSUInteger want = start + c * 4;
				if (o == want)
					snprintf(label, sizeof label, "own");
				else
					snprintf(label, sizeof label, "tgt@%lu", (unsigned long)o);
				c++;
				while (c < count && WORD(c) == x + (uint32_t)(c - c0))
					c++;
			} else if (kind == 0xB) {
				const unsigned k = (x >> 16) & 0xFFF;
				const NSUInteger o = (x & 0xFFFF) * 4;
				long long g = 0, m = 0;
				if (k < nb.count) {
					g = (long long)nb[k].gpuAddress - (long long)target.gpuAddress;
					m = (long long)(intptr_t)nb[k].contents - (long long)(intptr_t)target.contents;
				}
				snprintf(label, sizeof label, "nb%u@%lu (gpuAddress target%+lld, contents target%+lld)",
				         k, (unsigned long)o, g, m);
				c++;
				while (c < count && WORD(c) == x + (uint32_t)(c - c0))
					c++;
			} else if (kind == 0xC) {
				const uint32_t t = (x >> 13) & 0x7FFF;
				const char *who = t == tag ? "written" : "stale";
				char by[16] = "";
				if (t != tag)
					snprintf(by, sizeof by, "T%u ", t);
				if (s->tb)
					snprintf(label, sizeof label, "%s(%si%u)", who, by, x & 0x1FFF);
				else
					snprintf(label, sizeof label, "%s(%sr%u c%u)", who, by, (x >> 6) & 0x7F, x & 0x3F);
				c++;
				while (c < count && WORD(c) == x + (uint32_t)(c - c0))
					c++;
			} else if (x == 0) {
				snprintf(label, sizeof label, "zero");
				c++;
				while (c < count && WORD(c) == 0)
					c++;
			} else {
				snprintf(label, sizeof label, "other(%08x)", x);
				c++;
				while (c < count && WORD(c) == x)
					c++;
			}
			n += snprintf(line + n, sizeof line - n, " %s x%lu", label, (unsigned long)(c - c0));
			if (n > 3800)
				break;
		}
#undef WORD
		printf("%s\n", line);
	}
	fflush(stdout);
}

static void changed(id<MTLBuffer> target, NSArray<id<MTLBuffer>> *nb, NSUInteger length)
{
	const uint32_t *w = target.contents;
	NSUInteger t = 0;
	for (NSUInteger o = 0; o < length; o += 4)
		t += w[o / 4] != targetWord(o);
	printf("  target words changed: %lu of %lu\n", (unsigned long)t, (unsigned long)(length / 4));
	NSUInteger total = 0;
	for (unsigned k = 0; k < nb.count; k++) {
		const uint32_t *x = nb[k].contents;
		NSUInteger m = 0, first = 0;
		for (NSUInteger o = 0; o < nb[k].length; o += 4)
			if (x[o / 4] != neighbourWord(k, o)) {
				if (!m)
					first = o;
				m++;
			}
		if (m)
			printf("  neighbour %u (gpuAddress target%+lld, contents target%+lld) words changed: "
			       "%lu, from offset %lu\n",
			       k, (long long)nb[k].gpuAddress - (long long)target.gpuAddress,
			       (long long)(intptr_t)nb[k].contents - (long long)(intptr_t)target.contents,
			       (unsigned long)m, (unsigned long)first);
		total += m;
	}
	printf("  neighbour words changed: %lu\n", (unsigned long)total);
	fflush(stdout);
}

static int runPath(const struct Shape *s, const char *path)
{
	const NSUInteger page = (NSUInteger)getpagesize();
	const NSUInteger length = s->nocopy ? (s->length ? MIN(s->length, page) : page) : s->length;
	const NSUInteger offset = s->nocopy && s->tb ? length - s->offset : s->offset;
	const NSUInteger width = s->width;
	const NSUInteger rows = s->tb ? 1 : (s->nocopy ? length / kPitch + 1 : s->height);
	tag = (uint32_t)getpid() & 0x7FFF;
	NSError *err = nil;
	lib = [dev newLibraryWithSource:@(kSource) options:nil error:&err];
	if (!lib) {
		printf("  ! library: %s\n", err.description.UTF8String);
		return 1;
	}
	queue = [dev newCommandQueue];
	const MTLResourceOptions opts = optionsOf(s->storage);
	NSMutableArray<id<MTLBuffer>> *nb = [NSMutableArray array];
	id<MTLBuffer> target = nil;
	for (unsigned k = 0; k < kNeighbours / 2; k++)
		[nb addObject:[dev newBufferWithLength:kNeighbourLength options:opts]];
	if (s->nocopy) {
		uint8_t *m = mmap(NULL, 2 * page, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
		if (m == MAP_FAILED || mprotect(m + page, page, PROT_NONE) != 0) {
			printf("  ! mmap or mprotect failed\n");
			return 1;
		}
		target = [dev newBufferWithBytesNoCopy:m + page - length
		                                length:length
		                               options:opts
		                           deallocator:nil];
	} else {
		target = [dev newBufferWithLength:length options:opts];
	}
	for (unsigned k = kNeighbours / 2; k < kNeighbours; k++)
		[nb addObject:[dev newBufferWithLength:kNeighbourLength options:opts]];
	if (target == nil) {
		printf("%s %s: no buffer\n", s->name, path);
		return 0;
	}
	fill(target, ^uint32_t(NSUInteger o) { return targetWord(o); });
	for (unsigned k = 0; k < kNeighbours; k++)
		fill(nb[k], ^uint32_t(NSUInteger o) { return neighbourWord(k, o); });
	long long ga = 0, gb = 0, ca = 0, cb = 0;
	for (unsigned k = 0; k < kNeighbours; k++) {
		const long long g = (long long)nb[k].gpuAddress - (long long)target.gpuAddress;
		const long long c = (long long)(intptr_t)nb[k].contents - (long long)(intptr_t)target.contents;
		if (g > 0 && (!ga || g < ga))
			ga = g;
		if (g < 0 && (!gb || g > gb))
			gb = g;
		if (c > 0 && (!ca || c < ca))
			ca = c;
		if (c < 0 && (!cb || c > cb))
			cb = c;
	}
	printf("%s %s: tag %u, buffer %lu bytes (storage %lu), allocatedSize %lu, page %lu; nearest neighbours "
	       "by gpuAddress at target%+lld and target%+lld, by contents at target%+lld and target%+lld\n",
	       s->name, path, tag, (unsigned long)target.length, (unsigned long)target.storageMode,
	       (unsigned long)target.allocatedSize, (unsigned long)page, gb, ga, cb, ca);
	const BOOL writes = !strcmp(path, "kwrite") || !strcmp(path, "bwrite") || !strcmp(path, "replace");
	MTLTextureDescriptor *d = nil;
	if (s->tb) {
		d = [MTLTextureDescriptor textureBufferDescriptorWithPixelFormat:MTLPixelFormatR32Uint
		                                                           width:width
		                                                 resourceOptions:opts
		                                                           usage:MTLTextureUsageShaderRead];
	} else {
		d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint
		                                                       width:width
		                                                      height:rows
		                                                   mipmapped:NO];
		d.storageMode = s->storage;
	}
	d.usage = writes ? (MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite) : MTLTextureUsageShaderRead;
	const NSUInteger bpr = s->tb ? width * 4 : kPitch;
	id<MTLTexture> t = [target newTextureWithDescriptor:d offset:offset bytesPerRow:bpr];
	const NSUInteger end = s->tb ? offset + width * 4 : offset + kPitch * (rows - 1) + width * 4;
	printf("  texture %s %lu x %lu at offset %lu (ends at %lu, %ld past the length): %s",
	       s->tb ? "buffer" : "2D", (unsigned long)width, (unsigned long)rows,
	       (unsigned long)offset, (unsigned long)end, (long)end - (long)length, t ? "made" : "nil\n");
	if (!t)
		return 0;
	printf(", type %lu, allocatedSize %lu, bufferOffset %lu, bufferBytesPerRow %lu\n",
	       (unsigned long)t.textureType, (unsigned long)t.allocatedSize,
	       (unsigned long)t.bufferOffset, (unsigned long)t.bufferBytesPerRow);
	fflush(stdout);
	if (!strcmp(path, "made"))
		return 0;
	// The first row (2D) or texel (texture buffer) with a byte past the length.
	NSUInteger firstPast = 0;
	if (s->tb) {
		while (firstPast < width && offset + (firstPast + 1) * 4 <= length)
			firstPast++;
	} else {
		while (firstPast < rows && offset + firstPast * kPitch + width * 4 <= length)
			firstPast++;
	}
	const NSUInteger texels = width * rows;
	const NSUInteger bytes = texels * 4;
	id<MTLBuffer> out = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
	const MTLSize grid = s->tb ? MTLSizeMake(width, 1, 1) : MTLSizeMake(width, rows, 1);
	void (^readBack)(const char *, const char *) = ^(const char *how, const char *label) {
		memset(out.contents, 0x5A, bytes);
		if (!strcmp(how, "read")) {
			dispatchKernel(s->tb ? @"rdb" : @"rd2", t, out, grid);
			classify(label, out.contents, s, offset, length, width, rows, nb, target);
		} else if (!strcmp(how, "blit")) {
			id<MTLCommandBuffer> c = [queue commandBuffer];
			id<MTLBlitCommandEncoder> e = [c blitCommandEncoder];
			[e copyFromTexture:t
			             sourceSlice:0
			             sourceLevel:0
			            sourceOrigin:MTLOriginMake(0, 0, 0)
			              sourceSize:MTLSizeMake(width, rows, 1)
			                toBuffer:out
			       destinationOffset:0
			  destinationBytesPerRow:width * 4
			destinationBytesPerImage:bytes];
			[e endEncoding];
			waitFor(c, "blit");
			classify(label, out.contents, s, offset, length, width, rows, nb, target);
		} else if (!strcmp(how, "sync")) {
			id<MTLCommandBuffer> c = [queue commandBuffer];
			id<MTLBlitCommandEncoder> e = [c blitCommandEncoder];
			[e synchronizeResource:target];
			[e endEncoding];
			waitFor(c, "synchronize");
		} else {
			uint32_t *v = malloc(bytes);
			memset(v, 0x5A, bytes);
			[t getBytes:v
			    bytesPerRow:width * 4
			     fromRegion:MTLRegionMake2D(0, 0, width, rows)
			    mipmapLevel:0];
			classify(label, v, s, offset, length, width, rows, nb, target);
			free(v);
		}
	};
	if (!writes) {
		readBack(path, path);
		changed(target, nb, length);
		return 0;
	}
	const NSUInteger past = (s->tb ? width : rows) - firstPast;
	if (past == 0) {
		printf("  nothing past the length to write\n");
		return 0;
	}
	printf("  writing %s %lu..%lu (a byte or more past the length)\n", s->tb ? "texels" : "rows",
	       (unsigned long)firstPast, (unsigned long)(s->tb ? width : rows));
	fflush(stdout);
	const NSUInteger srcTexels = s->tb ? past : past * width;
	uint32_t *src = malloc(srcTexels * 4);
	for (NSUInteger i = 0; i < srcTexels; i++) {
		if (s->tb)
			src[i] = 0xC0000000u | tag << 13 | (uint32_t)(firstPast + i);
		else
			src[i] = 0xC0000000u | tag << 13 | (uint32_t)((firstPast + i / width) << 6) |
			         (uint32_t)(i % width);
	}
	const MTLOrigin origin = s->tb ? MTLOriginMake(firstPast, 0, 0) : MTLOriginMake(0, firstPast, 0);
	const MTLSize size = s->tb ? MTLSizeMake(past, 1, 1) : MTLSizeMake(width, past, 1);
	const NSUInteger srcRow = s->tb ? past * 4 : width * 4;
	if (!strcmp(path, "kwrite")) {
		const uint32_t f[2] = {(uint32_t)firstPast, tag};
		id<MTLBuffer> fb = [dev newBufferWithBytes:f length:sizeof f options:MTLResourceStorageModeShared];
		dispatchKernel(s->tb ? @"wrb" : @"wr2", t, fb, grid);
	} else if (!strcmp(path, "bwrite")) {
		id<MTLBuffer> sb = [dev newBufferWithBytes:src
		                                    length:srcTexels * 4
		                                   options:MTLResourceStorageModeShared];
		id<MTLCommandBuffer> c = [queue commandBuffer];
		id<MTLBlitCommandEncoder> e = [c blitCommandEncoder];
		[e copyFromBuffer:sb
		           sourceOffset:0
		      sourceBytesPerRow:srcRow
		    sourceBytesPerImage:srcTexels * 4
		             sourceSize:size
		              toTexture:t
		       destinationSlice:0
		       destinationLevel:0
		      destinationOrigin:origin];
		[e endEncoding];
		waitFor(c, "bwrite");
	} else {
		[t replaceRegion:(MTLRegion){origin, size} mipmapLevel:0 withBytes:src bytesPerRow:srcRow];
	}
	free(src);
	printf("  written\n");
	changed(target, nb, length);
	readBack("read", "read after");
	readBack("blit", "blit after");
	readBack("getbytes", "getbytes after");
	if (s->storage == MTLStorageModeManaged) {
		readBack("sync", "synchronize");
		readBack("getbytes", "getbytes after a synchronize");
	}
	changed(target, nb, length);
	return 0;
}

// --- limits ------------------------------------------------------------------------------------

static NSUInteger roundUp(NSUInteger v, NSUInteger to) { return (v + to - 1) / to * to; }

// Whether `form` makes a texture over `b` ending at `end` (see the header); `step` is the form's
// granularity of ends.
static BOOL makes(id<MTLBuffer> b, const char *form, NSUInteger end)
{
	@autoreleasepool {
		const MTLResourceOptions opts = b.resourceOptions;
		const BOOL tb = !strncmp(form, "tb", 2);
		const NSUInteger off = !strcmp(form, "tb.off") ? roundUp(b.length, 256) : 0;
		MTLTextureDescriptor *d = nil;
		NSUInteger bpr = 0;
		if (tb) {
			const NSUInteger w = end - off;
			d = [MTLTextureDescriptor textureBufferDescriptorWithPixelFormat:MTLPixelFormatR8Uint
			                                                           width:w
			                                                 resourceOptions:opts
			                                                           usage:MTLTextureUsageShaderRead];
			bpr = !strcmp(form, "tb.pad") ? w + 4096 : w;
		} else {
			const BOOL pad = !strncmp(form, "2d.pad", 6);
			const BOOL padoff = !strcmp(form, "2d.padoff");
			bpr = pad ? 2 * kPitch : kPitch;
			const NSUInteger h = padoff ? end / bpr : pad ? (end - kPitch) / bpr + 1 : end / kPitch;
			d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint
			                                                       width:kW2D
			                                                      height:h
			                                                   mipmapped:NO];
			d.storageMode = b.storageMode;
			d.usage = MTLTextureUsageShaderRead;
		}
		const BOOL arrayOff = !strcmp(form, "2da.padoff");
		if (!strncmp(form, "2da.pad", 7)) {
			d.textureType = MTLTextureType2DArray;
			d.arrayLength = 2;
			d.height = arrayOff ? end / (4 * kPitch) : (end + kPitch) / (4 * kPitch);
			bpr = 2 * kPitch;
		}
		id<MTLTexture> t = nil;
		if (!strncmp(form, "2d.lin", 6) || !strncmp(form, "2da.pad", 7)) {
			SEL sel = NSSelectorFromString(@"newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:");
			if (![(id)b respondsToSelector:sel])
				return NO;
			t = ((id (*)(id, SEL, id, NSUInteger, NSUInteger, NSUInteger))objc_msgSend)(
			    b, sel, d, arrayOff ? kPitch : off, bpr, bpr * (end / kPitch));
		} else {
			t = [b newTextureWithDescriptor:d
			                         offset:!strcmp(form, "2d.padoff") ? kPitch : off
			                    bytesPerRow:bpr];
		}
		return t != nil;
	}
}

static void limitOf(id<MTLBuffer> b, const char *form, char *line, size_t size)
{
	const BOOL tb = !strncmp(form, "tb", 2);
	const BOOL pad = !strcmp(form, "2d.pad");
	const BOOL padoff = !strcmp(form, "2d.padoff");
	const BOOL array = !strcmp(form, "2da.pad");
	const BOOL arrayOff = !strcmp(form, "2da.padoff");
	// End number n (from 1) of the form: a texture buffer's n bytes from its offset, a 2D
	// texture's n rows.
	const NSUInteger off = !strcmp(form, "tb.off") ? roundUp(b.length, 256) : 0;
	NSUInteger (^endOf)(NSUInteger) = ^NSUInteger(NSUInteger n) {
		return tb ? off + n : arrayOff ? 4 * kPitch * n : array ? 4 * kPitch * n - kPitch : padoff ? 2 * kPitch * n
		     : pad ? 2 * kPitch * (n - 1) + kPitch : kPitch * n;
	};
	NSUInteger (^rowsEndOf)(NSUInteger) = ^NSUInteger(NSUInteger n) {
		return arrayOff ? 4 * kPitch * n + kPitch : array ? 4 * kPitch * n
		     : padoff ? 2 * kPitch * n + kPitch : 2 * kPitch * n;
	};
	const NSUInteger top = tb ? ((NSUInteger)1 << 24) : 16384;
	if (!makes(b, form, endOf(1))) {
		snprintf(line, size, "%s nil at %lu", form, (unsigned long)endOf(1));
		return;
	}
	if (makes(b, form, endOf(top))) {
		snprintf(line, size, "%s made to all (%lu), nil at -", form, (unsigned long)endOf(top));
		return;
	}
	NSUInteger lo = 1, hi = top; // made at lo, nil at hi
	while (hi - lo > 1) {
		const NSUInteger mid = lo + (hi - lo) / 2;
		if (makes(b, form, endOf(mid)))
			lo = mid;
		else
			hi = mid;
	}
	if (pad || padoff || array || arrayOff)
		snprintf(line, size, "%s made to %lu (rows' end %lu), nil at %lu (rows' end %lu)", form,
		         (unsigned long)endOf(lo), (unsigned long)rowsEndOf(lo),
		         (unsigned long)endOf(hi), (unsigned long)rowsEndOf(hi));
	else
		snprintf(line, size, "%s made to %lu, nil at %lu", form, (unsigned long)endOf(lo),
		         (unsigned long)endOf(hi));
}

static NSUInteger resourceSizeOf(id<MTLBuffer> b)
{
	SEL sel = NSSelectorFromString(@"resourceSize");
	if (![(id)b respondsToSelector:sel])
		return 0;
	return ((NSUInteger (*)(id, SEL))objc_msgSend)(b, sel);
}

static int runLimits(const char *only)
{
	static const NSUInteger kLengths[] = {16,    256,   1024,  4096,  4097,   8192,   12288,
	                                      16383, 16384, 16385, 20000, 32768,  65536,  100000,
	                                      131072, 131073, 262144};
	static const char *kKinds[] = {"shared", "bytes", "managed", "private", "heap", "nocopy",
	                               "nocopy.mis"};
	static const char *kForms[] = {"tb",     "tb.pad", "tb.off", "2d",
	                               "2d.pad", "2d.padoff", "2d.lin", "2da.pad", "2da.padoff"};
	const NSUInteger page = (NSUInteger)getpagesize();
	printf("# page %lu, minimumTextureBufferAlignment R8Uint %lu, minimumLinearTextureAlignment "
	       "R32Uint %lu\n",
	       (unsigned long)page,
	       (unsigned long)[dev minimumTextureBufferAlignmentForPixelFormat:MTLPixelFormatR8Uint],
	       (unsigned long)[dev minimumLinearTextureAlignmentForPixelFormat:MTLPixelFormatR32Uint]);
	for (unsigned k = 0; k < sizeof kKinds / sizeof kKinds[0]; k++) {
		const char *kind = kKinds[k];
		if (only && strcmp(only, kind))
			continue;
		const BOOL managed = !strcmp(kind, "managed");
		const MTLResourceOptions opts = managed                     ? MTLResourceStorageModeManaged
		                                : !strcmp(kind, "private") ? MTLResourceStorageModePrivate
		                                                           : MTLResourceStorageModeShared;
		id<MTLHeap> heap = nil;
		if (!strcmp(kind, "heap")) {
			if (![dev supportsFamily:MTLGPUFamilyApple7]) {
				printf("limit heap: skipped, no Apple family (no Shared heap)\n");
				continue;
			}
			MTLHeapDescriptor *hd = [MTLHeapDescriptor new];
			hd.size = 4 << 20;
			hd.storageMode = MTLStorageModeShared;
			heap = [dev newHeapWithDescriptor:hd];
			if (!heap) {
				printf("limit heap: no heap\n");
				continue;
			}
		}
		uint8_t *arena = NULL;
		NSUInteger arenaAt = 0;
		const BOOL mis = !strcmp(kind, "nocopy.mis");
		if (!strncmp(kind, "nocopy", 6)) {
			arena = mmap(NULL, 64 << 20, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
			if (arena == MAP_FAILED) {
				printf("limit nocopy: no arena\n");
				continue;
			}
		}
		for (unsigned l = 0; l < sizeof kLengths / sizeof kLengths[0]; l++) {
			const NSUInteger len = kLengths[l];
			NSMutableArray<id<MTLBuffer>> *keep = [NSMutableArray array];
			for (unsigned i = 0; i < 4; i++) {
				id<MTLBuffer> b = nil;
				if (heap) {
					b = [heap newBufferWithLength:len options:opts];
				} else if (arena) {
					const NSUInteger lead = mis ? page - 4096 : 0;
					const NSUInteger span = roundUp(lead + len, page);
					if (arenaAt + span > (NSUInteger)(64 << 20))
						arenaAt = 0;
					b = [dev newBufferWithBytesNoCopy:arena + arenaAt + lead
					                           length:len
					                          options:opts
					                      deallocator:nil];
					arenaAt += span;
				} else if (!strcmp(kind, "bytes")) {
					void *zero = calloc(1, len);
					b = [dev newBufferWithBytes:zero length:len options:opts];
					free(zero);
				} else {
					b = [dev newBufferWithLength:len options:opts];
				}
				if (b == nil) {
					printf("limit %s %lu #%u: no buffer\n", kind, (unsigned long)len, i);
					continue;
				}
				[keep addObject:b];
				for (unsigned f = 0; f < sizeof kForms / sizeof kForms[0]; f++) {
					char what[160];
					limitOf(b, kForms[f], what, sizeof what);
					printf("limit %s %lu #%u gpu 0x%llx (mod 128 KiB %llu) alloc %lu rsize %lu "
					       "heapOffset %lu | %s\n",
					       kind, (unsigned long)b.length, i, (unsigned long long)b.gpuAddress,
					       (unsigned long long)(b.gpuAddress % 131072), (unsigned long)b.allocatedSize,
					       (unsigned long)resourceSizeOf(b), (unsigned long)(heap ? b.heapOffset : 0),
					       what);
					fflush(stdout);
				}
			}
		}
		if (arena)
			munmap(arena, 64 << 20);
	}
	return 0;
}

int main(int argc, char **argv)
{
	@autoreleasepool {
		if (argc < 3) {
			fprintf(stderr, "usage: %s DEVICE paths | path SHAPE PATH | limits [KIND]\n", argv[0]);
			return 2;
		}
		dev = oracle_device(argv[1]);
		if (!strcmp(argv[2], "path") && argc == 5) {
			for (int i = 0; i < kShapeCount; i++)
				if (!strcmp(kShapes[i].name, argv[3]))
					return runPath(&kShapes[i], argv[4]);
			return 2;
		}
		oracle_header("oracle_pastlen", dev);
		fflush(stdout);
		if (!strcmp(argv[2], "limits"))
			return runLimits(argc > 3 ? argv[3] : NULL);
		if (strcmp(argv[2], "paths"))
			return 2;
		for (int i = 0; i < kShapeCount; i++) {
			for (int p = 0; p < kPathCount; p++) {
				char *args[] = {argv[0], argv[1], "path", (char *)kShapes[i].name,
				                (char *)kPaths[p], NULL};
				pid_t pid;
				if (posix_spawn(&pid, argv[0], NULL, NULL, args, environ))
					return 1;
				int st = 0;
				waitpid(pid, &st, 0);
				if (WIFSIGNALED(st))
					printf("=> %s %s died on signal %d\n", kShapes[i].name, kPaths[p], WTERMSIG(st));
				else if (WEXITSTATUS(st))
					printf("=> %s %s exited %d\n", kShapes[i].name, kPaths[p], WEXITSTATUS(st));
				fflush(stdout);
			}
		}
	}
	return 0;
}
