// What Metal answers for GPU and CPU writes into read-only memory: memory of Metal's private
// option bit 24 (a no-copy buffer over the process's pages, or a heap made with it), and pages the
// process maps read-only. dgpu-oracle.md Q38 holds what it found.
//
// One shape a process, named on the command line, with its backing:
//   oracle_readonly DEVICE <bit24|plain> <shape>
// DEVICE is a substring of the device's name (`AMD`, `Intel`, `Apple`). `bit24` makes the buffer
// (or the heap a shape's target lies in) with bit 24, `plain` without it: the writable control. A
// `ro-pages-` prefix makes the buffer over the pages after the process maps them read-only. Each
// stage prints a line and flushes before the call it names, so the last `stage=` line of a process
// that ended is the call that ended it; an alarm ends one stuck past 30 s.
// `oracle_readonly.sh DEVICE` runs every shape, both backings, RUNS runs each (default 2).
//
// The shapes: a render pass over a texture over the buffer (`store`; `custom-color` and
// `custom-color-late`, kept by `CustomSampleDepthStore`, the descriptor's or set late), a 4x
// attachment of Metal's own resolved into it (`resolve-into`, `smsr-into`, each `-late` for the
// action set late; `late-resolve-texture-known` and `-late`, the texture named by the setter's SPI
// under a known or a late action; `untouched-into` and `untouched-smsr`, a format pair Metal leaves
// untouched); a depth or stencil attachment in a heap made with the bit (`depth-store`,
// `depth-dontcare`, `stencil-store`, `stencil-dontcare`, `custom-depth`, `custom-depth-late`,
// `custom-stencil`, `custom-stencil-late`) or resolved into one (`depth-resolve`, `depth-smsr`,
// `depth-resolve-late`, `depth-late-texture`, `stencil-resolve`, `stencil-smsr`,
// `stencil-resolve-late`); a draw whose fragment or vertex function `write()`s a texture over the
// buffer (`frag-write`, `vert-write`) or that makes it resident for writing (`resident-write`); a
// kernel's `write()` (`kernel-write`); a blit's copy, colour fill and bytes fill into it
// (`blit-copy-into`, `fill-color`, `fill-bytes`, the last two the `fillTexture:` SPI); mipmaps and
// an invalidation of a heap's texture (`mipmaps`, `mipmaps-shared`, `invalidate`); the CPU's
// `replaceRegion:` of a texture over the buffer or in a Shared heap (`replace`, `replace-heap`);
// and an indirect command buffer made with the bit (`icb-cpu`, a command the CPU encodes in a
// Shared one; `icb-gpu`, one a kernel encodes in a Private one; `icb-gpu-over-cpu`, a kernel's
// command over one the CPU encoded in a Shared one), its command executed.
//
// After the shape's command buffer (cb1): its status, error and event signal; the pages' bytes
// that changed (a target over the buffer), or the target's first texels as a blit reads them on a
// new queue (a target in a heap); then a fill and a signal on the same queue at once, on a new
// queue, and on the same queue 100 ms on; then the same shape again on the same queue (cb2), and a
// fill and a signal after it on that queue and on a new one, which answer whether a second fault
// revokes the queue. CPU shapes (`replace`...) report the pages, the texels the GPU reads, and the
// same later command buffers.
//
// Build (universal, on a Mac with the command line tools; build.sh builds it with the others):
//   clang -arch arm64 -arch x86_64 -mmacosx-version-min=14.0 -fobjc-arc -framework Metal \
//     -framework Foundation -o oracle_readonly oracle_readonly.m

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/message.h>
#import <sys/mman.h>
#import <unistd.h>

static const NSUInteger kBit24 = 1ull << 24;
static const size_t kRegion = 64 * 1024;
static const uint8_t kPad = 0xAB;
static const NSUInteger kOffset = 0x1000;
static const NSUInteger kSide = 4;
static const NSUInteger kPitch = 256;

static id<MTLDevice> gDevice;
static id<MTLLibrary> gLibrary;
static BOOL gBit24;
static uint8_t *gPages;
static id<MTLBuffer> gBuffer;
static NSMutableArray *gKeep;
// The target a shape made in a heap, its level read back after the command buffer.
static id<MTLTexture> gHeapTarget;
static NSUInteger gHeapLevel;

static void say(const char *key, NSString *value) {
  printf("%s=%s\n", key, value.UTF8String);
  fflush(stdout);
}

static void stage(const char *name) {
  printf("stage=%s\n", name);
  fflush(stdout);
}

static NSString *statusName(MTLCommandBufferStatus s) {
  switch (s) {
    case MTLCommandBufferStatusNotEnqueued: return @"NotEnqueued";
    case MTLCommandBufferStatusEnqueued: return @"Enqueued";
    case MTLCommandBufferStatusCommitted: return @"Committed";
    case MTLCommandBufferStatusScheduled: return @"Scheduled";
    case MTLCommandBufferStatusCompleted: return @"Completed";
    case MTLCommandBufferStatusError: return @"Error";
  }
  return @"?";
}

static void sayBuffer(const char *prefix, id<MTLCommandBuffer> cb) {
  char key[128];
  snprintf(key, sizeof key, "%s.status", prefix);
  say(key, statusName(cb.status));
  snprintf(key, sizeof key, "%s.error", prefix);
  say(key, cb.error == nil ? @"none"
                           : [NSString stringWithFormat:@"%@ code %ld: %@", cb.error.domain,
                                                        (long)cb.error.code,
                                                        cb.error.localizedDescription]);
}

static NSUInteger pagesMoved(void) {
  NSUInteger n = 0;
  for (size_t i = 0; i < kRegion; i++) n += gPages[i] != kPad;
  return n;
}

static NSString *const kSource =
    @"#include <metal_stdlib>\n"
    @"using namespace metal;\n"
    @"struct VOut { float4 position [[position]]; };\n"
    @"constant float2 kTri[3] = { float2(-1, -3), float2(-1, 1), float2(3, 1) };\n"
    @"vertex VOut v_main(uint vid [[vertex_id]]) {\n"
    @"  VOut o; o.position = float4(kTri[vid], 0.5, 1); return o;\n"
    @"}\n"
    @"vertex VOut v_write(uint vid [[vertex_id]],\n"
    @"                    texture2d<float, access::write> t [[texture(0)]]) {\n"
    @"  for (uint y = 0; y < 4; y++) for (uint x = 0; x < 4; x++)\n"
    @"    t.write(float4(0.25, 0.5, 0.75, 1), uint2(x, y));\n"
    @"  VOut o; o.position = float4(kTri[vid], 0.5, 1); return o;\n"
    @"}\n"
    @"fragment float4 f_main() { return float4(0.25, 0.5, 0.75, 1); }\n"
    @"fragment float4 f_write(VOut in [[stage_in]],\n"
    @"                        texture2d<float, access::write> t [[texture(0)]]) {\n"
    @"  t.write(float4(0.25, 0.5, 0.75, 1), uint2(in.position.xy));\n"
    @"  return float4(0.25, 0.5, 0.75, 1);\n"
    @"}\n"
    @"kernel void k_write(texture2d<float, access::write> t [[texture(0)]],\n"
    @"                    uint2 at [[thread_position_in_grid]]) {\n"
    @"  t.write(float4(0.25, 0.5, 0.75, 1), at);\n"
    @"}\n";

static id<MTLRenderPipelineState> pipeline(NSString *vertex, NSString *fragment,
                                           MTLPixelFormat color, NSUInteger samples,
                                           MTLPixelFormat depth, MTLPixelFormat stencil) {
  MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
  d.vertexFunction = [gLibrary newFunctionWithName:vertex];
  d.fragmentFunction = [gLibrary newFunctionWithName:fragment];
  d.colorAttachments[0].pixelFormat = color;
  d.depthAttachmentPixelFormat = depth;
  d.stencilAttachmentPixelFormat = stencil;
  d.rasterSampleCount = samples;
  NSError *error = nil;
  id<MTLRenderPipelineState> s = [gDevice newRenderPipelineStateWithDescriptor:d error:&error];
  if (s == nil) say("pipeline", [NSString stringWithFormat:@"failed: %@", error]);
  return s;
}

// A kSide x kSide texture of `format` over the buffer at kOffset, rows kPitch apart.
static id<MTLTexture> overBuffer(MTLPixelFormat format, MTLTextureUsage usage) {
  MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                               width:kSide
                                                                              height:kSide
                                                                           mipmapped:NO];
  d.usage = usage;
  d.storageMode = MTLStorageModeShared;
  d.allowGPUOptimizedContents = NO;
  stage("new-texture-over-buffer");
  id<MTLTexture> t = [gBuffer newTextureWithDescriptor:d offset:kOffset bytesPerRow:kPitch];
  if (t == nil) say("texture", @"nil");
  return t;
}

// A Private texture of its own memory.
static id<MTLTexture> own(MTLPixelFormat format, NSUInteger samples, MTLTextureUsage usage) {
  MTLTextureDescriptor *d = [MTLTextureDescriptor new];
  d.textureType = samples > 1 ? MTLTextureType2DMultisample : MTLTextureType2D;
  d.pixelFormat = format;
  d.width = kSide;
  d.height = kSide;
  d.sampleCount = samples;
  d.usage = usage;
  d.storageMode = MTLStorageModePrivate;
  return [gDevice newTextureWithDescriptor:d];
}

// A kSide x kSide texture of `format` and `levels` in a heap of `storage`, with bit 24 where the
// backing is `bit24`; kept as the shape's heap target, read back at `level`.
static id<MTLTexture> inHeap(MTLPixelFormat format, NSUInteger levels, MTLTextureUsage usage,
                             MTLStorageMode storage, NSUInteger level) {
  MTLTextureDescriptor *d = [MTLTextureDescriptor new];
  d.textureType = MTLTextureType2D;
  d.pixelFormat = format;
  d.width = kSide;
  d.height = kSide;
  d.mipmapLevelCount = levels;
  d.usage = usage;
  d.storageMode = storage;
  MTLSizeAndAlign sized = [gDevice heapTextureSizeAndAlignWithDescriptor:d];
  MTLHeapDescriptor *h = [MTLHeapDescriptor new];
  h.size = sized.size + sized.align + 0x10000;
  MTLResourceOptions mode = storage == MTLStorageModeShared ? MTLResourceStorageModeShared
                                                            : MTLResourceStorageModePrivate;
  h.resourceOptions = mode | (gBit24 ? kBit24 : 0);
  stage("new-heap");
  id<MTLHeap> heap = [gDevice newHeapWithDescriptor:h];
  if (heap == nil) {
    say("heap", @"nil");
    return nil;
  }
  [gKeep addObject:heap];
  stage("new-texture-in-heap");
  id<MTLTexture> t = [heap newTextureWithDescriptor:d];
  if (t == nil) {
    say("texture", @"nil");
    return nil;
  }
  gHeapTarget = t;
  gHeapLevel = level;
  return t;
}

static NSUInteger texelBytes(MTLPixelFormat f) {
  switch (f) {
    case MTLPixelFormatStencil8: return 1;
    case MTLPixelFormatRGBA16Float: return 8;
    default: return 4;
  }
}

// The first texels of `t`'s `level`, as a blit on a new queue copies them into a buffer of its
// own: that command buffer's status, and the first 8 bytes in hex.
static void readBack(const char *prefix, id<MTLTexture> t, NSUInteger level) {
  NSUInteger side = MAX(kSide >> level, 1);
  NSUInteger row = side * texelBytes(t.pixelFormat);
  id<MTLBuffer> out = [gDevice newBufferWithLength:row * side options:MTLResourceStorageModeShared];
  memset(out.contents, 0x5C, out.length);
  id<MTLCommandBuffer> cb = [[gDevice newCommandQueue] commandBuffer];
  id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
  [blit copyFromTexture:t
                   sourceSlice:0
                   sourceLevel:level
                  sourceOrigin:MTLOriginMake(0, 0, 0)
                    sourceSize:MTLSizeMake(side, side, 1)
                      toBuffer:out
             destinationOffset:0
        destinationBytesPerRow:row
      destinationBytesPerImage:row * side];
  [blit endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  char key[128];
  snprintf(key, sizeof key, "%s.read", prefix);
  sayBuffer(key, cb);
  const uint8_t *b = out.contents;
  NSMutableString *hex = [NSMutableString string];
  for (NSUInteger i = 0; i < MIN(8, out.length); i++) [hex appendFormat:@"%02x", b[i]];
  snprintf(key, sizeof key, "%s.bytes", prefix);
  say(key, hex);
}

typedef void (^Late)(id<MTLRenderCommandEncoder>);

// A render pass of `pass` that draws a triangle with `state`, `bind` run before the draw and
// `late` after it.
static void drawPass(id<MTLCommandBuffer> cb, MTLRenderPassDescriptor *pass,
                     id<MTLRenderPipelineState> state, Late bind, Late late) {
  stage("render-command-encoder");
  id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:pass];
  [e setRenderPipelineState:state];
  if (bind) bind(e);
  stage("draw");
  [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
  if (late) {
    stage("late");
    late(e);
  }
  stage("end-encoding");
  [e endEncoding];
}

static BOOL spi(id object, NSString *name) {
  if ([object respondsToSelector:NSSelectorFromString(name)]) return YES;
  say("spi", [NSString stringWithFormat:@"absent %@", name]);
  return NO;
}

static void lateColorResolve(id<MTLRenderCommandEncoder> e, id<MTLTexture> t) {
  NSString *name = @"setColorResolveTexture:slice:depthPlane:level:yInvert:atIndex:";
  if (!spi(e, name)) return;
  ((void (*)(id, SEL, id, NSUInteger, NSUInteger, NSUInteger, BOOL, NSUInteger))objc_msgSend)(
      e, NSSelectorFromString(name), t, 0, 0, 0, NO, 0);
}

static void lateDepthResolve(id<MTLRenderCommandEncoder> e, id<MTLTexture> t) {
  NSString *name = @"setDepthResolveTexture:slice:depthPlane:level:yInvert:";
  if (!spi(e, name)) return;
  ((void (*)(id, SEL, id, NSUInteger, NSUInteger, NSUInteger, BOOL))objc_msgSend)(
      e, NSSelectorFromString(name), t, 0, 0, 0, NO);
}

static const MTLTextureUsage kTarget = MTLTextureUsageRenderTarget;
static const MTLTextureUsage kTargetRead = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;

// A colour pass over a Private texture of its own (1 sample) that does not store it: what a
// depth or stencil shape draws beside.
static void ownColor(MTLRenderPassDescriptor *pass, NSUInteger samples) {
  pass.colorAttachments[0].texture = own(MTLPixelFormatRGBA8Unorm, samples, kTarget);
  pass.colorAttachments[0].loadAction = MTLLoadActionClear;
  pass.colorAttachments[0].storeAction = MTLStoreActionDontCare;
}

// Encodes `shape` into `cb`: NO where something it needs was not made.
static BOOL encodeShape(id<MTLCommandBuffer> cb, NSString *shape) {
  MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
  pass.colorAttachments[0].clearColor = MTLClearColorMake(1, 0.5, 0.25, 1);
  pass.depthAttachment.clearDepth = 0.25;
  pass.stencilAttachment.clearStencil = 0x5A;
  MTLPixelFormat none = MTLPixelFormatInvalid;

  // Colour attachments over the read-only buffer, and colour resolves into it.
  if ([shape isEqualToString:@"store"] || [shape hasPrefix:@"custom-color"]) {
    BOOL custom = [shape hasPrefix:@"custom-color"];
    BOOL late = [shape hasSuffix:@"-late"];
    id<MTLTexture> t = overBuffer(MTLPixelFormatRGBA8Unorm, kTarget);
    id<MTLRenderPipelineState> s = pipeline(@"v_main", @"f_main", MTLPixelFormatRGBA8Unorm, 1,
                                            none, none);
    if (t == nil || s == nil) return NO;
    MTLStoreAction action = custom ? MTLStoreActionCustomSampleDepthStore : MTLStoreActionStore;
    pass.colorAttachments[0].texture = t;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = late ? MTLStoreActionUnknown : action;
    drawPass(cb, pass, s, nil, late ? ^(id<MTLRenderCommandEncoder> e) {
      [e setColorStoreAction:action atIndex:0];
    } : nil);
    return YES;
  }
  if ([shape hasPrefix:@"resolve-into"] || [shape hasPrefix:@"smsr-into"] ||
      [shape hasPrefix:@"untouched"] || [shape hasPrefix:@"late-resolve-texture"]) {
    BOOL smsr = [shape hasPrefix:@"smsr"] || [shape isEqualToString:@"untouched-smsr"];
    BOOL late = [shape hasSuffix:@"-late"];
    BOOL untouched = [shape hasPrefix:@"untouched"];
    BOOL lateTexture = [shape hasPrefix:@"late-resolve-texture"];
    MTLPixelFormat from = untouched ? MTLPixelFormatBGRA8Unorm : MTLPixelFormatRGBA8Unorm;
    MTLPixelFormat into = untouched ? MTLPixelFormatRGBA16Float : MTLPixelFormatRGBA8Unorm;
    id<MTLTexture> t = overBuffer(into, kTarget);
    id<MTLTexture> samples = own(from, 4, kTarget);
    id<MTLRenderPipelineState> s = pipeline(@"v_main", @"f_main", from, 4, none, none);
    if (t == nil || samples == nil || s == nil) return NO;
    MTLStoreAction action =
        smsr ? MTLStoreActionStoreAndMultisampleResolve : MTLStoreActionMultisampleResolve;
    pass.colorAttachments[0].texture = samples;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = late ? MTLStoreActionUnknown : action;
    // late-resolve-texture-known: the descriptor resolves into a texture of its own, and the
    // encoder names the read-only one; late-resolve-texture-late: the action is Unknown too.
    pass.colorAttachments[0].resolveTexture =
        lateTexture ? (late ? nil : own(into, 1, kTarget)) : t;
    drawPass(cb, pass, s, nil, (late || lateTexture) ? ^(id<MTLRenderCommandEncoder> e) {
      if (lateTexture) lateColorResolve(e, t);
      if (late) [e setColorStoreAction:action atIndex:0];
    } : nil);
    return YES;
  }

  // Depth and stencil attachments in a read-only heap, and resolves into one.
  BOOL depth = [shape hasPrefix:@"depth-"] || [shape isEqualToString:@"custom-depth"] ||
               [shape isEqualToString:@"custom-depth-late"];
  BOOL stencil = [shape hasPrefix:@"stencil-"] || [shape hasPrefix:@"custom-stencil"];
  if (depth || stencil) {
    MTLPixelFormat format = depth ? MTLPixelFormatDepth32Float : MTLPixelFormatStencil8;
    MTLRenderPassAttachmentDescriptor *a =
        depth ? (MTLRenderPassAttachmentDescriptor *)pass.depthAttachment
              : (MTLRenderPassAttachmentDescriptor *)pass.stencilAttachment;
    BOOL resolve = [shape containsString:@"resolve"] || [shape hasSuffix:@"-smsr"] ||
                   [shape hasSuffix:@"late-texture"];
    BOOL late = [shape hasSuffix:@"-late"];
    BOOL lateTexture = [shape hasSuffix:@"late-texture"];
    NSUInteger samples = resolve ? 4 : 1;
    MTLStoreAction action;
    if ([shape hasSuffix:@"-store"]) action = MTLStoreActionStore;
    else if ([shape hasSuffix:@"-dontcare"]) action = MTLStoreActionDontCare;
    else if ([shape hasPrefix:@"custom-"]) action = MTLStoreActionCustomSampleDepthStore;
    else if ([shape hasSuffix:@"-smsr"]) action = MTLStoreActionStoreAndMultisampleResolve;
    else action = MTLStoreActionMultisampleResolve;
    id<MTLTexture> target = inHeap(format, 1, kTarget, MTLStorageModePrivate, 0);
    id<MTLRenderPipelineState> s =
        pipeline(@"v_main", @"f_main", MTLPixelFormatRGBA8Unorm, samples, depth ? format : none,
                 depth ? none : format);
    if (target == nil || s == nil) return NO;
    ownColor(pass, samples);
    a.loadAction = MTLLoadActionClear;
    a.storeAction = late ? MTLStoreActionUnknown : action;
    if (resolve) {
      a.texture = own(format, 4, kTarget);
      a.resolveTexture = lateTexture ? own(format, 1, kTarget) : target;
    } else {
      a.texture = target;
    }
    drawPass(cb, pass, s, nil, (late || lateTexture) ? ^(id<MTLRenderCommandEncoder> e) {
      if (lateTexture) lateDepthResolve(e, target);
      if (late && depth) [e setDepthStoreAction:action];
      if (late && !depth) [e setStencilStoreAction:action];
    } : nil);
    return YES;
  }

  // A draw's writes of a texture over the read-only buffer.
  if ([shape isEqualToString:@"frag-write"] || [shape isEqualToString:@"vert-write"] ||
      [shape isEqualToString:@"resident-write"]) {
    id<MTLTexture> t =
        overBuffer(MTLPixelFormatRGBA8Unorm, MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
    NSString *vertex = [shape isEqualToString:@"vert-write"] ? @"v_write" : @"v_main";
    NSString *fragment = [shape isEqualToString:@"frag-write"] ? @"f_write" : @"f_main";
    id<MTLRenderPipelineState> s =
        pipeline(vertex, fragment, MTLPixelFormatRGBA8Unorm, 1, none, none);
    if (t == nil || s == nil) return NO;
    pass.colorAttachments[0].texture = own(MTLPixelFormatRGBA8Unorm, 1, kTarget);
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    drawPass(cb, pass, s, ^(id<MTLRenderCommandEncoder> e) {
      if ([shape isEqualToString:@"resident-write"]) {
        [e useResource:t
                 usage:MTLResourceUsageRead | MTLResourceUsageWrite
                stages:MTLRenderStageVertex | MTLRenderStageFragment];
      } else {
        [e setVertexTexture:t atIndex:0];
        [e setFragmentTexture:t atIndex:0];
      }
    }, nil);
    return YES;
  }
  if ([shape isEqualToString:@"kernel-write"]) {
    id<MTLTexture> t =
        overBuffer(MTLPixelFormatRGBA8Unorm, MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
    NSError *error = nil;
    id<MTLComputePipelineState> s =
        [gDevice newComputePipelineStateWithFunction:[gLibrary newFunctionWithName:@"k_write"]
                                               error:&error];
    if (t == nil || s == nil) return NO;
    stage("compute-command-encoder");
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:s];
    [e setTexture:t atIndex:0];
    stage("dispatch");
    [e dispatchThreads:MTLSizeMake(kSide, kSide, 1) threadsPerThreadgroup:MTLSizeMake(kSide, kSide, 1)];
    stage("end-encoding");
    [e endEncoding];
    return YES;
  }

  // A blit's writes of a texture over the read-only buffer, or in a read-only heap.
  if ([shape isEqualToString:@"blit-copy-into"] || [shape hasPrefix:@"fill-"]) {
    id<MTLTexture> t = overBuffer(MTLPixelFormatRGBA8Unorm, MTLTextureUsageShaderRead);
    if (t == nil) return NO;
    stage("blit-command-encoder");
    id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
    MTLRegion region = MTLRegionMake2D(0, 0, kSide, kSide);
    if ([shape isEqualToString:@"blit-copy-into"]) {
      id<MTLBuffer> from = [gDevice newBufferWithLength:kPitch * kSide
                                                options:MTLResourceStorageModeShared];
      memset(from.contents, 0x3C, from.length);
      stage("copy-from-buffer");
      [e copyFromBuffer:from
                 sourceOffset:0
            sourceBytesPerRow:kPitch
          sourceBytesPerImage:kPitch * kSide
                   sourceSize:MTLSizeMake(kSide, kSide, 1)
                    toTexture:t
             destinationSlice:0
             destinationLevel:0
            destinationOrigin:MTLOriginMake(0, 0, 0)];
    } else if ([shape isEqualToString:@"fill-color"]) {
      NSString *name = @"fillTexture:level:slice:region:color:pixelFormat:";
      if (spi(e, name)) {
        stage("fill-texture-color");
        ((void (*)(id, SEL, id, NSUInteger, NSUInteger, MTLRegion, MTLClearColor,
                   MTLPixelFormat))objc_msgSend)(e, NSSelectorFromString(name), t, 0, 0, region,
                                                 MTLClearColorMake(0.25, 0.5, 0.75, 1),
                                                 MTLPixelFormatRGBA8Unorm);
      }
    } else {
      NSString *name = @"fillTexture:level:slice:region:bytes:length:";
      uint8_t texel[4] = {0x3C, 0x3C, 0x3C, 0x3C};
      if (spi(e, name)) {
        stage("fill-texture-bytes");
        ((void (*)(id, SEL, id, NSUInteger, NSUInteger, MTLRegion, const void *,
                   NSUInteger))objc_msgSend)(e, NSSelectorFromString(name), t, 0, 0, region, texel,
                                             sizeof texel);
      }
    }
    stage("end-encoding");
    [e endEncoding];
    return YES;
  }
  if ([shape hasPrefix:@"mipmaps"] || [shape isEqualToString:@"invalidate"]) {
    BOOL mipmaps = [shape hasPrefix:@"mipmaps"];
    // mipmaps-shared: a Shared heap's texture whose level 0 the CPU wrote first (a CPU write of
    // such a texture lands: `replace-heap`), so level 1 tells a written mipmap from a dropped one.
    BOOL shared = [shape isEqualToString:@"mipmaps-shared"];
    id<MTLTexture> t = inHeap(MTLPixelFormatRGBA8Unorm, mipmaps ? 3 : 1, kTargetRead,
                              shared ? MTLStorageModeShared : MTLStorageModePrivate,
                              mipmaps ? 1 : 0);
    if (t == nil) return NO;
    if (shared) {
      uint8_t texels[4 * 4 * 4];
      memset(texels, 0x3C, sizeof texels);
      [t replaceRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0 withBytes:texels bytesPerRow:16];
    }
    stage("blit-command-encoder");
    id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
    if (mipmaps) {
      stage("generate-mipmaps");
      [e generateMipmapsForTexture:t];
    } else if (spi(e, @"invalidateCompressedTexture:")) {
      stage("invalidate-compressed-texture");
      ((void (*)(id, SEL, id))objc_msgSend)(e, NSSelectorFromString(@"invalidateCompressedTexture:"),
                                            t);
    }
    stage("end-encoding");
    [e endEncoding];
    return YES;
  }
  say("shape", [NSString stringWithFormat:@"unknown %@", shape]);
  return NO;
}

// A command buffer on `queue` of a fill of a buffer of its own and a signal of `event` to
// `value`, committed and waited for, reported under `prefix`.
static void later(const char *prefix, id<MTLCommandQueue> queue, id<MTLSharedEvent> event,
                  uint64_t value) {
  id<MTLBuffer> out = [gDevice newBufferWithLength:256 options:MTLResourceStorageModeShared];
  memset(out.contents, kPad, out.length);
  id<MTLCommandBuffer> cb = [queue commandBuffer];
  id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
  [blit fillBuffer:out range:NSMakeRange(0, out.length) value:0x11];
  [blit endEncoding];
  [cb encodeSignalEvent:event value:value];
  [cb commit];
  [cb waitUntilCompleted];
  sayBuffer(prefix, cb);
  char key[128];
  snprintf(key, sizeof key, "%s.filled", prefix);
  say(key, ((uint8_t *)out.contents)[0] == 0x11 ? @"yes" : @"no");
  snprintf(key, sizeof key, "%s.signal", prefix);
  say(key, [NSString stringWithFormat:@"%llu", event.signaledValue]);
}

// What a shape left in its target, under `prefix`.
static void target(const char *prefix) {
  char key[128];
  snprintf(key, sizeof key, "%s.pages_moved", prefix);
  say(key, [NSString stringWithFormat:@"%lu", (unsigned long)pagesMoved()]);
  if (gHeapTarget != nil) {
    snprintf(key, sizeof key, "%s.heap", prefix);
    readBack(key, gHeapTarget, gHeapLevel);
  }
}

// The CPU's writes of a texture over the read-only buffer, or in a Shared read-only heap.
static int cpuShape(NSString *shape) {
  id<MTLTexture> t;
  if ([shape isEqualToString:@"replace-heap"]) {
    t = inHeap(MTLPixelFormatRGBA8Unorm, 1, MTLTextureUsageShaderRead, MTLStorageModeShared, 0);
  } else {
    t = overBuffer(MTLPixelFormatRGBA8Unorm, MTLTextureUsageShaderRead);
  }
  if (t == nil) return 2;
  uint8_t texels[kSide * kSide * 4];
  memset(texels, 0x3C, sizeof texels);
  stage("replace-region");
  [t replaceRegion:MTLRegionMake2D(0, 0, kSide, kSide)
       mipmapLevel:0
         withBytes:texels
       bytesPerRow:kSide * 4];
  stage("replaced");
  uint8_t back[kSide * kSide * 4];
  memset(back, 0x5C, sizeof back);
  [t getBytes:back bytesPerRow:kSide * 4 fromRegion:MTLRegionMake2D(0, 0, kSide, kSide)
      mipmapLevel:0];
  say("get_bytes_first", [NSString stringWithFormat:@"%02x%02x%02x%02x", back[0], back[1],
                                                    back[2], back[3]]);
  target("cpu");
  readBack("cpu.gpu", t, 0);
  return 0;
}

// The kernels of the indirect command buffer shapes: k_add adds v[0] to out[0]; k_encode writes
// command 0 of the argument buffer's buffer, k_add's state from it, and out and v.
static NSString *const kIcbSource =
    @"#include <metal_stdlib>\n"
    @"using namespace metal;\n"
    @"kernel void k_add(device uint *out [[buffer(0)]], device const uint *v [[buffer(1)]]) {\n"
    @"  out[0] += v[0];\n"
    @"}\n"
    @"struct IcbArgs { command_buffer cmds [[id(0)]]; compute_pipeline_state pso [[id(1)]]; };\n"
    @"kernel void k_encode(device IcbArgs &args [[buffer(0)]], device uint *out [[buffer(1)]],\n"
    @"                     device uint *v [[buffer(2)]]) {\n"
    @"  compute_command c(args.cmds, 0);\n"
    @"  c.set_compute_pipeline_state(args.pso);\n"
    @"  c.set_kernel_buffer(out, 0);\n"
    @"  c.set_kernel_buffer(v, 1);\n"
    @"  c.concurrent_dispatch_threadgroups(uint3(1, 1, 1), uint3(1, 1, 1));\n"
    @"}\n";

// An indirect command buffer of bit 24 (`bit24`) or not, its command 0 a dispatch of k_add of 5
// over out[0], 16 first: the CPU's (`icb-cpu`, Shared), a kernel's (`icb-gpu`, Private), or a
// kernel's over the CPU's of 1 (`icb-gpu-over-cpu`, Shared); what out[0] reads after the
// execution (21 where the add of 5 ran, 17 where the CPU's add of 1 did, 16 where none did), and
// the command buffers' status and error.
static int icbShape(NSString *shape, id<MTLCommandQueue> queue) {
  NSError *error = nil;
  id<MTLLibrary> lib = [gDevice newLibraryWithSource:kIcbSource options:nil error:&error];
  if (lib == nil) {
    say("icb_library", [NSString stringWithFormat:@"failed: %@", error]);
    return 2;
  }
  MTLComputePipelineDescriptor *pd = [MTLComputePipelineDescriptor new];
  pd.computeFunction = [lib newFunctionWithName:@"k_add"];
  pd.supportIndirectCommandBuffers = YES;
  id<MTLComputePipelineState> add =
      [gDevice newComputePipelineStateWithDescriptor:pd options:0 reflection:nil error:&error];
  id<MTLFunction> ef = [lib newFunctionWithName:@"k_encode"];
  id<MTLComputePipelineState> enc = [gDevice newComputePipelineStateWithFunction:ef error:&error];
  if (add == nil || enc == nil) {
    say("icb_pipelines", [NSString stringWithFormat:@"failed: %@", error]);
    return 2;
  }
  BOOL gpu = [shape hasPrefix:@"icb-gpu"];
  BOOL over = [shape isEqualToString:@"icb-gpu-over-cpu"];
  MTLResourceOptions storage = gpu && !over ? MTLResourceStorageModePrivate
                                            : MTLResourceStorageModeShared;
  MTLIndirectCommandBufferDescriptor *d = [MTLIndirectCommandBufferDescriptor new];
  d.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
  d.inheritPipelineState = NO;
  d.inheritBuffers = NO;
  d.maxKernelBufferBindCount = 2;
  stage("new-indirect-command-buffer");
  id<MTLIndirectCommandBuffer> icb =
      [gDevice newIndirectCommandBufferWithDescriptor:d
                                      maxCommandCount:1
                                              options:storage | (gBit24 ? kBit24 : 0)];
  if (icb == nil) {
    say("icb", @"nil");
    return 0;
  }
  say("icb.options", [NSString stringWithFormat:@"%#lx", (unsigned long)icb.resourceOptions]);
  id<MTLBuffer> out = [gDevice newBufferWithLength:16 options:MTLResourceStorageModeShared];
  id<MTLBuffer> five = [gDevice newBufferWithLength:16 options:MTLResourceStorageModeShared];
  id<MTLBuffer> one = [gDevice newBufferWithLength:16 options:MTLResourceStorageModeShared];
  ((uint32_t *)out.contents)[0] = 16;
  ((uint32_t *)five.contents)[0] = 5;
  ((uint32_t *)one.contents)[0] = 1;
  if (!gpu || over) {
    stage("cpu-encode");
    id<MTLIndirectComputeCommand> c = [icb indirectComputeCommandAtIndex:0];
    [c setComputePipelineState:add];
    [c setKernelBuffer:out offset:0 atIndex:0];
    [c setKernelBuffer:over ? one : five offset:0 atIndex:1];
    [c concurrentDispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
  }
  id<MTLCommandBuffer> cb = [queue commandBuffer];
  if (gpu) {
    id<MTLArgumentEncoder> ae = [ef newArgumentEncoderWithBufferIndex:0];
    id<MTLBuffer> args = [gDevice newBufferWithLength:ae.encodedLength
                                              options:MTLResourceStorageModeShared];
    [ae setArgumentBuffer:args offset:0];
    [ae setIndirectCommandBuffer:icb atIndex:0];
    [ae setComputePipelineState:add atIndex:1];
    stage("gpu-encode");
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:enc];
    [e setBuffer:args offset:0 atIndex:0];
    [e setBuffer:out offset:0 atIndex:1];
    [e setBuffer:five offset:0 atIndex:2];
    [e useResource:icb usage:MTLResourceUsageWrite];
    [e dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [e endEncoding];
  }
  stage("execute");
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  [e useResource:icb usage:MTLResourceUsageRead];
  [e useResource:out usage:MTLResourceUsageRead | MTLResourceUsageWrite];
  [e useResource:five usage:MTLResourceUsageRead];
  [e useResource:one usage:MTLResourceUsageRead];
  [e executeCommandsInBuffer:icb withRange:NSMakeRange(0, 1)];
  [e endEncoding];
  stage("commit");
  [cb commit];
  [cb waitUntilCompleted];
  sayBuffer("cb1", cb);
  say("cb1.out", [NSString stringWithFormat:@"%u", ((uint32_t *)out.contents)[0]]);
  // Executed again in a command buffer of its own: what the buffer's command holds now.
  ((uint32_t *)out.contents)[0] = 16;
  id<MTLCommandBuffer> again = [queue commandBuffer];
  e = [again computeCommandEncoder];
  [e useResource:icb usage:MTLResourceUsageRead];
  [e useResource:out usage:MTLResourceUsageRead | MTLResourceUsageWrite];
  [e useResource:five usage:MTLResourceUsageRead];
  [e useResource:one usage:MTLResourceUsageRead];
  [e executeCommandsInBuffer:icb withRange:NSMakeRange(0, 1)];
  [e endEncoding];
  stage("again");
  [again commit];
  [again waitUntilCompleted];
  sayBuffer("cb2", again);
  say("cb2.out", [NSString stringWithFormat:@"%u", ((uint32_t *)out.contents)[0]]);
  return 0;
}

static id<MTLDevice> pickDevice(const char *want) {
  for (id<MTLDevice> d in MTLCopyAllDevices()) {
    if ([d.name rangeOfString:@(want) options:NSCaseInsensitiveSearch].location != NSNotFound) {
      return d;
    }
  }
  return nil;
}

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc != 4) {
      fprintf(stderr, "usage: oracle_readonly DEVICE <bit24|plain> <shape>\n");
      return 2;
    }
    alarm(30);
    gBit24 = strcmp(argv[2], "bit24") == 0;
    if (!gBit24 && strcmp(argv[2], "plain") != 0) return 2;
    NSString *shape = @(argv[3]);
    gKeep = [NSMutableArray array];
    gDevice = pickDevice(argv[1]);
    if (gDevice == nil) {
      say("device", @"none");
      return 2;
    }
    say("device", gDevice.name);
    say("os", NSProcessInfo.processInfo.operatingSystemVersionString);
#if defined(__arm64__)
    say("arch", @"arm64");
#else
    say("arch", @"x86_64");
#endif
    NSError *error = nil;
    gLibrary = [gDevice newLibraryWithSource:kSource options:nil error:&error];
    if (gLibrary == nil) {
      say("library", [NSString stringWithFormat:@"failed: %@", error]);
      return 2;
    }
    gPages = mmap(NULL, kRegion, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (gPages == MAP_FAILED) return 2;
    memset(gPages, kPad, kRegion);
    // ro-pages-*: the pages are the process's read-only mapping.
    if ([shape hasPrefix:@"ro-pages-"]) {
      if (mprotect(gPages, kRegion, PROT_READ) != 0) return 2;
      say("pages", @"read-only");
      shape = [shape substringFromIndex:9];
    }
    stage("new-buffer-no-copy");
    gBuffer = [gDevice newBufferWithBytesNoCopy:gPages
                                         length:kRegion
                                        options:MTLResourceStorageModeShared | (gBit24 ? kBit24 : 0)
                                    deallocator:nil];
    if (gBuffer == nil) {
      say("buffer", @"nil");
      return 2;
    }
    id<MTLCommandQueue> queue = [gDevice newCommandQueue];
    id<MTLSharedEvent> event = [gDevice newSharedEvent];
    if ([shape hasPrefix:@"icb-"]) {
      int status = icbShape(shape, queue);
      if (status != 0) return status;
      later("same_queue", queue, event, 2);
      stage("done");
      return 0;
    }
    if ([shape hasPrefix:@"replace"]) {
      int status = cpuShape(shape);
      if (status != 0) return status;
      later("same_queue", queue, event, 2);
      stage("done");
      return 0;
    }
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    if (!encodeShape(cb, shape)) return 2;
    [cb encodeSignalEvent:event value:1];
    stage("commit");
    [cb commit];
    [cb waitUntilCompleted];
    sayBuffer("cb1", cb);
    say("cb1.signal", [NSString stringWithFormat:@"%llu", event.signaledValue]);
    target("cb1");
    later("same_queue", queue, event, 2);
    later("new_queue", [gDevice newCommandQueue], event, 3);
    usleep(100 * 1000);
    later("same_queue_later", queue, event, 4);
    id<MTLCommandBuffer> again = [queue commandBuffer];
    stage("again");
    if (!encodeShape(again, shape)) return 2;
    [again encodeSignalEvent:event value:5];
    [again commit];
    [again waitUntilCompleted];
    sayBuffer("cb2", again);
    say("cb2.signal", [NSString stringWithFormat:@"%llu", event.signaledValue]);
    target("cb2");
    later("after_second", queue, event, 6);
    later("after_second_new_queue", [gDevice newCommandQueue], event, 7);
    stage("done");
  }
  return 0;
}
