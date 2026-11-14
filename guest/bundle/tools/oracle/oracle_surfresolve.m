// What a 4x colour attachment's resolve into a texture over an IOSurface does where the two
// formats differ (a format pair Metal leaves a texture of its own untouched in), by the surface's
// planes and the texture's storage, against a texture of its own and a resolve of the same
// format. dgpu-oracle.md Q41 holds what it found.
//
// One case a process, named on the command line:
//   oracle_surfresolve DEVICE <over> <storage> <from> <into> <action>
// DEVICE is a substring of the device's name (`AMD`, `Intel`, `Apple`). <over> is what the
// resolve texture lies over: `own` (memory of its own), `flat` (a surface without planes, of the
// texture's size, its OSType 'RGhA' for 8 bytes a texel, 'w30r' for 4, 'BGRA' for a BGRA8 one),
// `plane` (plane 0 of a 128 x 16 '420v' surface, CoreVideo's, whose rows of 128 bytes or more
// hold the texture's), or `plane1` (plane 1 of a 128 x 32 one: 16 rows of 64 two-byte chroma
// elements). <storage> is the texture's: `shared`, `managed` or `private`. <from> is the
// attachment's raw MTLPixelFormat (80 BGRA8Unorm, 81 BGRA8Unorm_sRGB) and <into> the resolve
// texture's (115 RGBA16Float, 554 BGR10_XR, 555 BGR10_XR_sRGB, or <from> itself: the control).
// <action> is the store action: `resolve` (MultisampleResolve) or `smsr`
// (StoreAndMultisampleResolve). Each stage prints a line and flushes before the call it names, so
// the last `stage=` line of a process that ended is the call that ended it; an alarm ends one
// stuck past 30 s. `oracle_surfresolve.sh DEVICE` runs every case.
//
// The case: the 16 x 16 resolve texture is filled with 0xa5 bytes by a blit, and the 4-sample
// attachment (Private, of its own) cleared to (0.5, 0.25, 0.125, 1) and stored (cb0). Then one
// pass loads the samples, draws a triangle over every texel in (0.25, 0.5, 0.75, 1) and stores
// them by <action> into the resolve texture (cb1). Then a blit copies the resolve texture into a
// buffer, and a second pass loads the samples and resolves them into a texture of <from> of its
// own, which a blit copies out too (cb2). Lines: each command buffer's status and error
// (`cbN.status`, `cbN.error`), the resolve texture's first 16 bytes (`resolved.bytes`), whether
// every byte is still 0xa5 (`resolved.kept`), and the samples' first texel as the second pass
// resolves them (`samples.bytes`).
//
// Build (universal, on a Mac with the command line tools; build.sh builds it with the others):
//   clang -arch arm64 -arch x86_64 -mmacosx-version-min=14.0 -fobjc-arc -framework Metal \
//     -framework IOSurface -framework CoreVideo -framework Foundation \
//     -o oracle_surfresolve oracle_surfresolve.m

#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <unistd.h>

static const NSUInteger kSide = 16;
static const uint8_t kFill = 0xA5;

static id<MTLDevice> gDevice;

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

// Commits `cb`, waits for it, and prints its status and error under `prefix`.
static void run(const char *prefix, id<MTLCommandBuffer> cb) {
  char key[64];
  snprintf(key, sizeof key, "commit-%s", prefix);
  stage(key);
  [cb commit];
  [cb waitUntilCompleted];
  snprintf(key, sizeof key, "%s.status", prefix);
  say(key, statusName(cb.status));
  snprintf(key, sizeof key, "%s.error", prefix);
  say(key, cb.error == nil ? @"none"
                           : [NSString stringWithFormat:@"%@ code %ld: %@", cb.error.domain,
                                                        (long)cb.error.code,
                                                        cb.error.localizedDescription]);
}

static NSString *const kSource =
    @"#include <metal_stdlib>\n"
    @"using namespace metal;\n"
    @"struct VOut { float4 position [[position]]; };\n"
    @"constant float2 kTri[3] = { float2(-1, -3), float2(-1, 1), float2(3, 1) };\n"
    @"vertex VOut v_main(uint vid [[vertex_id]]) {\n"
    @"  VOut o; o.position = float4(kTri[vid], 0.5, 1); return o;\n"
    @"}\n"
    @"fragment float4 f_main() { return float4(0.25, 0.5, 0.75, 1); }\n";

// Bytes a texel of the formats the cases name.
static NSUInteger texelBytes(MTLPixelFormat f) {
  return f == MTLPixelFormatRGBA16Float ? 8 : 4;
}

static MTLStorageMode storageOf(const char *name) {
  if (strcmp(name, "shared") == 0) return MTLStorageModeShared;
  if (strcmp(name, "managed") == 0) return MTLStorageModeManaged;
  if (strcmp(name, "private") == 0) return MTLStorageModePrivate;
  return (MTLStorageMode)-1;
}

// A surface without planes of `width` x `height` texels of `bytes` bytes, of `ostype`.
static IOSurfaceRef flatSurface(OSType ostype, NSUInteger width, NSUInteger height,
                                NSUInteger bytes) {
  NSDictionary *p = @{
    (id)kIOSurfaceWidth : @(width),
    (id)kIOSurfaceHeight : @(height),
    (id)kIOSurfaceBytesPerElement : @(bytes),
    (id)kIOSurfacePixelFormat : @(ostype),
  };
  return IOSurfaceCreate((__bridge CFDictionaryRef)p);
}

// A 128 x `height` '420v' surface, as CoreVideo lays one out; the pixel buffer is kept in `keep`.
static IOSurfaceRef planarSurface(NSUInteger height, CVPixelBufferRef *keep) {
  NSDictionary *attrs = @{(id)kCVPixelBufferIOSurfacePropertiesKey : @{}};
  CVReturn r = CVPixelBufferCreate(kCFAllocatorDefault, 128, height,
                                   kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
                                   (__bridge CFDictionaryRef)attrs, keep);
  if (r != kCVReturnSuccess) return NULL;
  return CVPixelBufferGetIOSurface(*keep);
}

// The resolve texture: of `into`, `storage`, over what `over` names.
static id<MTLTexture> resolveTexture(const char *over, MTLPixelFormat into, MTLStorageMode storage,
                                     CVPixelBufferRef *keep) {
  MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:into
                                                                               width:kSide
                                                                              height:kSide
                                                                           mipmapped:NO];
  d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
  d.storageMode = storage;
  if (strcmp(over, "own") == 0) {
    stage("new-texture");
    return [gDevice newTextureWithDescriptor:d];
  }
  IOSurfaceRef s = NULL;
  size_t plane = 0;
  if (strcmp(over, "flat") == 0) {
    OSType ostype = texelBytes(into) == 8                                 ? 'RGhA'
                    : (into == MTLPixelFormatBGRA8Unorm ||
                       into == MTLPixelFormatBGRA8Unorm_sRGB)              ? 'BGRA'
                                                                           : 'w30r';
    stage("new-surface");
    s = flatSurface(ostype, kSide, kSide, texelBytes(into));
  } else if (strcmp(over, "plane") == 0) {
    stage("new-surface");
    s = planarSurface(kSide, keep);
  } else if (strcmp(over, "plane1") == 0) {
    stage("new-surface");
    s = planarSurface(2 * kSide, keep);
    plane = 1;
  }
  if (s == NULL) {
    say("surface", @"nil");
    return nil;
  }
  say("surface.planes", [NSString stringWithFormat:@"%zu", IOSurfaceGetPlaneCount(s)]);
  say("surface.row_bytes",
      [NSString stringWithFormat:@"%zu", IOSurfaceGetPlaneCount(s) == 0
                                             ? IOSurfaceGetBytesPerRow(s)
                                             : IOSurfaceGetBytesPerRowOfPlane(s, plane)]);
  stage("new-texture-over-surface");
  return [gDevice newTextureWithDescriptor:d iosurface:s plane:plane];
}

// A 4-sample Private texture of `format` of its own, or a 1-sample one.
static id<MTLTexture> own(MTLPixelFormat format, NSUInteger samples) {
  MTLTextureDescriptor *d = [MTLTextureDescriptor new];
  d.textureType = samples > 1 ? MTLTextureType2DMultisample : MTLTextureType2D;
  d.pixelFormat = format;
  d.width = kSide;
  d.height = kSide;
  d.sampleCount = samples;
  d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
  d.storageMode = MTLStorageModePrivate;
  return [gDevice newTextureWithDescriptor:d];
}

// A blit of `t` into `out`, its rows `row` bytes apart.
static void copyOut(id<MTLCommandBuffer> cb, id<MTLTexture> t, id<MTLBuffer> out, NSUInteger row) {
  id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
  [blit copyFromTexture:t
                   sourceSlice:0
                   sourceLevel:0
                  sourceOrigin:MTLOriginMake(0, 0, 0)
                    sourceSize:MTLSizeMake(kSide, kSide, 1)
                      toBuffer:out
             destinationOffset:0
        destinationBytesPerRow:row
      destinationBytesPerImage:row * kSide];
  [blit endEncoding];
}

static NSString *hex(const uint8_t *b, NSUInteger n) {
  NSMutableString *s = [NSMutableString string];
  for (NSUInteger i = 0; i < n; i++) [s appendFormat:@"%02x", b[i]];
  return s;
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
    if (argc != 7) {
      fprintf(stderr, "usage: oracle_surfresolve DEVICE <own|flat|plane|plane1> "
                      "<shared|managed|private> FROM INTO <resolve|smsr>\n");
      return 2;
    }
    alarm(30);
    const char *over = argv[2];
    MTLStorageMode storage = storageOf(argv[3]);
    MTLPixelFormat from = (MTLPixelFormat)strtoul(argv[4], NULL, 10);
    MTLPixelFormat into = (MTLPixelFormat)strtoul(argv[5], NULL, 10);
    BOOL smsr = strcmp(argv[6], "smsr") == 0;
    if (storage == (MTLStorageMode)-1 || (!smsr && strcmp(argv[6], "resolve") != 0)) return 2;
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
    id<MTLLibrary> library = [gDevice newLibraryWithSource:kSource options:nil error:&error];
    if (library == nil) {
      say("library", [NSString stringWithFormat:@"failed: %@", error]);
      return 2;
    }
    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction = [library newFunctionWithName:@"v_main"];
    pd.fragmentFunction = [library newFunctionWithName:@"f_main"];
    pd.colorAttachments[0].pixelFormat = from;
    pd.rasterSampleCount = 4;
    id<MTLRenderPipelineState> state = [gDevice newRenderPipelineStateWithDescriptor:pd
                                                                               error:&error];
    if (state == nil) {
      say("pipeline", [NSString stringWithFormat:@"failed: %@", error]);
      return 2;
    }
    CVPixelBufferRef keep = NULL;
    id<MTLTexture> target = resolveTexture(over, into, storage, &keep);
    if (target == nil) {
      say("texture", @"nil");
      return 0;
    }
    say("texture.storage", [NSString stringWithFormat:@"%lu", (unsigned long)target.storageMode]);
    id<MTLTexture> samples = own(from, 4);
    id<MTLTexture> check = own(from, 1);
    if (samples == nil || check == nil) {
      say("samples", @"nil");
      return 2;
    }
    NSUInteger row = kSide * texelBytes(into);
    id<MTLBuffer> fill = [gDevice newBufferWithLength:row * kSide
                                              options:MTLResourceStorageModeShared];
    memset(fill.contents, kFill, fill.length);
    id<MTLCommandQueue> queue = [gDevice newCommandQueue];

    // cb0: the fill, and the samples cleared and stored.
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:fill
               sourceOffset:0
          sourceBytesPerRow:row
        sourceBytesPerImage:row * kSide
                 sourceSize:MTLSizeMake(kSide, kSide, 1)
                  toTexture:target
           destinationSlice:0
           destinationLevel:0
          destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit endEncoding];
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = samples;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(0.5, 0.25, 0.125, 1);
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    [[cb renderCommandEncoderWithDescriptor:pass] endEncoding];
    run("cb0", cb);

    // cb1: the case's pass.
    cb = [queue commandBuffer];
    pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = samples;
    pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
    pass.colorAttachments[0].storeAction =
        smsr ? MTLStoreActionStoreAndMultisampleResolve : MTLStoreActionMultisampleResolve;
    pass.colorAttachments[0].resolveTexture = target;
    stage("render-command-encoder");
    id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:pass];
    [e setRenderPipelineState:state];
    stage("draw");
    [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    stage("end-encoding");
    [e endEncoding];
    run("cb1", cb);

    // cb2: the resolve texture copied out, and the samples resolved into a texture of their own
    // format and copied out.
    cb = [queue commandBuffer];
    id<MTLBuffer> out = [gDevice newBufferWithLength:row * kSide
                                             options:MTLResourceStorageModeShared];
    memset(out.contents, 0x5C, out.length);
    copyOut(cb, target, out, row);
    pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = samples;
    pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
    pass.colorAttachments[0].storeAction = MTLStoreActionMultisampleResolve;
    pass.colorAttachments[0].resolveTexture = check;
    [[cb renderCommandEncoderWithDescriptor:pass] endEncoding];
    id<MTLBuffer> checked = [gDevice newBufferWithLength:kSide * 4 * kSide
                                                 options:MTLResourceStorageModeShared];
    memset(checked.contents, 0x5C, checked.length);
    copyOut(cb, check, checked, kSide * 4);
    run("cb2", cb);

    const uint8_t *b = out.contents;
    BOOL kept = YES;
    for (NSUInteger i = 0; i < out.length; i++) kept = kept && b[i] == kFill;
    say("resolved.bytes", hex(b, 16));
    say("resolved.kept", kept ? @"yes" : @"no");
    say("samples.bytes", hex(checked.contents, 4));
    stage("done");
    if (keep != NULL) CVPixelBufferRelease(keep);
  }
  return 0;
}
