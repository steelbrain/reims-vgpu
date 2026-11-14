// What Metal stores into an IOSurface's two planes through a two-plane Y'CbCr texture over plane
// 0, by format: the 4:4:4 `_sRGB` twins 523 (YCBCR8_444_2P_sRGB) and 527 (YCBCR10_444_2P_sRGB),
// and beside them 503 (YCBCR8_444_2P) and 507 (YCBCR10_444_2P), whose stores metal2vulkan holds,
// and 500 (YCBCR8_420_2P), whose writes it drops. dgpu-oracle.md Q40 holds what it found.
//
// One format and one write a process, named on the command line:
//   oracle_twinwrites DEVICE FORMAT <kernel|fragment|clear|argbuf>
// DEVICE is a substring of the device's name (`AMD`, `Intel`, `Apple`), FORMAT the raw
// MTLPixelFormat (500, 503, 507, 523 or 527). The surface is 8 by 2 texels, of OSType '444v' for
// an 8-bit 4:4:4 format, 'x444' for a 10-bit one and '420v' for 500 (plane 1 of 4 by 1 Cb, Cr
// pairs), made by IOSurfaceCreate with its planes named (rows of 64 bytes, plane 1 at 4096) and
// no matrix, every byte of both planes 0xAB before each write.
// Column x is written colour x of kColours, alpha 1:
//   0 black, 1 white, 2 grey 0.5, 3 red, 4 green, 5 blue, 6 (0.2, 0.6, 0.9), 7 (0.9, 0.3, 0.1).
// `kernel` dispatches 8 x 2 threads of a kernel that write()s each texel's colour; `argbuf` does
// the same through an argument buffer that holds the texture, made resident for writing
// (`useResource:usage:` of `MTLResourceUsageWrite`); `fragment`
// draws over an 8 x 2 BGRA8Unorm target with a fragment function that write()s it at its
// position; `clear` runs, for each colour, a render pass into the texture that clears it to that
// colour and stores it, then reads the planes. Each stage prints a line and flushes before the
// call it names, so the last `stage=` line of a process that ended is the call that ended it; an
// alarm ends one stuck past 30 s. `oracle_twinwrites.sh DEVICE` runs every format and write.
//
// After each command buffer: its status and error, then each plane's rows of the written
// texels' bytes in hex (plane 0: 8 luma samples a row; plane 1: 8 Cb, Cr pairs a row, or 4 in
// one row for 500; a 10-bit sample is a little-endian 16-bit word, its code in the top 10 bits).
//
// Build (universal, on a Mac with the command line tools; build.sh builds it with the others):
//   clang -arch arm64 -arch x86_64 -mmacosx-version-min=14.0 -fobjc-arc -framework Metal \
//     -framework IOSurface -framework Foundation -o oracle_twinwrites oracle_twinwrites.m

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <unistd.h>

static const NSUInteger kWidth = 8;
static const NSUInteger kHeight = 2;
static const NSUInteger kRow = 64;
static const NSUInteger kChroma = 4096;
static const uint8_t kPad = 0xAB;

static const float kColours[8][4] = {
    {0, 0, 0, 1},       {1, 1, 1, 1},       {0.5, 0.5, 0.5, 1}, {1, 0, 0, 1},
    {0, 1, 0, 1},       {0, 0, 1, 1},       {0.2, 0.6, 0.9, 1}, {0.9, 0.3, 0.1, 1},
};

static id<MTLDevice> gDevice;
static id<MTLLibrary> gLibrary;

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

static void sayBuffer(NSString *prefix, id<MTLCommandBuffer> cb) {
  say([prefix stringByAppendingString:@".status"].UTF8String, statusName(cb.status));
  NSError *e = cb.error;
  say([prefix stringByAppendingString:@".error"].UTF8String,
      e == nil ? @"none"
               : [NSString stringWithFormat:@"%@ code %ld: %@", e.domain, (long)e.code,
                                            e.localizedDescription]);
}

static NSString *const kSource =
    @"#include <metal_stdlib>\n"
    @"using namespace metal;\n"
    @"struct VOut { float4 position [[position]]; };\n"
    @"constant float2 kTri[3] = { float2(-1, -3), float2(-1, 1), float2(3, 1) };\n"
    @"vertex VOut v_cover(uint vid [[vertex_id]]) {\n"
    @"  VOut o; o.position = float4(kTri[vid], 0.5, 1); return o;\n"
    @"}\n"
    @"kernel void k_write(texture2d<float, access::write> t [[texture(0)]],\n"
    @"                    constant float4 *colours [[buffer(0)]],\n"
    @"                    uint2 g [[thread_position_in_grid]]) {\n"
    @"  t.write(colours[g.x], g);\n"
    @"}\n"
    @"struct Held { texture2d<float, access::write> t [[id(0)]]; };\n"
    @"kernel void k_write_held(constant float4 *colours [[buffer(0)]],\n"
    @"                         constant Held &held [[buffer(1)]],\n"
    @"                         uint2 g [[thread_position_in_grid]]) {\n"
    @"  held.t.write(colours[g.x], g);\n"
    @"}\n"
    @"fragment float4 f_write(VOut in [[stage_in]],\n"
    @"                        texture2d<float, access::write> t [[texture(0)]],\n"
    @"                        constant float4 *colours [[buffer(0)]]) {\n"
    @"  uint2 at = uint2(in.position.xy);\n"
    @"  t.write(colours[at.x], at);\n"
    @"  return float4(0);\n"
    @"}\n";

// A surface of `ostype` of kWidth by kHeight texels, two planes of samples of `bytes` bytes each
// (a chroma element two of them), its chroma 4:4:4, or 4:2:0 where `sub`.
static IOSurfaceRef surface(uint32_t ostype, NSUInteger bytes, BOOL sub) {
  NSDictionary *plane0 = @{
    (id)kIOSurfacePlaneWidth : @(kWidth),
    (id)kIOSurfacePlaneHeight : @(kHeight),
    (id)kIOSurfacePlaneBytesPerElement : @(bytes),
    (id)kIOSurfacePlaneBytesPerRow : @(kRow),
    (id)kIOSurfacePlaneOffset : @0,
    (id)kIOSurfacePlaneSize : @(kRow * kHeight),
  };
  NSDictionary *plane1 = @{
    (id)kIOSurfacePlaneWidth : @(sub ? kWidth / 2 : kWidth),
    (id)kIOSurfacePlaneHeight : @(sub ? kHeight / 2 : kHeight),
    (id)kIOSurfacePlaneBytesPerElement : @(2 * bytes),
    (id)kIOSurfacePlaneBytesPerRow : @(kRow),
    (id)kIOSurfacePlaneOffset : @(kChroma),
    (id)kIOSurfacePlaneSize : @(kRow * kHeight),
  };
  NSDictionary *p = @{
    (id)kIOSurfaceWidth : @(kWidth),
    (id)kIOSurfaceHeight : @(kHeight),
    (id)kIOSurfacePixelFormat : @(ostype),
    (id)kIOSurfaceAllocSize : @(16384),
    (id)kIOSurfacePlaneInfo : @[ plane0, plane1 ],
  };
  return IOSurfaceCreate((__bridge CFDictionaryRef)p);
}

static void pad(IOSurfaceRef s) {
  IOSurfaceLock(s, 0, NULL);
  memset(IOSurfaceGetBaseAddress(s), kPad, IOSurfaceGetAllocSize(s));
  IOSurfaceUnlock(s, 0, NULL);
}

// Each plane's rows of the written texels' bytes, under `prefix`; plane 1's 4:2:0 where `sub`.
static void sayPlanes(NSString *prefix, IOSurfaceRef s, NSUInteger bytes, BOOL sub) {
  IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
  for (NSUInteger plane = 0; plane < 2; plane++) {
    const uint8_t *base = IOSurfaceGetBaseAddressOfPlane(s, plane);
    NSUInteger row = IOSurfaceGetBytesPerRowOfPlane(s, plane);
    NSUInteger used = kWidth * bytes * (plane == 1 && !sub ? 2 : 1);
    NSUInteger rows = plane == 1 && sub ? kHeight / 2 : kHeight;
    for (NSUInteger y = 0; y < rows; y++) {
      NSMutableString *hex = [NSMutableString string];
      for (NSUInteger i = 0; i < used; i++) [hex appendFormat:@"%02x", base[y * row + i]];
      NSString *key = [NSString stringWithFormat:@"%@.plane%lu.row%lu", prefix,
                                                 (unsigned long)plane, (unsigned long)y];
      say(key.UTF8String, hex);
    }
  }
  IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
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
      fprintf(stderr, "usage: oracle_twinwrites DEVICE FORMAT <kernel|fragment|clear|argbuf>\n");
      return 2;
    }
    alarm(30);
    NSUInteger format = strtoul(argv[2], NULL, 0);
    NSString *op = @(argv[3]);
    BOOL tenBit = format == 507 || format == 527;
    BOOL sub = format == 500;
    if (format != 500 && format != 503 && format != 507 && format != 523 && format != 527) return 2;
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
    NSUInteger bytes = tenBit ? 2 : 1;
    uint32_t ostype = sub ? '420v' : tenBit ? 'x444' : '444v';
    IOSurfaceRef s = surface(ostype, bytes, sub);
    if (s == NULL) {
      say("surface", @"none");
      return 2;
    }
    say("surface", [NSString stringWithFormat:@"%@ planes %zu rows %zu,%zu chroma at %zu",
                                              sub ? @"420v" : tenBit ? @"x444" : @"444v",
                                              IOSurfaceGetPlaneCount(s),
                                              IOSurfaceGetBytesPerRowOfPlane(s, 0),
                                              IOSurfaceGetBytesPerRowOfPlane(s, 1),
                                              (size_t)((uint8_t *)IOSurfaceGetBaseAddressOfPlane(s, 1) -
                                                       (uint8_t *)IOSurfaceGetBaseAddress(s))]);
    MTLTextureDescriptor *td =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:(MTLPixelFormat)format
                                                           width:kWidth
                                                          height:kHeight
                                                       mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite |
               MTLTextureUsageRenderTarget;
    td.storageMode = MTLStorageModeShared;
    stage("new-texture-over-surface");
    id<MTLTexture> t = [gDevice newTextureWithDescriptor:td iosurface:s plane:0];
    if (t == nil) {
      say("texture", @"nil");
      return 0;
    }
    say("texture", @"made");
    id<MTLCommandQueue> queue = [gDevice newCommandQueue];
    id<MTLBuffer> colours = [gDevice newBufferWithBytes:kColours
                                                 length:sizeof kColours
                                                options:MTLResourceStorageModeShared];
    NSString *prefix = [NSString stringWithFormat:@"%lu.%@", (unsigned long)format, op];
    if ([op isEqualToString:@"kernel"]) {
      id<MTLComputePipelineState> state =
          [gDevice newComputePipelineStateWithFunction:[gLibrary newFunctionWithName:@"k_write"]
                                                 error:&error];
      if (state == nil) {
        say("compute", [NSString stringWithFormat:@"failed: %@", error]);
        return 2;
      }
      pad(s);
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      stage("compute-encoder");
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      [e setComputePipelineState:state];
      [e setTexture:t atIndex:0];
      [e setBuffer:colours offset:0 atIndex:0];
      stage("dispatch");
      [e dispatchThreads:MTLSizeMake(kWidth, kHeight, 1)
          threadsPerThreadgroup:MTLSizeMake(kWidth, kHeight, 1)];
      [e endEncoding];
      stage("commit");
      [cb commit];
      [cb waitUntilCompleted];
      sayBuffer(prefix, cb);
      sayPlanes(prefix, s, bytes, sub);
    } else if ([op isEqualToString:@"argbuf"]) {
      id<MTLFunction> f = [gLibrary newFunctionWithName:@"k_write_held"];
      id<MTLComputePipelineState> state = [gDevice newComputePipelineStateWithFunction:f
                                                                                 error:&error];
      if (state == nil) {
        say("compute", [NSString stringWithFormat:@"failed: %@", error]);
        return 2;
      }
      id<MTLArgumentEncoder> ae = [f newArgumentEncoderWithBufferIndex:1];
      id<MTLBuffer> held = [gDevice newBufferWithLength:ae.encodedLength
                                                options:MTLResourceStorageModeShared];
      [ae setArgumentBuffer:held offset:0];
      [ae setTexture:t atIndex:0];
      pad(s);
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      stage("compute-encoder");
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      [e setComputePipelineState:state];
      [e setBuffer:colours offset:0 atIndex:0];
      [e setBuffer:held offset:0 atIndex:1];
      [e useResource:t usage:MTLResourceUsageWrite];
      stage("dispatch");
      [e dispatchThreads:MTLSizeMake(kWidth, kHeight, 1)
          threadsPerThreadgroup:MTLSizeMake(kWidth, kHeight, 1)];
      [e endEncoding];
      stage("commit");
      [cb commit];
      [cb waitUntilCompleted];
      sayBuffer(prefix, cb);
      sayPlanes(prefix, s, bytes, sub);
    } else if ([op isEqualToString:@"fragment"]) {
      MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
      d.vertexFunction = [gLibrary newFunctionWithName:@"v_cover"];
      d.fragmentFunction = [gLibrary newFunctionWithName:@"f_write"];
      d.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
      id<MTLRenderPipelineState> state = [gDevice newRenderPipelineStateWithDescriptor:d
                                                                                 error:&error];
      if (state == nil) {
        say("render", [NSString stringWithFormat:@"failed: %@", error]);
        return 2;
      }
      MTLTextureDescriptor *od =
          [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                             width:kWidth
                                                            height:kHeight
                                                         mipmapped:NO];
      od.usage = MTLTextureUsageRenderTarget;
      od.storageMode = MTLStorageModePrivate;
      MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
      pass.colorAttachments[0].texture = [gDevice newTextureWithDescriptor:od];
      pass.colorAttachments[0].loadAction = MTLLoadActionClear;
      pass.colorAttachments[0].storeAction = MTLStoreActionDontCare;
      pad(s);
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      stage("render-encoder");
      id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:pass];
      [e setRenderPipelineState:state];
      [e setFragmentTexture:t atIndex:0];
      [e setFragmentBuffer:colours offset:0 atIndex:0];
      stage("draw");
      [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
      [e endEncoding];
      stage("commit");
      [cb commit];
      [cb waitUntilCompleted];
      sayBuffer(prefix, cb);
      sayPlanes(prefix, s, bytes, sub);
    } else if ([op isEqualToString:@"clear"]) {
      for (NSUInteger c = 0; c < kWidth; c++) {
        MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[0].texture = t;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        pass.colorAttachments[0].clearColor =
            MTLClearColorMake(kColours[c][0], kColours[c][1], kColours[c][2], kColours[c][3]);
        pad(s);
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        stage([NSString stringWithFormat:@"render-encoder-%lu", (unsigned long)c].UTF8String);
        id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:pass];
        [e endEncoding];
        stage("commit");
        [cb commit];
        [cb waitUntilCompleted];
        NSString *each = [NSString stringWithFormat:@"%@.c%lu", prefix, (unsigned long)c];
        sayBuffer(each, cb);
        sayPlanes(each, s, bytes, sub);
      }
    } else {
      return 2;
    }
    CFRelease(s);
    stage("done");
  }
  return 0;
}
