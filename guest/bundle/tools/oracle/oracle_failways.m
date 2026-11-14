// What Metal answers for a command buffer its GPU fails in two ways at once: a render pass that
// stores into read-only memory (a no-copy buffer made with Metal's private option bit 24, which the
// M3 Max's GPU faults with kIOGPUCommandBufferCallbackErrorPageFault; dgpu-oracle.md Q38) beside
// encoders made under protection options (`setProtectionOptions:`, the command buffer's SPI,
// which fails the command buffer from the first encoder made under options other than 0, with
// code 16 below 0x10 and code 1 from 0x10 up, and with code 1 for options from 0x10 up set after
// every encoder too). dgpu-oracle.md Q39 holds what it found.
//
// One shape a process, named on the command line, with its backing and the protection options:
//   oracle_failways DEVICE <bit24|plain> <shape> [OPTIONS]
// DEVICE is a substring of the device's name (`AMD`, `Intel`, `Apple`). `bit24` makes the buffer
// the pass stores into with bit 24, `plain` without it: the writable control. OPTIONS is the value
// `setProtectionOptions:` is given (default 2; `0x` for hex). Each stage prints a line and flushes
// before the call it names, so the last `stage=` line of a process that ended is the call that
// ended it; an alarm ends one stuck past 30 s. `oracle_failways.sh DEVICE` runs every shape.
//
// The encoders, in order, by shape: `m` a compute encoder whose kernel writes its mark into a
// buffer of marks, `S` the render pass that clears a 4x4 RGBA8Unorm texture over the buffer and
// stores it (a store-out), `|` `setProtectionOptions:OPTIONS` (no encoder open):
//   store         m S m        the fault alone
//   prot          m | m m      the protection failure alone
//   store-prot    m S | m m    the fault first, then encoders made under the options
//   prot-store    m | m S m    encoders made under the options, the store-out among them
//   store-late    m S m |      the options set after every encoder
//   late          m m m |      the options set after every encoder, no store-out
// Each encoder is labelled e0, e1, ... in order, and is followed by the signal of an event of
// its own (to 1), so which signals fired shows where.
//
// After the command buffer: its status, its error (domain, code, description), each encoder's
// state from the error's encoder infos (Completed, Affected, Pending, Faulted, Unknown;
// `MTLCommandBufferErrorOptionEncoderExecutionStatus`), each marker encoder's mark (whether it
// ran), each event's signalled value, and the bytes of the buffer's pages that changed (whether
// the store landed); then a command buffer of one marker encoder on the same queue at once and on
// a new queue, which answer whether the queue runs on.
//
// Build (universal, on a Mac with the command line tools; build.sh builds it with the others):
//   clang -arch arm64 -arch x86_64 -mmacosx-version-min=14.0 -fobjc-arc -framework Metal \
//     -framework Foundation -o oracle_failways oracle_failways.m

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
static uint8_t *gPages;
static id<MTLBuffer> gBuffer;
static id<MTLBuffer> gMarks;
static id<MTLComputePipelineState> gMark;
static id<MTLRenderPipelineState> gDraw;

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

static NSString *encoderStateName(MTLCommandEncoderErrorState s) {
  switch (s) {
    case MTLCommandEncoderErrorStateUnknown: return @"Unknown";
    case MTLCommandEncoderErrorStateCompleted: return @"Completed";
    case MTLCommandEncoderErrorStateAffected: return @"Affected";
    case MTLCommandEncoderErrorStatePending: return @"Pending";
    case MTLCommandEncoderErrorStateFaulted: return @"Faulted";
  }
  return [NSString stringWithFormat:@"%ld", (long)s];
}

static void sayBuffer(const char *prefix, id<MTLCommandBuffer> cb) {
  char key[128];
  snprintf(key, sizeof key, "%s.status", prefix);
  say(key, statusName(cb.status));
  snprintf(key, sizeof key, "%s.error", prefix);
  NSError *e = cb.error;
  say(key, e == nil ? @"none"
                    : [NSString stringWithFormat:@"%@ code %ld: %@", e.domain, (long)e.code,
                                                 e.localizedDescription]);
  if (e == nil) return;
  NSArray *infos = e.userInfo[MTLCommandBufferEncoderInfoErrorKey];
  snprintf(key, sizeof key, "%s.encoders", prefix);
  NSMutableArray *states = [NSMutableArray array];
  for (id<MTLCommandBufferEncoderInfo> info in infos) {
    [states addObject:[NSString stringWithFormat:@"%@:%@", info.label,
                                                 encoderStateName(info.errorState)]];
  }
  say(key, infos == nil ? @"none" : [states componentsJoinedByString:@","]);
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
    @"vertex VOut v_cover(uint vid [[vertex_id]]) {\n"
    @"  VOut o; o.position = float4(kTri[vid], 0.5, 1); return o;\n"
    @"}\n"
    @"fragment float4 f_colour() { return float4(0.25, 0.5, 0.75, 1); }\n"
    @"kernel void k_mark(device uint *marks [[buffer(0)]], constant uint &i [[buffer(1)]],\n"
    @"                   uint t [[thread_position_in_grid]]) {\n"
    @"  if (t == 0) marks[i] = 0x5A00u + i;\n"
    @"}\n";

static BOOL makeStates(void) {
  NSError *error = nil;
  gLibrary = [gDevice newLibraryWithSource:kSource options:nil error:&error];
  if (gLibrary == nil) {
    say("library", [NSString stringWithFormat:@"failed: %@", error]);
    return NO;
  }
  gMark = [gDevice newComputePipelineStateWithFunction:[gLibrary newFunctionWithName:@"k_mark"]
                                                 error:&error];
  if (gMark == nil) {
    say("compute", [NSString stringWithFormat:@"failed: %@", error]);
    return NO;
  }
  MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
  d.vertexFunction = [gLibrary newFunctionWithName:@"v_cover"];
  d.fragmentFunction = [gLibrary newFunctionWithName:@"f_colour"];
  d.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
  gDraw = [gDevice newRenderPipelineStateWithDescriptor:d error:&error];
  if (gDraw == nil) {
    say("render", [NSString stringWithFormat:@"failed: %@", error]);
    return NO;
  }
  return YES;
}

// A compute encoder labelled `label` whose kernel writes mark `i`.
static void markEncoder(id<MTLCommandBuffer> cb, NSString *label, uint32_t i) {
  stage([NSString stringWithFormat:@"compute-encoder-%@", label].UTF8String);
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  e.label = label;
  [e setComputePipelineState:gMark];
  [e setBuffer:gMarks offset:0 atIndex:0];
  [e setBytes:&i length:sizeof i atIndex:1];
  [e dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
  [e endEncoding];
}

// The render pass labelled `label` that clears a texture over the buffer and stores it.
static BOOL storeEncoder(id<MTLCommandBuffer> cb, NSString *label) {
  MTLTextureDescriptor *d =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                         width:kSide
                                                        height:kSide
                                                     mipmapped:NO];
  d.usage = MTLTextureUsageRenderTarget;
  d.storageMode = MTLStorageModeShared;
  stage("new-texture-over-buffer");
  id<MTLTexture> t = [gBuffer newTextureWithDescriptor:d offset:kOffset bytesPerRow:kPitch];
  if (t == nil) {
    say("texture", @"nil");
    return NO;
  }
  MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
  pass.colorAttachments[0].texture = t;
  pass.colorAttachments[0].loadAction = MTLLoadActionClear;
  pass.colorAttachments[0].storeAction = MTLStoreActionStore;
  pass.colorAttachments[0].clearColor = MTLClearColorMake(1, 0.5, 0.25, 1);
  stage([NSString stringWithFormat:@"render-encoder-%@", label].UTF8String);
  id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:pass];
  e.label = label;
  [e setRenderPipelineState:gDraw];
  [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
  [e endEncoding];
  return YES;
}

static BOOL protect(id<MTLCommandBuffer> cb, uint64_t options) {
  SEL sel = NSSelectorFromString(@"setProtectionOptions:");
  if (![(id)cb respondsToSelector:sel]) {
    say("spi", @"absent setProtectionOptions:");
    return NO;
  }
  stage("set-protection-options");
  ((void (*)(id, SEL, uint64_t))objc_msgSend)(cb, sel, options);
  return YES;
}

// One marker encoder in a command buffer of `queue`, labelled `prefix`'s: whether it ran.
static void later(const char *prefix, id<MTLCommandQueue> queue, uint32_t i) {
  id<MTLCommandBuffer> cb = [queue commandBuffer];
  markEncoder(cb, @"later", i);
  [cb commit];
  [cb waitUntilCompleted];
  sayBuffer(prefix, cb);
  char key[128];
  snprintf(key, sizeof key, "%s.marked", prefix);
  say(key, ((uint32_t *)gMarks.contents)[i] == 0x5A00u + i ? @"yes" : @"no");
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
    if (argc != 4 && argc != 5) {
      fprintf(stderr, "usage: oracle_failways DEVICE <bit24|plain> <shape> [OPTIONS]\n");
      return 2;
    }
    alarm(30);
    BOOL bit24 = strcmp(argv[2], "bit24") == 0;
    if (!bit24 && strcmp(argv[2], "plain") != 0) return 2;
    NSString *shape = @(argv[3]);
    uint64_t options = argc == 5 ? strtoull(argv[4], NULL, 0) : 2;
    // The encoders of the shape, as the header lists them.
    NSDictionary<NSString *, NSString *> *shapes = @{
      @"store" : @"mSm",
      @"prot" : @"m|mm",
      @"store-prot" : @"mS|mm",
      @"prot-store" : @"m|mSm",
      @"store-late" : @"mSm|",
      @"late" : @"mmm|",
    };
    NSString *plan = shapes[shape];
    if (plan == nil) return 2;
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
    say("plan", plan);
    say("options", [NSString stringWithFormat:@"%#llx", options]);
    if (!makeStates()) return 2;
    gPages = mmap(NULL, kRegion, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (gPages == MAP_FAILED) return 2;
    memset(gPages, kPad, kRegion);
    stage("new-buffer-no-copy");
    gBuffer = [gDevice newBufferWithBytesNoCopy:gPages
                                         length:kRegion
                                        options:MTLResourceStorageModeShared | (bit24 ? kBit24 : 0)
                                    deallocator:nil];
    if (gBuffer == nil) {
      say("buffer", @"nil");
      return 2;
    }
    gMarks = [gDevice newBufferWithLength:256 options:MTLResourceStorageModeShared];
    memset(gMarks.contents, 0, gMarks.length);
    id<MTLCommandQueue> queue = [gDevice newCommandQueue];
    MTLCommandBufferDescriptor *cd = [MTLCommandBufferDescriptor new];
    cd.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
    id<MTLCommandBuffer> cb = [queue commandBufferWithDescriptor:cd];
    NSMutableArray<id<MTLSharedEvent>> *events = [NSMutableArray array];
    NSMutableArray<NSString *> *kinds = [NSMutableArray array];
    for (NSUInteger c = 0; c < plan.length; c++) {
      unichar step = [plan characterAtIndex:c];
      if (step == '|') {
        if (!protect(cb, options)) return 2;
        continue;
      }
      uint32_t i = (uint32_t)events.count;
      NSString *label = [NSString stringWithFormat:@"e%u", i];
      if (step == 'S') {
        if (!storeEncoder(cb, label)) return 2;
      } else {
        markEncoder(cb, label, i);
      }
      [kinds addObject:step == 'S' ? @"store" : @"mark"];
      id<MTLSharedEvent> event = [gDevice newSharedEvent];
      [cb encodeSignalEvent:event value:1];
      [events addObject:event];
    }
    stage("commit");
    [cb commit];
    [cb waitUntilCompleted];
    sayBuffer("cb", cb);
    NSMutableArray *ran = [NSMutableArray array];
    NSMutableArray *signals = [NSMutableArray array];
    for (NSUInteger i = 0; i < events.count; i++) {
      BOOL marked = ((uint32_t *)gMarks.contents)[i] == 0x5A00u + i;
      [ran addObject:[kinds[i] isEqualToString:@"store"]
                         ? [NSString stringWithFormat:@"e%lu:store", (unsigned long)i]
                         : [NSString stringWithFormat:@"e%lu:%@", (unsigned long)i,
                                                      marked ? @"ran" : @"not-run"]];
      [signals addObject:[NSString stringWithFormat:@"e%lu:%llu", (unsigned long)i,
                                                    events[i].signaledValue]];
    }
    say("cb.marks", [ran componentsJoinedByString:@","]);
    say("cb.signals", [signals componentsJoinedByString:@","]);
    say("cb.pages_moved", [NSString stringWithFormat:@"%lu", (unsigned long)pagesMoved()]);
    later("same_queue", queue, 8);
    later("new_queue", [gDevice newCommandQueue], 9);
    stage("done");
  }
  return 0;
}
