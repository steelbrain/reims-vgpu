// Ground-truth generator for reims-vgpu-wire.
//
// Loads Apple's paravirt Metal serializer on the host, drives it through the
// Metal API, and writes the wire bytes it produced together with what we asked
// for. The Rust side then maps its views onto those bytes and checks it reads
// back what Metal was told.
//
// WHY THIS RUNS AT ALL, on a Mac that is not a VM:
//
//   * The bundle ships x86_64 + arm64e. There is no plain-arm64 slice and
//     third-party arm64e needs a preview-ABI boot-arg, so this is built
//     -arch x86_64 and run under Rosetta.
//   * -[PGSerializer initWithDevice:objectRefAllocator:] takes id<MTLDevice> --
//     a protocol -- so the host's own GPU satisfies it. AppleParavirtDevice
//     needs an AppleParavirtGPU IOService and bare metal has none, but
//     PGSerializer sits below that and never asks for one.
//   * PGSerializerAllocator requires exactly one method,
//     -allocateOperationBytes:(size_t), so a conforming object receives every
//     operation the serializer writes, and the size it requests is the
//     operation's true length.
//
// EXPECTED VALUES ARE NOT READ BACK FROM THE BUFFER. Each case records what we
// set on the MTLTextureDescriptor, read from the descriptor object itself, so
// enum ordinals come from Metal rather than from anything hand-copied here. A
// fixture whose expectations were read out of the bytes it is meant to check
// would pass no matter what the layout did.
//
// Emits no Apple bytes into the repository: the JSON lands in a gitignored
// fixtures directory and is regenerated on demand. See ../AGENTS.md.
#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <dlfcn.h>
#import <float.h>
#import <objc/message.h>
#import <objc/runtime.h>
#import <setjmp.h>
#import <signal.h>

static const char *kBundleBin =
    "/System/Library/Extensions/AppleParavirtGPUMetal.bundle/Contents/MacOS/"
    "AppleParavirtGPUMetal";
static NSString *const kBundlePlist =
    @"/System/Library/Extensions/AppleParavirtGPUMetal.bundle/Contents/Info.plist";

#define ARENA_CAP (1u << 20)
#define ARENA_POISON 0xAA // untouched bytes stay recognisable

// The second fill, and the reason there are two.
//
// A byte left at `0xAA` reads as "the serializer never wrote here", and that
// reading is what several views in this crate rest on. It is not sound on its
// own: a serializer that genuinely writes `0xAA` is indistinguishable from one
// that writes nothing, and a serializer that writes a *bitfield* leaves the
// surrounding bits at the fill while the byte as a whole looks written.
//
// So every case is captured twice, once under each fill, and the two buffers
// are XORed. A bit that differs was never written; a bit that agrees was. The
// result is `written_mask`, a per-bit measurement rather than an eyeballed one,
// and the two fills are bitwise complements so no bit can agree by accident.
#define ARENA_POISON_ALT 0x55

static unsigned char *gArena;
static size_t gUsed;
static size_t gOpOff[64], gOpLen[64];
static int gOpCount;
static id<MTLDevice> gDevice;

/// Hand out `n` bytes and record them as one record.
static void *arenaTake(size_t n) {
  if (gUsed + n > ARENA_CAP || gOpCount >= 64) return NULL;
  void *p = gArena + gUsed;
  gOpOff[gOpCount] = gUsed;
  gOpLen[gOpCount] = n;
  gOpCount++;
  gUsed += n;
  return p;
}

/// Hand out `n` bytes of inline argument data *without* counting a record.
/// Bind tables and index lists come through this path; counting them would make
/// a one-record case look like several and trip `caseFromCapture`.
static void *arenaTakeUncounted(size_t n, size_t align) {
  if (align < 1) align = 1;
  gUsed = (gUsed + align - 1) & ~(align - 1);
  if (gUsed + n > ARENA_CAP) return NULL;
  void *p = gArena + gUsed;
  gUsed += n;
  return p;
}

@interface CaptureAllocator : NSObject
@end
@implementation CaptureAllocator
- (char *)allocateOperationBytes:(size_t)n {
  return (char *)arenaTake(n);
}
@end

@interface RefAllocator : NSObject
@end
@implementation RefAllocator
static unsigned gNextRef = 1;
- (unsigned int)allocateObjectRef { return gNextRef++; }
@end

#import "encoder.h"

/// The fill this pass uses. See [`ARENA_POISON_ALT`].
static unsigned char gPoison = ARENA_POISON;

/// Whether this pass records the non-`cases` outcomes.
///
/// The second pass drives every selector again, so a refusal or a silence
/// recorded on both would appear twice and the manifest tests that count them
/// would be counting passes rather than selectors.
static BOOL gRecordOutcomes = YES;

static void arenaReset(void) {
  memset(gArena, gPoison, 64 * 1024);
  gUsed = 0;
  gOpCount = 0;
}

// --- Surviving a selector Apple refuses -------------------------------------
//
// Some selectors on these classes do not serialize anything: they fail an
// assertion *inside* Apple's encoder and abort the process --
// "-[PGSerializerRenderCommandEncoder setProvokingVertexMode:]: failed
// assertion `Not supported'". With 364 selectors to triage, one of those taking
// the whole capture down means every case after it silently goes missing, and
// the JSON is only written at the end, so the run leaves no trace of how far it
// got.
//
// So SIGABRT is caught and unwound back to the case boundary. The refusal is
// then a *result* -- recorded in the output as an unsupported selector, with
// the message Apple printed -- rather than the end of the run. That is the
// evidence a `Coverage::Excluded` row needs, and it is regenerated every
// capture instead of being asserted once and trusted forever.
//
// Each case builds a fresh encoder, so an unwound one is dropped rather than
// reused. The handler is armed only across the serializer call itself.
static sigjmp_buf gAbortJmp;
static volatile sig_atomic_t gAbortArmed;
static volatile sig_atomic_t gLastSignal;

static void abortHandler(int sig) {
  if (gAbortArmed) {
    gAbortArmed = 0;
    gLastSignal = sig;
    siglongjmp(gAbortJmp, 1);
  }
  _exit(128 + sig);
}

/// Run `body`, returning 0 if it died on a signal rather than returning.
///
/// SIGSEGV and SIGBUS are caught alongside SIGABRT, and they mean something
/// different: an abort is Apple's own assertion and is evidence about the
/// serializer, while a fault is almost always *this* harness handing a stub
/// object back where the serializer wanted a real pointer. The two are recorded
/// separately for exactly that reason -- see `gCrashed`.
///
/// Unwinding a fault with siglongjmp leaves the encoder in an unknown state, so
/// the caller drops it rather than ending it, as it already does for an abort.
static int runSurvivingAbort(void (^body)(void)) {
  gLastSignal = 0;
  gAbortArmed = 1;
  if (sigsetjmp(gAbortJmp, 1) != 0) {
    gAbortArmed = 0;
    return 0;
  }
  body();
  gAbortArmed = 0;
  return 1;
}

static NSString *hexOf(const unsigned char *p, size_t n) {
  NSMutableString *s = [NSMutableString stringWithCapacity:n * 2];
  for (size_t i = 0; i < n; i++) [s appendFormat:@"%02x", p[i]];
  return s;
}

static NSString *sha256Of(const unsigned char *p, size_t n) {
  unsigned char d[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256(p, (CC_LONG)n, d);
  return hexOf(d, sizeof(d));
}

static NSString *sha256OfFile(const char *path) {
  NSData *data = [NSData dataWithContentsOfFile:@(path)];
  if (!data) return @"(unreadable)";
  return sha256Of(data.bytes, data.length);
}

// One captured operation, plus the Metal-side truth it should encode.
//
// Returns nil and reports how many records the case actually produced. A case
// that produced none is not necessarily broken -- see `gSilent` -- so the
// caller decides what to make of it; `*produced` is how it finds out.
/// One fixture from operation `index` of a capture that produced `wanted` of
/// them.
///
/// `wanted` is the caller's claim about the selector, not a tolerance. A case
/// that says one and produced two is refused here and recorded on the `multi`
/// list rather than being trimmed to its first record -- a selector whose
/// second operation went unrecorded is a wire record with no view and nothing
/// saying so.
static NSDictionary *caseFromCaptureAt(NSString *name, NSString *cls, NSString *sel,
                                       NSDictionary *expect, int index, int wanted,
                                       int *produced) {
  if (produced) *produced = gOpCount;
  if (gOpCount != wanted) {
    fprintf(stderr, "case %s produced %d operations; expected exactly %d\n",
            name.UTF8String, gOpCount, wanted);
    return nil;
  }
  const unsigned char *p = gArena + gOpOff[index];
  size_t len = gOpLen[index];
  return @{
    @"name" : name,
    @"class" : cls,
    @"selector" : sel,
    @"allocated_len" : @(len),
    @"buffer" : hexOf(p, len),
    @"sha256" : sha256Of(p, len),
    @"expect" : expect,
  };
}

static NSDictionary *caseFromCapture(NSString *name, NSString *cls, NSString *sel,
                                     NSDictionary *expect, int *produced) {
  return caseFromCaptureAt(name, cls, sel, expect, 0, 1, produced);
}

// Everything the Rust view should report, taken from the descriptor object so
// no Metal enum ordinal is transcribed by hand.
static NSDictionary *expectFromTextureDescriptor(MTLTextureDescriptor *d) {
  unsigned framebufferOnly = ((char (*)(id, SEL))objc_msgSend)(
      d, sel_getUid("framebufferOnly"));
  unsigned isDrawable = ((char (*)(id, SEL))objc_msgSend)(
      d, sel_getUid("isDrawable"));
  NSMutableDictionary *expect = [@{
    @"texture_type" : @((unsigned)d.textureType),
    @"usage" : @((unsigned)d.usage),
    @"pixel_format" : @((unsigned)d.pixelFormat),
    @"width" : @((unsigned long long)d.width),
    @"height" : @((unsigned long long)d.height),
    @"depth" : @((unsigned long long)d.depth),
    @"mipmap_level_count" : @((unsigned long long)d.mipmapLevelCount),
    @"sample_count" : @((unsigned long long)d.sampleCount),
    @"array_length" : @((unsigned long long)d.arrayLength),
    @"storage_mode" : @((unsigned)d.storageMode),
    @"allow_gpu_optimized_contents" : @(d.allowGPUOptimizedContents ? 1 : 0),
    @"framebuffer_only" : @(framebufferOnly),
    @"is_drawable" : @(isDrawable),
    @"hazard_tracking_mode" : @((unsigned)d.hazardTrackingMode),
    @"cpu_cache_mode" : @((unsigned)d.cpuCacheMode),
    @"compression_type" : @((unsigned)d.compressionType),
    // Carried on every texture case even though only the wide form has
    // somewhere to put them. The 32-byte body has no room for a swizzle, and a
    // reader comparing the two forms wants to see that the *same* descriptor
    // property reaches one and not the other.
    @"swizzle_red" : @((unsigned)d.swizzle.red),
    @"swizzle_green" : @((unsigned)d.swizzle.green),
    @"swizzle_blue" : @((unsigned)d.swizzle.blue),
    @"swizzle_alpha" : @((unsigned)d.swizzle.alpha),
  } mutableCopy];
  expect[@"force_resource_index"] = @(((char (*)(id, SEL))objc_msgSend)(
      d, sel_getUid("forceResourceIndex")) ? 1 : 0);
  expect[@"write_swizzle_enabled"] = @(((char (*)(id, SEL))objc_msgSend)(
      d, sel_getUid("writeSwizzleEnabled")) ? 1 : 0);
  expect[@"resource_index"] = @(((unsigned long long (*)(id, SEL))objc_msgSend)(
      d, sel_getUid("resourceIndex")));
  expect[@"protection_options"] = @(((unsigned long long (*)(id, SEL))objc_msgSend)(
      d, sel_getUid("protectionOptions")));
  expect[@"rotation"] = @(((unsigned long long (*)(id, SEL))objc_msgSend)(
      d, sel_getUid("rotation")));
  expect[@"sparse_surface_default_value"] =
      @(((unsigned long long (*)(id, SEL))objc_msgSend)(
          d, sel_getUid("sparseSurfaceDefaultValue")));
  return expect;
}

static MTLTextureDescriptor *baselineTexture(void) {
  MTLTextureDescriptor *d = [[MTLTextureDescriptor alloc] init];
  d.textureType = MTLTextureType2D;
  d.pixelFormat = MTLPixelFormatBGRA8Unorm;
  d.width = 0x1111;   // distinctive, so a byte landing anywhere is recognisable
  d.height = 0x2222;
  d.depth = 1;
  d.mipmapLevelCount = 1;
  d.sampleCount = 1;
  d.arrayLength = 1;
  d.storageMode = MTLStorageModePrivate;
  d.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
  return d;
}

static void addTextureCase(NSMutableArray *cases, id ser, id cap, NSString *name,
                           MTLTextureDescriptor *d) {
  arenaReset();
  ((unsigned (*)(id, SEL, id, id))objc_msgSend)(
      ser, sel_getUid("newTextureWithDescriptor:allocator:"), d, cap);
  NSDictionary *c = caseFromCapture(name, @"PGSerializer",
                                    @"newTextureWithDescriptor:allocator:",
                                    expectFromTextureDescriptor(d), NULL);
  if (c) [cases addObject:c];
}

/// Independently perturb the private descriptor properties that could account
/// for the remaining written fields. `prefix` distinguishes the serializer's
/// narrow and wide modes without changing the experiment.
static void addPrivateTextureCases(NSMutableArray *cases, id ser, id cap,
                                   NSString *prefix) {
  MTLTextureDescriptor *d = baselineTexture();
  ((void (*)(id, SEL, char))objc_msgSend)(
      d, sel_getUid("setForceResourceIndex:"), (char)1);
  addTextureCase(cases, ser, cap,
                 [prefix stringByAppendingString:@"_force_resource_index"], d);

  d = baselineTexture();
  ((void (*)(id, SEL, char))objc_msgSend)(
      d, sel_getUid("setWriteSwizzleEnabled:"), (char)1);
  addTextureCase(cases, ser, cap,
                 [prefix stringByAppendingString:@"_write_swizzle_enabled"], d);

  d = baselineTexture();
  ((void (*)(id, SEL, unsigned long long))objc_msgSend)(
      d, sel_getUid("setResourceIndex:"), 0x1122334455667788ULL);
  addTextureCase(cases, ser, cap,
                 [prefix stringByAppendingString:@"_resource_index"], d);

  d = baselineTexture();
  ((void (*)(id, SEL, unsigned long long))objc_msgSend)(
      d, sel_getUid("setProtectionOptions:"), 0x8877665544332211ULL);
  addTextureCase(cases, ser, cap,
                 [prefix stringByAppendingString:@"_protection_options"], d);

  d = baselineTexture();
  ((void (*)(id, SEL, unsigned long long))objc_msgSend)(
      d, sel_getUid("setRotation:"), 3);
  addTextureCase(cases, ser, cap,
                 [prefix stringByAppendingString:@"_rotation"], d);

  d = baselineTexture();
  ((void (*)(id, SEL, unsigned long long))objc_msgSend)(
      d, sel_getUid("setSparseSurfaceDefaultValue:"), 0x1234);
  addTextureCase(cases, ser, cap,
                 [prefix stringByAppendingString:@"_sparse_surface_default_value"], d);
}

// Perturbation sweep: baseline, then one property changed per case. Each is a
// separate fixture so a layout error shows up as one failing case naming the
// field, rather than as a diff a reader has to bisect.
/// Drive `body` with one serializer capability forced on, then restore it.
///
/// Several selector families are gated on a `-supportsX` flag, and a capture
/// taken with the flag off records them as *silent* -- which is a statement
/// about this harness, not about Apple, and would become a false
/// `EMITS_NO_OPERATION` row in the manifest. Forcing the flag is what separates
/// "the serializer writes nothing for this call" from "the serializer was not
/// asked to".
///
/// Restoring matters as much as setting. A serializer left in a different
/// capability state changes what every later case emits, and that failure looks
/// like a layout error somewhere else entirely -- which is the same reason
/// `capabilityCases` inverts and restores rather than re-writing.
static void withCapability(id ser, NSString *flag, void (^body)(void)) {
  SEL getSel = sel_getUid([NSString stringWithFormat:@"supports%@", flag].UTF8String);
  SEL setSel = sel_getUid([NSString stringWithFormat:@"setSupports%@:", flag].UTF8String);
  if (![ser respondsToSelector:getSel] || ![ser respondsToSelector:setSel]) {
    fprintf(stderr, "capability %s is absent on this serializer; skipping\n",
            flag.UTF8String);
    return;
  }
  char was = ((char (*)(id, SEL))objc_msgSend)(ser, getSel);
  fprintf(stderr, "capability %s was %d, forcing on\n", flag.UTF8String, (int)was);
  ((void (*)(id, SEL, char))objc_msgSend)(ser, setSel, (char)1);
  body();
  ((void (*)(id, SEL, char))objc_msgSend)(ser, setSel, was);
}

static NSArray *textureCases(id ser, id cap) {
  NSMutableArray *cases = [NSMutableArray array];
  MTLTextureDescriptor *d;

  addTextureCase(cases, ser, cap, @"texture_baseline", baselineTexture());

  // The same baseline descriptor with `-supportsTextureDescriptor2` on.
  //
  // `-serializeTextureDescriptor2:textureDescriptor:` declares a *different*
  // struct from its unsuffixed sibling: `b4b1b1b1b1b16IIIISSSSQCCCC` against
  // `b4b1b1b1b1b8b16IIISSSSQ`. The `b8` usage leaves the packed word, a fourth
  // `I` appears, and four `C` bytes trail -- 40 bytes against 32.
  //
  // Whether that second layout ever reaches the *wire* is the question, and it
  // is one this capture can answer rather than reason about: if the creation
  // record grows or its fields shift under this flag, every texture a guest
  // that negotiated the capability creates is mis-decoded by a reader built for
  // the 32-byte body. If it does not, the second struct is a host-side
  // serialization helper and nothing on the wire changes.
  withCapability(ser, @"TextureDescriptor2", ^{
    addTextureCase(cases, ser, cap, @"texture_baseline_descriptor2", baselineTexture());
  });

  // The wide creation record, and the flag that actually selects it.
  //
  // The comment above asked whether the second layout reaches the wire and this
  // answers yes -- but not under the flag its name suggests. Under
  // `-setSupportsSwizzledTextures:` this selector emits a **different opcode**,
  // 0x34 instead of 1, whose body is the 40-byte `b4b1b1b1b1b16IIIISSSSQCCCC`
  // form. `TextureDescriptor2` leaves *this* record alone (the case above is
  // byte-identical to the baseline) and switches the other four instead, which
  // is why one negative result about one selector could not settle the family.
  //
  // Two cases, because one cannot. The four trailing bytes read 02 03 04 05 on
  // the baseline, which is `MTLTextureSwizzleChannelsDefault` -- Red, Green,
  // Blue, Alpha -- and against a single fixture that is indistinguishable from
  // a constant the serializer writes. The permuted case moves all four to
  // values no default could produce.
  withCapability(ser, @"SwizzledTextures", ^{
    addTextureCase(cases, ser, cap, @"texture_swizzled", baselineTexture());

    MTLTextureDescriptor *sw = baselineTexture();
    sw.swizzle = MTLTextureSwizzleChannelsMake(
        MTLTextureSwizzleAlpha, MTLTextureSwizzleZero, MTLTextureSwizzleOne,
        MTLTextureSwizzleRed);
    addTextureCase(cases, ser, cap, @"texture_swizzled_permuted", sw);

    MTLTextureDescriptor *framebuffer = baselineTexture();
    ((void (*)(id, SEL, char))objc_msgSend)(
        framebuffer, sel_getUid("setFramebufferOnly:"), (char)1);
    addTextureCase(cases, ser, cap, @"texture_swizzled_framebuffer_only", framebuffer);
    MTLTextureDescriptor *drawable = baselineTexture();
    ((void (*)(id, SEL, char))objc_msgSend)(
        drawable, sel_getUid("setIsDrawable:"), (char)1);
    addTextureCase(cases, ser, cap, @"texture_swizzled_is_drawable", drawable);
    addPrivateTextureCases(cases, ser, cap, @"texture_swizzled");
  });

  addPrivateTextureCases(cases, ser, cap, @"texture");

  d = baselineTexture(); d.width = 0x3333;
  addTextureCase(cases, ser, cap, @"texture_width", d);
  d = baselineTexture(); d.height = 0x4444;
  addTextureCase(cases, ser, cap, @"texture_height", d);
  d = baselineTexture(); d.textureType = MTLTextureType3D; d.depth = 0x55;
  addTextureCase(cases, ser, cap, @"texture_depth_3d", d);
  d = baselineTexture(); d.pixelFormat = MTLPixelFormatRGBA8Unorm;
  addTextureCase(cases, ser, cap, @"texture_format_rgba8", d);
  d = baselineTexture(); d.pixelFormat = MTLPixelFormatR8Unorm;
  addTextureCase(cases, ser, cap, @"texture_format_r8", d);
  d = baselineTexture(); d.mipmapLevelCount = 7;
  addTextureCase(cases, ser, cap, @"texture_mips", d);
  d = baselineTexture(); d.textureType = MTLTextureType2DMultisample; d.sampleCount = 4;
  addTextureCase(cases, ser, cap, @"texture_samples", d);
  d = baselineTexture(); d.textureType = MTLTextureType2DArray; d.arrayLength = 6;
  addTextureCase(cases, ser, cap, @"texture_array", d);
  d = baselineTexture(); d.textureType = MTLTextureTypeCube;
  addTextureCase(cases, ser, cap, @"texture_cube", d);
  d = baselineTexture(); d.storageMode = MTLStorageModeShared;
  addTextureCase(cases, ser, cap, @"texture_storage_shared", d);
  d = baselineTexture(); d.storageMode = MTLStorageModeManaged;
  addTextureCase(cases, ser, cap, @"texture_storage_managed", d);
  d = baselineTexture(); d.usage = MTLTextureUsageShaderRead;
  addTextureCase(cases, ser, cap, @"texture_usage_read", d);
  d = baselineTexture(); d.usage = MTLTextureUsageShaderWrite;
  addTextureCase(cases, ser, cap, @"texture_usage_write", d);
  d = baselineTexture(); d.usage = MTLTextureUsageRenderTarget;
  addTextureCase(cases, ser, cap, @"texture_usage_rendertarget", d);

  // These properties independently attribute the remaining packed descriptor
  // fields and the resource-options aggregate.
  // Each defaults to something other than the value set here, so each is a real
  // perturbation rather than a restatement of the baseline.
  d = baselineTexture(); d.allowGPUOptimizedContents = NO;
  addTextureCase(cases, ser, cap, @"texture_no_gpu_optimized_contents", d);
  d = baselineTexture();
  ((void (*)(id, SEL, char))objc_msgSend)(d, sel_getUid("setFramebufferOnly:"), (char)1);
  addTextureCase(cases, ser, cap, @"texture_framebuffer_only", d);
  d = baselineTexture();
  ((void (*)(id, SEL, char))objc_msgSend)(d, sel_getUid("setIsDrawable:"), (char)1);
  addTextureCase(cases, ser, cap, @"texture_is_drawable", d);
  d = baselineTexture(); d.hazardTrackingMode = MTLHazardTrackingModeUntracked;
  addTextureCase(cases, ser, cap, @"texture_hazard_untracked", d);
  d = baselineTexture(); d.hazardTrackingMode = MTLHazardTrackingModeTracked;
  addTextureCase(cases, ser, cap, @"texture_hazard_tracked", d);
  // Write-combined needs a storage mode the CPU can reach; Private has no cache
  // mode to speak of and Metal is entitled to normalize the pair away.
  d = baselineTexture();
  d.storageMode = MTLStorageModeShared;
  d.cpuCacheMode = MTLCPUCacheModeWriteCombined;
  addTextureCase(cases, ser, cap, @"texture_cpu_cache_write_combined", d);
  d = baselineTexture(); d.compressionType = MTLTextureCompressionTypeLossy;
  addTextureCase(cases, ser, cap, @"texture_compression_lossy", d);

  return cases;
}

// --- Encoder records -------------------------------------------------------

/// Build a render encoder over a pass with one colour attachment.
///
/// Creating it emits records of its own (the render-pass record, and an 8-byte
/// allocation the serializer leaves unwritten until `endEncoding` — the segment
/// header). Callers that want a single command's record reset the arena after
/// this returns, so those never land in a case.
static id makeRenderEncoder(id ser, id stream) {
  MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
  rp.colorAttachments[0].texture = (id<MTLTexture>)[[StubTexture alloc] init];
  rp.colorAttachments[0].loadAction = MTLLoadActionClear;
  rp.colorAttachments[0].storeAction = MTLStoreActionStore;
  rp.colorAttachments[0].clearColor = MTLClearColorMake(0.25, 0.5, 0.75, 1.0);
  return ((id (*)(id, SEL, id, id, id))objc_msgSend)(
      [objc_getClass("PGSerializerRenderCommandEncoder") alloc],
      sel_getUid("initWithCommandBuffer:descriptor:serializer:"), stream, rp, ser);
}

/// Build a blit encoder.
///
/// Same designated initializer as the render encoder — the class differs, the
/// pass descriptor differs, and nothing else does.
static id makeBlitEncoder(id ser, id stream) {
  MTLBlitPassDescriptor *bp = [MTLBlitPassDescriptor blitPassDescriptor];
  return ((id (*)(id, SEL, id, id, id))objc_msgSend)(
      [objc_getClass("PGSerializerBlitCommandEncoder") alloc],
      sel_getUid("initWithCommandBuffer:descriptor:serializer:"), stream, bp, ser);
}

/// Selectors that aborted inside Apple's own encoder rather than serializing.
static NSMutableArray *gUnsupported;

/// Selectors that returned normally and emitted no operation at all.
///
/// A third outcome, distinct from both "here are its bytes" and "Apple
/// refused": the selector ran and wrote no record, so the guest can issue it
/// and no wire operation exists to decode. Recording it here makes the
/// exclusion evidence that is re-measured every capture rather than a claim
/// someone made once.
///
/// **A silence measured at the default capability state is not that claim**,
/// and this doc used to say it was — it named `fillBuffer:range:pattern4:` as
/// the worked example of a selector that writes nothing, which is wrong.
/// `fillBuffer:range:pattern4:` emits once `-setSupportsBlitEncoderSPI:` is on,
/// and it is one of nineteen selectors this list held that a capability
/// unlocks. Sixteen flags default off, so *any* entry here can be one of those.
///
/// `silent_with_every_capability` in the output is the same list measured with
/// all sixteen forced on, and the difference between the two is the set that
/// may not claim `EMITS_NO_OPERATION`. Those nineteen were found the moment
/// that pass first ran; before it, each had to be guessed at one family at a
/// time.
///
/// This is also how a *broken* case announces itself: a stub that answered zero
/// where the serializer wanted a resource produces the same silence. The two
/// are told apart by the `note: stubbed` lines the case printed, which is why
/// each case prints its own name first.
static NSMutableArray *gSilent;

/// Selectors that faulted rather than returning or asserting.
///
/// A fourth outcome, and the only one that is **not** evidence about Apple. A
/// SIGSEGV inside the serializer means it dereferenced something a stub handed
/// it -- `heapTextureDescriptorSizeAndAlign:sizeAndAlign:` asks its heap for
/// `descriptorPrivate` and follows the answer, and a forwarding stub answers
/// zero. So these get `Coverage::Unimplemented`, not an exclusion: the
/// selector's behaviour is unmeasured, and the missing piece is a stub rather
/// than a fact.
///
/// Catching them at all is the point. The JSON is written once at the end, so
/// before this a single fault threw away every case in the run -- which is the
/// same failure the SIGABRT handler was written for.
static NSMutableArray *gCrashed;
/// Selectors that emitted a record count no case claimed. See `captureCase`.
static NSMutableArray *gMulti;

/// Set for the sweep pass only: force every capability on before any case runs.
///
/// See `onePass`. The pass's fixtures are thrown away -- capability state can
/// change what a record *contains*, and the fixtures this crate pins must come
/// from a serializer in its default state. What is kept is its `silent` list,
/// which is the set of selectors that emit nothing even with everything on.
static BOOL gForceAllCapabilities;

/// Set for an attribution pass: force exactly this one capability on.
///
/// The sweep above answers "is this selector gated on *something*". It does not
/// answer "on what", and that second question is the one an agent has to settle
/// before it can add a case: `withCapability` takes one flag name. Answering it
/// by trying flags in turn is guessing with a fast oracle behind it, which is
/// still guessing -- and it is the same shape as the guess the sweep pass was
/// written to delete.
///
/// So each flag gets its own pass, and the difference between its `silent` list
/// and the default one is the set that flag alone unlocks. Exclusive with
/// `gForceAllCapabilities`; `onePass` forces one or the other, never both.
static NSString *gForceOneCapability;

/// Drive one selector, sort the outcome into the run's four buckets.
///
/// The caller has already announced the case, because on the encoder path the
/// announcement has to precede building the encoder. Returns whether the
/// invocation completed: after `NO` the callee is mid-unwind and a caller with
/// an object to tear down must not touch it.
static BOOL captureCase(NSMutableArray *cases, NSString *cls, NSString *name,
                        NSString *sel, NSDictionary *expect, int wanted,
                        NSArray<NSDictionary *> *expects, void (^invoke)(void)) {
  arenaReset();
  if (!runSurvivingAbort(invoke)) {
    int sig = (int)gLastSignal;
    if (sig == SIGABRT) {
      fprintf(stderr, "case %s: -[%s] asserted inside the serializer; recorded as "
                      "unsupported\n",
              name.UTF8String, sel.UTF8String);
      if (gRecordOutcomes) [gUnsupported addObject:@{
        @"name" : name,
        @"class" : cls,
        @"selector" : sel,
        @"reason" : @"the serializer fails an assertion instead of emitting an operation",
      }];
    } else {
      fprintf(stderr, "case %s: -[%s] faulted (signal %d); the stub could not answer "
                      "something the serializer dereferenced, so this selector is "
                      "UNMEASURED rather than unsupported\n",
              name.UTF8String, sel.UTF8String, sig);
      if (gRecordOutcomes) [gCrashed addObject:@{
        @"name" : name,
        @"class" : cls,
        @"selector" : sel,
        @"signal" : @(sig),
        @"reason" : @"the harness could not drive it; this says nothing about Apple",
      }];
    }
    return NO;
  }
  // Hex the records before any teardown: an encoder's -endEncoding backfills
  // the segment header into arena bytes this case has already read.
  int produced = 0;
  for (int i = 0; i < wanted; i++) {
    NSString *slot = wanted == 1 ? name : [NSString stringWithFormat:@"%@_%d", name, i];
    NSDictionary *expect_i = wanted == 1 ? expect : expects[i];
    NSDictionary *c = caseFromCaptureAt(slot, cls, sel, expect_i, i, wanted, &produced);
    if (c) [cases addObject:c];
  }
  if (!gRecordOutcomes) {
    return YES;
  }
  if (produced == 0) {
    [gSilent addObject:@{
      @"name" : name,
      @"class" : cls,
      @"selector" : sel,
      @"reason" : @"the serializer returns without emitting an operation",
    }];
  } else if (produced != wanted) {
    // The fifth outcome, and the one that used to be no outcome at all: a
    // selector that emitted a different number of records than the case
    // claimed vanished from every list, which is indistinguishable from a
    // selector nobody drove. Recorded so the manifest has to answer for it.
    [gMulti addObject:@{
      @"name" : name,
      @"class" : cls,
      @"selector" : sel,
      @"operations" : @(produced),
      @"expected" : @(wanted),
      @"reason" : @"the selector emits a different number of records than the "
                  @"case claimed; it needs one case per operation",
    }];
  }
  return YES;
}

/// Capture exactly the record one encoder call emits.
///
/// A fresh encoder per case keeps the pass record out of the capture without
/// depending on which allocation index a command lands at. `make` names the
/// encoder class so a refusal is recorded against the class that refused; a
/// blit selector attributed to the render encoder would send the next reader to
/// the wrong 152 selectors.
static void addCaseOnEncoder(NSMutableArray *cases, NSString *cls, id (^make)(void),
                             NSString *name, NSString *sel, NSDictionary *expect,
                             void (^invoke)(id enc)) {
  // The stub classes print every selector they were asked for and did not
  // model. Those notes are unattributable on their own -- a case that emits
  // nothing usually did so because one of them answered zero -- so the case
  // announces itself first and the notes that follow belong to it.
  fprintf(stderr, "case %s\n", name.UTF8String);
  id enc = make();
  if (!enc) {
    fprintf(stderr, "case %s: -[%s init] returned nil\n", name.UTF8String,
            cls.UTF8String);
    return;
  }
  if (!captureCase(cases, cls, name, sel, expect, 1, nil, ^{
        invoke(enc);
      })) {
    return; // the encoder is mid-unwind; drop it rather than ending it
  }
  // _MTLCommandEncoder asserts in -dealloc if it was never ended.
  ((void (*)(id, SEL))objc_msgSend)(enc, sel_getUid("endEncoding"));
}

/// Capture each of the several records one encoder call emits.
///
/// AGENTS.md's rule is one operation per case, and the alternative it names is
/// "a wrapper that splits them". This is that wrapper: the selector is driven
/// once, and one fixture is recorded per operation, named `<name>_0`, `<name>_1`
/// and so on with its own expectation. The count is asserted, not discovered --
/// a selector that emitted a different number lands on `multi` instead, because
/// the claim being tested is that this selector writes exactly these records.
static void addEncoderCaseSplit(NSMutableArray *cases, NSString *cls, id (^make)(void),
                                NSString *name, NSString *sel,
                                NSArray<NSDictionary *> *expects, void (^invoke)(id enc)) {
  fprintf(stderr, "case %s\n", name.UTF8String);
  id enc = make();
  if (!enc) {
    fprintf(stderr, "case %s: -[%s init] returned nil\n", name.UTF8String,
            cls.UTF8String);
    return;
  }
  if (!captureCase(cases, cls, name, sel, nil, (int)expects.count, expects, ^{
        invoke(enc);
      })) {
    return;
  }
  ((void (*)(id, SEL))objc_msgSend)(enc, sel_getUid("endEncoding"));
}

/// Capture the record a call on the bare serializer emits.
///
/// No encoder to build or end, so the announcement and the drive are adjacent.
static void addSerializerCase(NSMutableArray *cases, NSString *name, NSString *sel,
                              NSDictionary *expect, void (^invoke)(void)) {
  fprintf(stderr, "case %s\n", name.UTF8String);
  captureCase(cases, @"PGSerializer", name, sel, expect, 1, nil, invoke);
}

static void addEncoderCase(NSMutableArray *cases, id ser, id stream, NSString *name,
                           NSString *sel, NSDictionary *expect, void (^invoke)(id enc)) {
  addCaseOnEncoder(cases, @"PGSerializerRenderCommandEncoder",
                   ^id {
                     return makeRenderEncoder(ser, stream);
                   },
                   name, sel, expect, invoke);
}

/// Build a render-pass case's expectations out of the descriptor itself.
///
/// Every value is read back off `rp` after the case configured it, so a
/// property Metal normalized is expected at the value Metal kept rather than at
/// the one the case asked for. Object refs are the one exception and come from
/// the caller: a stub's ref is a property of the stub, not of the descriptor.
static NSDictionary *expectFromRenderPass(MTLRenderPassDescriptor *rp,
                                          NSDictionary *refs) {
  MTLRenderPassColorAttachmentDescriptor *c0 = rp.colorAttachments[0];
  NSMutableDictionary *e = [NSMutableDictionary dictionaryWithDictionary:@{
    @"color0_load_action" : @((unsigned)c0.loadAction),
    @"color0_store_action" : @((unsigned)c0.storeAction),
    @"color0_level" : @((unsigned)c0.level),
    @"color0_slice" : @((unsigned)c0.slice),
    @"color0_depth_plane" : @((unsigned)c0.depthPlane),
    @"color0_resolve_level" : @((unsigned)c0.resolveLevel),
    @"color0_resolve_slice" : @((unsigned)c0.resolveSlice),
    @"color0_resolve_depth_plane" : @((unsigned)c0.resolveDepthPlane),
    @"color0_clear_red" : @(c0.clearColor.red),
    @"color0_clear_green" : @(c0.clearColor.green),
    @"color0_clear_blue" : @(c0.clearColor.blue),
    @"color0_clear_alpha" : @(c0.clearColor.alpha),
    @"color0_store_action_options" : @((unsigned)c0.storeActionOptions),
    @"depth_resolve_filter" : @((unsigned)rp.depthAttachment.depthResolveFilter),
    @"stencil_resolve_filter" : @((unsigned)rp.stencilAttachment.stencilResolveFilter),
    @"depth_load_action" : @((unsigned)rp.depthAttachment.loadAction),
    @"depth_store_action" : @((unsigned)rp.depthAttachment.storeAction),
    @"depth_level" : @((unsigned)rp.depthAttachment.level),
    @"clear_depth" : @(rp.depthAttachment.clearDepth),
    @"stencil_load_action" : @((unsigned)rp.stencilAttachment.loadAction),
    @"stencil_store_action" : @((unsigned)rp.stencilAttachment.storeAction),
    @"clear_stencil" : @((unsigned)rp.stencilAttachment.clearStencil),
    @"render_target_width" : @((unsigned)rp.renderTargetWidth),
    @"render_target_height" : @((unsigned)rp.renderTargetHeight),
    @"render_target_array_length" : @((unsigned)rp.renderTargetArrayLength),
    @"default_raster_sample_count" : @((unsigned)rp.defaultRasterSampleCount),
    @"imageblock_sample_length" : @((unsigned)rp.imageblockSampleLength),
    @"threadgroup_memory_length" : @((unsigned)rp.threadgroupMemoryLength),
    @"tile_width" : @((unsigned)rp.tileWidth),
    @"tile_height" : @((unsigned)rp.tileHeight),
  }];
  [e addEntriesFromDictionary:refs];
  return e;
}

/// Drive `writeDescriptor` over one configured render pass descriptor.
///
/// The record this emits *is* the pass descriptor, so perturbing a field means
/// building a different **encoder** rather than calling a different selector:
/// the descriptor is consumed by
/// `initWithCommandBuffer:descriptor:serializer:` and `writeDescriptor`
/// re-emits whatever that encoder was built with. That is also why the record
/// was never captured before this — `makeRenderEncoder`'s own doc says
/// constructing an encoder emits it, and every case resets the arena
/// afterwards, so the one record no case could reach was the one every case
/// produced.
///
/// The baseline is a single colour attachment, cleared to four distinguishable
/// components, and each case moves exactly one property off it.
/// `records` is how many operations this descriptor is claimed to serialize as.
/// Two capabilities make it two rather than one, and asserting the count is
/// what keeps that from being discovered silently -- a case whose descriptor
/// grows a second record without saying so lands on `multi` instead.
static void addRenderPassCaseN(NSMutableArray *cases, id ser, id stream, NSString *name,
                               NSDictionary *refs, int records,
                               void (^configure)(MTLRenderPassDescriptor *rp)) {
  MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
  rp.colorAttachments[0].texture = (id<MTLTexture>)[[StubTexture alloc] init];
  rp.colorAttachments[0].loadAction = MTLLoadActionClear;
  rp.colorAttachments[0].storeAction = MTLStoreActionStore;
  rp.colorAttachments[0].clearColor = MTLClearColorMake(0.25, 0.5, 0.75, 1.0);
  configure(rp);
  NSDictionary *expect = expectFromRenderPass(rp, refs);
  id (^make)(void) = ^id {
    return ((id (*)(id, SEL, id, id, id))objc_msgSend)(
        [objc_getClass("PGSerializerRenderCommandEncoder") alloc],
        sel_getUid("initWithCommandBuffer:descriptor:serializer:"), stream, rp, ser);
  };
  void (^invoke)(id) = ^(id enc) {
    (void)((char (*)(id, SEL))objc_msgSend)(enc, sel_getUid("writeDescriptor"));
  };
  if (records == 1) {
    addCaseOnEncoder(cases, @"PGSerializerRenderCommandEncoder", make, name,
                     @"writeDescriptor", expect, invoke);
    return;
  }
  // Every record of a split descriptor describes the same descriptor, so they
  // share one expectation dictionary; the fixture test dispatches on the
  // opcode each record carries rather than on its index.
  NSMutableArray *expects = [NSMutableArray array];
  for (int i = 0; i < records; i++) [expects addObject:expect];
  addEncoderCaseSplit(cases, @"PGSerializerRenderCommandEncoder", make, name,
                      @"writeDescriptor", expects, invoke);
}

static void addRenderPassCase(NSMutableArray *cases, id ser, id stream, NSString *name,
                              NSDictionary *refs,
                              void (^configure)(MTLRenderPassDescriptor *rp)) {
  addRenderPassCaseN(cases, ser, stream, name, refs, 1, configure);
}

/// Drive one shader stage's five ray-tracing binds.
///
/// `setXAccelerationStructure:atBufferIndex:` and the visible / intersection
/// function tables exist identically on the vertex, fragment, mesh, object and
/// tile stages -- twenty selectors that differ only in an infix. Written once
/// and parameterised, because twenty hand-copied blocks is twenty chances to
/// leave a stage's own selector name behind in one of them.
///
/// The tile stage is deliberately not driven through here: it lives beside the
/// rest of the tile family under `withCapability`, and all five of its forms
/// are refused by the serializer.
static void addRayTracingBindCases(NSMutableArray *cases, id ser, id stream,
                                   NSString *stage, id accel, id visFnTable,
                                   id isectFnTable, unsigned base) {
  NSString *lower = stage.lowercaseString;
  id visTables[2] = {visFnTable, visFnTable};
  id isectTables[2] = {isectFnTable, isectFnTable};
  const id *visArray = visTables;
  const id *isectArray = isectTables;

  NSString *sel = [NSString stringWithFormat:@"set%@AccelerationStructure:atBufferIndex:", stage];
  addEncoderCase(cases, ser, stream,
                 [NSString stringWithFormat:@"render_set_%@_acceleration_structure", lower], sel,
                 @{@"acceleration_structure_ref" : @(STUB_ACCEL_STRUCT_REF),
                   @"index" : @(base)},
                 ^(id enc) {
                   ((void (*)(id, SEL, id, unsigned long))objc_msgSend)(
                       enc, sel_getUid(sel.UTF8String), accel, base);
                 });

  sel = [NSString stringWithFormat:@"set%@VisibleFunctionTable:atBufferIndex:", stage];
  addEncoderCase(cases, ser, stream,
                 [NSString stringWithFormat:@"render_set_%@_visible_function_table", lower], sel,
                 @{@"visible_function_table_ref" : @(STUB_VISIBLE_FN_TABLE_REF),
                   @"index" : @(base + 1)},
                 ^(id enc) {
                   ((void (*)(id, SEL, id, unsigned long))objc_msgSend)(
                       enc, sel_getUid(sel.UTF8String), visFnTable, base + 1);
                 });

  sel = [NSString stringWithFormat:@"set%@VisibleFunctionTables:withBufferRange:", stage];
  addEncoderCase(
      cases, ser, stream,
      [NSString stringWithFormat:@"render_set_%@_visible_function_tables_range", lower], sel,
      @{@"visible_function_table_ref" : @(STUB_VISIBLE_FN_TABLE_REF),
        @"first" : @(base + 2), @"count" : @2},
      ^(id enc) {
        ((void (*)(id, SEL, const id *, NSRange))objc_msgSend)(
            enc, sel_getUid(sel.UTF8String), visArray, NSMakeRange(base + 2, 2));
      });

  sel = [NSString stringWithFormat:@"set%@IntersectionFunctionTable:atBufferIndex:", stage];
  addEncoderCase(cases, ser, stream,
                 [NSString stringWithFormat:@"render_set_%@_intersection_function_table", lower],
                 sel,
                 @{@"intersection_function_table_ref" : @(STUB_INTERSECTION_FN_TABLE_REF),
                   @"index" : @(base + 4)},
                 ^(id enc) {
                   ((void (*)(id, SEL, id, unsigned long))objc_msgSend)(
                       enc, sel_getUid(sel.UTF8String), isectFnTable, base + 4);
                 });

  sel = [NSString stringWithFormat:@"set%@IntersectionFunctionTables:withBufferRange:", stage];
  addEncoderCase(
      cases, ser, stream,
      [NSString stringWithFormat:@"render_set_%@_intersection_function_tables_range", lower], sel,
      @{@"intersection_function_table_ref" : @(STUB_INTERSECTION_FN_TABLE_REF),
        @"first" : @(base + 5), @"count" : @2},
      ^(id enc) {
        ((void (*)(id, SEL, const id *, NSRange))objc_msgSend)(
            enc, sel_getUid(sel.UTF8String), isectArray, NSMakeRange(base + 5, 2));
      });
}

/// Build a compute encoder. Same designated initializer again.
static id makeComputeEncoder(id ser, id stream) {
  MTLComputePassDescriptor *cp = [MTLComputePassDescriptor computePassDescriptor];
  return ((id (*)(id, SEL, id, id, id))objc_msgSend)(
      [objc_getClass("PGSerializerComputeCommandEncoder") alloc],
      sel_getUid("initWithCommandBuffer:descriptor:serializer:"), stream, cp, ser);
}

static void addComputeCase(NSMutableArray *cases, id ser, id stream, NSString *name,
                           NSString *sel, NSDictionary *expect, void (^invoke)(id enc)) {
  addCaseOnEncoder(cases, @"PGSerializerComputeCommandEncoder",
                   ^id {
                     return makeComputeEncoder(ser, stream);
                   },
                   name, sel, expect, invoke);
}

static void addBlitCase(NSMutableArray *cases, id ser, id stream, NSString *name,
                        NSString *sel, NSDictionary *expect, void (^invoke)(id enc)) {
  addCaseOnEncoder(cases, @"PGSerializerBlitCommandEncoder",
                   ^id {
                     return makeBlitEncoder(ser, stream);
                   },
                   name, sel, expect, invoke);
}

static NSArray *encoderCases(id ser) {
  NSMutableArray *cases = [NSMutableArray array];
  id stream = [[CaptureCommandStream alloc] init];

  addEncoderCase(cases, ser, stream, @"render_draw_primitives",
                 @"drawPrimitives:vertexStart:vertexCount:",
                 @{@"primitive_type" : @(MTLPrimitiveTypeTriangle),
                   @"vertex_start" : @7,
                   @"vertex_count" : @11},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long, unsigned long))objc_msgSend)(
                       enc, sel_getUid("drawPrimitives:vertexStart:vertexCount:"),
                       MTLPrimitiveTypeTriangle, 7, 11);
                 });

  addEncoderCase(cases, ser, stream, @"render_draw_primitives_strip",
                 @"drawPrimitives:vertexStart:vertexCount:",
                 @{@"primitive_type" : @(MTLPrimitiveTypeTriangleStrip),
                   @"vertex_start" : @2,
                   @"vertex_count" : @5},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long, unsigned long))objc_msgSend)(
                       enc, sel_getUid("drawPrimitives:vertexStart:vertexCount:"),
                       MTLPrimitiveTypeTriangleStrip, 2, 5);
                 });

  // The opcode reims-vgpu has no constant for; see ops::render.
  addEncoderCase(cases, ser, stream, @"render_draw_primitives_instanced_base",
                 @"drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:",
                 @{@"primitive_type" : @(MTLPrimitiveTypeTriangle),
                   @"vertex_start" : @1,
                   @"vertex_count" : @2,
                   @"instance_count" : @3,
                   @"base_instance" : @4},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long, unsigned long,
                              unsigned long, unsigned long))objc_msgSend)(
                       enc,
                       sel_getUid("drawPrimitives:vertexStart:vertexCount:instanceCount:"
                                  "baseInstance:"),
                       MTLPrimitiveTypeTriangle, 1, 2, 3, 4);
                 });

  // Segment framing, driven on this class so the byte at `+4` can be shown to
  // be a *type* rather than a constant: the blit encoder writes 2 there and
  // this one writes something else, from the same code with the same
  // arguments. See `ops::segment`.
  addEncoderCase(cases, ser, stream, @"render_begin_segment",
                 @"beginSegment:protectionOptions:",
                 @{@"flag" : @1, @"protection_options" : @0x33}, ^(id enc) {
                   ((void (*)(id, SEL, char, unsigned long))objc_msgSend)(
                       enc, sel_getUid("beginSegment:protectionOptions:"), 1, 0x33);
                 });

  // The three draws above all fit their arguments in 16 bits. These repeat each
  // one with every argument above 0xffff, which is the experiment that decides
  // whether the serializer truncates, refuses, or switches to another encoding.
  addEncoderCase(cases, ser, stream, @"render_draw_primitives_wide",
                 @"drawPrimitives:vertexStart:vertexCount:",
                 @{@"primitive_type" : @(MTLPrimitiveTypeTriangle),
                   @"vertex_start" : @0x11111,
                   @"vertex_count" : @0x22222},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long, unsigned long))objc_msgSend)(
                       enc, sel_getUid("drawPrimitives:vertexStart:vertexCount:"),
                       MTLPrimitiveTypeTriangle, 0x11111, 0x22222);
                 });

  addEncoderCase(cases, ser, stream, @"render_draw_primitives_instanced",
                 @"drawPrimitives:vertexStart:vertexCount:instanceCount:",
                 @{@"primitive_type" : @(MTLPrimitiveTypeTriangleStrip),
                   @"vertex_start" : @0x1111,
                   @"vertex_count" : @0x2222,
                   @"instance_count" : @0x3333},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long, unsigned long,
                              unsigned long))objc_msgSend)(
                       enc,
                       sel_getUid("drawPrimitives:vertexStart:vertexCount:instanceCount:"),
                       MTLPrimitiveTypeTriangleStrip, 0x1111, 0x2222, 0x3333);
                 });

  addEncoderCase(cases, ser, stream, @"render_draw_primitives_instanced_wide",
                 @"drawPrimitives:vertexStart:vertexCount:instanceCount:",
                 @{@"primitive_type" : @(MTLPrimitiveTypeTriangleStrip),
                   @"vertex_start" : @0x11111,
                   @"vertex_count" : @0x22222,
                   @"instance_count" : @0x33333},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long, unsigned long,
                              unsigned long))objc_msgSend)(
                       enc,
                       sel_getUid("drawPrimitives:vertexStart:vertexCount:instanceCount:"),
                       MTLPrimitiveTypeTriangleStrip, 0x11111, 0x22222, 0x33333);
                 });

  addEncoderCase(cases, ser, stream, @"render_draw_primitives_instanced_base_wide",
                 @"drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:",
                 @{@"primitive_type" : @(MTLPrimitiveTypeTriangle),
                   @"vertex_start" : @0x11111,
                   @"vertex_count" : @0x22222,
                   @"instance_count" : @0x33333,
                   @"base_instance" : @0x44444},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long, unsigned long,
                              unsigned long, unsigned long))objc_msgSend)(
                       enc,
                       sel_getUid("drawPrimitives:vertexStart:vertexCount:instanceCount:"
                                  "baseInstance:"),
                       MTLPrimitiveTypeTriangle, 0x11111, 0x22222, 0x33333, 0x44444);
                 });

  // Where the switch happens, one argument at a time. `0xffff` is the largest
  // value a 16-bit field holds and `0x10000` the smallest it does not, so this
  // pair brackets the boundary rather than assuming it, and each later case
  // moves exactly one argument across it so the choice is attributable.
  addEncoderCase(cases, ser, stream, @"render_draw_primitives_count_at_16bit_max",
                 @"drawPrimitives:vertexStart:vertexCount:",
                 @{@"primitive_type" : @(MTLPrimitiveTypeTriangle),
                   @"vertex_start" : @0,
                   @"vertex_count" : @0xffff},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long, unsigned long))objc_msgSend)(
                       enc, sel_getUid("drawPrimitives:vertexStart:vertexCount:"),
                       MTLPrimitiveTypeTriangle, 0, 0xffff);
                 });

  addEncoderCase(cases, ser, stream, @"render_draw_primitives_count_over_16bit",
                 @"drawPrimitives:vertexStart:vertexCount:",
                 @{@"primitive_type" : @(MTLPrimitiveTypeTriangle),
                   @"vertex_start" : @0,
                   @"vertex_count" : @0x10000},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long, unsigned long))objc_msgSend)(
                       enc, sel_getUid("drawPrimitives:vertexStart:vertexCount:"),
                       MTLPrimitiveTypeTriangle, 0, 0x10000);
                 });

  addEncoderCase(cases, ser, stream, @"render_draw_primitives_start_over_16bit",
                 @"drawPrimitives:vertexStart:vertexCount:",
                 @{@"primitive_type" : @(MTLPrimitiveTypeTriangle),
                   @"vertex_start" : @0x10000,
                   @"vertex_count" : @3},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long, unsigned long))objc_msgSend)(
                       enc, sel_getUid("drawPrimitives:vertexStart:vertexCount:"),
                       MTLPrimitiveTypeTriangle, 0x10000, 3);
                 });

  addEncoderCase(cases, ser, stream, @"render_draw_primitives_instances_over_16bit",
                 @"drawPrimitives:vertexStart:vertexCount:instanceCount:",
                 @{@"primitive_type" : @(MTLPrimitiveTypeTriangle),
                   @"vertex_start" : @1,
                   @"vertex_count" : @2,
                   @"instance_count" : @0x10000},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long, unsigned long,
                              unsigned long))objc_msgSend)(
                       enc,
                       sel_getUid("drawPrimitives:vertexStart:vertexCount:instanceCount:"),
                       MTLPrimitiveTypeTriangle, 1, 2, 0x10000);
                 });

  addEncoderCase(cases, ser, stream, @"render_draw_primitives_base_over_16bit",
                 @"drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:",
                 @{@"primitive_type" : @(MTLPrimitiveTypeTriangle),
                   @"vertex_start" : @1,
                   @"vertex_count" : @2,
                   @"instance_count" : @3,
                   @"base_instance" : @0x10000},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long, unsigned long,
                              unsigned long, unsigned long))objc_msgSend)(
                       enc,
                       sel_getUid("drawPrimitives:vertexStart:vertexCount:instanceCount:"
                                  "baseInstance:"),
                       MTLPrimitiveTypeTriangle, 1, 2, 3, 0x10000);
                 });

  addEncoderCase(cases, ser, stream, @"render_set_scissor", @"setScissorRect:",
                 @{@"x" : @1, @"y" : @2, @"width" : @300, @"height" : @400},
                 ^(id enc) {
                   ((void (*)(id, SEL, MTLScissorRect))objc_msgSend)(
                       enc, sel_getUid("setScissorRect:"),
                       (MTLScissorRect){1, 2, 300, 400});
                 });

  addEncoderCase(cases, ser, stream, @"render_set_viewport", @"setViewport:",
                 @{@"origin_x" : @0, @"origin_y" : @0, @"width" : @640,
                   @"height" : @480, @"znear" : @0, @"zfar" : @1},
                 ^(id enc) {
                   ((void (*)(id, SEL, MTLViewport))objc_msgSend)(
                       enc, sel_getUid("setViewport:"),
                       (MTLViewport){0, 0, 640, 480, 0, 1});
                 });

  addEncoderCase(cases, ser, stream, @"render_set_cull_mode", @"setCullMode:",
                 @{@"cull_mode" : @(MTLCullModeBack)}, ^(id enc) {
                   ((void (*)(id, SEL, unsigned long))objc_msgSend)(
                       enc, sel_getUid("setCullMode:"), MTLCullModeBack);
                 });

  addEncoderCase(cases, ser, stream, @"render_set_front_facing",
                 @"setFrontFacingWinding:",
                 @{@"winding" : @(MTLWindingCounterClockwise)}, ^(id enc) {
                   ((void (*)(id, SEL, unsigned long))objc_msgSend)(
                       enc, sel_getUid("setFrontFacingWinding:"),
                       MTLWindingCounterClockwise);
                 });

  // Values chosen to be exact in binary floating point, so a float/double
  // confusion shows as a wrong value rather than a rounding difference.
  addEncoderCase(cases, ser, stream, @"render_set_blend_color",
                 @"setBlendColorRed:green:blue:alpha:",
                 @{@"red" : @0.25, @"green" : @0.5, @"blue" : @0.75, @"alpha" : @1.0},
                 ^(id enc) {
                   ((void (*)(id, SEL, float, float, float, float))objc_msgSend)(
                       enc, sel_getUid("setBlendColorRed:green:blue:alpha:"), 0.25f,
                       0.5f, 0.75f, 1.0f);
                 });

  // --- State records that take no object -----------------------------------
  //
  // Argument widths for every one of these come from the selector's own type
  // encoding in inventory.json, so the only thing left to derive is where each
  // argument lands. Floats are values that are exact in binary, so a
  // float/double confusion reads as a wrong number rather than a rounding
  // difference.

  addEncoderCase(cases, ser, stream, @"render_set_triangle_fill_mode",
                 @"setTriangleFillMode:", @{@"fill_mode" : @(MTLTriangleFillModeLines)},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long))objc_msgSend)(
                       enc, sel_getUid("setTriangleFillMode:"), MTLTriangleFillModeLines);
                 });

  addEncoderCase(cases, ser, stream, @"render_set_depth_clip_mode",
                 @"setDepthClipMode:", @{@"mode" : @(MTLDepthClipModeClamp)}, ^(id enc) {
                   ((void (*)(id, SEL, unsigned long))objc_msgSend)(
                       enc, sel_getUid("setDepthClipMode:"), MTLDepthClipModeClamp);
                 });

  addEncoderCase(cases, ser, stream, @"render_set_stencil_reference",
                 @"setStencilReferenceValue:", @{@"reference" : @0x11223344}, ^(id enc) {
                   ((void (*)(id, SEL, unsigned int))objc_msgSend)(
                       enc, sel_getUid("setStencilReferenceValue:"), 0x11223344u);
                 });

  addEncoderCase(cases, ser, stream, @"render_set_stencil_reference_front_back",
                 @"setStencilFrontReferenceValue:backReferenceValue:",
                 @{@"front" : @0x11223344, @"back" : @0x55667788}, ^(id enc) {
                   ((void (*)(id, SEL, unsigned int, unsigned int))objc_msgSend)(
                       enc, sel_getUid("setStencilFrontReferenceValue:backReferenceValue:"),
                       0x11223344u, 0x55667788u);
                 });

  addEncoderCase(cases, ser, stream, @"render_set_depth_bias",
                 @"setDepthBias:slopeScale:clamp:",
                 @{@"bias" : @0.25, @"slope_scale" : @1.5, @"clamp" : @2.25}, ^(id enc) {
                   ((void (*)(id, SEL, float, float, float))objc_msgSend)(
                       enc, sel_getUid("setDepthBias:slopeScale:clamp:"), 0.25f, 1.5f,
                       2.25f);
                 });

  addEncoderCase(cases, ser, stream, @"render_set_visibility_result_mode",
                 @"setVisibilityResultMode:offset:",
                 @{@"mode" : @(MTLVisibilityResultModeCounting), @"offset" : @0x1234},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long))objc_msgSend)(
                       enc, sel_getUid("setVisibilityResultMode:offset:"),
                       MTLVisibilityResultModeCounting, 0x1234);
                 });

  // These two are expected to land in `unsupported` rather than in `cases`:
  // both fail `Not supported` inside Apple's encoder. They are driven anyway,
  // because that refusal is the evidence their manifest rows cite, and driving
  // them keeps it re-measured on every capture instead of remembered.
  addEncoderCase(cases, ser, stream, @"render_set_triangle_front_back_fill_mode",
                 @"setTriangleFrontFillMode:backFillMode:",
                 @{@"front" : @(MTLTriangleFillModeLines),
                   @"back" : @(MTLTriangleFillModeFill)},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long))objc_msgSend)(
                       enc, sel_getUid("setTriangleFrontFillMode:backFillMode:"),
                       MTLTriangleFillModeLines, MTLTriangleFillModeFill);
                 });

  addEncoderCase(cases, ser, stream, @"render_set_provoking_vertex_mode",
                 @"setProvokingVertexMode:", @{@"mode" : @1}, ^(id enc) {
                   ((void (*)(id, SEL, unsigned long))objc_msgSend)(
                       enc, sel_getUid("setProvokingVertexMode:"), 1);
                 });

  addEncoderCase(cases, ser, stream, @"render_set_depth_test_bounds",
                 @"setDepthTestMinBound:maxBound:",
                 @{@"min_bound" : @0.25, @"max_bound" : @0.75}, ^(id enc) {
                   ((void (*)(id, SEL, float, float))objc_msgSend)(
                       enc, sel_getUid("setDepthTestMinBound:maxBound:"), 0.25f, 0.75f);
                 });

  addEncoderCase(cases, ser, stream, @"render_set_alpha_test_reference",
                 @"setAlphaTestReferenceValue:", @{@"reference" : @0.75}, ^(id enc) {
                   ((void (*)(id, SEL, float))objc_msgSend)(
                       enc, sel_getUid("setAlphaTestReferenceValue:"), 0.75f);
                 });

  addEncoderCase(cases, ser, stream, @"render_set_line_width", @"setLineWidth:",
                 @{@"width" : @2.5}, ^(id enc) {
                   ((void (*)(id, SEL, float))objc_msgSend)(
                       enc, sel_getUid("setLineWidth:"), 2.5f);
                 });

  addEncoderCase(cases, ser, stream, @"render_set_point_size", @"setPointSize:",
                 @{@"size" : @3.5}, ^(id enc) {
                   ((void (*)(id, SEL, float))objc_msgSend)(
                       enc, sel_getUid("setPointSize:"), 3.5f);
                 });

  addEncoderCase(cases, ser, stream, @"render_set_tessellation_factor_scale",
                 @"setTessellationFactorScale:", @{@"scale" : @1.25}, ^(id enc) {
                   ((void (*)(id, SEL, float))objc_msgSend)(
                       enc, sel_getUid("setTessellationFactorScale:"), 1.25f);
                 });

  addEncoderCase(cases, ser, stream, @"render_set_primitive_restart_enabled",
                 @"setPrimitiveRestartEnabled:", @{@"enabled" : @1}, ^(id enc) {
                   ((void (*)(id, SEL, char))objc_msgSend)(
                       enc, sel_getUid("setPrimitiveRestartEnabled:"), 1);
                 });

  addEncoderCase(cases, ser, stream, @"render_set_viewport_transform_enabled",
                 @"setViewportTransformEnabled:", @{@"enabled" : @1}, ^(id enc) {
                   ((void (*)(id, SEL, char))objc_msgSend)(
                       enc, sel_getUid("setViewportTransformEnabled:"), 1);
                 });

  // --- Bind records --------------------------------------------------------
  //
  // The highest-traffic surface `reims_vgpu::runtime::exec` decodes. Each names
  // an object, so the record carries a serializer ref rather than the object;
  // the stubs supply distinctive ones and no storage is ever read.
  id pipeline = [[StubPipelineState alloc] init];
  id dss = [[StubDepthStencilState alloc] init];
  id sampler = [[StubSamplerState alloc] init];
  id fence = [[StubFence alloc] init];
  id src_icb = [[StubICB alloc] initWithRef:STUB_ICB_REF];
  id dst_icb = [[StubICB alloc] initWithRef:STUB_ICB_DST_REF];
  id tex = [[StubTexture alloc] init];
  id vbuf = [[StubBuffer alloc] init];

  addEncoderCase(cases, ser, stream, @"render_set_render_pipeline_state",
                 @"setRenderPipelineState:", @{@"pipeline_ref" : @(STUB_PIPELINE_REF)},
                 ^(id enc) {
                   ((void (*)(id, SEL, id))objc_msgSend)(
                       enc, sel_getUid("setRenderPipelineState:"), pipeline);
                 });

  addEncoderCase(cases, ser, stream, @"render_set_depth_stencil_state",
                 @"setDepthStencilState:",
                 @{@"depth_stencil_ref" : @(STUB_DEPTH_STENCIL_REF)}, ^(id enc) {
                   ((void (*)(id, SEL, id))objc_msgSend)(
                       enc, sel_getUid("setDepthStencilState:"), dss);
                 });

  addEncoderCase(cases, ser, stream, @"render_set_vertex_buffer",
                 @"setVertexBuffer:offset:atIndex:",
                 @{@"buffer_ref" : @(STUB_BUFFER_REF), @"offset" : @0x1234,
                   @"index" : @5},
                 ^(id enc) {
                   ((void (*)(id, SEL, id, unsigned long, unsigned long))objc_msgSend)(
                       enc, sel_getUid("setVertexBuffer:offset:atIndex:"), vbuf, 0x1234, 5);
                 });

  addEncoderCase(cases, ser, stream, @"render_set_fragment_buffer",
                 @"setFragmentBuffer:offset:atIndex:",
                 @{@"buffer_ref" : @(STUB_BUFFER_REF), @"offset" : @0x5678,
                   @"index" : @6},
                 ^(id enc) {
                   ((void (*)(id, SEL, id, unsigned long, unsigned long))objc_msgSend)(
                       enc, sel_getUid("setFragmentBuffer:offset:atIndex:"), vbuf, 0x5678,
                       6);
                 });

  addEncoderCase(cases, ser, stream, @"render_set_vertex_buffer_offset",
                 @"setVertexBufferOffset:atIndex:",
                 @{@"offset" : @0x1234, @"index" : @5}, ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long))objc_msgSend)(
                       enc, sel_getUid("setVertexBufferOffset:atIndex:"), 0x1234, 5);
                 });

  addEncoderCase(cases, ser, stream, @"render_set_fragment_buffer_offset",
                 @"setFragmentBufferOffset:atIndex:",
                 @{@"offset" : @0x5678, @"index" : @6}, ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long))objc_msgSend)(
                       enc, sel_getUid("setFragmentBufferOffset:atIndex:"), 0x5678, 6);
                 });

  // The vertex binds that carry an attribute stride.
  //
  // Metal 3.1 added a stride argument to four selectors that already existed
  // without one, and the question these ask is the same one the sampler LOD
  // clamps asked: whether the stride form is a *different opcode*. It was for
  // the samplers -- `0x80`/`0x71` reached no arm in this device, so the sampler
  // stayed unbound -- and a vertex buffer that stays unbound is worse.
  //
  // Every value is distinct and none is a stride a real layout would use, so a
  // field that lands somewhere unexpected is recognisable on sight and a
  // decoder that read the stride as the index could not read back correct.
  withCapability(ser, @"DynamicAttributeStride", ^{
    addEncoderCase(cases, ser, stream, @"render_set_vertex_buffer_attribute_stride",
                   @"setVertexBuffer:offset:attributeStride:atIndex:",
                   @{@"buffer_ref" : @(STUB_BUFFER_REF), @"offset" : @0x2345,
                     @"attribute_stride" : @0x3456, @"index" : @7},
                   ^(id enc) {
                     ((void (*)(id, SEL, id, unsigned long, unsigned long,
                                unsigned long))objc_msgSend)(
                         enc, sel_getUid("setVertexBuffer:offset:attributeStride:atIndex:"),
                         vbuf, 0x2345, 0x3456, 7);
                   });

    addEncoderCase(cases, ser, stream, @"render_set_vertex_buffer_offset_attribute_stride",
                   @"setVertexBufferOffset:attributeStride:atIndex:",
                   @{@"offset" : @0x4567, @"attribute_stride" : @0x5678, @"index" : @8},
                   ^(id enc) {
                     ((void (*)(id, SEL, unsigned long, unsigned long,
                                unsigned long))objc_msgSend)(
                         enc, sel_getUid("setVertexBufferOffset:attributeStride:atIndex:"),
                         0x4567, 0x5678, 8);
                   });

    // `setVertexBytes:length:atIndex:` emits the buffer-bind opcode, staging
    // the bytes through the command stream. Whether the stride form does the
    // same is the point of driving it rather than assuming from the sibling.
    static const unsigned char strideBytes[16] = {0xde, 0xad, 0xbe, 0xef, 0x11, 0x22,
                                                  0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
                                                  0x99, 0xaa, 0xbb, 0xcc};
    addEncoderCase(cases, ser, stream, @"render_set_vertex_bytes_attribute_stride",
                   @"setVertexBytes:length:attributeStride:atIndex:",
                   // The staging ref and offset are the stub's, exactly as on
                   // the non-stride sibling: the serializer picks the buffer it
                   // copies into, so those come from the harness's own object
                   // rather than from anything this case passed.
                   @{@"length" : @16, @"attribute_stride" : @0x6789, @"index" : @9,
                     @"buffer_ref" : @(STUB_STAGING_REF),
                     @"offset" : @(STUB_STAGING_OFFSET)},
                   ^(id enc) {
                     ((void (*)(id, SEL, const void *, unsigned long, unsigned long,
                                unsigned long))objc_msgSend)(
                         enc,
                         sel_getUid("setVertexBytes:length:attributeStride:atIndex:"),
                         strideBytes, 16, 0x6789, 9);
                   });
  });

  // Vertex amplification, the second capability-gated family to be driven.
  //
  // `MTLVertexAmplificationViewMapping` is two `uint32_t` -- the type encoding
  // says `r^{?=II}`, so the array element is eight bytes and the count leads as
  // a `Q`. Two mappings with four distinct offsets, which is what shows whether
  // the mappings reach the wire per entry or not at all.
  withCapability(ser, @"VertexAmplification", ^{
    static const unsigned int viewMappings[4] = {0x1111, 0x2222, 0x3333, 0x4444};
    addEncoderCase(cases, ser, stream, @"render_set_vertex_amplification_count",
                   @"setVertexAmplificationCount:viewMappings:",
                   @{@"count" : @2,
                     @"viewport_offset0" : @0x1111, @"rt_offset0" : @0x2222,
                     @"viewport_offset1" : @0x3333, @"rt_offset1" : @0x4444},
                   ^(id enc) {
                     ((void (*)(id, SEL, unsigned long, const void *))objc_msgSend)(
                         enc, sel_getUid("setVertexAmplificationCount:viewMappings:"), 2,
                         viewMappings);
                   });

    addEncoderCase(cases, ser, stream, @"render_set_vertex_amplification_mode",
                   @"setVertexAmplificationMode:value:",
                   @{@"mode" : @0x5555, @"value" : @0x6666}, ^(id enc) {
                     ((void (*)(id, SEL, unsigned long, unsigned long))objc_msgSend)(
                         enc, sel_getUid("setVertexAmplificationMode:value:"), 0x5555,
                         0x6666);
                   });
  });

  addEncoderCase(cases, ser, stream, @"render_set_vertex_texture",
                 @"setVertexTexture:atIndex:",
                 @{@"texture_ref" : @(STUB_TEXTURE_REF), @"index" : @3}, ^(id enc) {
                   ((void (*)(id, SEL, id, unsigned long))objc_msgSend)(
                       enc, sel_getUid("setVertexTexture:atIndex:"), tex, 3);
                 });

  addEncoderCase(cases, ser, stream, @"render_set_fragment_texture",
                 @"setFragmentTexture:atIndex:",
                 @{@"texture_ref" : @(STUB_TEXTURE_REF), @"index" : @4}, ^(id enc) {
                   ((void (*)(id, SEL, id, unsigned long))objc_msgSend)(
                       enc, sel_getUid("setFragmentTexture:atIndex:"), tex, 4);
                 });

  addEncoderCase(cases, ser, stream, @"render_set_vertex_sampler",
                 @"setVertexSamplerState:atIndex:",
                 @{@"sampler_ref" : @(STUB_SAMPLER_REF), @"index" : @2}, ^(id enc) {
                   ((void (*)(id, SEL, id, unsigned long))objc_msgSend)(
                       enc, sel_getUid("setVertexSamplerState:atIndex:"), sampler, 2);
                 });

  addEncoderCase(cases, ser, stream, @"render_set_fragment_sampler",
                 @"setFragmentSamplerState:atIndex:",
                 @{@"sampler_ref" : @(STUB_SAMPLER_REF), @"index" : @7}, ^(id enc) {
                   ((void (*)(id, SEL, id, unsigned long))objc_msgSend)(
                       enc, sel_getUid("setFragmentSamplerState:atIndex:"), sampler, 7);
                 });

  addEncoderCase(cases, ser, stream, @"render_update_fence", @"updateFence:afterStages:",
                 @{@"fence_ref" : @(STUB_FENCE_REF), @"stages" : @2}, ^(id enc) {
                   ((void (*)(id, SEL, id, unsigned long))objc_msgSend)(
                       enc, sel_getUid("updateFence:afterStages:"), fence, 2);
                 });

  addEncoderCase(cases, ser, stream, @"render_wait_for_fence",
                 @"waitForFence:beforeStages:",
                 @{@"fence_ref" : @(STUB_FENCE_REF), @"stages" : @1}, ^(id enc) {
                   ((void (*)(id, SEL, id, unsigned long))objc_msgSend)(
                       enc, sel_getUid("waitForFence:beforeStages:"), fence, 1);
                 });

  addEncoderCase(cases, ser, stream, @"render_use_resource",
                 @"useResource:usage:stages:",
                 @{@"resource_ref" : @(STUB_BUFFER_REF), @"usage" : @1, @"stages" : @2},
                 ^(id enc) {
                   ((void (*)(id, SEL, id, unsigned long, unsigned long))objc_msgSend)(
                       enc, sel_getUid("useResource:usage:stages:"), vbuf, 1, 2);
                 });

  // Index 3 against action 2: equal values here would leave the two fields
  // indistinguishable, which is exactly what a first pass at this case did.
  addEncoderCase(cases, ser, stream, @"render_use_heap", @"useHeap:stages:",
                 @{@"heap_ref" : @(STUB_HEAP_REF), @"stages" : @2}, ^(id enc) {
                   ((void (*)(id, SEL, id, unsigned long))objc_msgSend)(
                       enc, sel_getUid("useHeap:stages:"), [[StubHeap alloc] init], 2);
                 });

  // The rest of the residency and barrier cluster.
  //
  // `reims_vgpu::runtime::decode::render` carries OP_USE_HEAP = 0x86 and
  // OP_USE_RESOURCE = 0x87 while the serializer writes 0x1b and 0x89 for those
  // selectors, so two opcodes in that module belong to something else and this
  // is the neighbourhood to look in. The plural heap form and both memory
  // barriers are the only residency-adjacent selectors left on the class.
  {
    id heaps[2] = {[[StubHeap alloc] init], [[StubHeap2 alloc] init]};
    const id *heap_list = heaps; // a block cannot capture a C array
    addEncoderCase(cases, ser, stream, @"render_use_heaps_count",
                   @"useHeaps:count:stages:",
                   @{@"heap_ref" : @(STUB_HEAP_REF),
                     @"heap_ref_2" : @(STUB_HEAP2_REF),
                     @"count" : @2,
                     @"stages" : @2},
                   ^(id enc) {
                     ((void (*)(id, SEL, const id *, unsigned long,
                                unsigned long))objc_msgSend)(
                         enc, sel_getUid("useHeaps:count:stages:"), heap_list, 2, 2);
                   });

    id resources[2] = {[[StubBuffer alloc] init],
                       [[StubTexture alloc] initWithRef:STUB_TEXTURE_DST_REF]};
    const id *resource_list = resources;
    addEncoderCase(cases, ser, stream, @"render_memory_barrier_resources",
                   @"memoryBarrierWithResources:count:afterStages:beforeStages:",
                   @{@"resource_ref" : @(STUB_BUFFER_REF),
                     @"resource_ref_2" : @(STUB_TEXTURE_DST_REF),
                     @"count" : @2,
                     @"after_stages" : @1,
                     @"before_stages" : @2},
                   ^(id enc) {
                     ((void (*)(id, SEL, const id *, unsigned long, unsigned long,
                                unsigned long))objc_msgSend)(
                         enc,
                         sel_getUid("memoryBarrierWithResources:count:afterStages:"
                                    "beforeStages:"),
                         resource_list, 2, 1, 2);
                   });
  }

  addEncoderCase(cases, ser, stream, @"render_memory_barrier_scope",
                 @"memoryBarrierWithScope:afterStages:beforeStages:",
                 @{@"scope" : @4, @"after_stages" : @1, @"before_stages" : @2},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long,
                              unsigned long))objc_msgSend)(
                       enc,
                       sel_getUid("memoryBarrierWithScope:afterStages:beforeStages:"), 4,
                       1, 2);
                 });

  addEncoderCase(cases, ser, stream, @"render_memory_barrier_scope_alt",
                 @"memoryBarrierWithScope:afterStages:beforeStages:",
                 @{@"scope" : @1, @"after_stages" : @4, @"before_stages" : @8},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long,
                              unsigned long))objc_msgSend)(
                       enc,
                       sel_getUid("memoryBarrierWithScope:afterStages:beforeStages:"), 1,
                       4, 8);
                 });

  addEncoderCase(cases, ser, stream, @"render_texture_barrier", @"textureBarrier", @{},
                 ^(id enc) {
                   ((void (*)(id, SEL))objc_msgSend)(enc, sel_getUid("textureBarrier"));
                 });

  // Inline constant data. The bytes go through the stream's *uncounted*
  // allocator rather than the record, so what lands in the record is the
  // length, the index, and however the serializer names the staged bytes.
  addEncoderCase(cases, ser, stream, @"render_set_vertex_bytes",
                 @"setVertexBytes:length:atIndex:",
                 @{@"first" : @3,
                   @"count" : @1,
                   @"buffer_ref" : @(STUB_STAGING_REF),
                   @"offset" : @(STUB_STAGING_OFFSET),
                   @"length" : @8},
                 ^(id enc) {
                   static const unsigned char blob[8] = {0x5a, 0x5b, 0x5c, 0x5d,
                                                         0x5e, 0x5f, 0x60, 0x61};
                   ((void (*)(id, SEL, const void *, unsigned long,
                              unsigned long))objc_msgSend)(
                       enc, sel_getUid("setVertexBytes:length:atIndex:"), blob,
                       sizeof(blob), 3);
                 });

  addEncoderCase(cases, ser, stream, @"render_set_fragment_bytes",
                 @"setFragmentBytes:length:atIndex:",
                 @{@"first" : @5,
                   @"count" : @1,
                   @"buffer_ref" : @(STUB_STAGING_REF),
                   @"offset" : @(STUB_STAGING_OFFSET),
                   @"length" : @12},
                 ^(id enc) {
                   static const unsigned char blob[12] = {0x5a, 0x5b, 0x5c, 0x5d,
                                                          0x5e, 0x5f, 0x60, 0x61,
                                                          0x62, 0x63, 0x64, 0x65};
                   ((void (*)(id, SEL, const void *, unsigned long,
                              unsigned long))objc_msgSend)(
                       enc, sel_getUid("setFragmentBytes:length:atIndex:"), blob,
                       sizeof(blob), 5);
                 });

  // Indirect draws: the counts come from a buffer rather than the record, so
  // the record is a primitive type, a buffer ref and an offset.
  addEncoderCase(cases, ser, stream, @"render_draw_primitives_indirect",
                 @"drawPrimitives:indirectBuffer:indirectBufferOffset:",
                 @{@"primitive_type" : @(MTLPrimitiveTypeTriangle),
                   @"indirect_buffer_ref" : @(STUB_BUFFER_REF),
                   @"indirect_buffer_offset" : @0x1111},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, id, unsigned long))objc_msgSend)(
                       enc,
                       sel_getUid("drawPrimitives:indirectBuffer:indirectBufferOffset:"),
                       MTLPrimitiveTypeTriangle, [[StubBuffer alloc] init], 0x1111);
                 });

  {
    id index_buffer = [[StubBuffer alloc] init];
    id indirect_buffer = [[StubBuffer alloc] initWithRef:STUB_BUFFER_DST_REF];
    addEncoderCase(
        cases, ser, stream, @"render_draw_indexed_indirect",
        @"drawIndexedPrimitives:indexType:indexBuffer:indexBufferOffset:indirectBuffer:"
        @"indirectBufferOffset:",
        @{@"primitive_type" : @(MTLPrimitiveTypeTriangleStrip),
          @"index_type" : @(MTLIndexTypeUInt32),
          @"index_buffer_ref" : @(STUB_BUFFER_REF),
          @"index_buffer_offset" : @0x1111,
          @"indirect_buffer_ref" : @(STUB_BUFFER_DST_REF),
          @"indirect_buffer_offset" : @0x2222},
        ^(id enc) {
          ((void (*)(id, SEL, unsigned long, unsigned long, id, unsigned long, id,
                     unsigned long))objc_msgSend)(
              enc,
              sel_getUid("drawIndexedPrimitives:indexType:indexBuffer:"
                         "indexBufferOffset:indirectBuffer:indirectBufferOffset:"),
              MTLPrimitiveTypeTriangleStrip, MTLIndexTypeUInt32, index_buffer, 0x1111,
              indirect_buffer, 0x2222);
        });
  }

  // Indirect command buffer execution. The device has an `icb_exec_seen`
  // counter for this and has never seen one fire, so the record's layout has
  // never been checked against Apple's.
  addEncoderCase(cases, ser, stream, @"render_execute_commands_range",
                 @"executeCommandsInBuffer:withRange:",
                 @{@"icb_ref" : @(STUB_ICB_REF),
                   @"range_location" : @0x1100,
                   @"range_length" : @0x2200},
                 ^(id enc) {
                   ((void (*)(id, SEL, id, NSRange))objc_msgSend)(
                       enc, sel_getUid("executeCommandsInBuffer:withRange:"),
                       [[StubICB alloc] init], NSMakeRange(0x1100, 0x2200));
                 });

  {
    id icb = [[StubICB alloc] init];
    id indirect_buffer = [[StubBuffer alloc] init];
    addEncoderCase(cases, ser, stream, @"render_execute_commands_indirect",
                   @"executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:",
                   @{@"icb_ref" : @(STUB_ICB_REF),
                     @"indirect_buffer_ref" : @(STUB_BUFFER_REF),
                     @"indirect_buffer_offset" : @0x1111},
                   ^(id enc) {
                     ((void (*)(id, SEL, id, id, unsigned long))objc_msgSend)(
                         enc,
                         sel_getUid("executeCommandsInBuffer:indirectBuffer:"
                                    "indirectBufferOffset:"),
                         icb, indirect_buffer, 0x1111);
                   });
  }

  // The plural viewport and scissor forms. The singular ones are already
  // covered, so these are the cases that show whether a count-plus-array record
  // exists or whether the serializer expands them into repeated singulars.
  {
    MTLScissorRect rects[2] = {{0x11, 0x22, 0x33, 0x44}, {0x55, 0x66, 0x77, 0x88}};
    const MTLScissorRect *rect_list = rects;
    addEncoderCase(cases, ser, stream, @"render_set_scissor_rects",
                   @"setScissorRects:count:",
                   @{@"count" : @2,
                     @"x" : @0x11,
                     @"y" : @0x22,
                     @"width" : @0x33,
                     @"height" : @0x44,
                     @"x_2" : @0x55,
                     @"y_2" : @0x66,
                     @"width_2" : @0x77,
                     @"height_2" : @0x88},
                   ^(id enc) {
                     ((void (*)(id, SEL, const MTLScissorRect *,
                                unsigned long))objc_msgSend)(
                         enc, sel_getUid("setScissorRects:count:"), rect_list, 2);
                   });

    MTLViewport ports[2] = {{1.0, 2.0, 3.0, 4.0, 0.25, 0.75},
                            {5.0, 6.0, 7.0, 8.0, 0.125, 0.875}};
    const MTLViewport *port_list = ports;
    addEncoderCase(cases, ser, stream, @"render_set_viewports",
                   @"setViewports:count:",
                   @{@"count" : @2,
                     @"origin_x" : @1.0,
                     @"origin_y" : @2.0,
                     @"width" : @3.0,
                     @"height" : @4.0,
                     @"znear" : @0.25,
                     @"zfar" : @0.75,
                     @"origin_x_2" : @5.0,
                     @"origin_y_2" : @6.0,
                     @"width_2" : @7.0,
                     @"height_2" : @8.0,
                     @"znear_2" : @0.125,
                     @"zfar_2" : @0.875},
                   ^(id enc) {
                     ((void (*)(id, SEL, const MTLViewport *, unsigned long))objc_msgSend)(
                         enc, sel_getUid("setViewports:count:"), port_list, 2);
                   });
  }

  addEncoderCase(cases, ser, stream, @"render_set_color_store_action",
                 @"setColorStoreAction:atIndex:",
                 @{@"store_action" : @(MTLStoreActionMultisampleResolve), @"index" : @3},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long))objc_msgSend)(
                       enc, sel_getUid("setColorStoreAction:atIndex:"),
                       MTLStoreActionMultisampleResolve, 3);
                 });

  // Second reading of useResource, moving both packed halves at once. Neither
  // MTLResourceUsage nor MTLRenderStages has a value above 0xffff, so no case
  // can prove the pair is two 16-bit fields rather than one 32-bit word by
  // magnitude; what distinguishes them is that they occupy one word between
  // `+4` and the resource ref at `+8`, and that each moves independently.
  addEncoderCase(cases, ser, stream, @"render_use_resource_write_tile",
                 @"useResource:usage:stages:",
                 @{@"resource_ref" : @(STUB_BUFFER_REF), @"usage" : @2, @"stages" : @4},
                 ^(id enc) {
                   ((void (*)(id, SEL, id, unsigned long, unsigned long))objc_msgSend)(
                       enc, sel_getUid("useResource:usage:stages:"), vbuf, 2, 4);
                 });

  // The plural forms, which are what settle whether the leading word of a bind
  // record is a count: a singular record could have any constant there.
  {
    // Bound to pointers: a block cannot capture an array.
    id texArray[3] = {tex, tex, tex};
    const id *textures = texArray;
    addEncoderCase(cases, ser, stream, @"render_set_vertex_textures_range",
                   @"setVertexTextures:withRange:",
                   @{@"texture_ref" : @(STUB_TEXTURE_REF), @"first" : @2, @"count" : @3},
                   ^(id enc) {
                     ((void (*)(id, SEL, const id *, NSRange))objc_msgSend)(
                         enc, sel_getUid("setVertexTextures:withRange:"), textures,
                         NSMakeRange(2, 3));
                   });

    id bufArray[2] = {vbuf, vbuf};
    const id *buffers = bufArray;
    NSUInteger offArray[2] = {0x1111, 0x2222};
    const NSUInteger *offsets = offArray;
    addEncoderCase(cases, ser, stream, @"render_set_fragment_buffers_range",
                   @"setFragmentBuffers:offsets:withRange:",
                   @{@"buffer_ref" : @(STUB_BUFFER_REF), @"first" : @4, @"count" : @2,
                     @"offset0" : @0x1111, @"offset1" : @0x2222},
                   ^(id enc) {
                     ((void (*)(id, SEL, const id *, const NSUInteger *,
                                NSRange))objc_msgSend)(
                         enc, sel_getUid("setFragmentBuffers:offsets:withRange:"), buffers,
                         offsets, NSMakeRange(4, 2));
                   });

    // Forty, which is past the 32-entry cap `reims_vgpu::runtime::decode::render`
    // used to apply. That cap had no citation and this case is what disproved
    // it: the serializer emits the record, so refusing it lost all forty binds.
    static id manyTextures[40];
    for (int i = 0; i < 40; i++) manyTextures[i] = tex;
    const id *many = manyTextures;
    addEncoderCase(cases, ser, stream, @"render_set_vertex_textures_range_40",
                   @"setVertexTextures:withRange:",
                   @{@"texture_ref" : @(STUB_TEXTURE_REF), @"first" : @0, @"count" : @40},
                   ^(id enc) {
                     ((void (*)(id, SEL, const id *, NSRange))objc_msgSend)(
                         enc, sel_getUid("setVertexTextures:withRange:"), many,
                         NSMakeRange(0, 40));
                   });

    // The plural forms of the other three bind tables. Each names its own
    // first/count so a record that picked up a sibling's range is visible, and
    // together they show the plural encoding is one shape rather than four.
    id fragTexArray[2] = {tex, tex};
    const id *fragTextures = fragTexArray;
    addEncoderCase(cases, ser, stream, @"render_set_fragment_textures_range",
                   @"setFragmentTextures:withRange:",
                   @{@"texture_ref" : @(STUB_TEXTURE_REF), @"first" : @5, @"count" : @2},
                   ^(id enc) {
                     ((void (*)(id, SEL, const id *, NSRange))objc_msgSend)(
                         enc, sel_getUid("setFragmentTextures:withRange:"), fragTextures,
                         NSMakeRange(5, 2));
                   });

    id sampArray[3] = {sampler, sampler, sampler};
    const id *samplers = sampArray;
    addEncoderCase(cases, ser, stream, @"render_set_vertex_samplers_range",
                   @"setVertexSamplerStates:withRange:",
                   @{@"sampler_ref" : @(STUB_SAMPLER_REF), @"first" : @1, @"count" : @3},
                   ^(id enc) {
                     ((void (*)(id, SEL, const id *, NSRange))objc_msgSend)(
                         enc, sel_getUid("setVertexSamplerStates:withRange:"), samplers,
                         NSMakeRange(1, 3));
                   });
    addEncoderCase(cases, ser, stream, @"render_set_fragment_samplers_range",
                   @"setFragmentSamplerStates:withRange:",
                   @{@"sampler_ref" : @(STUB_SAMPLER_REF), @"first" : @6, @"count" : @3},
                   ^(id enc) {
                     ((void (*)(id, SEL, const id *, NSRange))objc_msgSend)(
                         enc, sel_getUid("setFragmentSamplerStates:withRange:"), samplers,
                         NSMakeRange(6, 3));
                   });

    id vbufArray[2] = {vbuf, vbuf};
    const id *vbuffers = vbufArray;
    NSUInteger voffArray[2] = {0x3333, 0x4444};
    const NSUInteger *voffsets = voffArray;
    addEncoderCase(cases, ser, stream, @"render_set_vertex_buffers_range",
                   @"setVertexBuffers:offsets:withRange:",
                   @{@"buffer_ref" : @(STUB_BUFFER_REF), @"first" : @7, @"count" : @2,
                     @"offset0" : @0x3333, @"offset1" : @0x4444},
                   ^(id enc) {
                     ((void (*)(id, SEL, const id *, const NSUInteger *,
                                NSRange))objc_msgSend)(
                         enc, sel_getUid("setVertexBuffers:offsets:withRange:"), vbuffers,
                         voffsets, NSMakeRange(7, 2));
                   });

    // The plural attribute-stride form. Both strides differ from each other and
    // from both offsets, which is what shows whether the stride is per entry
    // (as the sampler LOD clamps are) or once per record.
    static NSUInteger vstrideArray[2] = {0x5555, 0x6666};
    const NSUInteger *vstrides = vstrideArray;
    withCapability(ser, @"DynamicAttributeStride", ^{
      addEncoderCase(cases, ser, stream, @"render_set_vertex_buffers_range_attribute_stride",
                     @"setVertexBuffers:offsets:attributeStrides:withRange:",
                     @{@"buffer_ref" : @(STUB_BUFFER_REF), @"first" : @9, @"count" : @2,
                       @"offset0" : @0x3333, @"offset1" : @0x4444,
                       @"attribute_stride0" : @0x5555, @"attribute_stride1" : @0x6666},
                     ^(id enc) {
                       ((void (*)(id, SEL, const id *, const NSUInteger *,
                                  const NSUInteger *, NSRange))objc_msgSend)(
                           enc,
                           sel_getUid("setVertexBuffers:offsets:attributeStrides:withRange:"),
                           vbuffers, voffsets, vstrides, NSMakeRange(9, 2));
                     });
    });

    // The sampler binds that carry LOD clamps. The compute encoder has this
    // record already (`compute_set_sampler_lod`), where the clamps are per
    // entry rather than per record -- a difference that only a plural case can
    // show, so both a singular and a plural form are driven here.
    //
    // The `lodBias:` form is a fifth argument Metal's own API does not have on
    // this selector family; whether it reaches the wire is the question.
    addEncoderCase(cases, ser, stream, @"render_set_vertex_sampler_lod",
                   @"setVertexSamplerState:lodMinClamp:lodMaxClamp:atIndex:",
                   @{@"sampler_ref" : @(STUB_SAMPLER_REF), @"first" : @2,
                     @"lod_min_clamp" : @0.25, @"lod_max_clamp" : @0.75},
                   ^(id enc) {
                     ((void (*)(id, SEL, id, float, float, unsigned long))objc_msgSend)(
                         enc,
                         sel_getUid("setVertexSamplerState:lodMinClamp:lodMaxClamp:atIndex:"),
                         sampler, 0.25f, 0.75f, 2);
                   });
    addEncoderCase(cases, ser, stream, @"render_set_fragment_sampler_lod",
                   @"setFragmentSamplerState:lodMinClamp:lodMaxClamp:atIndex:",
                   @{@"sampler_ref" : @(STUB_SAMPLER_REF), @"first" : @3,
                     @"lod_min_clamp" : @0.125, @"lod_max_clamp" : @0.875},
                   ^(id enc) {
                     ((void (*)(id, SEL, id, float, float, unsigned long))objc_msgSend)(
                         enc,
                         sel_getUid("setFragmentSamplerState:lodMinClamp:lodMaxClamp:atIndex:"),
                         sampler, 0.125f, 0.875f, 3);
                   });
    addEncoderCase(cases, ser, stream, @"render_set_vertex_sampler_lod_bias",
                   @"setVertexSamplerState:lodMinClamp:lodMaxClamp:lodBias:atIndex:",
                   @{@"sampler_ref" : @(STUB_SAMPLER_REF), @"first" : @4,
                     @"lod_min_clamp" : @0.25, @"lod_max_clamp" : @0.75,
                     @"lod_bias" : @0.5},
                   ^(id enc) {
                     ((void (*)(id, SEL, id, float, float, float,
                                unsigned long))objc_msgSend)(
                         enc,
                         sel_getUid("setVertexSamplerState:lodMinClamp:lodMaxClamp:"
                                    "lodBias:atIndex:"),
                         sampler, 0.25f, 0.75f, 0.5f, 4);
                   });
    addEncoderCase(cases, ser, stream, @"render_set_fragment_sampler_lod_bias",
                   @"setFragmentSamplerState:lodMinClamp:lodMaxClamp:lodBias:atIndex:",
                   @{@"sampler_ref" : @(STUB_SAMPLER_REF), @"first" : @5,
                     @"lod_min_clamp" : @0.125, @"lod_max_clamp" : @0.875,
                     @"lod_bias" : @0.375},
                   ^(id enc) {
                     ((void (*)(id, SEL, id, float, float, float,
                                unsigned long))objc_msgSend)(
                         enc,
                         sel_getUid("setFragmentSamplerState:lodMinClamp:lodMaxClamp:"
                                    "lodBias:atIndex:"),
                         sampler, 0.125f, 0.875f, 0.375f, 5);
                   });

    // Two entries with *different* clamps, which is the only way to tell a
    // per-entry pair from one pair at the head of the record.
    float minArray[2] = {0.25f, 0.5f};
    float maxArray[2] = {0.75f, 0.875f};
    const float *lodMins = minArray;
    const float *lodMaxes = maxArray;
    addEncoderCase(cases, ser, stream, @"render_set_vertex_samplers_lod_range",
                   @"setVertexSamplerStates:lodMinClamps:lodMaxClamps:withRange:",
                   @{@"sampler_ref" : @(STUB_SAMPLER_REF), @"first" : @2, @"count" : @2,
                     @"lod_min_clamp" : @0.25, @"lod_max_clamp" : @0.75,
                     @"lod_min_clamp_2" : @0.5, @"lod_max_clamp_2" : @0.875},
                   ^(id enc) {
                     ((void (*)(id, SEL, const id *, const float *, const float *,
                                NSRange))objc_msgSend)(
                         enc,
                         sel_getUid("setVertexSamplerStates:lodMinClamps:lodMaxClamps:"
                                    "withRange:"),
                         samplers, lodMins, lodMaxes, NSMakeRange(2, 2));
                   });
    addEncoderCase(cases, ser, stream, @"render_set_fragment_samplers_lod_range",
                   @"setFragmentSamplerStates:lodMinClamps:lodMaxClamps:withRange:",
                   @{@"sampler_ref" : @(STUB_SAMPLER_REF), @"first" : @8, @"count" : @2,
                     @"lod_min_clamp" : @0.25, @"lod_max_clamp" : @0.75,
                     @"lod_min_clamp_2" : @0.5, @"lod_max_clamp_2" : @0.875},
                   ^(id enc) {
                     ((void (*)(id, SEL, const id *, const float *, const float *,
                                NSRange))objc_msgSend)(
                         enc,
                         sel_getUid("setFragmentSamplerStates:lodMinClamps:lodMaxClamps:"
                                    "withRange:"),
                         samplers, lodMins, lodMaxes, NSMakeRange(8, 2));
                   });

    // One selector, two tables, and **two records**: it is a convenience over
    // the singular texture and sampler binds rather than a combined record.
    // Driven as a split case so both are pinned; a case claiming one would have
    // recorded the texture bind and lost the sampler bind with nothing saying
    // so, which is the hole `multi` now closes.
    addEncoderCaseSplit(cases, @"PGSerializerRenderCommandEncoder",
                        ^id {
                          return makeRenderEncoder(ser, stream);
                        },
                        @"render_set_fragment_texture_and_sampler",
                        @"setFragmentTexture:atTextureIndex:samplerState:atSamplerIndex:",
                        @[
                          @{@"texture_ref" : @(STUB_TEXTURE_REF), @"first" : @9,
                            @"count" : @1},
                          @{@"sampler_ref" : @(STUB_SAMPLER_REF), @"first" : @4,
                            @"count" : @1},
                        ],
                        ^(id enc) {
                          ((void (*)(id, SEL, id, unsigned long, id,
                                     unsigned long))objc_msgSend)(
                              enc,
                              sel_getUid("setFragmentTexture:atTextureIndex:samplerState:"
                                         "atSamplerIndex:"),
                              tex, 9, sampler, 4);
                        });

    id resArray[2] = {vbuf, tex};
    const id *resources = resArray;
    addEncoderCase(cases, ser, stream, @"render_use_resources_count",
                   @"useResources:count:usage:stages:",
                   @{@"count" : @2, @"usage" : @1, @"stages" : @2}, ^(id enc) {
                     ((void (*)(id, SEL, const id *, unsigned long, unsigned long,
                                unsigned long))objc_msgSend)(
                         enc, sel_getUid("useResources:count:usage:stages:"), resources, 2,
                         1, 2);
                   });
  }

  addEncoderCase(cases, ser, stream, @"render_set_depth_store_action",
                 @"setDepthStoreAction:", @{@"store_action" : @(MTLStoreActionStore)},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long))objc_msgSend)(
                       enc, sel_getUid("setDepthStoreAction:"), MTLStoreActionStore);
                 });

  addEncoderCase(cases, ser, stream, @"render_set_stencil_store_action",
                 @"setStencilStoreAction:", @{@"store_action" : @(MTLStoreActionDontCare)},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long))objc_msgSend)(
                       enc, sel_getUid("setStencilStoreAction:"), MTLStoreActionDontCare);
                 });

  // --- Indexed draws -------------------------------------------------------
  //
  // These name a buffer, so the record carries a resource ref rather than the
  // bytes. The stub supplies a distinctive one; nothing reads its storage.
  id ibuf = [[StubBuffer alloc] init];

  addEncoderCase(cases, ser, stream, @"render_draw_indexed",
                 @"drawIndexedPrimitives:indexCount:indexType:indexBuffer:"
                 @"indexBufferOffset:",
                 @{@"primitive_type" : @(MTLPrimitiveTypeTriangle),
                   @"index_count" : @0x1111,
                   @"index_type" : @(MTLIndexTypeUInt16),
                   @"index_buffer_ref" : @(STUB_BUFFER_REF),
                   @"index_buffer_offset" : @0x2222},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long, unsigned long, id,
                              unsigned long))objc_msgSend)(
                       enc,
                       sel_getUid("drawIndexedPrimitives:indexCount:indexType:indexBuffer:"
                                  "indexBufferOffset:"),
                       MTLPrimitiveTypeTriangle, 0x1111, MTLIndexTypeUInt16, ibuf, 0x2222);
                 });

  // Index type is the one enum in this family, so it gets its own case rather
  // than riding along with a magnitude change.
  addEncoderCase(cases, ser, stream, @"render_draw_indexed_uint32",
                 @"drawIndexedPrimitives:indexCount:indexType:indexBuffer:"
                 @"indexBufferOffset:",
                 @{@"primitive_type" : @(MTLPrimitiveTypeTriangleStrip),
                   @"index_count" : @0x1111,
                   @"index_type" : @(MTLIndexTypeUInt32),
                   @"index_buffer_ref" : @(STUB_BUFFER_REF),
                   @"index_buffer_offset" : @0x2222},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long, unsigned long, id,
                              unsigned long))objc_msgSend)(
                       enc,
                       sel_getUid("drawIndexedPrimitives:indexCount:indexType:indexBuffer:"
                                  "indexBufferOffset:"),
                       MTLPrimitiveTypeTriangleStrip, 0x1111, MTLIndexTypeUInt32, ibuf,
                       0x2222);
                 });

  addEncoderCase(cases, ser, stream, @"render_draw_indexed_count_over_16bit",
                 @"drawIndexedPrimitives:indexCount:indexType:indexBuffer:"
                 @"indexBufferOffset:",
                 @{@"primitive_type" : @(MTLPrimitiveTypeTriangle),
                   @"index_count" : @0x11111,
                   @"index_type" : @(MTLIndexTypeUInt16),
                   @"index_buffer_ref" : @(STUB_BUFFER_REF),
                   @"index_buffer_offset" : @0x2222},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long, unsigned long, id,
                              unsigned long))objc_msgSend)(
                       enc,
                       sel_getUid("drawIndexedPrimitives:indexCount:indexType:indexBuffer:"
                                  "indexBufferOffset:"),
                       MTLPrimitiveTypeTriangle, 0x11111, MTLIndexTypeUInt16, ibuf, 0x2222);
                 });

  // The offset is a byte offset into a buffer, so whether it has a 16-bit form
  // at all is a separate question from the counts'.
  addEncoderCase(cases, ser, stream, @"render_draw_indexed_offset_over_16bit",
                 @"drawIndexedPrimitives:indexCount:indexType:indexBuffer:"
                 @"indexBufferOffset:",
                 @{@"primitive_type" : @(MTLPrimitiveTypeTriangle),
                   @"index_count" : @0x1111,
                   @"index_type" : @(MTLIndexTypeUInt16),
                   @"index_buffer_ref" : @(STUB_BUFFER_REF),
                   @"index_buffer_offset" : @0x22222},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long, unsigned long, id,
                              unsigned long))objc_msgSend)(
                       enc,
                       sel_getUid("drawIndexedPrimitives:indexCount:indexType:indexBuffer:"
                                  "indexBufferOffset:"),
                       MTLPrimitiveTypeTriangle, 0x1111, MTLIndexTypeUInt16, ibuf, 0x22222);
                 });

  addEncoderCase(cases, ser, stream, @"render_draw_indexed_instanced",
                 @"drawIndexedPrimitives:indexCount:indexType:indexBuffer:"
                 @"indexBufferOffset:instanceCount:",
                 @{@"primitive_type" : @(MTLPrimitiveTypeTriangle),
                   @"index_count" : @0x1111,
                   @"index_type" : @(MTLIndexTypeUInt16),
                   @"index_buffer_ref" : @(STUB_BUFFER_REF),
                   @"index_buffer_offset" : @0x2222,
                   @"instance_count" : @0x3333},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long, unsigned long, id,
                              unsigned long, unsigned long))objc_msgSend)(
                       enc,
                       sel_getUid("drawIndexedPrimitives:indexCount:indexType:indexBuffer:"
                                  "indexBufferOffset:instanceCount:"),
                       MTLPrimitiveTypeTriangle, 0x1111, MTLIndexTypeUInt16, ibuf, 0x2222,
                       0x3333);
                 });

  addEncoderCase(cases, ser, stream, @"render_draw_indexed_instances_over_16bit",
                 @"drawIndexedPrimitives:indexCount:indexType:indexBuffer:"
                 @"indexBufferOffset:instanceCount:",
                 @{@"primitive_type" : @(MTLPrimitiveTypeTriangle),
                   @"index_count" : @0x1111,
                   @"index_type" : @(MTLIndexTypeUInt16),
                   @"index_buffer_ref" : @(STUB_BUFFER_REF),
                   @"index_buffer_offset" : @0x2222,
                   @"instance_count" : @0x10000},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long, unsigned long, id,
                              unsigned long, unsigned long))objc_msgSend)(
                       enc,
                       sel_getUid("drawIndexedPrimitives:indexCount:indexType:indexBuffer:"
                                  "indexBufferOffset:instanceCount:"),
                       MTLPrimitiveTypeTriangle, 0x1111, MTLIndexTypeUInt16, ibuf, 0x2222,
                       0x10000);
                 });

  addEncoderCase(cases, ser, stream, @"render_draw_indexed_instanced_base",
                 @"drawIndexedPrimitives:indexCount:indexType:indexBuffer:"
                 @"indexBufferOffset:instanceCount:baseVertex:baseInstance:",
                 @{@"primitive_type" : @(MTLPrimitiveTypeTriangle),
                   @"index_count" : @0x1111,
                   @"index_type" : @(MTLIndexTypeUInt16),
                   @"index_buffer_ref" : @(STUB_BUFFER_REF),
                   @"index_buffer_offset" : @0x2222,
                   @"instance_count" : @0x3333,
                   @"base_vertex" : @0x44,
                   @"base_instance" : @0x55},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long, unsigned long, id,
                              unsigned long, unsigned long, long,
                              unsigned long))objc_msgSend)(
                       enc,
                       sel_getUid("drawIndexedPrimitives:indexCount:indexType:indexBuffer:"
                                  "indexBufferOffset:instanceCount:baseVertex:baseInstance:"),
                       MTLPrimitiveTypeTriangle, 0x1111, MTLIndexTypeUInt16, ibuf, 0x2222,
                       0x3333, 0x44, 0x55);
                 });

  // Second reading of the same record with every field moved, because this is
  // the one form whose count and offset appear in the opposite order to its
  // siblings' and a single case cannot tell a swap from a coincidence.
  addEncoderCase(cases, ser, stream, @"render_draw_indexed_instanced_base_alt",
                 @"drawIndexedPrimitives:indexCount:indexType:indexBuffer:"
                 @"indexBufferOffset:instanceCount:baseVertex:baseInstance:",
                 @{@"primitive_type" : @(MTLPrimitiveTypeTriangleStrip),
                   @"index_count" : @0x6666,
                   @"index_type" : @(MTLIndexTypeUInt32),
                   @"index_buffer_ref" : @(STUB_BUFFER_REF),
                   @"index_buffer_offset" : @0x7777,
                   @"instance_count" : @0x8888,
                   @"base_vertex" : @0x99,
                   @"base_instance" : @0xbb},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long, unsigned long, id,
                              unsigned long, unsigned long, long,
                              unsigned long))objc_msgSend)(
                       enc,
                       sel_getUid("drawIndexedPrimitives:indexCount:indexType:indexBuffer:"
                                  "indexBufferOffset:instanceCount:baseVertex:baseInstance:"),
                       MTLPrimitiveTypeTriangleStrip, 0x6666, MTLIndexTypeUInt32, ibuf,
                       0x7777, 0x8888, 0x99, 0xbb);
                 });

  addEncoderCase(cases, ser, stream, @"render_draw_indexed_base_instances_over_16bit",
                 @"drawIndexedPrimitives:indexCount:indexType:indexBuffer:"
                 @"indexBufferOffset:instanceCount:baseVertex:baseInstance:",
                 @{@"primitive_type" : @(MTLPrimitiveTypeTriangle),
                   @"index_count" : @0x1111,
                   @"index_type" : @(MTLIndexTypeUInt16),
                   @"index_buffer_ref" : @(STUB_BUFFER_REF),
                   @"index_buffer_offset" : @0x2222,
                   @"instance_count" : @0x10000,
                   @"base_vertex" : @0x44,
                   @"base_instance" : @0x55},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long, unsigned long, id,
                              unsigned long, unsigned long, long,
                              unsigned long))objc_msgSend)(
                       enc,
                       sel_getUid("drawIndexedPrimitives:indexCount:indexType:indexBuffer:"
                                  "indexBufferOffset:instanceCount:baseVertex:baseInstance:"),
                       MTLPrimitiveTypeTriangle, 0x1111, MTLIndexTypeUInt16, ibuf, 0x2222,
                       0x10000, 0x44, 0x55);
                 });

  // `baseVertex` is the family's only signed argument. Whether a negative one
  // reaches the compact form as a truncated two's complement, or forces the
  // wide form the way an over-large unsigned argument does, is not something
  // the unsigned cases can answer.
  addEncoderCase(cases, ser, stream, @"render_draw_indexed_negative_base_vertex",
                 @"drawIndexedPrimitives:indexCount:indexType:indexBuffer:"
                 @"indexBufferOffset:instanceCount:baseVertex:baseInstance:",
                 @{@"primitive_type" : @(MTLPrimitiveTypeTriangle),
                   @"index_count" : @0x1111,
                   @"index_type" : @(MTLIndexTypeUInt16),
                   @"index_buffer_ref" : @(STUB_BUFFER_REF),
                   @"index_buffer_offset" : @0x2222,
                   @"instance_count" : @0x3333,
                   @"base_vertex" : @(-2),
                   @"base_instance" : @0x55},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long, unsigned long, id,
                              unsigned long, unsigned long, long,
                              unsigned long))objc_msgSend)(
                       enc,
                       sel_getUid("drawIndexedPrimitives:indexCount:indexType:indexBuffer:"
                                  "indexBufferOffset:instanceCount:baseVertex:baseInstance:"),
                       MTLPrimitiveTypeTriangle, 0x1111, MTLIndexTypeUInt16, ibuf, 0x2222,
                       0x3333, -2, 0x55);
                 });

  // The pair that decides whether `baseVertex`'s fit test is signed or
  // unsigned. `0xffff` fits an unsigned 16-bit field and does not fit a signed
  // one; `-70000` fits neither. If the first goes wide the test is signed.
  addEncoderCase(cases, ser, stream, @"render_draw_indexed_base_vertex_at_u16_max",
                 @"drawIndexedPrimitives:indexCount:indexType:indexBuffer:"
                 @"indexBufferOffset:instanceCount:baseVertex:baseInstance:",
                 @{@"primitive_type" : @(MTLPrimitiveTypeTriangle),
                   @"index_count" : @0x1111,
                   @"index_type" : @(MTLIndexTypeUInt16),
                   @"index_buffer_ref" : @(STUB_BUFFER_REF),
                   @"index_buffer_offset" : @0x2222,
                   @"instance_count" : @0x3333,
                   @"base_vertex" : @0xffff,
                   @"base_instance" : @0x55},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long, unsigned long, id,
                              unsigned long, unsigned long, long,
                              unsigned long))objc_msgSend)(
                       enc,
                       sel_getUid("drawIndexedPrimitives:indexCount:indexType:indexBuffer:"
                                  "indexBufferOffset:instanceCount:baseVertex:baseInstance:"),
                       MTLPrimitiveTypeTriangle, 0x1111, MTLIndexTypeUInt16, ibuf, 0x2222,
                       0x3333, 0xffff, 0x55);
                 });

  addEncoderCase(cases, ser, stream, @"render_draw_indexed_base_vertex_below_i16",
                 @"drawIndexedPrimitives:indexCount:indexType:indexBuffer:"
                 @"indexBufferOffset:instanceCount:baseVertex:baseInstance:",
                 @{@"primitive_type" : @(MTLPrimitiveTypeTriangle),
                   @"index_count" : @0x1111,
                   @"index_type" : @(MTLIndexTypeUInt16),
                   @"index_buffer_ref" : @(STUB_BUFFER_REF),
                   @"index_buffer_offset" : @0x2222,
                   @"instance_count" : @0x3333,
                   @"base_vertex" : @(-70000),
                   @"base_instance" : @0x55},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long, unsigned long, id,
                              unsigned long, unsigned long, long,
                              unsigned long))objc_msgSend)(
                       enc,
                       sel_getUid("drawIndexedPrimitives:indexCount:indexType:indexBuffer:"
                                  "indexBufferOffset:instanceCount:baseVertex:baseInstance:"),
                       MTLPrimitiveTypeTriangle, 0x1111, MTLIndexTypeUInt16, ibuf, 0x2222,
                       0x3333, -70000, 0x55);
                 });

  addEncoderCase(cases, ser, stream, @"render_draw_indexed_base_instance_over_16bit",
                 @"drawIndexedPrimitives:indexCount:indexType:indexBuffer:"
                 @"indexBufferOffset:instanceCount:baseVertex:baseInstance:",
                 @{@"primitive_type" : @(MTLPrimitiveTypeTriangle),
                   @"index_count" : @0x1111,
                   @"index_type" : @(MTLIndexTypeUInt16),
                   @"index_buffer_ref" : @(STUB_BUFFER_REF),
                   @"index_buffer_offset" : @0x2222,
                   @"instance_count" : @0x3333,
                   @"base_vertex" : @0x44,
                   @"base_instance" : @0x10000},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long, unsigned long, id,
                              unsigned long, unsigned long, long,
                              unsigned long))objc_msgSend)(
                       enc,
                       sel_getUid("drawIndexedPrimitives:indexCount:indexType:indexBuffer:"
                                  "indexBufferOffset:instanceCount:baseVertex:baseInstance:"),
                       MTLPrimitiveTypeTriangle, 0x1111, MTLIndexTypeUInt16, ibuf, 0x2222,
                       0x3333, 0x44, 0x10000);
                 });

  // A negative `baseVertex` in a record another argument pushed wide: the only
  // way to see the wide field's full width, since `baseVertex` never widens a
  // record by itself.
  addEncoderCase(cases, ser, stream, @"render_draw_indexed_wide_negative_base_vertex",
                 @"drawIndexedPrimitives:indexCount:indexType:indexBuffer:"
                 @"indexBufferOffset:instanceCount:baseVertex:baseInstance:",
                 @{@"primitive_type" : @(MTLPrimitiveTypeTriangle),
                   @"index_count" : @0x1111,
                   @"index_type" : @(MTLIndexTypeUInt16),
                   @"index_buffer_ref" : @(STUB_BUFFER_REF),
                   @"index_buffer_offset" : @0x2222,
                   @"instance_count" : @0x10000,
                   @"base_vertex" : @(-70000),
                   @"base_instance" : @0x55},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long, unsigned long, id,
                              unsigned long, unsigned long, long,
                              unsigned long))objc_msgSend)(
                       enc,
                       sel_getUid("drawIndexedPrimitives:indexCount:indexType:indexBuffer:"
                                  "indexBufferOffset:instanceCount:baseVertex:baseInstance:"),
                       MTLPrimitiveTypeTriangle, 0x1111, MTLIndexTypeUInt16, ibuf, 0x2222,
                       0x10000, -70000, 0x55);
                 });

  // --- Tile shaders --------------------------------------------------------
  //
  // The largest capability-gated family: nineteen selectors that emit nothing
  // at all with `-supportsTileShaders` at its default of false, which is every
  // capture taken before this one. Driven with the flag forced, so a `silent`
  // outcome here is Apple's answer rather than this harness's.
  //
  // Tile binds are a fourth argument table beside vertex, fragment and the
  // compute encoder's — the same shapes (`atIndex:`, `withRange:`, the LOD
  // clamp forms, the staged `Bytes` form) against a different stage. Whether
  // they are the *same opcodes* with a stage selector or their own is the
  // question, and it is the one that decides whether a guest running a tile
  // shader binds anything at all: the attribute-stride binds were their own
  // opcodes and sat above this device's accepted window, so every one was
  // refused.
  //
  // Every index is distinct across the family so a record that wrote the wrong
  // one is recognisable on sight rather than by arithmetic.
  withCapability(ser, @"TileShaders", ^{
    id accel = [[StubAccelStruct alloc] init];
    id visFnTable = [[StubVisibleFnTable alloc] init];
    id isectFnTable = [[StubIntersectionFnTable alloc] init];

    // A tile dispatch's threads-per-tile is an `MTLSize`, three `Q` by value.
    // Three different values, none a power of two a real tile size would use,
    // so a record that dropped `depth` or transposed the pair is visible.
    addEncoderCase(cases, ser, stream, @"render_dispatch_threads_per_tile",
                   @"dispatchThreadsPerTile:",
                   @{@"width" : @0x11, @"height" : @0x22, @"depth" : @0x33},
                   ^(id enc) {
                     ((void (*)(id, SEL, MTLSize))objc_msgSend)(
                         enc, sel_getUid("dispatchThreadsPerTile:"),
                         MTLSizeMake(0x11, 0x22, 0x33));
                   });

    // The region form adds an `MTLRegion` — an origin and a size, six more
    // `Q`. Distinct from the threads-per-tile size in every component, which
    // is what separates "the region reached the wire" from "the size was
    // written twice".
    addEncoderCase(cases, ser, stream, @"render_dispatch_threads_per_tile_in_region",
                   @"dispatchThreadsPerTile:inRegion:",
                   @{@"width" : @0x11, @"height" : @0x22, @"depth" : @0x33,
                     @"origin_x" : @0x44, @"origin_y" : @0x55, @"origin_z" : @0x66,
                     @"region_width" : @0x77, @"region_height" : @0x88,
                     @"region_depth" : @0x99},
                   ^(id enc) {
                     ((void (*)(id, SEL, MTLSize, MTLRegion))objc_msgSend)(
                         enc, sel_getUid("dispatchThreadsPerTile:inRegion:"),
                         MTLSizeMake(0x11, 0x22, 0x33),
                         MTLRegionMake3D(0x44, 0x55, 0x66, 0x77, 0x88, 0x99));
                   });

    // The render-target array index trails as an `I` — a 32-bit field where
    // everything before it is 64-bit, which the type encoding settles and no
    // amount of reading the sibling would.
    addEncoderCase(cases, ser, stream,
                   @"render_dispatch_threads_per_tile_in_region_rt_index",
                   @"dispatchThreadsPerTile:inRegion:withRenderTargetArrayIndex:",
                   @{@"width" : @0x11, @"height" : @0x22, @"depth" : @0x33,
                     @"origin_x" : @0x44, @"origin_y" : @0x55, @"origin_z" : @0x66,
                     @"region_width" : @0x77, @"region_height" : @0x88,
                     @"region_depth" : @0x99, @"render_target_array_index" : @0xabc},
                   ^(id enc) {
                     ((void (*)(id, SEL, MTLSize, MTLRegion, unsigned int))objc_msgSend)(
                         enc,
                         sel_getUid("dispatchThreadsPerTile:inRegion:"
                                    "withRenderTargetArrayIndex:"),
                         MTLSizeMake(0x11, 0x22, 0x33),
                         MTLRegionMake3D(0x44, 0x55, 0x66, 0x77, 0x88, 0x99), 0xabc);
                   });

    addEncoderCase(cases, ser, stream, @"render_set_tile_buffer",
                   @"setTileBuffer:offset:atIndex:",
                   @{@"buffer_ref" : @(STUB_BUFFER_REF), @"offset" : @0x1234,
                     @"index" : @3},
                   ^(id enc) {
                     ((void (*)(id, SEL, id, unsigned long, unsigned long))objc_msgSend)(
                         enc, sel_getUid("setTileBuffer:offset:atIndex:"), vbuf, 0x1234, 3);
                   });

    addEncoderCase(cases, ser, stream, @"render_set_tile_buffer_offset",
                   @"setTileBufferOffset:atIndex:",
                   @{@"offset" : @0x2345, @"index" : @4}, ^(id enc) {
                     ((void (*)(id, SEL, unsigned long, unsigned long))objc_msgSend)(
                         enc, sel_getUid("setTileBufferOffset:atIndex:"), 0x2345, 4);
                   });

    // Two buffers at two *different* offsets: one offset would read back
    // correct whether the record carries them per entry or once at the head.
    id tileBuffers[2] = {vbuf, [[StubBuffer alloc] initWithRef:STUB_BUFFER_DST_REF]};
    unsigned long tileOffsets[2] = {0x3456, 0x4567};
    const id *tileBufferArray = tileBuffers;
    const unsigned long *tileOffsetArray = tileOffsets;
    addEncoderCase(cases, ser, stream, @"render_set_tile_buffers_range",
                   @"setTileBuffers:offsets:withRange:",
                   @{@"buffer_ref" : @(STUB_BUFFER_REF),
                     @"buffer_ref_2" : @(STUB_BUFFER_DST_REF), @"first" : @5,
                     @"count" : @2, @"offset" : @0x3456, @"offset_2" : @0x4567},
                   ^(id enc) {
                     ((void (*)(id, SEL, const id *, const unsigned long *,
                                NSRange))objc_msgSend)(
                         enc, sel_getUid("setTileBuffers:offsets:withRange:"),
                         tileBufferArray, tileOffsetArray, NSMakeRange(5, 2));
                   });

    // The staged form. Its ref and offset are the stub staging buffer's, not
    // this case's — the serializer picks the buffer it copies into.
    static const unsigned char tileBytes[16] = {0x0f, 0x1e, 0x2d, 0x3c, 0x4b, 0x5a,
                                                0x69, 0x78, 0x87, 0x96, 0xa5, 0xb4,
                                                0xc3, 0xd2, 0xe1, 0xf0};
    addEncoderCase(cases, ser, stream, @"render_set_tile_bytes",
                   @"setTileBytes:length:atIndex:",
                   @{@"length" : @16, @"index" : @6,
                     @"buffer_ref" : @(STUB_STAGING_REF),
                     @"offset" : @(STUB_STAGING_OFFSET)},
                   ^(id enc) {
                     ((void (*)(id, SEL, const void *, unsigned long,
                                unsigned long))objc_msgSend)(
                         enc, sel_getUid("setTileBytes:length:atIndex:"), tileBytes, 16, 6);
                   });

    addEncoderCase(cases, ser, stream, @"render_set_tile_texture",
                   @"setTileTexture:atIndex:",
                   @{@"texture_ref" : @(STUB_TEXTURE_REF), @"index" : @2}, ^(id enc) {
                     ((void (*)(id, SEL, id, unsigned long))objc_msgSend)(
                         enc, sel_getUid("setTileTexture:atIndex:"), tex, 2);
                   });

    id tileTextures[2] = {tex, [[StubTexture alloc] initWithRef:STUB_TEXTURE_DST_REF]};
    const id *tileTextureArray = tileTextures;
    addEncoderCase(cases, ser, stream, @"render_set_tile_textures_range",
                   @"setTileTextures:withRange:",
                   @{@"texture_ref" : @(STUB_TEXTURE_REF),
                     @"texture_ref_2" : @(STUB_TEXTURE_DST_REF), @"first" : @7,
                     @"count" : @2},
                   ^(id enc) {
                     ((void (*)(id, SEL, const id *, NSRange))objc_msgSend)(
                         enc, sel_getUid("setTileTextures:withRange:"), tileTextureArray,
                         NSMakeRange(7, 2));
                   });

    addEncoderCase(cases, ser, stream, @"render_set_tile_sampler",
                   @"setTileSamplerState:atIndex:",
                   @{@"sampler_ref" : @(STUB_SAMPLER_REF), @"index" : @4}, ^(id enc) {
                     ((void (*)(id, SEL, id, unsigned long))objc_msgSend)(
                         enc, sel_getUid("setTileSamplerState:atIndex:"), sampler, 4);
                   });

    addEncoderCase(cases, ser, stream, @"render_set_tile_sampler_lod",
                   @"setTileSamplerState:lodMinClamp:lodMaxClamp:atIndex:",
                   @{@"sampler_ref" : @(STUB_SAMPLER_REF), @"first" : @5,
                     @"lod_min_clamp" : @0.25, @"lod_max_clamp" : @0.75},
                   ^(id enc) {
                     ((void (*)(id, SEL, id, float, float, unsigned long))objc_msgSend)(
                         enc,
                         sel_getUid("setTileSamplerState:lodMinClamp:lodMaxClamp:atIndex:"),
                         sampler, 0.25f, 0.75f, 5);
                   });

    id tileSamplers[2] = {sampler, [[StubSamplerState alloc] init]};
    const id *tileSamplerArray = tileSamplers;
    addEncoderCase(cases, ser, stream, @"render_set_tile_samplers_range",
                   @"setTileSamplerStates:withRange:",
                   @{@"sampler_ref" : @(STUB_SAMPLER_REF), @"first" : @6, @"count" : @2},
                   ^(id enc) {
                     ((void (*)(id, SEL, const id *, NSRange))objc_msgSend)(
                         enc, sel_getUid("setTileSamplerStates:withRange:"),
                         tileSamplerArray, NSMakeRange(6, 2));
                   });

    float tileLodMins[2] = {0.25f, 0.5f};
    float tileLodMaxes[2] = {0.75f, 0.875f};
    const float *tileLodMinArray = tileLodMins;
    const float *tileLodMaxArray = tileLodMaxes;
    addEncoderCase(cases, ser, stream, @"render_set_tile_samplers_lod_range",
                   @"setTileSamplerStates:lodMinClamps:lodMaxClamps:withRange:",
                   @{@"sampler_ref" : @(STUB_SAMPLER_REF), @"first" : @8, @"count" : @2,
                     @"lod_min_clamp" : @0.25, @"lod_max_clamp" : @0.75,
                     @"lod_min_clamp_2" : @0.5, @"lod_max_clamp_2" : @0.875},
                   ^(id enc) {
                     ((void (*)(id, SEL, const id *, const float *, const float *,
                                NSRange))objc_msgSend)(
                         enc,
                         sel_getUid("setTileSamplerStates:lodMinClamps:lodMaxClamps:"
                                    "withRange:"),
                         tileSamplerArray, tileLodMinArray, tileLodMaxArray,
                         NSMakeRange(8, 2));
                   });

    // The three ray-tracing binds this stage carries. Each names a distinct
    // Metal protocol reached through its own ref accessor, so the stubs are
    // separate and their refs are far apart.
    addEncoderCase(cases, ser, stream, @"render_set_tile_acceleration_structure",
                   @"setTileAccelerationStructure:atBufferIndex:",
                   @{@"acceleration_structure_ref" : @(STUB_ACCEL_STRUCT_REF),
                     @"index" : @9},
                   ^(id enc) {
                     ((void (*)(id, SEL, id, unsigned long))objc_msgSend)(
                         enc, sel_getUid("setTileAccelerationStructure:atBufferIndex:"),
                         accel, 9);
                   });

    addEncoderCase(cases, ser, stream, @"render_set_tile_visible_function_table",
                   @"setTileVisibleFunctionTable:atBufferIndex:",
                   @{@"visible_function_table_ref" : @(STUB_VISIBLE_FN_TABLE_REF),
                     @"index" : @10},
                   ^(id enc) {
                     ((void (*)(id, SEL, id, unsigned long))objc_msgSend)(
                         enc, sel_getUid("setTileVisibleFunctionTable:atBufferIndex:"),
                         visFnTable, 10);
                   });

    id visFnTables[2] = {visFnTable, visFnTable};
    const id *visFnTableArray = visFnTables;
    addEncoderCase(cases, ser, stream, @"render_set_tile_visible_function_tables_range",
                   @"setTileVisibleFunctionTables:withBufferRange:",
                   @{@"visible_function_table_ref" : @(STUB_VISIBLE_FN_TABLE_REF),
                     @"first" : @11, @"count" : @2},
                   ^(id enc) {
                     ((void (*)(id, SEL, const id *, NSRange))objc_msgSend)(
                         enc, sel_getUid("setTileVisibleFunctionTables:withBufferRange:"),
                         visFnTableArray, NSMakeRange(11, 2));
                   });

    addEncoderCase(cases, ser, stream, @"render_set_tile_intersection_function_table",
                   @"setTileIntersectionFunctionTable:atBufferIndex:",
                   @{@"intersection_function_table_ref" :
                         @(STUB_INTERSECTION_FN_TABLE_REF),
                     @"index" : @13},
                   ^(id enc) {
                     ((void (*)(id, SEL, id, unsigned long))objc_msgSend)(
                         enc,
                         sel_getUid("setTileIntersectionFunctionTable:atBufferIndex:"),
                         isectFnTable, 13);
                   });

    id isectFnTables[2] = {isectFnTable, isectFnTable};
    const id *isectFnTableArray = isectFnTables;
    addEncoderCase(cases, ser, stream,
                   @"render_set_tile_intersection_function_tables_range",
                   @"setTileIntersectionFunctionTables:withBufferRange:",
                   @{@"intersection_function_table_ref" :
                         @(STUB_INTERSECTION_FN_TABLE_REF),
                     @"first" : @14, @"count" : @2},
                   ^(id enc) {
                     ((void (*)(id, SEL, const id *, NSRange))objc_msgSend)(
                         enc,
                         sel_getUid("setTileIntersectionFunctionTables:withBufferRange:"),
                         isectFnTableArray, NSMakeRange(14, 2));
                   });

    // `getTileDimensions:` takes `^{?=SS}` — it fills a caller's two-`u16`
    // struct rather than the stream. Driven anyway, because "the type encoding
    // says it writes elsewhere" and "it emits nothing" are two different
    // claims and only the capture settles the second.
    addEncoderCase(cases, ser, stream, @"render_get_tile_dimensions",
                   @"getTileDimensions:", @{}, ^(id enc) {
                     unsigned short dims[2] = {0, 0};
                     ((void (*)(id, SEL, void *))objc_msgSend)(
                         enc, sel_getUid("getTileDimensions:"), dims);
                   });

    // The render encoder's threadgroup-memory bind is the *tile* stage's
    // imageblock memory, and it is the one selector in this family whose name
    // does not say "tile". It emits `0x9c`, which fills the one hole in the
    // `0x9b`–`0xa4` run.
    //
    // Gated on `TileShaders` alone — driven under `TileShaders` and
    // `ImageBlocks` together first, then under `TileShaders` by itself, and it
    // emits either way, so `ImageBlocks` is not part of the gate. Its
    // compute-encoder namesake emits `0xd3` with no flag at all, which is what
    // made a silent reading here worth a second look rather than a manifest
    // row.
    addEncoderCase(cases, ser, stream, @"render_set_threadgroup_memory_length",
                   @"setThreadgroupMemoryLength:offset:atIndex:",
                   @{@"length" : @0x1234, @"offset" : @0x2345, @"index" : @5},
                   ^(id enc) {
                     ((void (*)(id, SEL, unsigned long, unsigned long,
                                unsigned long))objc_msgSend)(
                         enc, sel_getUid("setThreadgroupMemoryLength:offset:atIndex:"),
                         0x1234, 0x2345, 5);
                   });

    // The two bare property reads, driven rather than assumed. `getTileDimensions:`
    // above is the reason: it also looked like a pure query from its encoding
    // and it emits a record. These do not, and that is now measured — they land
    // on `silent` with the capability forced on, which is the only state in
    // which the answer means anything.
    addEncoderCase(cases, ser, stream, @"render_tile_width", @"tileWidth", @{},
                   ^(id enc) {
                     ((unsigned long (*)(id, SEL))objc_msgSend)(enc,
                                                                sel_getUid("tileWidth"));
                   });
    addEncoderCase(cases, ser, stream, @"render_tile_height", @"tileHeight", @{},
                   ^(id enc) {
                     ((unsigned long (*)(id, SEL))objc_msgSend)(enc,
                                                                sel_getUid("tileHeight"));
                   });
  });

  // --- The ray-tracing binds on the other four stages ----------------------
  //
  // Twenty selectors of one shape. The tile stage's five are refused by the
  // serializer, which is a statement about the tile stage and not about the
  // family -- so the other four are driven rather than generalised from it.
  // Each stage takes its own index base so a record that named the wrong stage
  // is visible in the index alone.
  {
    id accel = [[StubAccelStruct alloc] init];
    id visFnTable = [[StubVisibleFnTable alloc] init];
    id isectFnTable = [[StubIntersectionFnTable alloc] init];
    unsigned base = 10;
    for (NSString *stage in @[ @"Vertex", @"Fragment", @"Mesh", @"Object" ]) {
      addRayTracingBindCases(cases, ser, stream, stage, accel, visFnTable, isectFnTable, base);
      base += 10;
    }
  }

  // --- The remaining untriaged render state --------------------------------
  //
  // Everything left on this encoder that is neither a patch draw nor an
  // encoder-lifecycle hook. Driven at default capability state; anything that
  // comes back silent gets re-driven under the flag its family is gated on,
  // because a silent measured at the default state is a statement about this
  // harness.
  addEncoderCase(cases, ser, stream, @"render_set_color_store_action_options",
                 @"setColorStoreActionOptions:atIndex:",
                 @{@"options" : @0x1111, @"index" : @3}, ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long))objc_msgSend)(
                       enc, sel_getUid("setColorStoreActionOptions:atIndex:"), 0x1111, 3);
                 });
  addEncoderCase(cases, ser, stream, @"render_set_depth_store_action_options",
                 @"setDepthStoreActionOptions:", @{@"options" : @0x2222}, ^(id enc) {
                   ((void (*)(id, SEL, unsigned long))objc_msgSend)(
                       enc, sel_getUid("setDepthStoreActionOptions:"), 0x2222);
                 });
  addEncoderCase(cases, ser, stream, @"render_set_stencil_store_action_options",
                 @"setStencilStoreActionOptions:", @{@"options" : @0x3333}, ^(id enc) {
                   ((void (*)(id, SEL, unsigned long))objc_msgSend)(
                       enc, sel_getUid("setStencilStoreActionOptions:"), 0x3333);
                 });

  // The three MSAA resolve targets. `yInvert` is a `c` between two `Q`, which
  // is where a record that widened it would show.
  addEncoderCase(cases, ser, stream, @"render_set_color_resolve_texture",
                 @"setColorResolveTexture:slice:depthPlane:level:yInvert:atIndex:",
                 @{@"texture_ref" : @(STUB_TEXTURE_REF), @"slice" : @0x11,
                   @"depth_plane" : @0x22, @"level" : @0x33, @"y_invert" : @1,
                   @"index" : @2},
                 ^(id enc) {
                   ((void (*)(id, SEL, id, unsigned long, unsigned long, unsigned long,
                              char, unsigned long))objc_msgSend)(
                       enc,
                       sel_getUid("setColorResolveTexture:slice:depthPlane:level:"
                                  "yInvert:atIndex:"),
                       tex, 0x11, 0x22, 0x33, 1, 2);
                 });
  addEncoderCase(cases, ser, stream, @"render_set_depth_resolve_texture",
                 @"setDepthResolveTexture:slice:depthPlane:level:yInvert:",
                 @{@"texture_ref" : @(STUB_TEXTURE_REF), @"slice" : @0x44,
                   @"depth_plane" : @0x55, @"level" : @0x66, @"y_invert" : @0},
                 ^(id enc) {
                   ((void (*)(id, SEL, id, unsigned long, unsigned long, unsigned long,
                              char))objc_msgSend)(
                       enc,
                       sel_getUid("setDepthResolveTexture:slice:depthPlane:level:yInvert:"),
                       tex, 0x44, 0x55, 0x66, 0);
                 });
  addEncoderCase(cases, ser, stream, @"render_set_stencil_resolve_texture",
                 @"setStencilResolveTexture:slice:depthPlane:level:yInvert:",
                 @{@"texture_ref" : @(STUB_TEXTURE_REF), @"slice" : @0x77,
                   @"depth_plane" : @0x88, @"level" : @0x99, @"y_invert" : @1},
                 ^(id enc) {
                   ((void (*)(id, SEL, id, unsigned long, unsigned long, unsigned long,
                              char))objc_msgSend)(
                       enc,
                       sel_getUid("setStencilResolveTexture:slice:depthPlane:level:yInvert:"),
                       tex, 0x77, 0x88, 0x99, 1);
                 });

  // Four `f32` and an index. Distinct powers of two so each lands exactly.
  addEncoderCase(cases, ser, stream, @"render_set_clip_plane",
                 @"setClipPlane:p2:p3:p4:atIndex:",
                 @{@"p1" : @0.25, @"p2" : @0.5, @"p3" : @0.75, @"p4" : @0.125,
                   @"index" : @4},
                 ^(id enc) {
                   ((void (*)(id, SEL, float, float, float, float,
                              unsigned long))objc_msgSend)(
                       enc, sel_getUid("setClipPlane:p2:p3:p4:atIndex:"), 0.25f, 0.5f,
                       0.75f, 0.125f, 4);
                 });

  addEncoderCase(cases, ser, stream, @"render_set_tessellation_factor_buffer",
                 @"setTessellationFactorBuffer:offset:instanceStride:",
                 @{@"buffer_ref" : @(STUB_BUFFER_REF), @"offset" : @0x3456,
                   @"instance_stride" : @0x4567},
                 ^(id enc) {
                   ((void (*)(id, SEL, id, unsigned long, unsigned long))objc_msgSend)(
                       enc, sel_getUid("setTessellationFactorBuffer:offset:instanceStride:"),
                       vbuf, 0x3456, 0x4567);
                 });

  // `setPrimitiveRestartEnabled:` (no index) is already covered; this is the
  // two-argument form, whose `BOOL` sits before a `Q` at an offset the packing
  // makes 20 rather than 24.
  addEncoderCase(cases, ser, stream, @"render_set_primitive_restart_enabled_index",
                 @"setPrimitiveRestartEnabled:index:",
                 @{@"enabled" : @1, @"index" : @0x5678}, ^(id enc) {
                   ((void (*)(id, SEL, char, unsigned long))objc_msgSend)(
                       enc, sel_getUid("setPrimitiveRestartEnabled:index:"), 1, 0x5678);
                 });

  // Driven at both `MTLDepthClipMode` values, because one of them is the state
  // the encoder already holds and a serializer that skips a redundant write
  // would look silent at that one. Its non-SPI sibling `setDepthClipMode:`
  // emits `0x6d` at either.
  addEncoderCase(cases, ser, stream, @"render_set_depth_clip_mode_spi",
                 @"setDepthClipModeSPI:", @{@"mode" : @1}, ^(id enc) {
                   ((void (*)(id, SEL, unsigned long))objc_msgSend)(
                       enc, sel_getUid("setDepthClipModeSPI:"), 1);
                 });
  addEncoderCase(cases, ser, stream, @"render_set_depth_clip_mode_spi_clip",
                 @"setDepthClipModeSPI:", @{@"mode" : @0}, ^(id enc) {
                   ((void (*)(id, SEL, unsigned long))objc_msgSend)(
                       enc, sel_getUid("setDepthClipModeSPI:"), 0);
                 });
  addEncoderCase(cases, ser, stream, @"render_set_transform_feedback_state",
                 @"setTransformFeedbackState:", @{@"state" : @0x6789}, ^(id enc) {
                   ((void (*)(id, SEL, unsigned long))objc_msgSend)(
                       enc, sel_getUid("setTransformFeedbackState:"), 0x6789);
                 });

  // --- The four patch draws ------------------------------------------------
  //
  // Tessellation's consumers. `setTessellationFactorBuffer:offset:instanceStride:`
  // and `setTessellationFactorScale:` both emit, so the state a tessellated draw
  // needs reaches the wire; whether any draw can spend it is what these settle,
  // and it is not something the emitting siblings imply either way.
  // `ibuf` is STUB_BUFFER_REF, the same ref `vbuf` carries, so these declare
  // their own second and third buffers: an indexed indirect patch draw names
  // three, and three slots cannot be told apart by two values.
  id cpbuf = [[StubBuffer alloc] initWithRef:STUB_BUFFER_DST_REF];
  id patchIndirect = [[StubBuffer alloc] initWithRef:STUB_BUFFER_THIRD_REF];

  addEncoderCase(cases, ser, stream, @"render_draw_patches",
                 @"drawPatches:patchStart:patchCount:patchIndexBuffer:"
                 @"patchIndexBufferOffset:instanceCount:baseInstance:",
                 @{@"control_points" : @3, @"patch_start" : @0x11,
                   @"patch_count" : @0x22, @"patch_index_buffer_ref" : @(STUB_BUFFER_REF),
                   @"patch_index_buffer_offset" : @0x33, @"instance_count" : @0x44,
                   @"base_instance" : @0x55},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long, unsigned long, id,
                              unsigned long, unsigned long, unsigned long))objc_msgSend)(
                       enc,
                       sel_getUid("drawPatches:patchStart:patchCount:patchIndexBuffer:"
                                  "patchIndexBufferOffset:instanceCount:baseInstance:"),
                       3, 0x11, 0x22, vbuf, 0x33, 0x44, 0x55);
                 });

  // Every field of `0x0d` is narrowed to 16 bits from a `Q`, which is the same
  // thing the compact draws do — and those have a *wide* opcode the serializer
  // switches to above 16 bits. Whether the patch draws have one is the question
  // this asks, and getting it wrong is silent wrong geometry rather than a
  // refusal: a `patchCount` of 0x10000 truncated to 16 bits draws nothing.
  addEncoderCase(cases, ser, stream, @"render_draw_patches_over_16bit",
                 @"drawPatches:patchStart:patchCount:patchIndexBuffer:"
                 @"patchIndexBufferOffset:instanceCount:baseInstance:",
                 @{@"control_points" : @3, @"patch_start" : @0x11,
                   @"patch_count" : @0x10000, @"patch_index_buffer_ref" : @(STUB_BUFFER_REF),
                   @"patch_index_buffer_offset" : @0x33, @"instance_count" : @0x44,
                   @"base_instance" : @0x55},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long, unsigned long, id,
                              unsigned long, unsigned long, unsigned long))objc_msgSend)(
                       enc,
                       sel_getUid("drawPatches:patchStart:patchCount:patchIndexBuffer:"
                                  "patchIndexBufferOffset:instanceCount:baseInstance:"),
                       3, 0x11, 0x10000, vbuf, 0x33, 0x44, 0x55);
                 });

  addEncoderCase(cases, ser, stream, @"render_draw_patches_indirect",
                 @"drawPatches:patchIndexBuffer:patchIndexBufferOffset:"
                 @"indirectBuffer:indirectBufferOffset:",
                 @{@"control_points" : @3,
                   @"patch_index_buffer_ref" : @(STUB_BUFFER_REF),
                   @"patch_index_buffer_offset" : @0x11,
                   @"indirect_buffer_ref" : @(STUB_BUFFER_THIRD_REF),
                   @"indirect_buffer_offset" : @0x22},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, id, unsigned long, id,
                              unsigned long))objc_msgSend)(
                       enc,
                       sel_getUid("drawPatches:patchIndexBuffer:patchIndexBufferOffset:"
                                  "indirectBuffer:indirectBufferOffset:"),
                       3, vbuf, 0x11, patchIndirect, 0x22);
                 });

  addEncoderCase(cases, ser, stream, @"render_draw_indexed_patches",
                 @"drawIndexedPatches:patchStart:patchCount:patchIndexBuffer:"
                 @"patchIndexBufferOffset:controlPointIndexBuffer:"
                 @"controlPointIndexBufferOffset:instanceCount:baseInstance:",
                 @{@"control_points" : @3, @"patch_start" : @0x11,
                   @"patch_count" : @0x22, @"patch_index_buffer_ref" : @(STUB_BUFFER_REF),
                   @"patch_index_buffer_offset" : @0x33,
                   @"control_point_index_buffer_ref" : @(STUB_BUFFER_DST_REF),
                   @"control_point_index_buffer_offset" : @0x44,
                   @"instance_count" : @0x55, @"base_instance" : @0x66},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long, unsigned long, id,
                              unsigned long, id, unsigned long, unsigned long,
                              unsigned long))objc_msgSend)(
                       enc,
                       sel_getUid("drawIndexedPatches:patchStart:patchCount:"
                                  "patchIndexBuffer:patchIndexBufferOffset:"
                                  "controlPointIndexBuffer:controlPointIndexBufferOffset:"
                                  "instanceCount:baseInstance:"),
                       3, 0x11, 0x22, vbuf, 0x33, cpbuf, 0x44, 0x55, 0x66);
                 });

  // The indexed form's wide sibling. Driven rather than inferred from `0x0d`
  // having one: the wide opcode is not derivable from the compact one, only
  // captured — and `0x0e` being free is a hint, not evidence.
  addEncoderCase(cases, ser, stream, @"render_draw_indexed_patches_over_16bit",
                 @"drawIndexedPatches:patchStart:patchCount:patchIndexBuffer:"
                 @"patchIndexBufferOffset:controlPointIndexBuffer:"
                 @"controlPointIndexBufferOffset:instanceCount:baseInstance:",
                 @{@"control_points" : @3, @"patch_start" : @0x11,
                   @"patch_count" : @0x10000,
                   @"patch_index_buffer_ref" : @(STUB_BUFFER_REF),
                   @"patch_index_buffer_offset" : @0x33,
                   @"control_point_index_buffer_ref" : @(STUB_BUFFER_DST_REF),
                   @"control_point_index_buffer_offset" : @0x44,
                   @"instance_count" : @0x55, @"base_instance" : @0x66},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long, unsigned long, id,
                              unsigned long, id, unsigned long, unsigned long,
                              unsigned long))objc_msgSend)(
                       enc,
                       sel_getUid("drawIndexedPatches:patchStart:patchCount:"
                                  "patchIndexBuffer:patchIndexBufferOffset:"
                                  "controlPointIndexBuffer:controlPointIndexBufferOffset:"
                                  "instanceCount:baseInstance:"),
                       3, 0x11, 0x10000, vbuf, 0x33, cpbuf, 0x44, 0x55, 0x66);
                 });

  addEncoderCase(cases, ser, stream, @"render_draw_indexed_patches_indirect",
                 @"drawIndexedPatches:patchIndexBuffer:patchIndexBufferOffset:"
                 @"controlPointIndexBuffer:controlPointIndexBufferOffset:"
                 @"indirectBuffer:indirectBufferOffset:",
                 @{@"control_points" : @3,
                   @"patch_index_buffer_ref" : @(STUB_BUFFER_REF),
                   @"patch_index_buffer_offset" : @0x11,
                   @"control_point_index_buffer_ref" : @(STUB_BUFFER_DST_REF),
                   @"control_point_index_buffer_offset" : @0x22,
                   @"indirect_buffer_ref" : @(STUB_BUFFER_THIRD_REF),
                   @"indirect_buffer_offset" : @0x33},
                 ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, id, unsigned long, id, unsigned long,
                              id, unsigned long))objc_msgSend)(
                       enc,
                       sel_getUid("drawIndexedPatches:patchIndexBuffer:"
                                  "patchIndexBufferOffset:controlPointIndexBuffer:"
                                  "controlPointIndexBufferOffset:indirectBuffer:"
                                  "indirectBufferOffset:"),
                       3, vbuf, 0x11, cpbuf, 0x22, patchIndirect, 0x33);
                 });

  // --- Encoder lifecycle and the split machinery ---------------------------
  //
  // Everything left on this class. None of these is a Metal command a guest
  // issues; they are the serializer's own bookkeeping for splitting a long
  // encoder across command buffers, and for the render pass's cleared/loaded
  // state. Driven rather than assumed to be silent, because `getTileDimensions:`
  // was in this shape too and emitted a record.
  addEncoderCase(cases, ser, stream, @"render_force_load_actions", @"forceLoadActions",
                 @{}, ^(id enc) {
                   ((void (*)(id, SEL))objc_msgSend)(enc, sel_getUid("forceLoadActions"));
                 });
  addEncoderCase(cases, ser, stream, @"render_force_store_actions_for_position",
                 @"forceStoreActionsForPosition:", @{}, ^(id enc) {
                   ((void (*)(id, SEL, unsigned long))objc_msgSend)(
                       enc, sel_getUid("forceStoreActionsForPosition:"), 0);
                 });
  addEncoderCase(cases, ser, stream, @"render_set_encoder_position",
                 @"setEncoderPosition:", @{}, ^(id enc) {
                   ((void (*)(id, SEL, unsigned long))objc_msgSend)(
                       enc, sel_getUid("setEncoderPosition:"), 0);
                 });
  addEncoderCase(cases, ser, stream, @"render_set_depth_cleared", @"setDepthCleared", @{},
                 ^(id enc) {
                   ((void (*)(id, SEL))objc_msgSend)(enc, sel_getUid("setDepthCleared"));
                 });
  addEncoderCase(cases, ser, stream, @"render_set_stencil_cleared", @"setStencilCleared",
                 @{}, ^(id enc) {
                   ((void (*)(id, SEL))objc_msgSend)(enc, sel_getUid("setStencilCleared"));
                 });
  addEncoderCase(cases, ser, stream, @"render_is_memoryless_render", @"isMemorylessRender",
                 @{}, ^(id enc) {
                   ((char (*)(id, SEL))objc_msgSend)(enc, sel_getUid("isMemorylessRender"));
                 });
  addEncoderCase(cases, ser, stream, @"render_add_render_target_references",
                 @"addRenderTargetReferences", @{}, ^(id enc) {
                   ((char (*)(id, SEL))objc_msgSend)(
                       enc, sel_getUid("addRenderTargetReferences"));
                 });
  addEncoderCase(cases, ser, stream, @"render_split", @"split", @{}, ^(id enc) {
    ((void (*)(id, SEL))objc_msgSend)(enc, sel_getUid("split"));
  });
  addEncoderCase(cases, ser, stream, @"render_fix_store_actions", @"fixStoreActions:", @{},
                 ^(id enc) {
                   ((void (*)(id, SEL, id))objc_msgSend)(
                       enc, sel_getUid("fixStoreActions:"), nil);
                 });
  // `@?16` is a block. Registering a handler is bookkeeping by construction —
  // a function pointer cannot cross to the host — but the claim is still made
  // by a case rather than by that argument.
  addEncoderCase(cases, ser, stream, @"render_add_split_handler", @"addSplitHandler:", @{},
                 ^(id enc) {
                   ((void (*)(id, SEL, void (^)(void)))objc_msgSend)(
                       enc, sel_getUid("addSplitHandler:"), ^{
                       });
                 });

  // --- The five selectors this class shares with the other encoders ---------
  //
  // Each of these was driven on the blit or the compute encoder and left
  // undriven here, so five of this class's rows rested on nothing at all. A
  // selector's outcome on one class is not evidence about the same selector on
  // another -- `writeDescriptor` is silent on the compute encoder until
  // `ComputePassDescriptorDispatchType` is forced, and `getType` is silent on
  // blit -- which is the same "a family is not uniform" rule one axis over.
  //
  // They are driven at the default capability state; if one of them is gated,
  // the attribution passes name the flag rather than this file guessing it.
  addEncoderCase(cases, ser, stream, @"render_flush_writes", @"flushWrites", @{},
                 ^(id enc) {
                   ((void (*)(id, SEL))objc_msgSend)(enc, sel_getUid("flushWrites"));
                 });

  addEncoderCase(cases, ser, stream, @"render_handle_splits", @"handleSplits", @{},
                 ^(id enc) {
                   (void)((char (*)(id, SEL))objc_msgSend)(enc,
                                                           sel_getUid("handleSplits"));
                 });

  addEncoderCase(cases, ser, stream, @"render_get_type", @"getType", @{}, ^(id enc) {
    (void)((unsigned long (*)(id, SEL))objc_msgSend)(enc, sel_getUid("getType"));
  });

  // `writeDescriptor` is the fifth, and it is not bookkeeping: it re-emits the
  // render pass descriptor, opcode 0x1a. The perturbation family below moves
  // one property at a time off the baseline.
  addRenderPassCase(cases, ser, stream, @"render_write_descriptor",
                    @{@"color0_texture_ref" : @(STUB_TEXTURE_REF)},
                    ^(MTLRenderPassDescriptor *rp) {
                      (void)rp;
                    });

  // The pass-level scalars, one case each. Which of them reach the wire is the
  // question -- `MTLRenderPassDescriptor` declares ten and the record's tail is
  // 28 bytes, so they cannot all be there at 8 bytes apiece.
  addRenderPassCase(cases, ser, stream, @"render_pass_target_size",
                    @{@"color0_texture_ref" : @(STUB_TEXTURE_REF)},
                    ^(MTLRenderPassDescriptor *rp) {
                      rp.renderTargetWidth = 0x1234;
                      rp.renderTargetHeight = 0x5678;
                    });

  addRenderPassCase(cases, ser, stream, @"render_pass_array_length",
                    @{@"color0_texture_ref" : @(STUB_TEXTURE_REF)},
                    ^(MTLRenderPassDescriptor *rp) {
                      rp.renderTargetArrayLength = 0x11;
                    });

  // Refused outright at the default state -- the encoder's designated
  // initializer returns nil rather than asserting, which is a fifth way a case
  // can fail and one `addCaseOnEncoder` already reports by name. The flag whose
  // name matches is the one that unlocks it here, which is not something to
  // assume after `TextureDescriptor2`: it is checked by the case's own outcome.
  withCapability(ser, @"DefaultRasterSampleCount", ^{
    addRenderPassCaseN(cases, ser, stream, @"render_pass_raster_sample_count",
                       @{@"color0_texture_ref" : @(STUB_TEXTURE_REF)}, 2,
                       ^(MTLRenderPassDescriptor *rp) {
                         rp.defaultRasterSampleCount = 4;
                       });
  });

  withCapability(ser, @"RasterizationRateMap", ^{
    addRenderPassCaseN(cases, ser, stream, @"render_pass_rate_map",
                       @{@"color0_texture_ref" : @(STUB_TEXTURE_REF),
                         @"rate_map_ref" : @(STUB_RATE_MAP_REF)}, 2,
                       ^(MTLRenderPassDescriptor *rp) {
                         rp.rasterizationRateMap =
                             (id<MTLRasterizationRateMap>)[[StubRateMap alloc] init];
                       });
  });

  addRenderPassCase(cases, ser, stream, @"render_pass_tile_size",
                    @{@"color0_texture_ref" : @(STUB_TEXTURE_REF)},
                    ^(MTLRenderPassDescriptor *rp) {
                      rp.tileWidth = 0x21;
                      rp.tileHeight = 0x22;
                    });

  addRenderPassCase(cases, ser, stream, @"render_pass_imageblock",
                    @{@"color0_texture_ref" : @(STUB_TEXTURE_REF)},
                    ^(MTLRenderPassDescriptor *rp) {
                      rp.imageblockSampleLength = 0x40;
                      rp.threadgroupMemoryLength = 0x80;
                    });

  // The same four properties again with `TileShaders` forced on, where they do
  // reach the wire -- as a **second record** rather than as fields of the pass
  // descriptor. The two cases above are the default-state negative result and
  // stay: what a capability changes is only a finding next to what it changed
  // from.
  withCapability(ser, @"TileShaders", ^{
    addRenderPassCaseN(cases, ser, stream, @"render_pass_tile_size_capable",
                       @{@"color0_texture_ref" : @(STUB_TEXTURE_REF)}, 2,
                       ^(MTLRenderPassDescriptor *rp) {
                         rp.tileWidth = 0x21;
                         rp.tileHeight = 0x22;
                       });

    addRenderPassCaseN(cases, ser, stream, @"render_pass_imageblock_capable",
                       @{@"color0_texture_ref" : @(STUB_TEXTURE_REF)}, 3,
                       ^(MTLRenderPassDescriptor *rp) {
                         rp.imageblockSampleLength = 0x40;
                         rp.threadgroupMemoryLength = 0x80;
                       });
  });

  // `0x1e`..`0x24` is a run with two holes at `0x1f` and `0x20`, and a hole in
  // an opcode run is a property nobody has driven rather than a number Apple
  // skipped. `setSamplePositions:count:` is the pass property left that has a
  // capability of its own.
  withCapability(ser, @"ProgrammableSamplePositions", ^{
    addRenderPassCaseN(cases, ser, stream, @"render_pass_sample_positions",
                       @{@"color0_texture_ref" : @(STUB_TEXTURE_REF),
                         @"sample_position_count" : @2}, 2,
                       ^(MTLRenderPassDescriptor *rp) {
                         MTLSamplePosition pos[2] = {{0.25f, 0.75f}, {0.125f, 0.375f}};
                         [rp setSamplePositions:pos count:2];
                       });
  });

  // The last `MTLRenderPassDescriptor` property, and the last hole in the run.
  //
  // Driven twice. The stub-buffer case below is the honest negative result for
  // a forwarding stub, and it is not conclusive: a serializer that asks the
  // sample buffer for something a stub answers zero to would go silent for a
  // reason that says nothing about Apple, which is exactly what `gCrashed`
  // exists to keep separate. So the same property is driven again with a
  // **real** `MTLCounterSampleBuffer` off this host's device. That one is
  // conditional -- a GPU with no counter sets cannot create one -- so it
  // announces the reason it did not run rather than disappearing.
  {
    MTLCounterSampleBufferDescriptor *csd =
        [[MTLCounterSampleBufferDescriptor alloc] init];
    id<MTLCounterSet> set = gDevice.counterSets.firstObject;
    id<MTLCounterSampleBuffer> real = nil;
    if (set) {
      csd.counterSet = set;
      csd.storageMode = MTLStorageModeShared;
      csd.sampleCount = 4;
      NSError *err = nil;
      real = [gDevice newCounterSampleBufferWithDescriptor:csd error:&err];
      if (!real) {
        fprintf(stderr,
                "note: no MTLCounterSampleBuffer (%s); "
                "render_pass_sample_buffer_real not driven\n",
                err.localizedDescription.UTF8String ?: "unknown");
      }
    } else {
      fprintf(stderr, "note: this device exposes no counter sets; "
                      "render_pass_sample_buffer_real not driven\n");
    }
    if (real) {
      addRenderPassCase(cases, ser, stream, @"render_pass_sample_buffer_real",
                        @{@"color0_texture_ref" : @(STUB_TEXTURE_REF)},
                        ^(MTLRenderPassDescriptor *rp) {
                          MTLRenderPassSampleBufferAttachmentDescriptor *s =
                              rp.sampleBufferAttachments[0];
                          s.sampleBuffer = real;
                          s.startOfVertexSampleIndex = 0x11;
                          s.endOfVertexSampleIndex = 0x22;
                          s.startOfFragmentSampleIndex = 0x33;
                          s.endOfFragmentSampleIndex = 0x44;
                        });
    }
  }

  addRenderPassCase(cases, ser, stream, @"render_pass_sample_buffer",
                    @{@"color0_texture_ref" : @(STUB_TEXTURE_REF)},
                    ^(MTLRenderPassDescriptor *rp) {
                      MTLRenderPassSampleBufferAttachmentDescriptor *s =
                          rp.sampleBufferAttachments[0];
                      s.sampleBuffer =
                          (id<MTLCounterSampleBuffer>)[[StubBuffer alloc] init];
                      s.startOfVertexSampleIndex = 0x11;
                      s.endOfVertexSampleIndex = 0x22;
                      s.startOfFragmentSampleIndex = 0x33;
                      s.endOfFragmentSampleIndex = 0x44;
                    });

  addRenderPassCase(cases, ser, stream, @"render_pass_visibility_buffer",
                    @{@"color0_texture_ref" : @(STUB_TEXTURE_REF),
                      @"visibility_buffer_ref" : @(STUB_BUFFER_REF)},
                    ^(MTLRenderPassDescriptor *rp) {
                      rp.visibilityResultBuffer =
                          (id<MTLBuffer>)[[StubBuffer alloc] init];
                    });

  // The colour attachment's own fields. Slot 0 carries the baseline, so these
  // move level/slice/plane, then the resolve half, then the slot index itself
  // -- the last is what pins the stride rather than assuming 8 equal slots.
  addRenderPassCase(cases, ser, stream, @"render_pass_color_level_slice",
                    @{@"color0_texture_ref" : @(STUB_TEXTURE_REF)},
                    ^(MTLRenderPassDescriptor *rp) {
                      rp.colorAttachments[0].level = 2;
                      rp.colorAttachments[0].slice = 3;
                      rp.colorAttachments[0].depthPlane = 4;
                    });

  addRenderPassCase(cases, ser, stream, @"render_pass_color_resolve",
                    @{@"color0_texture_ref" : @(STUB_TEXTURE_REF),
                      @"color0_resolve_texture_ref" : @(STUB_TEXTURE_DST_REF)},
                    ^(MTLRenderPassDescriptor *rp) {
                      rp.colorAttachments[0].resolveTexture =
                          (id<MTLTexture>)[[StubTexture alloc]
                              initWithRef:STUB_TEXTURE_DST_REF];
                      rp.colorAttachments[0].resolveLevel = 5;
                      rp.colorAttachments[0].resolveSlice = 6;
                      rp.colorAttachments[0].resolveDepthPlane = 7;
                      rp.colorAttachments[0].storeAction =
                          MTLStoreActionMultisampleResolve;
                    });

  addRenderPassCase(cases, ser, stream, @"render_pass_color_slot_three",
                    @{@"color0_texture_ref" : @(STUB_TEXTURE_REF),
                      @"color3_texture_ref" : @(STUB_TEXTURE_DST_REF)},
                    ^(MTLRenderPassDescriptor *rp) {
                      rp.colorAttachments[3].texture =
                          (id<MTLTexture>)[[StubTexture alloc]
                              initWithRef:STUB_TEXTURE_DST_REF];
                      rp.colorAttachments[3].loadAction = MTLLoadActionLoad;
                      rp.colorAttachments[3].storeAction = MTLStoreActionDontCare;
                      rp.colorAttachments[3].clearColor =
                          MTLClearColorMake(0.125, 0.375, 0.625, 0.875);
                    });

  // Depth and stencil. Both slots are written in every capture, so a case that
  // only reads the baseline cannot tell a field from a constant; each of these
  // moves the load/store pair and the clear value together with the texture.
  addRenderPassCase(cases, ser, stream, @"render_pass_depth",
                    @{@"color0_texture_ref" : @(STUB_TEXTURE_REF),
                      @"depth_texture_ref" : @(STUB_TEXTURE_R8_REF)},
                    ^(MTLRenderPassDescriptor *rp) {
                      rp.depthAttachment.texture = (id<MTLTexture>)[[StubTexture alloc]
                          initWithRef:STUB_TEXTURE_R8_REF];
                      rp.depthAttachment.loadAction = MTLLoadActionClear;
                      rp.depthAttachment.storeAction = MTLStoreActionStore;
                      rp.depthAttachment.clearDepth = 0.375;
                      rp.depthAttachment.level = 1;
                    });

  addRenderPassCase(cases, ser, stream, @"render_pass_stencil",
                    @{@"color0_texture_ref" : @(STUB_TEXTURE_REF),
                      @"stencil_texture_ref" : @(STUB_TEXTURE_DST_REF)},
                    ^(MTLRenderPassDescriptor *rp) {
                      rp.stencilAttachment.texture =
                          (id<MTLTexture>)[[StubTexture alloc]
                              initWithRef:STUB_TEXTURE_DST_REF];
                      rp.stencilAttachment.loadAction = MTLLoadActionClear;
                      rp.stencilAttachment.storeAction = MTLStoreActionStore;
                      rp.stencilAttachment.clearStencil = 0x5a;
                    });

  // `MTLRenderPassDescriptor` hands back a depth and a stencil attachment whose
  // `loadAction` already reads Clear, so the two cases above set the value that
  // was there and cannot locate the field. These move it the other way.
  addRenderPassCase(cases, ser, stream, @"render_pass_depth_load",
                    @{@"color0_texture_ref" : @(STUB_TEXTURE_REF),
                      @"depth_texture_ref" : @(STUB_TEXTURE_R8_REF)},
                    ^(MTLRenderPassDescriptor *rp) {
                      rp.depthAttachment.texture = (id<MTLTexture>)[[StubTexture alloc]
                          initWithRef:STUB_TEXTURE_R8_REF];
                      rp.depthAttachment.loadAction = MTLLoadActionLoad;
                      rp.depthAttachment.storeAction = MTLStoreActionDontCare;
                      rp.depthAttachment.clearDepth = 0.625;
                    });

  // Each of the three attachment shapes has a four-byte slot no case above
  // moves: colour at +0x18, depth at +0x24, stencil at +0x20. These are the
  // three `MTLRenderPassAttachmentDescriptor` properties left, and driving them
  // is what tells a field apart from four bytes that are always zero.
  addRenderPassCase(cases, ser, stream, @"render_pass_color_store_options",
                    @{@"color0_texture_ref" : @(STUB_TEXTURE_REF)},
                    ^(MTLRenderPassDescriptor *rp) {
                      rp.colorAttachments[0].storeActionOptions =
                          MTLStoreActionOptionCustomSamplePositions;
                    });

  addRenderPassCase(cases, ser, stream, @"render_pass_depth_resolve_filter",
                    @{@"color0_texture_ref" : @(STUB_TEXTURE_REF),
                      @"depth_texture_ref" : @(STUB_TEXTURE_R8_REF)},
                    ^(MTLRenderPassDescriptor *rp) {
                      rp.depthAttachment.texture = (id<MTLTexture>)[[StubTexture alloc]
                          initWithRef:STUB_TEXTURE_R8_REF];
                      rp.depthAttachment.depthResolveFilter =
                          MTLMultisampleDepthResolveFilterMax;
                    });

  addRenderPassCase(cases, ser, stream, @"render_pass_stencil_resolve_filter",
                    @{@"color0_texture_ref" : @(STUB_TEXTURE_REF),
                      @"stencil_texture_ref" : @(STUB_TEXTURE_DST_REF)},
                    ^(MTLRenderPassDescriptor *rp) {
                      rp.stencilAttachment.texture =
                          (id<MTLTexture>)[[StubTexture alloc]
                              initWithRef:STUB_TEXTURE_DST_REF];
                      rp.stencilAttachment.stencilResolveFilter =
                          MTLMultisampleStencilResolveFilterDepthResolvedSample;
                    });

  // `storeActionOptions` is declared on the shared attachment base class, so
  // the depth and stencil slots have the same four bytes between `store_action`
  // and their clear value. Driven rather than carried over from the colour
  // case: naming a field on a sibling's evidence is the guess this crate exists
  // to refuse.
  addRenderPassCase(cases, ser, stream, @"render_pass_depth_store_options",
                    @{@"color0_texture_ref" : @(STUB_TEXTURE_REF),
                      @"depth_texture_ref" : @(STUB_TEXTURE_R8_REF)},
                    ^(MTLRenderPassDescriptor *rp) {
                      rp.depthAttachment.texture = (id<MTLTexture>)[[StubTexture alloc]
                          initWithRef:STUB_TEXTURE_R8_REF];
                      rp.depthAttachment.storeActionOptions =
                          MTLStoreActionOptionCustomSamplePositions;
                    });

  addRenderPassCase(cases, ser, stream, @"render_pass_stencil_store_options",
                    @{@"color0_texture_ref" : @(STUB_TEXTURE_REF),
                      @"stencil_texture_ref" : @(STUB_TEXTURE_DST_REF)},
                    ^(MTLRenderPassDescriptor *rp) {
                      rp.stencilAttachment.texture =
                          (id<MTLTexture>)[[StubTexture alloc]
                              initWithRef:STUB_TEXTURE_DST_REF];
                      rp.stencilAttachment.storeActionOptions =
                          MTLStoreActionOptionCustomSamplePositions;
                    });

  addRenderPassCase(cases, ser, stream, @"render_pass_stencil_load",
                    @{@"color0_texture_ref" : @(STUB_TEXTURE_REF),
                      @"stencil_texture_ref" : @(STUB_TEXTURE_DST_REF)},
                    ^(MTLRenderPassDescriptor *rp) {
                      rp.stencilAttachment.texture =
                          (id<MTLTexture>)[[StubTexture alloc]
                              initWithRef:STUB_TEXTURE_DST_REF];
                      rp.stencilAttachment.loadAction = MTLLoadActionLoad;
                      rp.stencilAttachment.storeAction = MTLStoreActionDontCare;
                      rp.stencilAttachment.clearStencil = 0xa5;
                    });

  // The only one of the five that takes an object. Two cases, because
  // `withBarrier:` is a `c` and a single case cannot tell a field that carries
  // it from a byte that is constant.
  addEncoderCase(cases, ser, stream, @"render_sample_counters",
                 @"sampleCountersInBuffer:atSampleIndex:withBarrier:",
                 @{@"counters_ref" : @(STUB_BUFFER_REF),
                   @"sample_index" : @0x2200,
                   @"barrier" : @1},
                 ^(id enc) {
                   ((void (*)(id, SEL, id, unsigned long, char))objc_msgSend)(
                       enc, sel_getUid("sampleCountersInBuffer:atSampleIndex:withBarrier:"),
                       vbuf, 0x2200, 1);
                 });

  addEncoderCase(cases, ser, stream, @"render_sample_counters_no_barrier",
                 @"sampleCountersInBuffer:atSampleIndex:withBarrier:",
                 @{@"counters_ref" : @(STUB_BUFFER_REF),
                   @"sample_index" : @0x3300,
                   @"barrier" : @0},
                 ^(id enc) {
                   ((void (*)(id, SEL, id, unsigned long, char))objc_msgSend)(
                       enc, sel_getUid("sampleCountersInBuffer:atSampleIndex:withBarrier:"),
                       vbuf, 0x3300, 0);
                 });

  return cases;
}

// --- Blit encoder records ---------------------------------------------------
//
// The 38 selectors on PGSerializerBlitCommandEncoder are the simplest bodies on
// the surface: almost all of them are a resource ref and some integers, with no
// descriptor to walk and no inline argument data. Two resources of each kind
// are used throughout so a record that wrote one ref into both slots, or
// swapped source and destination, cannot read back correct.
static NSArray *blitCases(id ser) {
  NSMutableArray *cases = [NSMutableArray array];
  id stream = [[CaptureCommandStream alloc] init];

  // Typed rather than `id`, because three cases read `pixelFormat` back off the
  // texture to build their expectation: the colour fill writes a format word
  // its selector never names, so the value has to come from the object it came
  // from rather than be transcribed.
  StubTexture *src_tex = [[StubTexture alloc] initWithRef:STUB_TEXTURE_REF];
  StubTexture *dst_tex = [[StubTexture alloc] initWithRef:STUB_TEXTURE_DST_REF];
  StubTexture *r8_tex = [[StubTexture alloc] initWithRef:STUB_TEXTURE_R8_REF
                                             pixelFormat:MTLPixelFormatR8Unorm];
  id src_buf = [[StubBuffer alloc] initWithRef:STUB_BUFFER_REF];
  id dst_buf = [[StubBuffer alloc] initWithRef:STUB_BUFFER_DST_REF];
  id fence = [[StubFence alloc] init];
  id src_icb = [[StubICB alloc] initWithRef:STUB_ICB_REF];
  id dst_icb = [[StubICB alloc] initWithRef:STUB_ICB_DST_REF];

  addBlitCase(cases, ser, stream, @"blit_update_fence", @"updateFence:",
              @{@"fence_ref" : @(STUB_FENCE_REF)}, ^(id enc) {
                ((void (*)(id, SEL, id))objc_msgSend)(enc, sel_getUid("updateFence:"),
                                                      fence);
              });

  addBlitCase(cases, ser, stream, @"blit_wait_for_fence", @"waitForFence:",
              @{@"fence_ref" : @(STUB_FENCE_REF)}, ^(id enc) {
                ((void (*)(id, SEL, id))objc_msgSend)(enc, sel_getUid("waitForFence:"),
                                                      fence);
              });

  addBlitCase(cases, ser, stream, @"blit_synchronize_resource", @"synchronizeResource:",
              @{@"resource_ref" : @(STUB_BUFFER_REF)}, ^(id enc) {
                ((void (*)(id, SEL, id))objc_msgSend)(
                    enc, sel_getUid("synchronizeResource:"), src_buf);
              });

  addBlitCase(cases, ser, stream, @"blit_synchronize_texture",
              @"synchronizeTexture:slice:level:",
              @{@"texture_ref" : @(STUB_TEXTURE_REF), @"slice" : @3, @"level" : @5},
              ^(id enc) {
                ((void (*)(id, SEL, id, unsigned long, unsigned long))objc_msgSend)(
                    enc, sel_getUid("synchronizeTexture:slice:level:"), src_tex, 3, 5);
              });

  addBlitCase(cases, ser, stream, @"blit_generate_mipmaps",
              @"generateMipmapsForTexture:",
              @{@"texture_ref" : @(STUB_TEXTURE_REF)}, ^(id enc) {
                ((void (*)(id, SEL, id))objc_msgSend)(
                    enc, sel_getUid("generateMipmapsForTexture:"), src_tex);
              });

  addBlitCase(cases, ser, stream, @"blit_fill_buffer", @"fillBuffer:range:value:",
              @{@"buffer_ref" : @(STUB_BUFFER_REF),
                @"range_location" : @0x1100,
                @"range_length" : @0x2200,
                @"value" : @0x5a},
              ^(id enc) {
                ((void (*)(id, SEL, id, NSRange, unsigned char))objc_msgSend)(
                    enc, sel_getUid("fillBuffer:range:value:"), src_buf,
                    NSMakeRange(0x1100, 0x2200), 0x5a);
              });

  addBlitCase(cases, ser, stream, @"blit_copy_buffer_to_buffer",
              @"copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:",
              @{@"source_ref" : @(STUB_BUFFER_REF),
                @"source_offset" : @0x1111,
                @"dest_ref" : @(STUB_BUFFER_DST_REF),
                @"dest_offset" : @0x2222,
                @"size" : @0x3333},
              ^(id enc) {
                ((void (*)(id, SEL, id, unsigned long, id, unsigned long,
                           unsigned long))objc_msgSend)(
                    enc,
                    sel_getUid("copyFromBuffer:sourceOffset:toBuffer:destinationOffset:"
                               "size:"),
                    src_buf, 0x1111, dst_buf, 0x2222, 0x3333);
              });

  // The whole-texture copy. Its slice and level counts are not arguments — the
  // serializer reads them off the source texture — so the expectation is read
  // off the same object rather than transcribed, which is what makes this case
  // evidence that the record is the slices form with everything at its
  // default.
  addBlitCase(cases, ser, stream, @"blit_copy_texture_to_texture",
              @"copyFromTexture:toTexture:",
              @{@"source_ref" : @(STUB_TEXTURE_REF),
                @"dest_ref" : @(STUB_TEXTURE_DST_REF),
                @"source_slice" : @0,
                @"source_level" : @0,
                @"dest_slice" : @0,
                @"dest_level" : @0,
                @"slice_count" : @([(id<MTLTexture>)src_tex arrayLength]),
                @"level_count" : @([(id<MTLTexture>)src_tex mipmapLevelCount])},
              ^(id enc) {
                ((void (*)(id, SEL, id, id))objc_msgSend)(
                    enc, sel_getUid("copyFromTexture:toTexture:"), src_tex, dst_tex);
              });

  addBlitCase(
      cases, ser, stream, @"blit_copy_texture_region",
      @"copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:"
      @"destinationSlice:destinationLevel:destinationOrigin:",
      @{@"source_ref" : @(STUB_TEXTURE_REF),
        @"source_slice" : @2,
        @"source_level" : @3,
        @"source_origin_x" : @0x11,
        @"source_origin_y" : @0x22,
        @"source_origin_z" : @0x33,
        @"size_width" : @0x44,
        @"size_height" : @0x55,
        @"size_depth" : @1,
        @"dest_ref" : @(STUB_TEXTURE_DST_REF),
        @"dest_slice" : @4,
        @"dest_level" : @5,
        @"dest_origin_x" : @0x66,
        @"dest_origin_y" : @0x77,
        @"dest_origin_z" : @0x88},
      ^(id enc) {
        ((void (*)(id, SEL, id, unsigned long, unsigned long, MTLOrigin, MTLSize, id,
                   unsigned long, unsigned long, MTLOrigin))objc_msgSend)(
            enc,
            sel_getUid("copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:"
                       "sourceSize:toTexture:destinationSlice:destinationLevel:"
                       "destinationOrigin:"),
            src_tex, 2, 3, MTLOriginMake(0x11, 0x22, 0x33), MTLSizeMake(0x44, 0x55, 1),
            dst_tex, 4, 5, MTLOriginMake(0x66, 0x77, 0x88));
      });

  addBlitCase(
      cases, ser, stream, @"blit_copy_texture_slices",
      @"copyFromTexture:sourceSlice:sourceLevel:toTexture:destinationSlice:"
      @"destinationLevel:sliceCount:levelCount:",
      @{@"source_ref" : @(STUB_TEXTURE_REF),
        @"source_slice" : @2,
        @"source_level" : @3,
        @"dest_ref" : @(STUB_TEXTURE_DST_REF),
        @"dest_slice" : @4,
        @"dest_level" : @5,
        @"slice_count" : @6,
        @"level_count" : @7},
      ^(id enc) {
        ((void (*)(id, SEL, id, unsigned long, unsigned long, id, unsigned long,
                   unsigned long, unsigned long, unsigned long))objc_msgSend)(
            enc,
            sel_getUid("copyFromTexture:sourceSlice:sourceLevel:toTexture:"
                       "destinationSlice:destinationLevel:sliceCount:levelCount:"),
            src_tex, 2, 3, dst_tex, 4, 5, 6, 7);
      });

  addBlitCase(
      cases, ser, stream, @"blit_copy_buffer_to_texture",
      @"copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:"
      @"toTexture:destinationSlice:destinationLevel:destinationOrigin:",
      @{@"source_ref" : @(STUB_BUFFER_REF),
        @"source_offset" : @0x1111,
        @"source_bytes_per_row" : @0x2222,
        @"source_bytes_per_image" : @0x3333,
        @"size_width" : @0x44,
        @"size_height" : @0x55,
        @"size_depth" : @1,
        @"dest_ref" : @(STUB_TEXTURE_DST_REF),
        @"dest_slice" : @6,
        @"dest_level" : @7,
        @"dest_origin_x" : @0x66,
        @"dest_origin_y" : @0x77,
        @"dest_origin_z" : @0x88},
      ^(id enc) {
        ((void (*)(id, SEL, id, unsigned long, unsigned long, unsigned long, MTLSize, id,
                   unsigned long, unsigned long, MTLOrigin))objc_msgSend)(
            enc,
            sel_getUid("copyFromBuffer:sourceOffset:sourceBytesPerRow:"
                       "sourceBytesPerImage:sourceSize:toTexture:destinationSlice:"
                       "destinationLevel:destinationOrigin:"),
            src_buf, 0x1111, 0x2222, 0x3333, MTLSizeMake(0x44, 0x55, 1), dst_tex, 6, 7,
            MTLOriginMake(0x66, 0x77, 0x88));
      });

  addBlitCase(
      cases, ser, stream, @"blit_copy_texture_to_buffer",
      @"copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:"
      @"destinationOffset:destinationBytesPerRow:destinationBytesPerImage:",
      @{@"source_ref" : @(STUB_TEXTURE_REF),
        @"source_slice" : @2,
        @"source_level" : @3,
        @"source_origin_x" : @0x11,
        @"source_origin_y" : @0x22,
        @"source_origin_z" : @0x33,
        @"size_width" : @0x44,
        @"size_height" : @0x55,
        @"size_depth" : @1,
        @"dest_ref" : @(STUB_BUFFER_DST_REF),
        @"dest_offset" : @0x1111,
        @"dest_bytes_per_row" : @0x2222,
        @"dest_bytes_per_image" : @0x3333},
      ^(id enc) {
        ((void (*)(id, SEL, id, unsigned long, unsigned long, MTLOrigin, MTLSize, id,
                   unsigned long, unsigned long, unsigned long))objc_msgSend)(
            enc,
            sel_getUid("copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:"
                       "sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:"
                       "destinationBytesPerImage:"),
            src_tex, 2, 3, MTLOriginMake(0x11, 0x22, 0x33), MTLSizeMake(0x44, 0x55, 1),
            dst_buf, 0x1111, 0x2222, 0x3333);
      });

  addBlitCase(cases, ser, stream, @"blit_optimize_for_gpu",
              @"optimizeContentsForGPUAccess:",
              @{@"texture_ref" : @(STUB_TEXTURE_REF)}, ^(id enc) {
                ((void (*)(id, SEL, id))objc_msgSend)(
                    enc, sel_getUid("optimizeContentsForGPUAccess:"), src_tex);
              });

  addBlitCase(cases, ser, stream, @"blit_optimize_for_gpu_slice_level",
              @"optimizeContentsForGPUAccess:slice:level:",
              @{@"texture_ref" : @(STUB_TEXTURE_REF), @"slice" : @3, @"level" : @5},
              ^(id enc) {
                ((void (*)(id, SEL, id, unsigned long, unsigned long))objc_msgSend)(
                    enc, sel_getUid("optimizeContentsForGPUAccess:slice:level:"), src_tex,
                    3, 5);
              });

  addBlitCase(cases, ser, stream, @"blit_optimize_for_cpu",
              @"optimizeContentsForCPUAccess:",
              @{@"texture_ref" : @(STUB_TEXTURE_REF)}, ^(id enc) {
                ((void (*)(id, SEL, id))objc_msgSend)(
                    enc, sel_getUid("optimizeContentsForCPUAccess:"), src_tex);
              });

  addBlitCase(cases, ser, stream, @"blit_optimize_for_cpu_slice_level",
              @"optimizeContentsForCPUAccess:slice:level:",
              @{@"texture_ref" : @(STUB_TEXTURE_REF), @"slice" : @3, @"level" : @5},
              ^(id enc) {
                ((void (*)(id, SEL, id, unsigned long, unsigned long))objc_msgSend)(
                    enc, sel_getUid("optimizeContentsForCPUAccess:slice:level:"), src_tex,
                    3, 5);
              });

  addBlitCase(cases, ser, stream, @"blit_reset_commands_in_buffer",
              @"resetCommandsInBuffer:withRange:",
              @{@"icb_ref" : @(STUB_ICB_REF),
                @"range_location" : @0x1100,
                @"range_length" : @0x2200},
              ^(id enc) {
                ((void (*)(id, SEL, id, NSRange))objc_msgSend)(
                    enc, sel_getUid("resetCommandsInBuffer:withRange:"), src_icb,
                    NSMakeRange(0x1100, 0x2200));
              });

  addBlitCase(cases, ser, stream, @"blit_optimize_indirect_command_buffer",
              @"optimizeIndirectCommandBuffer:withRange:",
              @{@"icb_ref" : @(STUB_ICB_REF),
                @"range_location" : @0x3300,
                @"range_length" : @0x4400},
              ^(id enc) {
                ((void (*)(id, SEL, id, NSRange))objc_msgSend)(
                    enc, sel_getUid("optimizeIndirectCommandBuffer:withRange:"), src_icb,
                    NSMakeRange(0x3300, 0x4400));
              });

  addBlitCase(cases, ser, stream, @"blit_copy_indirect_command_buffer",
              @"copyIndirectCommandBuffer:sourceRange:destination:destinationIndex:",
              @{@"source_ref" : @(STUB_ICB_REF),
                @"range_location" : @0x1100,
                @"range_length" : @0x2200,
                @"dest_ref" : @(STUB_ICB_DST_REF),
                @"dest_index" : @0x3300},
              ^(id enc) {
                ((void (*)(id, SEL, id, NSRange, id, unsigned long))objc_msgSend)(
                    enc,
                    sel_getUid("copyIndirectCommandBuffer:sourceRange:destination:"
                               "destinationIndex:"),
                    src_icb, NSMakeRange(0x1100, 0x2200), dst_icb, 0x3300);
              });

  // The `options:` variants. Each copy selector has one, and the plain form's
  // record already reserves room past its last named argument — this is the
  // case that decides whether that room is `options` or padding, and how wide
  // it is. `MTLBlitOptionRowLinearPVRTC` is used because it is 4: distinct from
  // 0, and distinct from every slice and level index in these cases.
  addBlitCase(
      cases, ser, stream, @"blit_copy_texture_region_options",
      @"copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:"
      @"destinationSlice:destinationLevel:destinationOrigin:options:",
      @{@"source_ref" : @(STUB_TEXTURE_REF),
        @"source_slice" : @9,
        @"source_level" : @0xa,
        @"source_origin_x" : @0x11,
        @"source_origin_y" : @0x22,
        @"source_origin_z" : @0x33,
        @"size_width" : @0x44,
        @"size_height" : @0x55,
        @"size_depth" : @1,
        @"dest_ref" : @(STUB_TEXTURE_DST_REF),
        @"dest_slice" : @0xb,
        @"dest_level" : @0xc,
        @"dest_origin_x" : @0x66,
        @"dest_origin_y" : @0x77,
        @"dest_origin_z" : @0x88,
        @"options" : @(MTLBlitOptionRowLinearPVRTC)},
      ^(id enc) {
        ((void (*)(id, SEL, id, unsigned long, unsigned long, MTLOrigin, MTLSize, id,
                   unsigned long, unsigned long, MTLOrigin, unsigned long))objc_msgSend)(
            enc,
            sel_getUid("copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:"
                       "sourceSize:toTexture:destinationSlice:destinationLevel:"
                       "destinationOrigin:options:"),
            src_tex, 9, 0xa, MTLOriginMake(0x11, 0x22, 0x33), MTLSizeMake(0x44, 0x55, 1),
            dst_tex, 0xb, 0xc, MTLOriginMake(0x66, 0x77, 0x88),
            MTLBlitOptionRowLinearPVRTC);
      });

  addBlitCase(
      cases, ser, stream, @"blit_copy_buffer_to_texture_options",
      @"copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:"
      @"toTexture:destinationSlice:destinationLevel:destinationOrigin:options:",
      @{@"source_ref" : @(STUB_BUFFER_REF),
        @"source_offset" : @0x1111,
        @"source_bytes_per_row" : @0x2222,
        @"source_bytes_per_image" : @0x3333,
        @"size_width" : @0x44,
        @"size_height" : @0x55,
        @"size_depth" : @1,
        @"dest_ref" : @(STUB_TEXTURE_DST_REF),
        @"dest_slice" : @9,
        @"dest_level" : @0xa,
        @"dest_origin_x" : @0x66,
        @"dest_origin_y" : @0x77,
        @"dest_origin_z" : @0x88,
        @"options" : @(MTLBlitOptionRowLinearPVRTC)},
      ^(id enc) {
        ((void (*)(id, SEL, id, unsigned long, unsigned long, unsigned long, MTLSize, id,
                   unsigned long, unsigned long, MTLOrigin, unsigned long))objc_msgSend)(
            enc,
            sel_getUid("copyFromBuffer:sourceOffset:sourceBytesPerRow:"
                       "sourceBytesPerImage:sourceSize:toTexture:destinationSlice:"
                       "destinationLevel:destinationOrigin:options:"),
            src_buf, 0x1111, 0x2222, 0x3333, MTLSizeMake(0x44, 0x55, 1), dst_tex, 9, 0xa,
            MTLOriginMake(0x66, 0x77, 0x88), MTLBlitOptionRowLinearPVRTC);
      });

  addBlitCase(
      cases, ser, stream, @"blit_copy_texture_to_buffer_options",
      @"copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:"
      @"destinationOffset:destinationBytesPerRow:destinationBytesPerImage:options:",
      @{@"source_ref" : @(STUB_TEXTURE_REF),
        @"source_slice" : @9,
        @"source_level" : @0xa,
        @"source_origin_x" : @0x11,
        @"source_origin_y" : @0x22,
        @"source_origin_z" : @0x33,
        @"size_width" : @0x44,
        @"size_height" : @0x55,
        @"size_depth" : @1,
        @"dest_ref" : @(STUB_BUFFER_DST_REF),
        @"dest_offset" : @0x1111,
        @"dest_bytes_per_row" : @0x2222,
        @"dest_bytes_per_image" : @0x3333,
        @"options" : @(MTLBlitOptionRowLinearPVRTC)},
      ^(id enc) {
        ((void (*)(id, SEL, id, unsigned long, unsigned long, MTLOrigin, MTLSize, id,
                   unsigned long, unsigned long, unsigned long,
                   unsigned long))objc_msgSend)(
            enc,
            sel_getUid("copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:"
                       "sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:"
                       "destinationBytesPerImage:options:"),
            src_tex, 9, 0xa, MTLOriginMake(0x11, 0x22, 0x33), MTLSizeMake(0x44, 0x55, 1),
            dst_buf, 0x1111, 0x2222, 0x3333, MTLBlitOptionRowLinearPVRTC);
      });

  addBlitCase(cases, ser, stream, @"blit_optimize_with_command",
              @"optimize:withCommand:",
              @{@"texture_ref" : @(STUB_TEXTURE_REF), @"command" : @0x77}, ^(id enc) {
                ((void (*)(id, SEL, id, unsigned int))objc_msgSend)(
                    enc, sel_getUid("optimize:withCommand:"), src_tex, 0x77);
              });

  addBlitCase(cases, ser, stream, @"blit_optimize_slice_level_with_command",
              @"optimize:slice:level:withCommand:",
              @{@"texture_ref" : @(STUB_TEXTURE_REF),
                @"slice" : @3,
                @"level" : @5,
                @"command" : @0x77},
              ^(id enc) {
                ((void (*)(id, SEL, id, unsigned long, unsigned long,
                           unsigned int))objc_msgSend)(
                    enc, sel_getUid("optimize:slice:level:withCommand:"), src_tex, 3, 5,
                    0x77);
              });

  addBlitCase(cases, ser, stream, @"blit_optimize_reset_with_command",
              @"optimizeReset:withRange:withCommand:",
              @{@"icb_ref" : @(STUB_ICB_REF),
                @"range_location" : @0x1100,
                @"range_length" : @0x2200,
                @"command" : @0x77},
              ^(id enc) {
                ((void (*)(id, SEL, id, NSRange, unsigned int))objc_msgSend)(
                    enc, sel_getUid("optimizeReset:withRange:withCommand:"), src_icb,
                    NSMakeRange(0x1100, 0x2200), 0x77);
              });

  // Second value for the `withCommand:` argument. The first capture came back
  // with the *opcode* equal to the value passed, which would make these three
  // selectors generic emitters whose caller picks the opcode rather than
  // selectors with an opcode of their own. One observation of that cannot tell
  // "the argument is the opcode" from "the opcode happens to be 0x77", so the
  // value moves and the opcode is checked against it.
  addBlitCase(cases, ser, stream, @"blit_optimize_with_command_alt",
              @"optimize:withCommand:",
              @{@"texture_ref" : @(STUB_TEXTURE_REF), @"command" : @0x55}, ^(id enc) {
                ((void (*)(id, SEL, id, unsigned int))objc_msgSend)(
                    enc, sel_getUid("optimize:withCommand:"), src_tex, 0x55);
              });

  addBlitCase(cases, ser, stream, @"blit_resolve_counters",
              @"resolveCounters:inRange:destinationBuffer:destinationOffset:",
              @{@"counters_ref" : @(STUB_BUFFER_REF),
                @"range_location" : @0x1100,
                @"range_length" : @0x2200,
                @"dest_ref" : @(STUB_BUFFER_DST_REF),
                @"dest_offset" : @0x3300},
              ^(id enc) {
                ((void (*)(id, SEL, id, NSRange, id, unsigned long))objc_msgSend)(
                    enc,
                    sel_getUid("resolveCounters:inRange:destinationBuffer:"
                               "destinationOffset:"),
                    src_buf, NSMakeRange(0x1100, 0x2200), dst_buf, 0x3300);
              });

  // The rest of the class, driven so every row in the manifest rests on an
  // observation. Several of these emit nothing; that is a result and lands in
  // the `silent` list rather than being asserted from the selector's name.
  addBlitCase(cases, ser, stream, @"blit_reset_texture_access_counters",
              @"resetTextureAccessCounters:region:mipLevel:slice:",
              @{@"texture_ref" : @(STUB_TEXTURE_REF), @"mip_level" : @3, @"slice" : @5},
              ^(id enc) {
                ((void (*)(id, SEL, id, MTLRegion, unsigned long,
                           unsigned long))objc_msgSend)(
                    enc, sel_getUid("resetTextureAccessCounters:region:mipLevel:slice:"),
                    src_tex, MTLRegionMake3D(0x11, 0x22, 0x33, 0x44, 0x55, 1), 3, 5);
              });

  addBlitCase(cases, ser, stream, @"blit_get_texture_access_counters",
              @"getTextureAccessCounters:region:mipLevel:slice:resetCounters:"
              @"countersBuffer:countersBufferOffset:",
              @{@"texture_ref" : @(STUB_TEXTURE_REF),
                @"mip_level" : @3,
                @"slice" : @5,
                @"reset_counters" : @1,
                @"dest_ref" : @(STUB_BUFFER_DST_REF),
                @"dest_offset" : @0x1111},
              ^(id enc) {
                ((void (*)(id, SEL, id, MTLRegion, unsigned long, unsigned long, char, id,
                           unsigned long))objc_msgSend)(
                    enc,
                    sel_getUid("getTextureAccessCounters:region:mipLevel:slice:"
                               "resetCounters:countersBuffer:countersBufferOffset:"),
                    src_tex, MTLRegionMake3D(0x11, 0x22, 0x33, 0x44, 0x55, 1), 3, 5, 1,
                    dst_buf, 0x1111);
              });

  // Segment framing rather than a command. Driven twice with different
  // arguments, because the first capture wrote seven of its eight bytes and
  // *neither argument appeared in them* — one observation cannot tell a header
  // that ignores its arguments from one whose fields happened to land on
  // values that look like constants.
  addBlitCase(cases, ser, stream, @"blit_begin_segment",
              @"beginSegment:protectionOptions:",
              @{@"flag" : @1, @"protection_options" : @0x33}, ^(id enc) {
                ((void (*)(id, SEL, char, unsigned long))objc_msgSend)(
                    enc, sel_getUid("beginSegment:protectionOptions:"), 1, 0x33);
              });

  addBlitCase(cases, ser, stream, @"blit_begin_segment_alt",
              @"beginSegment:protectionOptions:",
              @{@"flag" : @0, @"protection_options" : @0x44}, ^(id enc) {
                ((void (*)(id, SEL, char, unsigned long))objc_msgSend)(
                    enc, sel_getUid("beginSegment:protectionOptions:"), 0, 0x44);
              });

  // A continuation is relational state between two segment headers. The
  // serializer writes the second header's continuation argument at +5, then
  // asks the command stream to mark +6 in the preceding header. The capture
  // stream is told which first header to mark so this case records both ends
  // of that edge instead of inferring either direction from one nonzero byte.
  addEncoderCaseSplit(cases, @"PGSerializerBlitCommandEncoder",
                      ^id { return makeBlitEncoder(ser, stream); },
                      @"blit_segment_continuation_pair",
                      @"beginSegment:protectionOptions:",
                      @[
                        @{@"flag" : @0, @"protection_options" : @0,
                          @"segment_type" : @2, @"continues_next" : @1},
                        @{@"flag" : @1, @"protection_options" : @0,
                          @"segment_type" : @2, @"continues_next" : @0},
                      ],
                      ^(id enc) {
                        [stream setContinuationTarget:nil];
                        ((void (*)(id, SEL, char, unsigned long))objc_msgSend)(
                            enc, sel_getUid("beginSegment:protectionOptions:"), 0, 0);
                        [stream setContinuationTarget:gArena + gOpOff[0]];
                        ((void (*)(id, SEL, char, unsigned long))objc_msgSend)(
                            enc, sel_getUid("beginSegment:protectionOptions:"), 1, 0);
                        [stream setContinuationTarget:nil];
                      });

  // The protection-options envelope, and the two conditions it needs. The
  // options arrive as a second segment rather than in either continuation
  // byte of the ordinary segment header.
  //
  // Four cases, because two variables move and one observation of each cannot
  // separate them. The envelope needs the BOOL **clear** and the options
  // **non-zero**; either alone produces the ordinary single header. Driving only
  // (0, 0x44) against the default state would have attributed it to whichever
  // of the two the reader guessed.
  withCapability(ser, @"ProtectionOptionsEnvelope", ^{
    for (NSArray *pc in @[ @[ @"blit_begin_segment_protected", @0x44 ],
                           @[ @"blit_begin_segment_protected_alt", @0x33 ] ]) {
      unsigned long po = [pc[1] unsignedLongValue];
      addEncoderCaseSplit(cases, @"PGSerializerBlitCommandEncoder",
                          ^id { return makeBlitEncoder(ser, stream); },
                          pc[0], @"beginSegment:protectionOptions:",
                          @[
                            @{@"flag" : @0, @"protection_options" : @(po),
                              @"segment_type" : @5},
                            @{@"protection_options" : @(po)},
                            @{@"flag" : @0, @"protection_options" : @(po),
                              @"segment_type" : @2},
                          ],
                          ^(id enc) {
                            ((void (*)(id, SEL, char, unsigned long))objc_msgSend)(
                                enc, sel_getUid("beginSegment:protectionOptions:"), 0, po);
                          });
    }

    // The two arms that produce one record. Each holds the other variable at
    // the value the envelope needs, so each falsifies a single-cause reading.
    addBlitCase(cases, ser, stream, @"blit_begin_segment_protected_flag_set",
                @"beginSegment:protectionOptions:",
                @{@"flag" : @1, @"protection_options" : @0x44}, ^(id enc) {
                  ((void (*)(id, SEL, char, unsigned long))objc_msgSend)(
                      enc, sel_getUid("beginSegment:protectionOptions:"), 1, 0x44);
                });
    addBlitCase(cases, ser, stream, @"blit_begin_segment_protection_zero",
                @"beginSegment:protectionOptions:",
                @{@"flag" : @0, @"protection_options" : @0}, ^(id enc) {
                  ((void (*)(id, SEL, char, unsigned long))objc_msgSend)(
                      enc, sel_getUid("beginSegment:protectionOptions:"), 0, 0);
                });
  });

  addBlitCase(cases, ser, stream, @"blit_get_type", @"getType", @{}, ^(id enc) {
    (void)((unsigned long (*)(id, SEL))objc_msgSend)(enc, sel_getUid("getType"));
  });

  addBlitCase(cases, ser, stream, @"blit_sample_counters",
              @"sampleCountersInBuffer:atSampleIndex:withBarrier:",
              @{@"counters_ref" : @(STUB_BUFFER_REF),
                @"sample_index" : @0x1100,
                @"barrier" : @1},
              ^(id enc) {
                ((void (*)(id, SEL, id, unsigned long, char))objc_msgSend)(
                    enc, sel_getUid("sampleCountersInBuffer:atSampleIndex:withBarrier:"),
                    src_buf, 0x1100, 1);
              });

  // The six selectors `-setSupportsBlitEncoderSPI:` gates.
  //
  // All six were on the `silent` list, which the manifest turns into "Apple's
  // serializer emits no operation for this selector" -- a claim about Apple
  // that was wrong six times over. They emit. The flag is not guessed: the
  // capture's attribution passes drive each of the sixteen capabilities alone
  // and report which selectors stop being silent, and all six of these came
  // back under this one.
  //
  // Why it matters more than a coverage number: every one of these is a write
  // to guest-visible memory. A dropped `fillBuffer:` or `fillTexture:` leaves
  // the destination holding whatever it held before, so the guest reads stale
  // content back and nothing anywhere says a command was lost.
  withCapability(ser, @"BlitEncoderSPI", ^{
    // A record this class already emits *without* the flag, driven again with
    // it on. `withCapability` is safe for the family it wraps only if the flag
    // does not change what the rest of the class writes, and that is a
    // question a capture can answer rather than assume -- the same check
    // `texture_baseline_descriptor2` makes for `TextureDescriptor2`.
    addBlitCase(cases, ser, stream, @"blit_fill_buffer_under_spi",
                @"fillBuffer:range:value:",
                @{@"buffer_ref" : @(STUB_BUFFER_REF),
                  @"range_location" : @0x1100,
                  @"range_length" : @0x2200,
                  @"value" : @0x5a},
                ^(id enc) {
                  ((void (*)(id, SEL, id, NSRange, unsigned char))objc_msgSend)(
                      enc, sel_getUid("fillBuffer:range:value:"), src_buf,
                      NSMakeRange(0x1100, 0x2200), 0x5a);
                });

    addBlitCase(cases, ser, stream, @"blit_fill_buffer_pattern4",
                @"fillBuffer:range:pattern4:",
                @{@"buffer_ref" : @(STUB_BUFFER_REF),
                  @"range_location" : @0x3300,
                  @"range_length" : @0x4400,
                  @"pattern4" : @0x89abcdef},
                ^(id enc) {
                  ((void (*)(id, SEL, id, NSRange, unsigned int))objc_msgSend)(
                      enc, sel_getUid("fillBuffer:range:pattern4:"), src_buf,
                      NSMakeRange(0x3300, 0x4400), 0x89abcdefu);
                });

    // Second value for every field of the pattern4 fill, so each one has moved
    // once. A field that never moved is not derived however obvious it looks.
    addBlitCase(cases, ser, stream, @"blit_fill_buffer_pattern4_alt",
                @"fillBuffer:range:pattern4:",
                @{@"buffer_ref" : @(STUB_BUFFER_DST_REF),
                  @"range_location" : @0x5500,
                  @"range_length" : @0x6600,
                  @"pattern4" : @0x13572468},
                ^(id enc) {
                  ((void (*)(id, SEL, id, NSRange, unsigned int))objc_msgSend)(
                      enc, sel_getUid("fillBuffer:range:pattern4:"), dst_buf,
                      NSMakeRange(0x5500, 0x6600), 0x13572468u);
                });

    addBlitCase(cases, ser, stream, @"blit_invalidate_compressed_texture",
                @"invalidateCompressedTexture:",
                @{@"texture_ref" : @(STUB_TEXTURE_REF)}, ^(id enc) {
                  ((void (*)(id, SEL, id))objc_msgSend)(
                      enc, sel_getUid("invalidateCompressedTexture:"), src_tex);
                });

    addBlitCase(cases, ser, stream, @"blit_invalidate_compressed_texture_slice_level",
                @"invalidateCompressedTexture:slice:level:",
                @{@"texture_ref" : @(STUB_TEXTURE_REF), @"slice" : @3, @"level" : @5},
                ^(id enc) {
                  ((void (*)(id, SEL, id, unsigned long, unsigned long))objc_msgSend)(
                      enc, sel_getUid("invalidateCompressedTexture:slice:level:"),
                      src_tex, 3, 5);
                });

    addBlitCase(cases, ser, stream, @"blit_fill_texture_color",
                @"fillTexture:level:slice:region:color:",
                @{@"texture_ref" : @(STUB_TEXTURE_REF),
                  @"texture_pixel_format" : @((unsigned)src_tex.pixelFormat),
                  @"color_red" : @0.25,
                  @"color_green" : @0.5,
                  @"color_blue" : @0.75,
                  @"color_alpha" : @1.0,
                  @"level" : @3,
                  @"slice" : @5,
                  @"origin_x" : @0x11,
                  @"origin_y" : @0x22,
                  @"origin_z" : @0x33,
                  @"size_width" : @0x44,
                  @"size_height" : @0x55,
                  @"size_depth" : @1},
                ^(id enc) {
                  ((void (*)(id, SEL, id, unsigned long, unsigned long, MTLRegion,
                             MTLClearColor))objc_msgSend)(
                      enc, sel_getUid("fillTexture:level:slice:region:color:"), src_tex,
                      3, 5, MTLRegionMake3D(0x11, 0x22, 0x33, 0x44, 0x55, 1),
                      MTLClearColorMake(0.25, 0.5, 0.75, 1.0));
                });

    // Every argument of the colour fill moved: a second texture, a different
    // level and slice, a different region, and a colour whose four components
    // are distinct from each other *and* from the baseline's, so a record that
    // wrote one component into all four slots cannot read back correct.
    addBlitCase(cases, ser, stream, @"blit_fill_texture_color_alt",
                @"fillTexture:level:slice:region:color:",
                @{@"texture_ref" : @(STUB_TEXTURE_DST_REF),
                  @"texture_pixel_format" : @((unsigned)dst_tex.pixelFormat),
                  @"color_red" : @0.125,
                  @"color_green" : @0.375,
                  @"color_blue" : @0.625,
                  @"color_alpha" : @0.875,
                  @"level" : @7,
                  @"slice" : @2,
                  @"origin_x" : @0x66,
                  @"origin_y" : @0x77,
                  @"origin_z" : @0x88,
                  @"size_width" : @0x99,
                  @"size_height" : @0xaa,
                  @"size_depth" : @2},
                ^(id enc) {
                  ((void (*)(id, SEL, id, unsigned long, unsigned long, MTLRegion,
                             MTLClearColor))objc_msgSend)(
                      enc, sel_getUid("fillTexture:level:slice:region:color:"), dst_tex,
                      7, 2, MTLRegionMake3D(0x66, 0x77, 0x88, 0x99, 0xaa, 2),
                      MTLClearColorMake(0.125, 0.375, 0.625, 0.875));
                });

    // The same selector against a texture whose `pixelFormat` is not the
    // stub's default. `fillTexture:level:slice:region:color:` writes a format
    // word its selector never mentions, and against one texture that word
    // reads 80 in every capture -- which cannot tell "the serializer asked the
    // texture" from "the field is a constant". Expectation read off the
    // texture, so the case says which object it came from.
    addBlitCase(cases, ser, stream, @"blit_fill_texture_color_r8_texture",
                @"fillTexture:level:slice:region:color:",
                @{@"texture_ref" : @(STUB_TEXTURE_R8_REF),
                  @"level" : @3,
                  @"slice" : @5,
                  @"origin_x" : @0x11,
                  @"origin_y" : @0x22,
                  @"origin_z" : @0x33,
                  @"size_width" : @0x44,
                  @"size_height" : @0x55,
                  @"size_depth" : @1,
                  @"texture_pixel_format" : @((unsigned)r8_tex.pixelFormat),
                  @"color_red" : @0.25,
                  @"color_green" : @0.5,
                  @"color_blue" : @0.75,
                  @"color_alpha" : @1.0},
                ^(id enc) {
                  ((void (*)(id, SEL, id, unsigned long, unsigned long, MTLRegion,
                             MTLClearColor))objc_msgSend)(
                      enc, sel_getUid("fillTexture:level:slice:region:color:"), r8_tex, 3,
                      5, MTLRegionMake3D(0x11, 0x22, 0x33, 0x44, 0x55, 1),
                      MTLClearColorMake(0.25, 0.5, 0.75, 1.0));
                });

    addBlitCase(cases, ser, stream, @"blit_fill_texture_color_pixel_format",
                @"fillTexture:level:slice:region:color:pixelFormat:",
                @{@"texture_ref" : @(STUB_TEXTURE_REF),
                  @"level" : @3,
                  @"slice" : @5,
                  @"origin_x" : @0x11,
                  @"origin_y" : @0x22,
                  @"origin_z" : @0x33,
                  @"size_width" : @0x44,
                  @"size_height" : @0x55,
                  @"size_depth" : @1,
                  @"pixel_format" : @(MTLPixelFormatRGBA16Float),
                  @"color_red" : @0.25,
                  @"color_green" : @0.5,
                  @"color_blue" : @0.75,
                  @"color_alpha" : @1.0},
                ^(id enc) {
                  ((void (*)(id, SEL, id, unsigned long, unsigned long, MTLRegion,
                             MTLClearColor, unsigned long))objc_msgSend)(
                      enc,
                      sel_getUid("fillTexture:level:slice:region:color:pixelFormat:"),
                      src_tex, 3, 5, MTLRegionMake3D(0x11, 0x22, 0x33, 0x44, 0x55, 1),
                      MTLClearColorMake(0.25, 0.5, 0.75, 1.0), MTLPixelFormatRGBA16Float);
                });

    // Only the format moves against the case above, so whatever differs
    // between the two bodies is the format field and nothing else.
    addBlitCase(cases, ser, stream, @"blit_fill_texture_color_pixel_format_alt",
                @"fillTexture:level:slice:region:color:pixelFormat:",
                @{@"texture_ref" : @(STUB_TEXTURE_REF),
                  @"level" : @3,
                  @"slice" : @5,
                  @"origin_x" : @0x11,
                  @"origin_y" : @0x22,
                  @"origin_z" : @0x33,
                  @"size_width" : @0x44,
                  @"size_height" : @0x55,
                  @"size_depth" : @1,
                  @"pixel_format" : @(MTLPixelFormatR8Unorm),
                  @"color_red" : @0.25,
                  @"color_green" : @0.5,
                  @"color_blue" : @0.75,
                  @"color_alpha" : @1.0},
                ^(id enc) {
                  ((void (*)(id, SEL, id, unsigned long, unsigned long, MTLRegion,
                             MTLClearColor, unsigned long))objc_msgSend)(
                      enc,
                      sel_getUid("fillTexture:level:slice:region:color:pixelFormat:"),
                      src_tex, 3, 5, MTLRegionMake3D(0x11, 0x22, 0x33, 0x44, 0x55, 1),
                      MTLClearColorMake(0.25, 0.5, 0.75, 1.0), MTLPixelFormatR8Unorm);
                });

    addBlitCase(cases, ser, stream, @"blit_fill_texture_bytes",
                @"fillTexture:level:slice:region:bytes:length:",
                @{@"texture_ref" : @(STUB_TEXTURE_REF),
                  @"bytes_ref" : @(STUB_STAGING_REF),
                  @"bytes_offset" : @(STUB_STAGING_OFFSET),
                  @"level" : @3,
                  @"slice" : @5,
                  @"origin_x" : @0x11,
                  @"origin_y" : @0x22,
                  @"origin_z" : @0x33,
                  @"size_width" : @0x44,
                  @"size_height" : @0x55,
                  @"size_depth" : @1,
                  @"length" : @8},
                ^(id enc) {
                  static const unsigned char pattern[8] = {0x5a, 0x5b, 0x5c, 0x5d,
                                                           0x5e, 0x5f, 0x60, 0x61};
                  ((void (*)(id, SEL, id, unsigned long, unsigned long, MTLRegion,
                             const void *, unsigned long))objc_msgSend)(
                      enc, sel_getUid("fillTexture:level:slice:region:bytes:length:"),
                      src_tex, 3, 5, MTLRegionMake3D(0x11, 0x22, 0x33, 0x44, 0x55, 1),
                      pattern, sizeof(pattern));
                });

    // A different length and a different pattern. `length:` is the one
    // argument of this selector whose *effect* on the record cannot be guessed
    // — the bytes may be inlined after the header, staged through a buffer, or
    // ignored — and only a second length distinguishes those.
    addBlitCase(cases, ser, stream, @"blit_fill_texture_bytes_alt",
                @"fillTexture:level:slice:region:bytes:length:",
                @{@"texture_ref" : @(STUB_TEXTURE_REF),
                  @"bytes_ref" : @(STUB_STAGING_REF),
                  @"bytes_offset" : @(STUB_STAGING_OFFSET),
                  @"level" : @3,
                  @"slice" : @5,
                  @"origin_x" : @0x11,
                  @"origin_y" : @0x22,
                  @"origin_z" : @0x33,
                  @"size_width" : @0x44,
                  @"size_height" : @0x55,
                  @"size_depth" : @1,
                  @"length" : @4},
                ^(id enc) {
                  static const unsigned char pattern[4] = {0x71, 0x72, 0x73, 0x74};
                  ((void (*)(id, SEL, id, unsigned long, unsigned long, MTLRegion,
                             const void *, unsigned long))objc_msgSend)(
                      enc, sel_getUid("fillTexture:level:slice:region:bytes:length:"),
                      src_tex, 3, 5, MTLRegionMake3D(0x11, 0x22, 0x33, 0x44, 0x55, 1),
                      pattern, sizeof(pattern));
                });
  });

  return cases;
}

// --- Compute encoder records ------------------------------------------------
//
// 58 selectors, and the only class with *control flow*: encodeStartIf:,
// encodeStartWhile:, encodeStartDoWhile and their ends. The device has a
// `compute_ctrl_seen` counter that has never fired, so nothing has ever checked
// those records against Apple's.
static NSArray *computeCases(id ser) {
  NSMutableArray *cases = [NSMutableArray array];
  id stream = [[CaptureCommandStream alloc] init];

  id buf = [[StubBuffer alloc] init];
  id buf2 = [[StubBuffer alloc] initWithRef:STUB_BUFFER_DST_REF];
  id tex = [[StubTexture alloc] init];
  id tex2 = [[StubTexture alloc] initWithRef:STUB_TEXTURE_DST_REF];
  id sampler = [[StubSamplerState alloc] init];
  id pipeline = [[StubPipelineState alloc] init];
  id fence = [[StubFence alloc] init];
  id icb = [[StubICB alloc] init];

  addComputeCase(cases, ser, stream, @"compute_set_pipeline_state",
                 @"setComputePipelineState:",
                 @{@"pipeline_ref" : @(STUB_PIPELINE_REF)}, ^(id enc) {
                   ((void (*)(id, SEL, id))objc_msgSend)(
                       enc, sel_getUid("setComputePipelineState:"), pipeline);
                 });

  addComputeCase(cases, ser, stream, @"compute_dispatch_threadgroups",
                 @"dispatchThreadgroups:threadsPerThreadgroup:",
                 @{@"groups_width" : @0x11,
                   @"groups_height" : @0x22,
                   @"groups_depth" : @0x33,
                   @"threads_width" : @0x44,
                   @"threads_height" : @0x55,
                   @"threads_depth" : @0x66},
                 ^(id enc) {
                   ((void (*)(id, SEL, MTLSize, MTLSize))objc_msgSend)(
                       enc, sel_getUid("dispatchThreadgroups:threadsPerThreadgroup:"),
                       MTLSizeMake(0x11, 0x22, 0x33), MTLSizeMake(0x44, 0x55, 0x66));
                 });

  // A serial compute pass barriers after every dispatch, and the capability is
  // what turns that on.
  //
  // With `-setSupportsComputePassDescriptorDispatchType:` on, each of the five
  // dispatch and ICB-execute selectors emits **two** records: its own, then an
  // 0xd7 memory barrier at scope `Buffers|Textures`. At the default state it
  // emits one. That is not a layout change, so no fixture this crate pins is
  // wrong -- but a reader counting records per selector would be, and the
  // device beside this crate treats 0xd7 as a no-op, which is only safe as long
  // as its own submission already orders serial dispatches.
  //
  // Two things this pair establishes that one case could not. The barrier is a
  // property of the **pass's dispatch type**, not of the selector: driven again
  // after `setCurrentDispatchType:1` (concurrent) the same selector emits one
  // record. And the flag has to be on when the **encoder is created** -- forcing
  // it inside the case body, after `makeComputeEncoder`, produces one record,
  // so the encoder reads the capability at init.
  withCapability(ser, @"ComputePassDescriptorDispatchType", ^{
    addEncoderCaseSplit(cases, @"PGSerializerComputeCommandEncoder",
                        ^id { return makeComputeEncoder(ser, stream); },
                        @"compute_dispatch_threadgroups_serial",
                        @"dispatchThreadgroups:threadsPerThreadgroup:",
                        @[
                          @{@"groups_width" : @0x11, @"groups_height" : @0x22,
                            @"groups_depth" : @0x33, @"threads_width" : @0x44,
                            @"threads_height" : @0x55, @"threads_depth" : @0x66},
                          @{@"scope" : @(MTLBarrierScopeBuffers | MTLBarrierScopeTextures)},
                        ],
                        ^(id enc) {
                          ((void (*)(id, SEL, MTLSize, MTLSize))objc_msgSend)(
                              enc, sel_getUid("dispatchThreadgroups:threadsPerThreadgroup:"),
                              MTLSizeMake(0x11, 0x22, 0x33), MTLSizeMake(0x44, 0x55, 0x66));
                        });

    // The same selector, the same flag, one record. Without this the barrier
    // reads as something the capability adds unconditionally.
    addComputeCase(cases, ser, stream, @"compute_dispatch_threadgroups_concurrent",
                   @"dispatchThreadgroups:threadsPerThreadgroup:",
                   @{@"groups_width" : @0x11,
                     @"groups_height" : @0x22,
                     @"groups_depth" : @0x33,
                     @"threads_width" : @0x44,
                     @"threads_height" : @0x55,
                     @"threads_depth" : @0x66},
                   ^(id enc) {
                     ((void (*)(id, SEL, unsigned long))objc_msgSend)(
                         enc, sel_getUid("setCurrentDispatchType:"), 1);
                     ((void (*)(id, SEL, MTLSize, MTLSize))objc_msgSend)(
                         enc, sel_getUid("dispatchThreadgroups:threadsPerThreadgroup:"),
                         MTLSizeMake(0x11, 0x22, 0x33), MTLSizeMake(0x44, 0x55, 0x66));
                   });

    // An ICB execution, because "every dispatch barriers" and "every one of the
    // five selectors barriers" are different claims and the ICB forms are not
    // dispatches.
    addEncoderCaseSplit(cases, @"PGSerializerComputeCommandEncoder",
                        ^id { return makeComputeEncoder(ser, stream); },
                        @"compute_execute_commands_range_serial",
                        @"executeCommandsInBuffer:withRange:",
                        @[
                          @{@"icb_ref" : @(STUB_ICB_REF), @"range_location" : @0x1100,
                            @"range_length" : @0x2200},
                          @{@"scope" : @(MTLBarrierScopeBuffers | MTLBarrierScopeTextures)},
                        ],
                        ^(id enc) {
                          ((void (*)(id, SEL, id, NSRange))objc_msgSend)(
                              enc, sel_getUid("executeCommandsInBuffer:withRange:"), icb,
                              NSMakeRange(0x1100, 0x2200));
                        });

    // The other four of the six. The attribution pass already measures that all
    // six emit two operations; what it cannot say is *which* second record, and
    // "the family is uniform" is the kind of claim that is cheap to drive and
    // expensive to assume. Their manifest rows list 0xd7 alongside their own
    // opcode, and `every_covered_row_lists_the_opcode_apple_wrote` only accepts
    // that because these fixtures exist.
    addEncoderCaseSplit(cases, @"PGSerializerComputeCommandEncoder",
                        ^id { return makeComputeEncoder(ser, stream); },
                        @"compute_dispatch_threads_serial",
                        @"dispatchThreads:threadsPerThreadgroup:",
                        @[
                          @{@"groups_width" : @0x11, @"groups_height" : @0x22,
                            @"groups_depth" : @0x33, @"threads_width" : @0x44,
                            @"threads_height" : @0x55, @"threads_depth" : @0x66},
                          @{@"scope" : @(MTLBarrierScopeBuffers | MTLBarrierScopeTextures)},
                        ],
                        ^(id enc) {
                          ((void (*)(id, SEL, MTLSize, MTLSize))objc_msgSend)(
                              enc, sel_getUid("dispatchThreads:threadsPerThreadgroup:"),
                              MTLSizeMake(0x11, 0x22, 0x33), MTLSizeMake(0x44, 0x55, 0x66));
                        });

    addEncoderCaseSplit(cases, @"PGSerializerComputeCommandEncoder",
                        ^id { return makeComputeEncoder(ser, stream); },
                        @"compute_dispatch_threadgroups_indirect_serial",
                        @"dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:"
                        @"threadsPerThreadgroup:",
                        @[
                          @{@"indirect_buffer_ref" : @(STUB_BUFFER_REF),
                            @"indirect_buffer_offset" : @0x1111,
                            @"threads_width" : @0x44, @"threads_height" : @0x55,
                            @"threads_depth" : @0x66},
                          @{@"scope" : @(MTLBarrierScopeBuffers | MTLBarrierScopeTextures)},
                        ],
                        ^(id enc) {
                          ((void (*)(id, SEL, id, unsigned long, MTLSize))objc_msgSend)(
                              enc,
                              sel_getUid("dispatchThreadgroupsWithIndirectBuffer:"
                                         "indirectBufferOffset:threadsPerThreadgroup:"),
                              buf, 0x1111, MTLSizeMake(0x44, 0x55, 0x66));
                        });

    // Two flags at once, which is the only case here that needs a conjunction:
    // this selector asserts without `DispatchThreadsIndirect` and barriers only
    // with `ComputePassDescriptorDispatchType`.
    withCapability(ser, @"DispatchThreadsIndirect", ^{
      addEncoderCaseSplit(cases, @"PGSerializerComputeCommandEncoder",
                          ^id { return makeComputeEncoder(ser, stream); },
                          @"compute_dispatch_threads_indirect_serial",
                          @"dispatchThreadsWithIndirectBuffer:indirectBufferOffset:",
                          @[
                            @{@"indirect_buffer_ref" : @(STUB_BUFFER_REF),
                              @"indirect_buffer_offset" : @0x2222},
                            @{@"scope" : @(MTLBarrierScopeBuffers | MTLBarrierScopeTextures)},
                          ],
                          ^(id enc) {
                            ((void (*)(id, SEL, id, unsigned long))objc_msgSend)(
                                enc,
                                sel_getUid("dispatchThreadsWithIndirectBuffer:"
                                           "indirectBufferOffset:"),
                                buf, 0x2222);
                          });
    });

    addEncoderCaseSplit(cases, @"PGSerializerComputeCommandEncoder",
                        ^id { return makeComputeEncoder(ser, stream); },
                        @"compute_execute_commands_indirect_serial",
                        @"executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:",
                        @[
                          @{@"icb_ref" : @(STUB_ICB_REF),
                            @"indirect_buffer_ref" : @(STUB_BUFFER_REF),
                            @"indirect_buffer_offset" : @0x1111},
                          @{@"scope" : @(MTLBarrierScopeBuffers | MTLBarrierScopeTextures)},
                        ],
                        ^(id enc) {
                          ((void (*)(id, SEL, id, id, unsigned long))objc_msgSend)(
                              enc,
                              sel_getUid("executeCommandsInBuffer:indirectBuffer:"
                                         "indirectBufferOffset:"),
                              icb, buf, 0x1111);
                        });
  });

  addComputeCase(cases, ser, stream, @"compute_dispatch_threads",
                 @"dispatchThreads:threadsPerThreadgroup:",
                 @{@"groups_width" : @0x11,
                   @"groups_height" : @0x22,
                   @"groups_depth" : @0x33,
                   @"threads_width" : @0x44,
                   @"threads_height" : @0x55,
                   @"threads_depth" : @0x66},
                 ^(id enc) {
                   ((void (*)(id, SEL, MTLSize, MTLSize))objc_msgSend)(
                       enc, sel_getUid("dispatchThreads:threadsPerThreadgroup:"),
                       MTLSizeMake(0x11, 0x22, 0x33), MTLSizeMake(0x44, 0x55, 0x66));
                 });

  addComputeCase(cases, ser, stream, @"compute_dispatch_threadgroups_indirect",
                 @"dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:"
                 @"threadsPerThreadgroup:",
                 @{@"indirect_buffer_ref" : @(STUB_BUFFER_REF),
                   @"indirect_buffer_offset" : @0x1111,
                   @"threads_width" : @0x44,
                   @"threads_height" : @0x55,
                   @"threads_depth" : @0x66},
                 ^(id enc) {
                   ((void (*)(id, SEL, id, unsigned long, MTLSize))objc_msgSend)(
                       enc,
                       sel_getUid("dispatchThreadgroupsWithIndirectBuffer:"
                                  "indirectBufferOffset:threadsPerThreadgroup:"),
                       buf, 0x1111, MTLSizeMake(0x44, 0x55, 0x66));
                 });

  // The indirect *threads* dispatch, and the reason it took so long to reach a
  // fixture.
  //
  // At the default capability state the serializer asserts on it, so it landed
  // on `unsupported` and its manifest row said REFUSED_BY_SERIALIZER — a claim
  // about Apple. The capability sweep that exists to catch exactly that could
  // not: it diffs the two passes' `silent` lists, and a selector that asserts
  // is on `unsupported` instead. The byte diff in `capability_content_deltas`
  // is what found it, and it names the flag.
  withCapability(ser, @"DispatchThreadsIndirect", ^{
    addComputeCase(cases, ser, stream, @"compute_dispatch_threads_indirect",
                   @"dispatchThreadsWithIndirectBuffer:indirectBufferOffset:",
                   @{@"indirect_buffer_ref" : @(STUB_BUFFER_REF),
                     @"indirect_buffer_offset" : @0x2222},
                   ^(id enc) {
                     ((void (*)(id, SEL, id, unsigned long))objc_msgSend)(
                         enc,
                         sel_getUid("dispatchThreadsWithIndirectBuffer:"
                                    "indirectBufferOffset:"),
                         buf, 0x2222);
                   });
  });

  addComputeCase(cases, ser, stream, @"compute_set_buffer", @"setBuffer:offset:atIndex:",
                 @{@"buffer_ref" : @(STUB_BUFFER_REF),
                   @"offset" : @0x1234,
                   @"first" : @5,
                   @"count" : @1},
                 ^(id enc) {
                   ((void (*)(id, SEL, id, unsigned long, unsigned long))objc_msgSend)(
                       enc, sel_getUid("setBuffer:offset:atIndex:"), buf, 0x1234, 5);
                 });

  addComputeCase(cases, ser, stream, @"compute_set_buffer_offset",
                 @"setBufferOffset:atIndex:",
                 @{@"offset" : @0x5678, @"first" : @6}, ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long))objc_msgSend)(
                       enc, sel_getUid("setBufferOffset:atIndex:"), 0x5678, 6);
                 });

  {
    id bufs[2] = {buf, buf2};
    const id *buf_list = bufs;
    NSUInteger offsets[2] = {0x1111, 0x2222};
    const NSUInteger *offset_list = offsets;
    addComputeCase(cases, ser, stream, @"compute_set_buffers_range",
                   @"setBuffers:offsets:withRange:",
                   @{@"buffer_ref" : @(STUB_BUFFER_REF),
                     @"buffer_ref_2" : @(STUB_BUFFER_DST_REF),
                     @"first" : @4,
                     @"count" : @2,
                     @"offset" : @0x1111,
                     @"offset_2" : @0x2222},
                   ^(id enc) {
                     ((void (*)(id, SEL, const id *, const NSUInteger *,
                                NSRange))objc_msgSend)(
                         enc, sel_getUid("setBuffers:offsets:withRange:"), buf_list,
                         offset_list, NSMakeRange(4, 2));
                   });

    id texes[2] = {tex, tex2};
    const id *tex_list = texes;
    addComputeCase(cases, ser, stream, @"compute_set_textures_range",
                   @"setTextures:withRange:",
                   @{@"texture_ref" : @(STUB_TEXTURE_REF),
                     @"texture_ref_2" : @(STUB_TEXTURE_DST_REF),
                     @"first" : @2,
                     @"count" : @2},
                   ^(id enc) {
                     ((void (*)(id, SEL, const id *, NSRange))objc_msgSend)(
                         enc, sel_getUid("setTextures:withRange:"), tex_list,
                         NSMakeRange(2, 2));
                   });

    // The bind range is truncated at the stage's argument-table size, and the
    // three resource classes do not share one. Requesting 200 from index 0
    // yields 128 textures, 31 buffers and 16 samplers; the `_offset` case shows
    // the bound is on **first + count** rather than on count, which is what
    // separates "a cap on how many" from "a cap on how far".
    //
    // This is why one `MAX_BIND_ENTRIES` cannot be right: any single number is
    // four times too permissive for buffers and eight times for samplers, or
    // else it truncates textures. `runtime::decode` used to carry exactly that
    // — 128 on the compute rail, 32 on the render rail — and the render one
    // dropped a forty-slot texture bind whole.
    static id manyTex[200];
    static id manySmp[200];
    static id manyBuf[200];
    static NSUInteger manyOffsets[200];
    for (int i = 0; i < 200; i++) {
      manyTex[i] = tex;
      manySmp[i] = sampler;
      manyBuf[i] = buf;
      manyOffsets[i] = 0x1000 + i;
    }
    const id *texAll = manyTex;
    const id *smpAll = manySmp;
    const id *bufAll = manyBuf;
    const NSUInteger *offAll = manyOffsets;

    addComputeCase(cases, ser, stream, @"compute_set_textures_over_bind_limit",
                   @"setTextures:withRange:",
                   @{@"texture_ref" : @(STUB_TEXTURE_REF), @"first" : @0, @"requested" : @200},
                   ^(id enc) {
                     ((void (*)(id, SEL, const id *, NSRange))objc_msgSend)(
                         enc, sel_getUid("setTextures:withRange:"), texAll, NSMakeRange(0, 200));
                   });

    // Twenty from index 120: eight survive, so the limit counts from zero.
    addComputeCase(cases, ser, stream, @"compute_set_textures_over_bind_limit_offset",
                   @"setTextures:withRange:",
                   @{@"texture_ref" : @(STUB_TEXTURE_REF), @"first" : @120, @"requested" : @20},
                   ^(id enc) {
                     ((void (*)(id, SEL, const id *, NSRange))objc_msgSend)(
                         enc, sel_getUid("setTextures:withRange:"), texAll, NSMakeRange(120, 20));
                   });

    addComputeCase(cases, ser, stream, @"compute_set_samplers_over_bind_limit",
                   @"setSamplerStates:withRange:",
                   @{@"sampler_ref" : @(STUB_SAMPLER_REF), @"first" : @0, @"requested" : @200},
                   ^(id enc) {
                     ((void (*)(id, SEL, const id *, NSRange))objc_msgSend)(
                         enc, sel_getUid("setSamplerStates:withRange:"), smpAll,
                         NSMakeRange(0, 200));
                   });

    addComputeCase(cases, ser, stream, @"compute_set_buffers_over_bind_limit",
                   @"setBuffers:offsets:withRange:",
                   @{@"buffer_ref" : @(STUB_BUFFER_REF), @"first" : @0, @"requested" : @200},
                   ^(id enc) {
                     ((void (*)(id, SEL, const id *, const NSUInteger *, NSRange))objc_msgSend)(
                         enc, sel_getUid("setBuffers:offsets:withRange:"), bufAll, offAll,
                         NSMakeRange(0, 200));
                   });

    // The render encoder truncates at the same three sizes, so this is a
    // property of the stage's argument table rather than of the compute rail.
    addEncoderCase(cases, ser, stream, @"render_set_vertex_textures_over_bind_limit",
                   @"setVertexTextures:withRange:",
                   @{@"texture_ref" : @(STUB_TEXTURE_REF), @"first" : @0, @"requested" : @200},
                   ^(id enc) {
                     ((void (*)(id, SEL, const id *, NSRange))objc_msgSend)(
                         enc, sel_getUid("setVertexTextures:withRange:"), texAll,
                         NSMakeRange(0, 200));
                   });

    addEncoderCase(cases, ser, stream, @"render_set_vertex_buffers_over_bind_limit",
                   @"setVertexBuffers:offsets:withRange:",
                   @{@"buffer_ref" : @(STUB_BUFFER_REF), @"first" : @0, @"requested" : @200},
                   ^(id enc) {
                     ((void (*)(id, SEL, const id *, const NSUInteger *, NSRange))objc_msgSend)(
                         enc, sel_getUid("setVertexBuffers:offsets:withRange:"), bufAll, offAll,
                         NSMakeRange(0, 200));
                   });
  }

  addComputeCase(cases, ser, stream, @"compute_set_texture", @"setTexture:atIndex:",
                 @{@"texture_ref" : @(STUB_TEXTURE_REF), @"first" : @3, @"count" : @1},
                 ^(id enc) {
                   ((void (*)(id, SEL, id, unsigned long))objc_msgSend)(
                       enc, sel_getUid("setTexture:atIndex:"), tex, 3);
                 });

  addComputeCase(cases, ser, stream, @"compute_set_sampler_state",
                 @"setSamplerState:atIndex:",
                 @{@"sampler_ref" : @(STUB_SAMPLER_REF), @"first" : @4, @"count" : @1},
                 ^(id enc) {
                   ((void (*)(id, SEL, id, unsigned long))objc_msgSend)(
                       enc, sel_getUid("setSamplerState:atIndex:"), sampler, 4);
                 });

  addComputeCase(cases, ser, stream, @"compute_set_sampler_state_lod",
                 @"setSamplerState:lodMinClamp:lodMaxClamp:atIndex:",
                 @{@"sampler_ref" : @(STUB_SAMPLER_REF),
                   @"lod_min_clamp" : @0.25,
                   @"lod_max_clamp" : @0.75,
                   @"first" : @6,
                   @"count" : @1},
                 ^(id enc) {
                   ((void (*)(id, SEL, id, float, float, unsigned long))objc_msgSend)(
                       enc, sel_getUid("setSamplerState:lodMinClamp:lodMaxClamp:atIndex:"),
                       sampler, 0.25f, 0.75f, 6);
                 });

  // The plural lod-clamp form. With `count == 1` the two floats could be one
  // pair for the record; two slots with four distinct clamps show they are per
  // entry.
  {
    id samplers[2] = {sampler, [[StubSamplerState alloc] init]};
    const id *sampler_list = samplers;
    float mins[2] = {0.25f, 0.125f};
    float maxs[2] = {0.75f, 0.875f};
    const float *min_list = mins;
    const float *max_list = maxs;
    addComputeCase(cases, ser, stream, @"compute_set_sampler_states_lod",
                   @"setSamplerStates:lodMinClamps:lodMaxClamps:withRange:",
                   @{@"sampler_ref" : @(STUB_SAMPLER_REF),
                     @"first" : @2,
                     @"count" : @2,
                     @"lod_min_clamp" : @0.25,
                     @"lod_max_clamp" : @0.75,
                     @"lod_min_clamp_2" : @0.125,
                     @"lod_max_clamp_2" : @0.875},
                   ^(id enc) {
                     ((void (*)(id, SEL, const id *, const float *, const float *,
                                NSRange))objc_msgSend)(
                         enc,
                         sel_getUid("setSamplerStates:lodMinClamps:lodMaxClamps:"
                                    "withRange:"),
                         sampler_list, min_list, max_list, NSMakeRange(2, 2));
                   });
  }

  addComputeCase(cases, ser, stream, @"compute_set_bytes", @"setBytes:length:atIndex:",
                 @{@"buffer_ref" : @(STUB_STAGING_REF),
                   @"offset" : @(STUB_STAGING_OFFSET),
                   @"first" : @7,
                   @"count" : @1,
                   @"length" : @8},
                 ^(id enc) {
                   static const unsigned char blob[8] = {0x5a, 0x5b, 0x5c, 0x5d,
                                                         0x5e, 0x5f, 0x60, 0x61};
                   ((void (*)(id, SEL, const void *, unsigned long,
                              unsigned long))objc_msgSend)(
                       enc, sel_getUid("setBytes:length:atIndex:"), blob, sizeof(blob), 7);
                 });

  addComputeCase(cases, ser, stream, @"compute_set_threadgroup_memory_length",
                 @"setThreadgroupMemoryLength:atIndex:",
                 @{@"length" : @0x1100, @"index" : @3}, ^(id enc) {
                   ((void (*)(id, SEL, unsigned long, unsigned long))objc_msgSend)(
                       enc, sel_getUid("setThreadgroupMemoryLength:atIndex:"), 0x1100, 3);
                 });

  addComputeCase(cases, ser, stream, @"compute_set_stage_in_region",
                 @"setStageInRegion:",
                 @{@"origin_x" : @0x11,
                   @"origin_y" : @0x22,
                   @"origin_z" : @0x33,
                   @"size_width" : @0x44,
                   @"size_height" : @0x55,
                   @"size_depth" : @0x66},
                 ^(id enc) {
                   ((void (*)(id, SEL, MTLRegion))objc_msgSend)(
                       enc, sel_getUid("setStageInRegion:"),
                       MTLRegionMake3D(0x11, 0x22, 0x33, 0x44, 0x55, 0x66));
                 });

  // Gated on `-setSupportsImageBlocks:`, which is off by default -- so this
  // selector was captured writing nothing and carried EMITS_NO_OPERATION.
  // Driven twice, because a record with two same-typed arguments cannot show
  // which slot is which from one observation.
  withCapability(ser, @"ImageBlocks", ^{
    addComputeCase(cases, ser, stream, @"compute_set_imageblock_size",
                   @"setImageblockWidth:height:",
                   @{@"width" : @0x11, @"height" : @0x22}, ^(id enc) {
                     ((void (*)(id, SEL, unsigned long, unsigned long))objc_msgSend)(
                         enc, sel_getUid("setImageblockWidth:height:"), 0x11, 0x22);
                   });

    addComputeCase(cases, ser, stream, @"compute_set_imageblock_size_alt",
                   @"setImageblockWidth:height:",
                   @{@"width" : @0x3333, @"height" : @0x4444}, ^(id enc) {
                     ((void (*)(id, SEL, unsigned long, unsigned long))objc_msgSend)(
                         enc, sel_getUid("setImageblockWidth:height:"), 0x3333, 0x4444);
                   });
  });

  addComputeCase(cases, ser, stream, @"compute_update_fence", @"updateFence:",
                 @{@"fence_ref" : @(STUB_FENCE_REF)}, ^(id enc) {
                   ((void (*)(id, SEL, id))objc_msgSend)(enc, sel_getUid("updateFence:"),
                                                         fence);
                 });

  addComputeCase(cases, ser, stream, @"compute_wait_for_fence", @"waitForFence:",
                 @{@"fence_ref" : @(STUB_FENCE_REF)}, ^(id enc) {
                   ((void (*)(id, SEL, id))objc_msgSend)(enc, sel_getUid("waitForFence:"),
                                                         fence);
                 });

  addComputeCase(cases, ser, stream, @"compute_memory_barrier_scope",
                 @"memoryBarrierWithScope:", @{@"scope" : @4}, ^(id enc) {
                   ((void (*)(id, SEL, unsigned long))objc_msgSend)(
                       enc, sel_getUid("memoryBarrierWithScope:"), 4);
                 });

  {
    id resources[2] = {buf, tex2};
    const id *resource_list = resources;
    addComputeCase(cases, ser, stream, @"compute_memory_barrier_resources",
                   @"memoryBarrierWithResources:count:",
                   @{@"resource_ref" : @(STUB_BUFFER_REF),
                     @"resource_ref_2" : @(STUB_TEXTURE_DST_REF),
                     @"count" : @2},
                   ^(id enc) {
                     ((void (*)(id, SEL, const id *, unsigned long))objc_msgSend)(
                         enc, sel_getUid("memoryBarrierWithResources:count:"),
                         resource_list, 2);
                   });
  }

  addComputeCase(cases, ser, stream, @"compute_execute_commands_range",
                 @"executeCommandsInBuffer:withRange:",
                 @{@"icb_ref" : @(STUB_ICB_REF),
                   @"range_location" : @0x1100,
                   @"range_length" : @0x2200},
                 ^(id enc) {
                   ((void (*)(id, SEL, id, NSRange))objc_msgSend)(
                       enc, sel_getUid("executeCommandsInBuffer:withRange:"), icb,
                       NSMakeRange(0x1100, 0x2200));
                 });

  addComputeCase(cases, ser, stream, @"compute_execute_commands_indirect",
                 @"executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:",
                 @{@"icb_ref" : @(STUB_ICB_REF),
                   @"indirect_buffer_ref" : @(STUB_BUFFER_REF),
                   @"indirect_buffer_offset" : @0x1111},
                 ^(id enc) {
                   ((void (*)(id, SEL, id, id, unsigned long))objc_msgSend)(
                       enc,
                       sel_getUid("executeCommandsInBuffer:indirectBuffer:"
                                  "indirectBufferOffset:"),
                       icb, buf, 0x1111);
                 });

  addComputeCase(cases, ser, stream, @"compute_set_current_dispatch_type",
                 @"setCurrentDispatchType:", @{@"dispatch_type" : @1}, ^(id enc) {
                   ((void (*)(id, SEL, unsigned long))objc_msgSend)(
                       enc, sel_getUid("setCurrentDispatchType:"), 1);
                 });

  // Control flow, and it is gated on `-setSupportsCommandBufferJump:`.
  //
  // Every one of these seven landed on the `silent` list -- "Apple's serializer
  // emits no operation for this selector" -- for as long as this class was
  // driven at the default capability state, where all sixteen flags read false.
  // They emit. The flag is not a guess: the capture's per-flag attribution
  // passes drive each capability alone and report which selectors stop being
  // silent, and all seven came back under this one.
  //
  // `reims_vgpu::runtime::compute_exec` has a `compute_ctrl_seen` counter that
  // has never fired on a driven boot, and until now that was as consistent with
  // "the protocol has no control flow" as with "this workload uses none". It is
  // the second: a guest that used one was executing a predicate this device had
  // never decoded.
  //
  // Each argument-bearing selector is driven twice with every field moved, so
  // no field is named from a single observation. The three predicates are also
  // driven against *different* buffers, because a record that wrote one arm's
  // buffer ref into another's would read back correct against one stub.
  withCapability(ser, @"CommandBufferJump", ^{
    addComputeCase(cases, ser, stream, @"compute_encode_start_if",
                   @"encodeStartIf:offset:comparison:referenceValue:",
                   @{@"buffer_ref" : @(STUB_BUFFER_REF),
                     @"offset" : @0x1111,
                     @"comparison" : @0x22,
                     @"reference_value" : @0x33},
                   ^(id enc) {
                     ((void (*)(id, SEL, id, unsigned long, unsigned long,
                                unsigned int))objc_msgSend)(
                         enc,
                         sel_getUid("encodeStartIf:offset:comparison:referenceValue:"),
                         buf, 0x1111, 0x22, 0x33);
                   });

    addComputeCase(cases, ser, stream, @"compute_encode_start_if_alt",
                   @"encodeStartIf:offset:comparison:referenceValue:",
                   @{@"buffer_ref" : @(STUB_BUFFER_DST_REF),
                     @"offset" : @0x4444,
                     @"comparison" : @0x11,
                     @"reference_value" : @0x89abcdef},
                   ^(id enc) {
                     ((void (*)(id, SEL, id, unsigned long, unsigned long,
                                unsigned int))objc_msgSend)(
                         enc,
                         sel_getUid("encodeStartIf:offset:comparison:referenceValue:"),
                         buf2, 0x4444, 0x11, 0x89abcdefu);
                   });

    addComputeCase(cases, ser, stream, @"compute_encode_start_else", @"encodeStartElse",
                   @{}, ^(id enc) {
                     ((void (*)(id, SEL))objc_msgSend)(enc, sel_getUid("encodeStartElse"));
                   });

    addComputeCase(cases, ser, stream, @"compute_encode_end_if", @"encodeEndIf", @{},
                   ^(id enc) {
                     (void)((char (*)(id, SEL))objc_msgSend)(enc, sel_getUid("encodeEndIf"));
                   });

    addComputeCase(cases, ser, stream, @"compute_encode_start_while",
                   @"encodeStartWhile:offset:comparison:referenceValue:",
                   @{@"buffer_ref" : @(STUB_BUFFER_REF),
                     @"offset" : @0x2222,
                     @"comparison" : @0x44,
                     @"reference_value" : @0x55},
                   ^(id enc) {
                     ((void (*)(id, SEL, id, unsigned long, unsigned long,
                                unsigned int))objc_msgSend)(
                         enc,
                         sel_getUid("encodeStartWhile:offset:comparison:referenceValue:"),
                         buf, 0x2222, 0x44, 0x55);
                   });

    addComputeCase(cases, ser, stream, @"compute_encode_start_while_alt",
                   @"encodeStartWhile:offset:comparison:referenceValue:",
                   @{@"buffer_ref" : @(STUB_BUFFER_DST_REF),
                     @"offset" : @0x5555,
                     @"comparison" : @0x03,
                     @"reference_value" : @0x13572468},
                   ^(id enc) {
                     ((void (*)(id, SEL, id, unsigned long, unsigned long,
                                unsigned int))objc_msgSend)(
                         enc,
                         sel_getUid("encodeStartWhile:offset:comparison:referenceValue:"),
                         buf2, 0x5555, 0x03, 0x13572468u);
                   });

    addComputeCase(cases, ser, stream, @"compute_encode_end_while", @"encodeEndWhile", @{},
                   ^(id enc) {
                     (void)((char (*)(id, SEL))objc_msgSend)(enc,
                                                             sel_getUid("encodeEndWhile"));
                   });

    addComputeCase(cases, ser, stream, @"compute_encode_start_do_while",
                   @"encodeStartDoWhile", @{}, ^(id enc) {
                     ((void (*)(id, SEL))objc_msgSend)(enc,
                                                       sel_getUid("encodeStartDoWhile"));
                   });

    addComputeCase(cases, ser, stream, @"compute_encode_end_do_while",
                   @"encodeEndDoWhile:offset:comparison:referenceValue:",
                   @{@"buffer_ref" : @(STUB_BUFFER_REF),
                     @"offset" : @0x3333,
                     @"comparison" : @0x66,
                     @"reference_value" : @0x77},
                   ^(id enc) {
                     (void)((char (*)(id, SEL, id, unsigned long, unsigned long,
                                      unsigned int))objc_msgSend)(
                         enc,
                         sel_getUid("encodeEndDoWhile:offset:comparison:referenceValue:"),
                         buf, 0x3333, 0x66, 0x77);
                   });

    addComputeCase(cases, ser, stream, @"compute_encode_end_do_while_alt",
                   @"encodeEndDoWhile:offset:comparison:referenceValue:",
                   @{@"buffer_ref" : @(STUB_BUFFER_DST_REF),
                     @"offset" : @0x6666,
                     @"comparison" : @0x07,
                     @"reference_value" : @0xfeedface},
                   ^(id enc) {
                     (void)((char (*)(id, SEL, id, unsigned long, unsigned long,
                                      unsigned int))objc_msgSend)(
                         enc,
                         sel_getUid("encodeEndDoWhile:offset:comparison:referenceValue:"),
                         buf2, 0x6666, 0x07, 0xfeedfaceu);
                   });
  });

  // The remaining no-argument selectors, driven so their rows rest on an
  // observation rather than on their names.
  addComputeCase(cases, ser, stream, @"compute_flush_writes", @"flushWrites", @{},
                 ^(id enc) {
                   ((void (*)(id, SEL))objc_msgSend)(enc, sel_getUid("flushWrites"));
                 });

  // Gated on `-setSupportsComputePassDescriptorDispatchType:`, along with
  // `writeDescriptor` below. Both were captured writing nothing and carried
  // EMITS_NO_OPERATION; the flag is named by the attribution passes, not
  // guessed from the selector.
  //
  // `maybeEmitSerialBarrier` is driven twice, at both dispatch types. Its name
  // says it emits *conditionally*, and the condition it is named for is the
  // pass's dispatch type -- so one observation cannot tell a record that
  // always appears from one that appears at the type this encoder happens to
  // start in. `setCurrentDispatchType:` is itself silent at the default state,
  // which is why the state has to be set through it here rather than assumed.
  withCapability(ser, @"ComputePassDescriptorDispatchType", ^{
    // The scope is the serializer's choice, not an argument, so the
    // expectation is built from the SDK's own enum rather than transcribed out
    // of the bytes -- which would make the fixture agree with itself.
    addComputeCase(cases, ser, stream, @"compute_maybe_emit_serial_barrier",
                   @"maybeEmitSerialBarrier",
                   @{@"scope" : @(MTLBarrierScopeBuffers | MTLBarrierScopeTextures)},
                   ^(id enc) {
                     ((void (*)(id, SEL))objc_msgSend)(
                         enc, sel_getUid("maybeEmitSerialBarrier"));
                   });

    addComputeCase(cases, ser, stream, @"compute_maybe_emit_serial_barrier_concurrent",
                   @"maybeEmitSerialBarrier", @{}, ^(id enc) {
                     ((void (*)(id, SEL, unsigned long))objc_msgSend)(
                         enc, sel_getUid("setCurrentDispatchType:"), 1);
                     ((void (*)(id, SEL))objc_msgSend)(
                         enc, sel_getUid("maybeEmitSerialBarrier"));
                   });
  });

  // The other selector the silent-list sweep could not see: it asserts at the
  // default state and emits under its own flag. Its blit sibling
  // `invalidateCompressedTexture*` is gated on `BlitEncoderSPI` instead, so a
  // family split across two flags is a real thing this serializer does.
  withCapability(ser, @"InsertCompressedTextureReinterpretationFlush", ^{
    addComputeCase(cases, ser, stream,
                   @"compute_insert_compressed_texture_reinterpretation_flush",
                   @"insertCompressedTextureReinterpretationFlush", @{}, ^(id enc) {
                     ((void (*)(id, SEL))objc_msgSend)(
                         enc, sel_getUid("insertCompressedTextureReinterpretationFlush"));
                   });
  });

  // The compute encoder's three attribute-stride binds.
  //
  // `reims_vgpu::runtime::decode::compute` has constants for two of them, and
  // none had ever been driven: they were captured at the default capability
  // state, came back silent, and were filed as records Apple does not emit.
  // They are gated on `-supportsDynamicAttributeStride`, the same flag as the
  // render encoder's -- whose four forms were driven when that flag was first
  // chased, while these three were left behind. Driving a *family* is not the
  // same as driving a flag, and this is what the difference cost.
  withCapability(ser, @"DynamicAttributeStride", ^{
    addComputeCase(cases, ser, stream, @"compute_set_buffer_stride",
                   @"setBuffer:offset:attributeStride:atIndex:",
                   @{@"buffer_ref" : @(STUB_BUFFER_REF),
                     @"offset" : @0x1234,
                     @"attribute_stride" : @0x5678,
                     @"first" : @5,
                     @"count" : @1},
                   ^(id enc) {
                     ((void (*)(id, SEL, id, unsigned long, unsigned long,
                                unsigned long))objc_msgSend)(
                         enc, sel_getUid("setBuffer:offset:attributeStride:atIndex:"), buf,
                         0x1234, 0x5678, 5);
                   });

    addComputeCase(cases, ser, stream, @"compute_set_buffer_offset_stride",
                   @"setBufferOffset:attributeStride:atIndex:",
                   @{@"offset" : @0x1234, @"attribute_stride" : @0x5678, @"first" : @6},
                   ^(id enc) {
                     ((void (*)(id, SEL, unsigned long, unsigned long,
                                unsigned long))objc_msgSend)(
                         enc, sel_getUid("setBufferOffset:attributeStride:atIndex:"), 0x1234,
                         0x5678, 6);
                   });

    // The fourth attribute-stride form, and the one the two above do not
    // settle. `setBytes:length:atIndex:` stages its bytes through the command
    // stream and emits the buffer bind naming the staging pair; the render
    // encoder's stride form does the same. Whether this one follows that or
    // the *other* pair of stride selectors, which emit nothing, is what the
    // case measures rather than infers. The staging ref and offset come from
    // the harness's own command stream, exactly as on the non-stride sibling.
    static const unsigned char computeStrideBytes[12] = {
        0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xcb};
    addComputeCase(cases, ser, stream, @"compute_set_bytes_stride",
                   @"setBytes:length:attributeStride:atIndex:",
                   @{@"buffer_ref" : @(STUB_STAGING_REF),
                     @"offset" : @(STUB_STAGING_OFFSET),
                     @"length" : @12,
                     @"attribute_stride" : @0x789a,
                     @"first" : @4,
                     @"count" : @1},
                   ^(id enc) {
                     ((void (*)(id, SEL, const void *, unsigned long, unsigned long,
                                unsigned long))objc_msgSend)(
                         enc, sel_getUid("setBytes:length:attributeStride:atIndex:"),
                         computeStrideBytes, sizeof(computeStrideBytes), 0x789a, 4);
                   });
  });

  addComputeCase(cases, ser, stream, @"compute_set_stage_in_region_indirect",
                 @"setStageInRegionWithIndirectBuffer:indirectBufferOffset:",
                 @{@"indirect_buffer_ref" : @(STUB_BUFFER_REF),
                   @"indirect_buffer_offset" : @0x1111},
                 ^(id enc) {
                   ((void (*)(id, SEL, id, unsigned long))objc_msgSend)(
                       enc,
                       sel_getUid("setStageInRegionWithIndirectBuffer:"
                                  "indirectBufferOffset:"),
                       buf, 0x1111);
                 });

  addComputeCase(cases, ser, stream, @"compute_set_acceleration_structure",
                 @"setAccelerationStructure:atBufferIndex:",
                 @{@"object_ref" : @(STUB_BUFFER_REF), @"index" : @3}, ^(id enc) {
                   ((void (*)(id, SEL, id, unsigned long))objc_msgSend)(
                       enc, sel_getUid("setAccelerationStructure:atBufferIndex:"), buf, 3);
                 });

  addComputeCase(cases, ser, stream, @"compute_set_visible_function_table",
                 @"setVisibleFunctionTable:atBufferIndex:",
                 @{@"object_ref" : @(STUB_BUFFER_REF), @"index" : @4}, ^(id enc) {
                   ((void (*)(id, SEL, id, unsigned long))objc_msgSend)(
                       enc, sel_getUid("setVisibleFunctionTable:atBufferIndex:"), buf, 4);
                 });

  addComputeCase(cases, ser, stream, @"compute_set_intersection_function_table",
                 @"setIntersectionFunctionTable:atBufferIndex:",
                 @{@"object_ref" : @(STUB_BUFFER_REF), @"index" : @5}, ^(id enc) {
                   ((void (*)(id, SEL, id, unsigned long))objc_msgSend)(
                       enc, sel_getUid("setIntersectionFunctionTable:atBufferIndex:"), buf,
                       5);
                 });

  addComputeCase(cases, ser, stream, @"compute_sample_counters",
                 @"sampleCountersInBuffer:atSampleIndex:withBarrier:",
                 @{@"counters_ref" : @(STUB_BUFFER_REF),
                   @"sample_index" : @0x1100,
                   @"barrier" : @1},
                 ^(id enc) {
                   ((void (*)(id, SEL, id, unsigned long, char))objc_msgSend)(
                       enc, sel_getUid("sampleCountersInBuffer:atSampleIndex:withBarrier:"),
                       buf, 0x1100, 1);
                 });

  {
    id samplers[2] = {sampler, [[StubSamplerState alloc] init]};
    const id *sampler_list = samplers;
    addComputeCase(cases, ser, stream, @"compute_set_sampler_states_range",
                   @"setSamplerStates:withRange:",
                   @{@"sampler_ref" : @(STUB_SAMPLER_REF), @"first" : @3, @"count" : @2},
                   ^(id enc) {
                     ((void (*)(id, SEL, const id *, NSRange))objc_msgSend)(
                         enc, sel_getUid("setSamplerStates:withRange:"), sampler_list,
                         NSMakeRange(3, 2));
                   });

    id bufs2[2] = {buf, buf2};
    const id *buf_list2 = bufs2;
    NSUInteger offs2[2] = {0x1111, 0x2222};
    const NSUInteger *off_list2 = offs2;
    NSUInteger strides2[2] = {0x3333, 0x4444};
    const NSUInteger *stride_list2 = strides2;
    // The plural attribute-stride bind, the third of the three
    // `DynamicAttributeStride` forms on this encoder. Its two siblings are
    // driven under the flag above; this one stayed here beside the other
    // plural binds, and without the flag it wrote nothing. Two buffers at two
    // offsets *and* two strides, which is what shows both are per entry.
    withCapability(ser, @"DynamicAttributeStride", ^{
    addComputeCase(cases, ser, stream, @"compute_set_buffers_strides_range",
                   @"setBuffers:offsets:attributeStrides:withRange:",
                   @{@"buffer_ref" : @(STUB_BUFFER_REF),
                     @"buffer_ref_2" : @(STUB_BUFFER_DST_REF),
                     @"first" : @4,
                     @"count" : @2,
                     @"offset" : @0x1111,
                     @"offset_2" : @0x2222,
                     @"attribute_stride" : @0x3333,
                     @"attribute_stride_2" : @0x4444},
                   ^(id enc) {
                     ((void (*)(id, SEL, const id *, const NSUInteger *,
                                const NSUInteger *, NSRange))objc_msgSend)(
                         enc, sel_getUid("setBuffers:offsets:attributeStrides:withRange:"),
                         buf_list2, off_list2, stride_list2, NSMakeRange(4, 2));
                   });
    });

    id tables[2] = {buf, buf2};
    const id *table_list = tables;
    addComputeCase(cases, ser, stream, @"compute_set_visible_function_tables",
                   @"setVisibleFunctionTables:withBufferRange:",
                   @{@"object_ref" : @(STUB_BUFFER_REF), @"first" : @2, @"count" : @2},
                   ^(id enc) {
                     ((void (*)(id, SEL, const id *, NSRange))objc_msgSend)(
                         enc, sel_getUid("setVisibleFunctionTables:withBufferRange:"),
                         table_list, NSMakeRange(2, 2));
                   });

    addComputeCase(cases, ser, stream, @"compute_set_intersection_function_tables",
                   @"setIntersectionFunctionTables:withBufferRange:",
                   @{@"object_ref" : @(STUB_BUFFER_REF), @"first" : @3, @"count" : @2},
                   ^(id enc) {
                     ((void (*)(id, SEL, const id *, NSRange))objc_msgSend)(
                         enc, sel_getUid("setIntersectionFunctionTables:withBufferRange:"),
                         table_list, NSMakeRange(3, 2));
                   });
  }

  addComputeCase(cases, ser, stream, @"compute_reattach_to_command_stream",
                 @"reattachToCommandStream:", @{}, ^(id enc) {
                   ((void (*)(id, SEL, id))objc_msgSend)(
                       enc, sel_getUid("reattachToCommandStream:"), stream);
                 });

  addComputeCase(cases, ser, stream, @"compute_new_kernel_debug_info",
                 @"newKernelDebugInfo", @{}, ^(id enc) {
                   (void)((id (*)(id, SEL))objc_msgSend)(enc,
                                                         sel_getUid("newKernelDebugInfo"));
                 });

  addComputeCase(cases, ser, stream, @"compute_handle_splits", @"handleSplits", @{},
                 ^(id enc) {
                   (void)((char (*)(id, SEL))objc_msgSend)(enc,
                                                           sel_getUid("handleSplits"));
                 });

  addComputeCase(cases, ser, stream, @"compute_should_allow_reattach",
                 @"shouldAllowReattach", @{}, ^(id enc) {
                   (void)((char (*)(id, SEL))objc_msgSend)(
                       enc, sel_getUid("shouldAllowReattach"));
                 });

  withCapability(ser, @"ComputePassDescriptorDispatchType", ^{
    addComputeCase(cases, ser, stream, @"compute_write_descriptor", @"writeDescriptor",
                   @{@"dispatch_type" : @0}, ^(id enc) {
                     (void)((char (*)(id, SEL))objc_msgSend)(
                         enc, sel_getUid("writeDescriptor"));
                   });

    // The same selector after the pass's dispatch type has moved. If the
    // descriptor it writes carries that type, this record differs from the one
    // above and the field is located; if it does not, the two are identical and
    // that is the finding.
    addComputeCase(cases, ser, stream, @"compute_write_descriptor_concurrent",
                   @"writeDescriptor", @{@"dispatch_type" : @1}, ^(id enc) {
                     ((void (*)(id, SEL, unsigned long))objc_msgSend)(
                         enc, sel_getUid("setCurrentDispatchType:"), 1);
                     (void)((char (*)(id, SEL))objc_msgSend)(
                         enc, sel_getUid("writeDescriptor"));
                   });
  });

  addComputeCase(cases, ser, stream, @"compute_get_type", @"getType", @{}, ^(id enc) {
    (void)((unsigned long (*)(id, SEL))objc_msgSend)(enc, sel_getUid("getType"));
  });

  addComputeCase(cases, ser, stream, @"compute_dispatch_type", @"dispatchType", @{},
                 ^(id enc) {
                   (void)((unsigned long (*)(id, SEL))objc_msgSend)(
                       enc, sel_getUid("dispatchType"));
                 });

  addComputeCase(cases, ser, stream, @"compute_begin_segment",
                 @"beginSegment:protectionOptions:",
                 @{@"flag" : @1, @"protection_options" : @0x33}, ^(id enc) {
                   ((void (*)(id, SEL, char, unsigned long))objc_msgSend)(
                       enc, sel_getUid("beginSegment:protectionOptions:"), 1, 0x33);
                 });

  return cases;
}

// --- Info encoder records ---------------------------------------------------
//
// 21 selectors, and the only class whose selectors are *queries*: each takes an
// object and a pointer to a struct the answer lands in. The device decodes
// exactly one info opcode (`INFO_OP_ICB_HOST_RESOURCE`), so twenty of these
// have never been looked at.
//
// Its designated initializer takes no descriptor, unlike the other three.
static id makeInfoEncoder(id ser, id stream) {
  return ((id (*)(id, SEL, id, id))objc_msgSend)(
      [objc_getClass("PGSerializerInfoCommandEncoder") alloc],
      sel_getUid("initWithCommandBuffer:serializer:"), stream, ser);
}

static void addInfoCase(NSMutableArray *cases, id ser, id stream, NSString *name,
                        NSString *sel, NSDictionary *expect, void (^invoke)(id enc)) {
  addCaseOnEncoder(cases, @"PGSerializerInfoCommandEncoder",
                   ^id {
                     return makeInfoEncoder(ser, stream);
                   },
                   name, sel, expect, invoke);
}

/// Scratch the `info:` out-parameters are written into. Poisoned like the
/// arena, so a field the serializer leaves alone is recognisable -- these are
/// host-side answers, not wire bytes, and nothing here asserts on them.
static unsigned char gInfoOut[256];

/// `expect` for a query, with the reply pair the *stream* is about to hand back.
///
/// Every selector on this class takes an out-pointer, and the record has to name
/// where that lands on the far side of the wire. The encoder does not invent
/// that pair: it asks the command stream through
/// `-getBufferBytes:alignment:buffer:offset:` and records the answer. So the
/// expectation for those two fields comes from `CaptureCommandStream`, which is
/// the input object here in exactly the way a descriptor is for a creation
/// record -- and a fixture that omitted them could not tell a recorded reply
/// pair from two zeroed fields.
///
/// `copyRasterizationRateParameterBuffer:` deliberately does not use this: its
/// buffer and offset are the caller's, not the stream's, and that difference is
/// the only thing separating its record from a query's.
static NSDictionary *infoQueryExpect(NSDictionary *base) {
  NSMutableDictionary *d = [base mutableCopy];
  d[@"reply_buffer_ref"] = @(STUB_STAGING_REF);
  d[@"reply_offset"] = @(STUB_STAGING_OFFSET);
  return d;
}

/// A coordinate a rate-map mapper is asked to translate.
///
/// Declared once per case and used for both the call and its expectation, so
/// the two cannot drift apart the way a transcribed literal can.
struct InfoCoord {
  float x, y;
};

static NSArray *infoCases(id ser) {
  NSMutableArray *cases = [NSMutableArray array];
  id stream = [[CaptureCommandStream alloc] init];
  void *out = gInfoOut;

  id buf = [[StubBuffer alloc] init];
  id tex = [[StubTexture alloc] init];
  id icb = [[StubICB alloc] init];
  id heap = [[StubHeap alloc] init];
  id sampler = [[StubSamplerState alloc] init];
  id depth_stencil = [[StubDepthStencilState alloc] init];
  id pipeline = [[StubPipelineState alloc] init];
  id rate_map = [[StubRateMap alloc] init];

  // The `…HostResourceInfo:info:` family: one object, one out-pointer. Each
  // names a different stub so a record that picked up the wrong object is
  // visible rather than plausible.
  struct {
    NSString *name;
    NSString *sel;
    id object;
    unsigned ref;
  } host_info[] = {
      {@"info_icb_host_resource", @"icbHostResourceInfo:info:", icb, STUB_ICB_REF},
      {@"info_buffer_host_resource", @"bufferHostResourceInfo:info:", buf,
       STUB_BUFFER_REF},
      {@"info_texture_host_resource", @"textureHostResourceInfo:info:", tex,
       STUB_TEXTURE_REF},
      {@"info_heap_host_resource", @"heapHostResourceInfo:info:", heap, STUB_HEAP_REF},
      {@"info_sampler_host_resource", @"samplerStateHostResourceInfo:info:", sampler,
       STUB_SAMPLER_REF},
      {@"info_depth_stencil_host_resource", @"depthStencilHostResourceInfo:info:",
       depth_stencil, STUB_DEPTH_STENCIL_REF},
      {@"info_render_pipeline_host_resource", @"renderPipelineHostResourceInfo:info:",
       pipeline, STUB_PIPELINE_REF},
      {@"info_compute_pipeline_host_resource", @"computePipelineHostResourceInfo:info:",
       pipeline, STUB_PIPELINE_REF},
      {@"info_render_pipeline_state", @"renderPipelineStateInfo:info:", pipeline,
       STUB_PIPELINE_REF},
      {@"info_compute_pipeline_state", @"computePipelineStateInfo:info:", pipeline,
       STUB_PIPELINE_REF},
  };
  for (unsigned i = 0; i < sizeof(host_info) / sizeof(host_info[0]); i++) {
    NSString *sel = host_info[i].sel;
    id object = host_info[i].object;
    memset(gInfoOut, gPoison, sizeof(gInfoOut));
    addInfoCase(cases, ser, stream, host_info[i].name, sel,
                infoQueryExpect(@{@"object_ref" : @(host_info[i].ref)}), ^(id enc) {
                  ((void (*)(id, SEL, id, void *))objc_msgSend)(enc, sel_getUid(sel.UTF8String),
                                                                object, out);
                });
  }

  // `heapTextureDescriptorSizeAndAlign:sizeAndAlign:` is the one selector on
  // this class whose first argument is not a resource. Its name says
  // "descriptor" and its type encoding says `@`, and the table above had been
  // handing it a `StubHeap` on the strength of the "heap" prefix -- which
  // faulted, landed on `crashed`, and so measured nothing at all. Driven with
  // the descriptor the sizing query on `PGSerializer` takes, it emits.
  {
    MTLTextureDescriptor *sizingDesc = baselineTexture();
    memset(gInfoOut, gPoison, sizeof(gInfoOut));
    addInfoCase(cases, ser, stream, @"info_heap_texture_descriptor_size_and_align",
                @"heapTextureDescriptorSizeAndAlign:sizeAndAlign:",
                infoQueryExpect(expectFromTextureDescriptor(sizingDesc)), ^(id enc) {
                  ((void (*)(id, SEL, id, void *))objc_msgSend)(
                      enc, sel_getUid("heapTextureDescriptorSizeAndAlign:sizeAndAlign:"),
                      sizingDesc, out);
                });

    // The fifth record the wide descriptor reaches, and the only one that is a
    // query rather than a creation. Same flag as the three backed creations.
    withCapability(ser, @"TextureDescriptor2", ^{
      memset(gInfoOut, gPoison, sizeof(gInfoOut));
      addInfoCase(cases, ser, stream, @"info_heap_texture_descriptor_size_and_align_wide",
                  @"heapTextureDescriptorSizeAndAlign:sizeAndAlign:",
                  infoQueryExpect(expectFromTextureDescriptor(sizingDesc)), ^(id enc) {
                    ((void (*)(id, SEL, id, void *))objc_msgSend)(
                        enc, sel_getUid("heapTextureDescriptorSizeAndAlign:sizeAndAlign:"),
                        sizingDesc, out);
                  });
    });
  }

  // The two imageblock queries take a size by value between the object and the
  // out-pointer.
  addInfoCase(cases, ser, stream, @"info_render_pipeline_imageblock",
              @"renderPipelineStateImageBlockMemoryLength:imageblockDimensions:info:",
              infoQueryExpect(@{@"object_ref" : @(STUB_PIPELINE_REF),
                                @"width" : @0x11,
                                @"height" : @0x22,
                                @"depth" : @0x33}),
              ^(id enc) {
                ((void (*)(id, SEL, id, MTLSize, void *))objc_msgSend)(
                    enc,
                    sel_getUid("renderPipelineStateImageBlockMemoryLength:"
                               "imageblockDimensions:info:"),
                    pipeline, MTLSizeMake(0x11, 0x22, 0x33), out);
              });

  addInfoCase(cases, ser, stream, @"info_compute_pipeline_imageblock",
              @"computePipelineStateImageBlockMemoryLength:imageblockDimensions:info:",
              infoQueryExpect(@{@"object_ref" : @(STUB_PIPELINE_REF),
                                @"width" : @0x11,
                                @"height" : @0x22,
                                @"depth" : @0x33}),
              ^(id enc) {
                ((void (*)(id, SEL, id, MTLSize, void *))objc_msgSend)(
                    enc,
                    sel_getUid("computePipelineStateImageBlockMemoryLength:"
                               "imageblockDimensions:info:"),
                    pipeline, MTLSizeMake(0x11, 0x22, 0x33), out);
              });

  addInfoCase(cases, ser, stream, @"info_copy_rasterization_rate_parameter_buffer",
              @"copyRasterizationRateParameterBuffer:buffer:bufferOffset:",
              @{@"object_ref" : @(STUB_RATE_MAP_REF),
                @"buffer_ref" : @(STUB_BUFFER_REF),
                @"buffer_offset" : @0x1111},
              ^(id enc) {
                ((void (*)(id, SEL, id, id, unsigned long))objc_msgSend)(
                    enc,
                    sel_getUid("copyRasterizationRateParameterBuffer:buffer:"
                               "bufferOffset:"),
                    rate_map, buf, 0x1111);
              });

  addInfoCase(cases, ser, stream, @"info_get_rasterization_rate_map",
              @"getRasterizationRateMapInfo:layerCount:info:",
              infoQueryExpect(@{@"object_ref" : @(STUB_RATE_MAP_REF),
                                @"layer_count" : @2}), ^(id enc) {
                ((void (*)(id, SEL, id, unsigned int, void *))objc_msgSend)(
                    enc, sel_getUid("getRasterizationRateMapInfo:layerCount:info:"), rate_map,
                    2, out);
              });

  // A second layer count. The word at `+16` of the first capture read 28 for a
  // count of 2, which is not the count -- so either it is derived from it or it
  // is something else, and one observation cannot say which.
  addInfoCase(cases, ser, stream, @"info_get_rasterization_rate_map_alt",
              @"getRasterizationRateMapInfo:layerCount:info:",
              infoQueryExpect(@{@"object_ref" : @(STUB_RATE_MAP_REF),
                                @"layer_count" : @5}), ^(id enc) {
                ((void (*)(id, SEL, id, unsigned int, void *))objc_msgSend)(
                    enc, sel_getUid("getRasterizationRateMapInfo:layerCount:info:"),
                    rate_map, 5, out);
              });

  // The four coordinate mappers. Two take a coordinate by value, one takes an
  // array, and one is the "internal" form with an extra command word.
  // The two fixed-opcode mappers. Different layers and different coordinates,
  // so neither record can be confused with the other and no field pair can be
  // swapped unseen.
  const struct InfoCoord screen_coord = {0.25f, 0.75f};
  addInfoCase(cases, ser, stream, @"info_map_screen_to_physical",
              @"mapScreenToPhysicalCoordinates:forScreenCoordinate:forLayer:"
              @"toPhysicalCoordinate:",
              infoQueryExpect(@{@"object_ref" : @(STUB_RATE_MAP_REF),
                                @"layer" : @3,
                                @"x" : @(screen_coord.x),
                                @"y" : @(screen_coord.y)}), ^(id enc) {
                ((void (*)(id, SEL, id, struct InfoCoord, unsigned long,
                           void *))objc_msgSend)(
                    enc,
                    sel_getUid("mapScreenToPhysicalCoordinates:forScreenCoordinate:"
                               "forLayer:toPhysicalCoordinate:"),
                    rate_map, screen_coord, 3, out);
              });

  const struct InfoCoord physical_coord = {0.125f, 0.875f};
  addInfoCase(cases, ser, stream, @"info_map_physical_to_screen",
              @"mapPhysicalToScreenCoordinates:forPhysicalCoordinate:forLayer:"
              @"toScreenCoordinate:",
              infoQueryExpect(@{@"object_ref" : @(STUB_RATE_MAP_REF),
                                @"layer" : @4,
                                @"x" : @(physical_coord.x),
                                @"y" : @(physical_coord.y)}), ^(id enc) {
                ((void (*)(id, SEL, id, struct InfoCoord, unsigned long,
                           void *))objc_msgSend)(
                    enc,
                    sel_getUid("mapPhysicalToScreenCoordinates:forPhysicalCoordinate:"
                               "forLayer:toScreenCoordinate:"),
                    rate_map, physical_coord, 4, out);
              });

  addInfoCase(cases, ser, stream, @"info_map_physical_to_screen_multiple",
              @"mapPhysicalToScreenCoordinateMultiple:forPhysicalCoordinates:forLayer:"
              @"toScreenCoordinates:count:",
              @{@"object_ref" : @(STUB_RATE_MAP_REF), @"layer" : @5, @"count" : @2},
              ^(id enc) {
                static const float coords[4] = {0.25f, 0.75f, 0.125f, 0.875f};
                ((void (*)(id, SEL, id, const void *, unsigned long, void *,
                           unsigned long))objc_msgSend)(
                    enc,
                    sel_getUid("mapPhysicalToScreenCoordinateMultiple:"
                               "forPhysicalCoordinates:forLayer:toScreenCoordinates:"
                               "count:"),
                    rate_map, coords, 5, out, 2);
              });

  // The generic form the two above are wrappers over: its `command:` argument
  // lands in the opcode field. Two distinct commands, because one alone would
  // be satisfied by a fixed opcode that happened to equal it.
  addInfoCase(cases, ser, stream, @"info_map_coordinate_internal",
              @"mapCoordinateInternal:fromCoordinate:forLayer:toCoordinate:command:",
              infoQueryExpect(@{@"object_ref" : @(STUB_RATE_MAP_REF),
                                @"layer" : @6,
                                @"x" : @(screen_coord.x),
                                @"y" : @(screen_coord.y),
                                @"command" : @0x77}),
              ^(id enc) {
                ((void (*)(id, SEL, id, struct InfoCoord, unsigned long, void *,
                           unsigned int))objc_msgSend)(
                    enc,
                    sel_getUid("mapCoordinateInternal:fromCoordinate:forLayer:"
                               "toCoordinate:command:"),
                    rate_map, screen_coord, 6, out, 0x77);
              });

  addInfoCase(cases, ser, stream, @"info_map_coordinate_internal_alt",
              @"mapCoordinateInternal:fromCoordinate:forLayer:toCoordinate:command:",
              infoQueryExpect(@{@"object_ref" : @(STUB_RATE_MAP_REF),
                                @"layer" : @7,
                                @"x" : @(physical_coord.x),
                                @"y" : @(physical_coord.y),
                                @"command" : @0x55}),
              ^(id enc) {
                ((void (*)(id, SEL, id, struct InfoCoord, unsigned long, void *,
                           unsigned int))objc_msgSend)(
                    enc,
                    sel_getUid("mapCoordinateInternal:fromCoordinate:forLayer:"
                               "toCoordinate:command:"),
                    rate_map, physical_coord, 7, out, 0x55);
              });

  addInfoCase(cases, ser, stream, @"info_begin_segment",
              @"beginSegment:protectionOptions:",
              @{@"flag" : @1, @"protection_options" : @0x33}, ^(id enc) {
                ((void (*)(id, SEL, char, unsigned long))objc_msgSend)(
                    enc, sel_getUid("beginSegment:protectionOptions:"), 1, 0x33);
              });

  return cases;
}

// --- The remaining object-creation records ---------------------------------
//
// `newTextureWithDescriptor:allocator:` is one of several creation selectors
// that carry a descriptor onto the wire, and it was the only one driven. The
// rest were parked in the manifest as `Unimplemented` carrying an opcode each,
// which reads as "measured, no view yet" but was not measured here — no case,
// no `silent` entry, no `unsupported` entry. This drives them.
//
// The ref is the selector's own return value rather than a number predicted
// here: `-allocateObjectRef` hands them out in sequence, so a case that
// asserted the sequence would pass whatever the record's first word held.
static void addCreationCase(NSMutableArray *cases, NSString *name, NSString *sel,
                            NSMutableDictionary *expect, unsigned (^invoke)(void)) {
  fprintf(stderr, "case %s\n", name.UTF8String);
  captureCase(cases, @"PGSerializer", name, sel, expect, 1, nil, ^{
    expect[@"object_ref"] = @(invoke());
  });
}

/// Everything the sampler view should report, read off the descriptor.
///
/// Metal normalizes some of these on the way in, so this runs *after* the
/// properties are set and takes what Metal kept, not what it was handed.
static NSMutableDictionary *expectFromSamplerDescriptor(MTLSamplerDescriptor *d) {
  NSMutableDictionary *expect = [@{
    @"min_filter" : @((unsigned)d.minFilter),
    @"mag_filter" : @((unsigned)d.magFilter),
    @"mip_filter" : @((unsigned)d.mipFilter),
    @"s_address_mode" : @((unsigned)d.sAddressMode),
    @"t_address_mode" : @((unsigned)d.tAddressMode),
    @"r_address_mode" : @((unsigned)d.rAddressMode),
    @"max_anisotropy" : @((unsigned long long)d.maxAnisotropy),
    @"compare_function" : @((unsigned)d.compareFunction),
    @"border_color" : @((unsigned)d.borderColor),
    @"lod_min_clamp" : @(d.lodMinClamp),
    @"lod_max_clamp" : @(d.lodMaxClamp),
    @"lod_average" : @(d.lodAverage ? 1 : 0),
    @"normalized_coordinates" : @(d.normalizedCoordinates ? 1 : 0),
    @"support_argument_buffers" : @(d.supportArgumentBuffers ? 1 : 0),
  } mutableCopy];
  expect[@"force_resource_index"] = @(((char (*)(id, SEL))objc_msgSend)(
      d, sel_getUid("forceResourceIndex")) ? 1 : 0);
  expect[@"force_seams_on_cubemap_filtering"] = @(((char (*)(id, SEL))objc_msgSend)(
      d, sel_getUid("forceSeamsOnCubemapFiltering")) ? 1 : 0);
  expect[@"resource_index"] = @(((unsigned long long (*)(id, SEL))objc_msgSend)(
      d, sel_getUid("resourceIndex")));
  expect[@"pixel_format"] = @(((unsigned long long (*)(id, SEL))objc_msgSend)(
      d, sel_getUid("pixelFormat")));
  expect[@"reduction_mode"] = @(((unsigned long long (*)(id, SEL))objc_msgSend)(
      d, sel_getUid("reductionMode")));
  expect[@"lod_bias"] = @(((float (*)(id, SEL))objc_msgSend)(
      d, sel_getUid("lodBias")));
  expect[@"border_color_spi"] = @(((unsigned long long (*)(id, SEL))objc_msgSend)(
      d, sel_getUid("borderColorSPI")));
  return expect;
}

/// Baseline sampler: Metal's own defaults, with one landmark.
///
/// `lodMaxClamp` keeps its `FLT_MAX` default deliberately — `0x7f7fffff` is
/// recognisable on sight in a hex dump, which is what locates the float pair
/// before either clamp is perturbed.
static MTLSamplerDescriptor *baselineSampler(void) {
  MTLSamplerDescriptor *d = [[MTLSamplerDescriptor alloc] init];
  d.minFilter = MTLSamplerMinMagFilterNearest;
  d.magFilter = MTLSamplerMinMagFilterNearest;
  d.mipFilter = MTLSamplerMipFilterNotMipmapped;
  d.sAddressMode = MTLSamplerAddressModeClampToEdge;
  d.tAddressMode = MTLSamplerAddressModeClampToEdge;
  d.rAddressMode = MTLSamplerAddressModeClampToEdge;
  d.maxAnisotropy = 1;
  d.lodMinClamp = 0.0f;
  d.lodMaxClamp = FLT_MAX;
  d.normalizedCoordinates = YES;
  d.compareFunction = MTLCompareFunctionNever;
  d.borderColor = MTLSamplerBorderColorTransparentBlack;
  return d;
}

static void addSamplerCase(NSMutableArray *cases, id ser, id cap, NSString *name,
                           MTLSamplerDescriptor *d) {
  addCreationCase(cases, name, @"newSamplerStateWithDescriptor:allocator:",
                  expectFromSamplerDescriptor(d), ^unsigned {
                    return ((unsigned (*)(id, SEL, id, id))objc_msgSend)(
                        ser, sel_getUid("newSamplerStateWithDescriptor:allocator:"), d,
                        cap);
                  });
}

/// A stencil face, built rather than mutated in place.
///
/// `-[MTLDepthStencilDescriptor frontFaceStencil]` is documented to answer a
/// copy, so setting a field through the getter would set it on a temporary and
/// the case would perturb nothing. Assigning a whole face avoids the question.
static MTLStencilDescriptor *stencilFace(MTLCompareFunction cmp, MTLStencilOperation sfail,
                                         MTLStencilOperation dfail, MTLStencilOperation pass,
                                         uint32_t readMask, uint32_t writeMask) {
  MTLStencilDescriptor *s = [[MTLStencilDescriptor alloc] init];
  s.stencilCompareFunction = cmp;
  s.stencilFailureOperation = sfail;
  s.depthFailureOperation = dfail;
  s.depthStencilPassOperation = pass;
  s.readMask = readMask;
  s.writeMask = writeMask;
  return s;
}

static MTLStencilDescriptor *baselineStencilFace(void) {
  return stencilFace(MTLCompareFunctionAlways, MTLStencilOperationKeep,
                     MTLStencilOperationKeep, MTLStencilOperationKeep, 0xffffffffu,
                     0xffffffffu);
}

static MTLDepthStencilDescriptor *baselineDepthStencil(void) {
  MTLDepthStencilDescriptor *d = [[MTLDepthStencilDescriptor alloc] init];
  d.depthCompareFunction = MTLCompareFunctionAlways;
  d.depthWriteEnabled = NO;
  d.frontFaceStencil = baselineStencilFace();
  d.backFaceStencil = baselineStencilFace();
  return d;
}

/// Clear one object slot in the descriptor-private structure after checking
/// that the runtime still declares the expected object/object prefix. Public
/// setters normalize `nil` back to a default face and cannot drive this state.
static BOOL forceDepthStencilFaceAbsent(MTLDepthStencilDescriptor *d, unsigned face) {
  SEL privateSel = sel_getUid("depthStencilPrivate");
  Method method = class_getInstanceMethod(object_getClass(d), privateSel);
  const char *encoding = method ? method_getTypeEncoding(method) : NULL;
  if (!encoding || !strstr(encoding, "DepthStencilDescriptorPrivate=@@")) {
    fprintf(stderr, "depth-stencil private layout lacks its object/object prefix\n");
    return NO;
  }
  id __unsafe_unretained *fields = ((id __unsafe_unretained *(*)(id, SEL))objc_msgSend)(
      d, privateSel);
  if (!fields || face > 1) return NO;
  fields[face] = nil;
  return fields[face] == nil;
}

/// The two faces are read separately so a view that swaps them fails.
///
/// Every case below moves exactly one face, and gives it a value the other face
/// does not hold, so front/back cannot be told apart by luck.
static NSMutableDictionary *expectFromDepthStencilDescriptor(MTLDepthStencilDescriptor *d) {
  MTLStencilDescriptor *f = d.frontFaceStencil;
  MTLStencilDescriptor *b = d.backFaceStencil;
  // A nil face answers 0 to every accessor, which is a valid value for most of
  // them -- so presence is recorded separately rather than inferred from the
  // numbers, and the face fields of an absent face assert nothing.
  return [@{
    @"depth_compare_function" : @((unsigned)d.depthCompareFunction),
    @"depth_write_enabled" : @(d.depthWriteEnabled ? 1 : 0),
    @"front_face_present" : @(f ? 1 : 0),
    @"back_face_present" : @(b ? 1 : 0),
    @"front_stencil_compare_function" : @((unsigned)f.stencilCompareFunction),
    @"front_stencil_failure_operation" : @((unsigned)f.stencilFailureOperation),
    @"front_depth_failure_operation" : @((unsigned)f.depthFailureOperation),
    @"front_depth_stencil_pass_operation" : @((unsigned)f.depthStencilPassOperation),
    @"front_read_mask" : @((unsigned long long)f.readMask),
    @"front_write_mask" : @((unsigned long long)f.writeMask),
    @"back_stencil_compare_function" : @((unsigned)b.stencilCompareFunction),
    @"back_stencil_failure_operation" : @((unsigned)b.stencilFailureOperation),
    @"back_depth_failure_operation" : @((unsigned)b.depthFailureOperation),
    @"back_depth_stencil_pass_operation" : @((unsigned)b.depthStencilPassOperation),
    @"back_read_mask" : @((unsigned long long)b.readMask),
    @"back_write_mask" : @((unsigned long long)b.writeMask),
  } mutableCopy];
}

static void addDepthStencilCase(NSMutableArray *cases, id ser, id cap, NSString *name,
                                MTLDepthStencilDescriptor *d) {
  addCreationCase(cases, name, @"newDepthStencilStateWithDescriptor:allocator:",
                  expectFromDepthStencilDescriptor(d), ^unsigned {
                    return ((unsigned (*)(id, SEL, id, id))objc_msgSend)(
                        ser, sel_getUid("newDepthStencilStateWithDescriptor:allocator:"),
                        d, cap);
                  });
}

/// Serialize a descriptor after clearing its private face objects without
/// invoking the public getters, which normalize them back to defaults.
static BOOL addPrivateAbsentDepthStencilCase(NSMutableArray *cases, id ser, id cap,
                                             NSString *name, BOOL frontAbsent,
                                             BOOL backAbsent) {
  MTLDepthStencilDescriptor *d = baselineDepthStencil();
  NSMutableDictionary *expect = expectFromDepthStencilDescriptor(d);
  for (NSString *side in @[ @"front", @"back" ]) {
    BOOL absent = [side isEqualToString:@"front"] ? frontAbsent : backAbsent;
    if (!absent) continue;
    unsigned face = [side isEqualToString:@"front"] ? 0 : 1;
    if (!forceDepthStencilFaceAbsent(d, face)) return NO;
    expect[[NSString stringWithFormat:@"%@_face_present", side]] = @0;
    for (NSString *field in @[
           @"stencil_compare_function", @"stencil_failure_operation",
           @"depth_failure_operation", @"depth_stencil_pass_operation",
           @"read_mask", @"write_mask"
         ])
      expect[[NSString stringWithFormat:@"%@_%@", side, field]] = @0;
  }
  addCreationCase(cases, name, @"newDepthStencilStateWithDescriptor:allocator:", expect,
                  ^unsigned {
                    return ((unsigned (*)(id, SEL, id, id))objc_msgSend)(
                        ser, sel_getUid("newDepthStencilStateWithDescriptor:allocator:"),
                        d, cap);
                  });
  return YES;
}

static NSArray *creationCases(id ser, id cap) {
  NSMutableArray *cases = [NSMutableArray array];
  MTLSamplerDescriptor *s;

  addSamplerCase(cases, ser, cap, @"sampler_baseline", baselineSampler());

  s = baselineSampler(); s.minFilter = MTLSamplerMinMagFilterLinear;
  addSamplerCase(cases, ser, cap, @"sampler_min_filter_linear", s);
  s = baselineSampler(); s.magFilter = MTLSamplerMinMagFilterLinear;
  addSamplerCase(cases, ser, cap, @"sampler_mag_filter_linear", s);
  s = baselineSampler(); s.mipFilter = MTLSamplerMipFilterNearest;
  addSamplerCase(cases, ser, cap, @"sampler_mip_filter_nearest", s);
  s = baselineSampler(); s.mipFilter = MTLSamplerMipFilterLinear;
  addSamplerCase(cases, ser, cap, @"sampler_mip_filter_linear", s);
  // Each axis gets a different mode, so a view that reads the wrong one of the
  // three reports a value no other case produced rather than a plausible one.
  s = baselineSampler(); s.sAddressMode = MTLSamplerAddressModeMirrorRepeat;
  addSamplerCase(cases, ser, cap, @"sampler_address_s_mirror_repeat", s);
  s = baselineSampler(); s.tAddressMode = MTLSamplerAddressModeClampToZero;
  addSamplerCase(cases, ser, cap, @"sampler_address_t_clamp_to_zero", s);
  s = baselineSampler(); s.rAddressMode = MTLSamplerAddressModeRepeat;
  addSamplerCase(cases, ser, cap, @"sampler_address_r_repeat", s);
  s = baselineSampler(); s.maxAnisotropy = 13;
  addSamplerCase(cases, ser, cap, @"sampler_max_anisotropy", s);
  s = baselineSampler(); s.lodMinClamp = 0.25f;
  addSamplerCase(cases, ser, cap, @"sampler_lod_min_clamp", s);
  s = baselineSampler(); s.lodMaxClamp = 6.5f;
  addSamplerCase(cases, ser, cap, @"sampler_lod_max_clamp", s);
  s = baselineSampler(); s.lodAverage = YES;
  addSamplerCase(cases, ser, cap, @"sampler_lod_average", s);
  s = baselineSampler(); s.compareFunction = MTLCompareFunctionGreater;
  addSamplerCase(cases, ser, cap, @"sampler_compare_greater", s);
  // Border colour is only meaningful with an address mode that reaches it, and
  // Metal is entitled to normalize the pair; moving both keeps the descriptor
  // one Metal accepts, and the expectation is read back afterwards either way.
  s = baselineSampler();
  s.sAddressMode = MTLSamplerAddressModeClampToBorderColor;
  s.tAddressMode = MTLSamplerAddressModeClampToBorderColor;
  s.rAddressMode = MTLSamplerAddressModeClampToBorderColor;
  s.borderColor = MTLSamplerBorderColorOpaqueWhite;
  addSamplerCase(cases, ser, cap, @"sampler_border_opaque_white", s);
  // Unnormalized coordinates constrain the rest of the descriptor — no
  // mipmapping, no anisotropy, edge or zero addressing only — and the baseline
  // already satisfies every one of those.
  s = baselineSampler(); s.normalizedCoordinates = NO;
  addSamplerCase(cases, ser, cap, @"sampler_unnormalized_coordinates", s);
  s = baselineSampler(); s.supportArgumentBuffers = YES;
  addSamplerCase(cases, ser, cap, @"sampler_support_argument_buffers", s);
  s = baselineSampler();
  ((void (*)(id, SEL, char))objc_msgSend)(
      s, sel_getUid("setForceSeamsOnCubemapFiltering:"), (char)1);
  addSamplerCase(cases, ser, cap, @"sampler_force_seams_on_cubemap_filtering", s);
  s = baselineSampler();
  ((void (*)(id, SEL, char))objc_msgSend)(
      s, sel_getUid("setForceResourceIndex:"), (char)1);
  addSamplerCase(cases, ser, cap, @"sampler_force_resource_index", s);
  s = baselineSampler();
  ((void (*)(id, SEL, unsigned long long))objc_msgSend)(
      s, sel_getUid("setResourceIndex:"), 0x1122334455667788ULL);
  addSamplerCase(cases, ser, cap, @"sampler_resource_index", s);
  s = baselineSampler();
  ((void (*)(id, SEL, unsigned long long))objc_msgSend)(
      s, sel_getUid("setPixelFormat:"), MTLPixelFormatBGRA8Unorm);
  addSamplerCase(cases, ser, cap, @"sampler_pixel_format", s);
  s = baselineSampler();
  ((void (*)(id, SEL, unsigned long long))objc_msgSend)(
      s, sel_getUid("setReductionMode:"), 1);
  addSamplerCase(cases, ser, cap, @"sampler_reduction_mode", s);
  s = baselineSampler();
  ((void (*)(id, SEL, float))objc_msgSend)(s, sel_getUid("setLodBias:"), 2.5f);
  addSamplerCase(cases, ser, cap, @"sampler_lod_bias", s);
  s = baselineSampler();
  ((void (*)(id, SEL, unsigned long long))objc_msgSend)(
      s, sel_getUid("setBorderColorSPI:"), 3);
  addSamplerCase(cases, ser, cap, @"sampler_border_color_spi", s);

  MTLDepthStencilDescriptor *ds;

  addDepthStencilCase(cases, ser, cap, @"depth_stencil_baseline", baselineDepthStencil());

  ds = baselineDepthStencil(); ds.depthCompareFunction = MTLCompareFunctionGreater;
  addDepthStencilCase(cases, ser, cap, @"depth_stencil_depth_compare_greater", ds);
  ds = baselineDepthStencil(); ds.depthWriteEnabled = YES;
  addDepthStencilCase(cases, ser, cap, @"depth_stencil_depth_write_enabled", ds);

  ds = baselineDepthStencil();
  ds.frontFaceStencil = stencilFace(MTLCompareFunctionEqual, MTLStencilOperationKeep,
                                    MTLStencilOperationKeep, MTLStencilOperationKeep,
                                    0xffffffffu, 0xffffffffu);
  addDepthStencilCase(cases, ser, cap, @"depth_stencil_front_compare_equal", ds);
  ds = baselineDepthStencil();
  ds.frontFaceStencil = stencilFace(MTLCompareFunctionAlways,
                                    MTLStencilOperationIncrementClamp,
                                    MTLStencilOperationKeep, MTLStencilOperationKeep,
                                    0xffffffffu, 0xffffffffu);
  addDepthStencilCase(cases, ser, cap, @"depth_stencil_front_fail_increment_clamp", ds);
  ds = baselineDepthStencil();
  ds.frontFaceStencil = stencilFace(MTLCompareFunctionAlways, MTLStencilOperationKeep,
                                    MTLStencilOperationDecrementClamp,
                                    MTLStencilOperationKeep, 0xffffffffu, 0xffffffffu);
  addDepthStencilCase(cases, ser, cap, @"depth_stencil_front_depth_fail_decrement_clamp",
                      ds);
  ds = baselineDepthStencil();
  ds.frontFaceStencil = stencilFace(MTLCompareFunctionAlways, MTLStencilOperationKeep,
                                    MTLStencilOperationKeep, MTLStencilOperationInvert,
                                    0xffffffffu, 0xffffffffu);
  addDepthStencilCase(cases, ser, cap, @"depth_stencil_front_pass_invert", ds);
  ds = baselineDepthStencil();
  ds.frontFaceStencil = stencilFace(MTLCompareFunctionAlways, MTLStencilOperationKeep,
                                    MTLStencilOperationKeep, MTLStencilOperationKeep,
                                    0x11223344u, 0xffffffffu);
  addDepthStencilCase(cases, ser, cap, @"depth_stencil_front_read_mask", ds);
  ds = baselineDepthStencil();
  ds.frontFaceStencil = stencilFace(MTLCompareFunctionAlways, MTLStencilOperationKeep,
                                    MTLStencilOperationKeep, MTLStencilOperationKeep,
                                    0xffffffffu, 0x55667788u);
  addDepthStencilCase(cases, ser, cap, @"depth_stencil_front_write_mask", ds);

  ds = baselineDepthStencil();
  ds.backFaceStencil = stencilFace(MTLCompareFunctionNotEqual, MTLStencilOperationKeep,
                                   MTLStencilOperationKeep, MTLStencilOperationKeep,
                                   0xffffffffu, 0xffffffffu);
  addDepthStencilCase(cases, ser, cap, @"depth_stencil_back_compare_not_equal", ds);
  ds = baselineDepthStencil();
  ds.backFaceStencil = stencilFace(MTLCompareFunctionAlways,
                                   MTLStencilOperationIncrementWrap,
                                   MTLStencilOperationKeep, MTLStencilOperationKeep,
                                   0xffffffffu, 0xffffffffu);
  addDepthStencilCase(cases, ser, cap, @"depth_stencil_back_fail_increment_wrap", ds);
  ds = baselineDepthStencil();
  ds.backFaceStencil = stencilFace(MTLCompareFunctionAlways, MTLStencilOperationKeep,
                                   MTLStencilOperationDecrementWrap,
                                   MTLStencilOperationKeep, 0xffffffffu, 0xffffffffu);
  addDepthStencilCase(cases, ser, cap, @"depth_stencil_back_depth_fail_decrement_wrap",
                      ds);
  ds = baselineDepthStencil();
  ds.backFaceStencil = stencilFace(MTLCompareFunctionAlways, MTLStencilOperationKeep,
                                   MTLStencilOperationKeep, MTLStencilOperationReplace,
                                   0xffffffffu, 0xffffffffu);
  addDepthStencilCase(cases, ser, cap, @"depth_stencil_back_pass_replace", ds);
  ds = baselineDepthStencil();
  ds.backFaceStencil = stencilFace(MTLCompareFunctionAlways, MTLStencilOperationKeep,
                                   MTLStencilOperationKeep, MTLStencilOperationKeep,
                                   0x99aabbccu, 0xffffffffu);
  addDepthStencilCase(cases, ser, cap, @"depth_stencil_back_read_mask", ds);
  ds = baselineDepthStencil();
  ds.backFaceStencil = stencilFace(MTLCompareFunctionAlways, MTLStencilOperationKeep,
                                   MTLStencilOperationKeep, MTLStencilOperationKeep,
                                   0xffffffffu, 0xddeeff00u);
  addDepthStencilCase(cases, ser, cap, @"depth_stencil_back_write_mask", ds);

  // Public `nil` assignments normalize back to default face objects. These
  // cases pin that negative result.
  ds = baselineDepthStencil(); ds.frontFaceStencil = nil;
  addDepthStencilCase(cases, ser, cap, @"depth_stencil_front_face_absent", ds);
  ds = baselineDepthStencil(); ds.backFaceStencil = nil;
  addDepthStencilCase(cases, ser, cap, @"depth_stencil_back_face_absent", ds);
  ds = baselineDepthStencil();
  ds.frontFaceStencil = nil;
  ds.backFaceStencil = nil;
  addDepthStencilCase(cases, ser, cap, @"depth_stencil_both_faces_absent", ds);

  // Drive the state the public setters make unreachable by clearing the two
  // object slots in the runtime-typed private structure independently.
  if (!addPrivateAbsentDepthStencilCase(
          cases, ser, cap, @"depth_stencil_private_front_face_absent", YES, NO) ||
      !addPrivateAbsentDepthStencilCase(
          cases, ser, cap, @"depth_stencil_private_back_face_absent", NO, YES) ||
      !addPrivateAbsentDepthStencilCase(
          cases, ser, cap, @"depth_stencil_private_both_faces_absent", YES, YES))
    return nil;

  // A fence carries no descriptor at all, so one case is the whole surface --
  // and that is the finding worth pinning: if the record is longer than a ref,
  // something else is in it.
  addCreationCase(cases, @"fence_new", @"newFenceWithAllocator:",
                  [@{} mutableCopy], ^unsigned {
                    return ((unsigned (*)(id, SEL, id))objc_msgSend)(
                        ser, sel_getUid("newFenceWithAllocator:"), cap);
                  });

  // The texture views. `reims-vgpu` decodes three of these opcodes today and
  // has never seen a record Apple produced, so these are the first fixtures
  // that can say whether it reads them right.
  //
  // The base texture is a stub answering `textureRef`, because a real
  // `MTLTexture` has no paravirt ref for the serializer to write. Every case
  // gives its ranges distinctive bounds -- a level base of 3 and a slice base
  // of 5 are different numbers, so a view that swapped the two `_NSRange`s
  // would report a pair no case produced.
  id base = [[StubTexture alloc] init];

  {
    NSMutableDictionary *expect = [@{
      @"base_texture_ref" : @(STUB_TEXTURE_REF),
      @"pixel_format" : @(MTLPixelFormatRGBA8Unorm),
    } mutableCopy];
    addCreationCase(cases, @"texture_view_format",
                    @"newTextureViewWithPixelFormat:baseTexture:allocator:", expect,
                    ^unsigned {
                      return ((unsigned (*)(id, SEL, unsigned long long, id, id))objc_msgSend)(
                          ser,
                          sel_getUid("newTextureViewWithPixelFormat:baseTexture:allocator:"),
                          MTLPixelFormatRGBA8Unorm, base, cap);
                    });
  }

  for (NSArray *tv in @[
         // format, textureType, levelBase, levelCount, sliceBase, sliceCount
         @[ @"texture_view_ranged", @(MTLPixelFormatRGBA8Unorm), @(MTLTextureType2D), @3, @2, @5,
            @4 ],
         @[ @"texture_view_ranged_alt", @(MTLPixelFormatR8Unorm), @(MTLTextureType2DArray), @1,
            @7, @2, @6 ],
       ]) {
    unsigned long long fmt = [tv[1] unsignedLongLongValue];
    unsigned long long type = [tv[2] unsignedLongLongValue];
    NSRange levels = NSMakeRange([tv[3] unsignedLongValue], [tv[4] unsignedLongValue]);
    NSRange slices = NSMakeRange([tv[5] unsignedLongValue], [tv[6] unsignedLongValue]);
    NSMutableDictionary *expect = [@{
      @"base_texture_ref" : @(STUB_TEXTURE_REF),
      @"pixel_format" : @(fmt),
      @"texture_type" : @(type),
      @"level_base" : @(levels.location),
      @"level_count" : @(levels.length),
      @"slice_base" : @(slices.location),
      @"slice_count" : @(slices.length),
    } mutableCopy];
    addCreationCase(
        cases, tv[0],
        @"newTextureViewWithPixelFormat:textureType:levels:slices:baseTexture:allocator:",
        expect, ^unsigned {
          return ((unsigned (*)(id, SEL, unsigned long long, unsigned long long, NSRange,
                                NSRange, id, id))objc_msgSend)(
              ser,
              sel_getUid("newTextureViewWithPixelFormat:textureType:levels:slices:"
                         "baseTexture:allocator:"),
              fmt, type, levels, slices, base, cap);
        });
  }

  // The swizzled form. Its four channels are given four *different* values, so
  // a view that reads them in the wrong order reports a permutation no other
  // case produced -- the failure a same-value swizzle could not show.
  for (NSArray *sw in @[
         @[ @"texture_view_swizzle", @(MTLTextureSwizzleRed), @(MTLTextureSwizzleGreen),
            @(MTLTextureSwizzleBlue), @(MTLTextureSwizzleAlpha) ],
         @[ @"texture_view_swizzle_permuted", @(MTLTextureSwizzleAlpha), @(MTLTextureSwizzleZero),
            @(MTLTextureSwizzleOne), @(MTLTextureSwizzleRed) ],
       ]) {
    MTLTextureSwizzleChannels ch;
    ch.red = (MTLTextureSwizzle)[sw[1] unsignedCharValue];
    ch.green = (MTLTextureSwizzle)[sw[2] unsignedCharValue];
    ch.blue = (MTLTextureSwizzle)[sw[3] unsignedCharValue];
    ch.alpha = (MTLTextureSwizzle)[sw[4] unsignedCharValue];
    NSRange levels = NSMakeRange(3, 2);
    NSRange slices = NSMakeRange(5, 4);
    NSMutableDictionary *expect = [@{
      @"base_texture_ref" : @(STUB_TEXTURE_REF),
      @"pixel_format" : @(MTLPixelFormatRGBA8Unorm),
      @"texture_type" : @(MTLTextureType2D),
      @"level_base" : @(levels.location),
      @"level_count" : @(levels.length),
      @"slice_base" : @(slices.location),
      @"slice_count" : @(slices.length),
      @"swizzle_red" : @((unsigned)ch.red),
      @"swizzle_green" : @((unsigned)ch.green),
      @"swizzle_blue" : @((unsigned)ch.blue),
      @"swizzle_alpha" : @((unsigned)ch.alpha),
    } mutableCopy];
    addCreationCase(cases, sw[0],
                    @"newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:"
                    @"baseTexture:allocator:",
                    expect, ^unsigned {
                      return ((unsigned (*)(id, SEL, unsigned long long, unsigned long long,
                                            NSRange, NSRange, MTLTextureSwizzleChannels, id,
                                            id))objc_msgSend)(
                          ser,
                          sel_getUid("newTextureViewWithPixelFormat:textureType:levels:slices:"
                                     "swizzle:baseTexture:allocator:"),
                          MTLPixelFormatRGBA8Unorm, MTLTextureType2D, levels, slices, ch, base,
                          cap);
                    });
  }

  // A buffer-backed texture. `reims-vgpu` calls this opcode 9 and declines it
  // on every rail, so this fixture is what prices implementing it.
  {
    id buf = [[StubBuffer alloc] init];
    MTLTextureDescriptor *td = baselineTexture();
    NSMutableDictionary *expect = [expectFromTextureDescriptor(td) mutableCopy];
    expect[@"buffer_ref"] = @(STUB_BUFFER_REF);
    expect[@"offset"] = @0x2200;
    expect[@"bytes_per_row"] = @0x4400;
    addCreationCase(cases, @"buffer_texture",
                    @"newTextureWithBuffer:descriptor:offset:bytesPerRow:allocator:", expect,
                    ^unsigned {
                      return ((unsigned (*)(id, SEL, id, id, unsigned long long,
                                            unsigned long long, id))objc_msgSend)(
                          ser,
                          sel_getUid("newTextureWithBuffer:descriptor:offset:bytesPerRow:"
                                     "allocator:"),
                          buf, td, 0x2200ull, 0x4400ull, cap);
                    });
  }

  // An IOSurface-backed texture in two planes, proving that plane remains a
  // field after the descriptor rather than aliasing descriptor state.
  for (NSArray *ios in @[ @[ @"iosurface_texture_plane0", @0 ],
                          @[ @"iosurface_texture_plane1", @1 ] ]) {
    MTLTextureDescriptor *td = baselineTexture();
    unsigned long long plane = [ios[1] unsignedLongLongValue];
    NSMutableDictionary *expect = [expectFromTextureDescriptor(td) mutableCopy];
    expect[@"plane"] = @(plane);
    addCreationCase(cases, ios[0], @"newIOSurfaceTextureWithDescriptor:plane:allocator:",
                    expect, ^unsigned {
                      return ((unsigned (*)(id, SEL, id, unsigned long long, id))objc_msgSend)(
                          ser, sel_getUid("newIOSurfaceTextureWithDescriptor:plane:allocator:"),
                          td, plane, cap);
                    });
  }

  // A texture placed inside a heap. The heap is a stub answering `heapRef`,
  // because a real `MTLHeap` has no paravirt ref for the serializer to write.
  id heap = [[StubHeap alloc] init];
  for (NSArray *hc in @[
         @[ @"heap_texture_baseline", @0x1234ab0ull, @1 ],
         @[ @"heap_texture_no_offset", @0x1234ab0ull, @0 ],
         @[ @"heap_texture_offset_alt", @0x777000ull, @1 ],
       ]) {
    MTLTextureDescriptor *td = baselineTexture();
    unsigned long long off = [hc[1] unsignedLongLongValue];
    char useOffset = (char)[hc[2] intValue];
    NSMutableDictionary *expect = [expectFromTextureDescriptor(td) mutableCopy];
    expect[@"heap_ref"] = @(STUB_HEAP_REF);
    expect[@"offset"] = @(off);
    expect[@"use_offset"] = @(useOffset ? 1 : 0);
    addCreationCase(cases, hc[0],
                    @"newTextureWithDescriptor:heap:offset:useOffset:allocator:", expect,
                    ^unsigned {
                      return ((unsigned (*)(id, SEL, id, id, unsigned long long, char,
                                            id))objc_msgSend)(
                          ser,
                          sel_getUid(
                              "newTextureWithDescriptor:heap:offset:useOffset:allocator:"),
                          td, heap, off, useOffset, cap);
                    });
  }

  // The three backed creations again under `-setSupportsTextureDescriptor2:`,
  // where each moves to its own opcode carrying the 40-byte descriptor.
  //
  // The body is already derived from `texture_swizzled`; what these pin is each
  // record's **prefix**, which is the half a shared body cannot settle. A buffer
  // ref with an offset and a bytes-per-row, a heap ref with an offset and a
  // `useOffset` bit, a plane index -- all three sit ahead of the descriptor in
  // the narrow form, and nothing says they still do in the wide one.
  //
  // Note the flag: these three answer to `TextureDescriptor2` and the plain
  // `newTextureWithDescriptor:allocator:` answers to `SwizzledTextures`. One
  // family, two capabilities, which is what made a negative result about the
  // plain form read as a negative result about all four.
  withCapability(ser, @"TextureDescriptor2", ^{
    id buf2 = [[StubBuffer alloc] init];
    MTLTextureDescriptor *td = baselineTexture();
    NSMutableDictionary *expect = [expectFromTextureDescriptor(td) mutableCopy];
    expect[@"buffer_ref"] = @(STUB_BUFFER_REF);
    expect[@"offset"] = @0x2200;
    expect[@"bytes_per_row"] = @0x4400;
    addCreationCase(cases, @"buffer_texture_wide",
                    @"newTextureWithBuffer:descriptor:offset:bytesPerRow:allocator:", expect,
                    ^unsigned {
                      return ((unsigned (*)(id, SEL, id, id, unsigned long long,
                                            unsigned long long, id))objc_msgSend)(
                          ser,
                          sel_getUid("newTextureWithBuffer:descriptor:offset:bytesPerRow:"
                                     "allocator:"),
                          buf2, td, 0x2200ull, 0x4400ull, cap);
                    });

    // Two planes again, for the same reason the narrow pair exists: a plane
    // index has to be carried somewhere and one observation cannot say where.
    for (NSArray *ios in @[ @[ @"iosurface_texture_wide_plane0", @0 ],
                            @[ @"iosurface_texture_wide_plane1", @1 ] ]) {
      MTLTextureDescriptor *itd = baselineTexture();
      unsigned long long plane = [ios[1] unsignedLongLongValue];
      NSMutableDictionary *iexpect = [expectFromTextureDescriptor(itd) mutableCopy];
      iexpect[@"plane"] = @(plane);
      addCreationCase(cases, ios[0], @"newIOSurfaceTextureWithDescriptor:plane:allocator:",
                      iexpect, ^unsigned {
                        return ((unsigned (*)(id, SEL, id, unsigned long long, id))objc_msgSend)(
                            ser, sel_getUid("newIOSurfaceTextureWithDescriptor:plane:allocator:"),
                            itd, plane, cap);
                      });
    }

    // Both `useOffset` values, because the narrow form's `useOffset` is one bit
    // inside a four-byte slot and reading it wide was a real device bug. One
    // observation of the wide form cannot say which bit it is.
    id heap2 = [[StubHeap alloc] init];
    for (NSArray *hc in @[
           @[ @"heap_texture_wide", @0x1234ab0ull, @1 ],
           @[ @"heap_texture_wide_no_offset", @0x1234ab0ull, @0 ],
         ]) {
      MTLTextureDescriptor *htd = baselineTexture();
      unsigned long long off = [hc[1] unsignedLongLongValue];
      char useOffset = (char)[hc[2] intValue];
      NSMutableDictionary *hexpect = [expectFromTextureDescriptor(htd) mutableCopy];
      hexpect[@"heap_ref"] = @(STUB_HEAP_REF);
      hexpect[@"offset"] = @(off);
      hexpect[@"use_offset"] = @(useOffset ? 1 : 0);
      addCreationCase(cases, hc[0],
                      @"newTextureWithDescriptor:heap:offset:useOffset:allocator:", hexpect,
                      ^unsigned {
                        return ((unsigned (*)(id, SEL, id, id, unsigned long long, char,
                                              id))objc_msgSend)(
                            ser,
                            sel_getUid(
                                "newTextureWithDescriptor:heap:offset:useOffset:allocator:"),
                            htd, heap2, off, useOffset, cap);
                      });
    }
  });

  // --- The remaining PGSerializer selectors --------------------------------
  //
  // Everything on this class that is not a pipeline-state or function creation.
  // Each is driven rather than reasoned about, including the ones whose type
  // encoding already says they cannot put their argument on the wire: the point
  // of `getTileDimensions:` is that "the argument does not go on the wire" and
  // "no record is emitted" are different claims.

  // Both `serializeTextureDescriptor*` take a struct **pointer** and return
  // void, so they fill the caller's buffer. Whether they also emit is what this
  // measures. The unsuffixed encoding is `ops::texture::TextureDescriptorBody`'s
  // layout field for field, which is an independent derivation of a struct
  // arrived at by perturbation; the suffixed one is a different 40-byte struct
  // that `the_second_texture_descriptor_layout_does_not_reach_the_wire` already
  // pins as never reaching the wire.
  {
    MTLTextureDescriptor *td = baselineTexture();
    static unsigned char descOut[64];
    addSerializerCase(cases, @"serializer_serialize_texture_descriptor",
                      @"serializeTextureDescriptor:textureDescriptor:", @{}, ^{
                        memset(descOut, 0, sizeof(descOut));
                        ((void (*)(id, SEL, void *, id))objc_msgSend)(
                            ser, sel_getUid("serializeTextureDescriptor:textureDescriptor:"),
                            descOut, td);
                      });
    addSerializerCase(cases, @"serializer_serialize_texture_descriptor2",
                      @"serializeTextureDescriptor2:textureDescriptor:", @{}, ^{
                        memset(descOut, 0, sizeof(descOut));
                        ((void (*)(id, SEL, void *, id))objc_msgSend)(
                            ser,
                            sel_getUid("serializeTextureDescriptor2:textureDescriptor:"),
                            descOut, td);
                      });

    // A pure size computation: it returns a `Q` and names no object, so there
    // is nothing for it to record. Driven to say so rather than to assume it.
    addSerializerCase(cases, @"serializer_data_size_for_region",
                      @"dataSizeForRegion:pixelFormat:bytesPerRow:bytesPerImage:", @{}, ^{
                        MTLRegion r = MTLRegionMake2D(0, 0, 4, 4);
                        ((unsigned long long (*)(id, SEL, MTLRegion, unsigned long,
                                                 unsigned long, unsigned long))objc_msgSend)(
                            ser,
                            sel_getUid("dataSizeForRegion:pixelFormat:bytesPerRow:"
                                       "bytesPerImage:"),
                            r, MTLPixelFormatBGRA8Unorm, 16, 64);
                      });

    // Heap sizing. Returns void and takes an allocator, so it may well be a
    // query answered through a readback record the way `getTileDimensions:` is.
    addSerializerCase(cases, @"serializer_heap_texture_size_and_align",
                      @"heapTextureSizeAndAlignWithDescriptor:allocator:", @{}, ^{
                        ((void (*)(id, SEL, id, id))objc_msgSend)(
                            ser,
                            sel_getUid("heapTextureSizeAndAlignWithDescriptor:allocator:"),
                            td, cap);
                      });

    // The two shared-texture creations. The first takes the new ref as an
    // *input* rather than allocating one, which is why it returns a `B` and not
    // an `I` — so its expectation is the ref this case passed.
    addSerializerCase(cases, @"serializer_new_shared_texture_with_descriptor",
                      @"newSharedTextureWithDescriptor:newTextureRef:allocator:",
                      @{@"object_ref" : @0x3131}, ^{
                        ((char (*)(id, SEL, id, unsigned, id))objc_msgSend)(
                            ser,
                            sel_getUid("newSharedTextureWithDescriptor:newTextureRef:"
                                       "allocator:"),
                            td, 0x3131, cap);
                      });
    addSerializerCase(cases, @"serializer_new_shared_texture_with_handle",
                      @"newSharedTextureWithHandle:allocator:", @{}, ^{
                        ((unsigned (*)(id, SEL, unsigned, id))objc_msgSend)(
                            ser, sel_getUid("newSharedTextureWithHandle:allocator:"), 0x3232,
                            cap);
                      });

    // `copyImageBytesFromSource:toDestination:…` is a host-side image copy: two
    // `char *` and a region. Buffers are far larger than the region so a wrong
    // stride reads inside them rather than off the end, and the harness catches
    // a fault anyway.
    static unsigned char imgSrc[4096];
    static unsigned char imgDst[4096];
    addSerializerCase(cases, @"serializer_copy_image_bytes",
                      @"copyImageBytesFromSource:toDestination:dataSize:region:"
                      @"bytesPerRow:bytesPerImage:mipmapLevel:slice:texture:",
                      @{}, ^{
                        MTLRegion r = MTLRegionMake2D(0, 0, 2, 2);
                        ((void (*)(id, SEL, char *, char *, unsigned long, MTLRegion,
                                   unsigned long, unsigned long, unsigned long,
                                   unsigned long, id))objc_msgSend)(
                            ser,
                            sel_getUid("copyImageBytesFromSource:toDestination:dataSize:"
                                       "region:bytesPerRow:bytesPerImage:mipmapLevel:"
                                       "slice:texture:"),
                            (char *)imgSrc, (char *)imgDst, sizeof(imgDst), r, 8, 16, 0, 0,
                            [[StubTexture alloc] init]);
                      });
  }

  // The indirect-command-buffer creation, one property per case.
  //
  // Its `layout:` argument is a `^{?=SSSSIIIIIIIIIII}` the type encoding lays
  // out for free: four `u16` then eleven `u32`, 52 bytes. Filled with distinct
  // values so a field that reaches the wire is recognisable, and a field that
  // does not is visibly absent.
  //
  // The descriptor is the half the encoding says nothing about, and one case
  // cannot name a byte of it: `MTLIndirectCommandBufferDescriptor` has fifteen
  // properties and the sixteen bytes between the ref and the layout hold some
  // subset of them. Every case below moves exactly one, off a baseline that
  // holds every other at its default, so a byte that changes is attributable.
  {
    // The BOOL properties this descriptor carries, in the order Metal declares
    // them, paired with the expectation key each is read back into. Six of them
    // arrived in macOS 26, so a host whose SDK or serializer predates them
    // simply produces no case for that flag rather than a case that measured
    // nothing -- which is why they are reached by selector rather than by
    // property syntax.
    struct IcbFlag {
      const char *selector;
      NSString *key;
    };
    static const struct IcbFlag icbFlags[] = {
        {"inheritPipelineState", @"inherit_pipeline_state"},
        {"inheritBuffers", @"inherit_buffers"},
        {"supportRayTracing", @"support_ray_tracing"},
        {"supportDynamicAttributeStride", @"support_dynamic_attribute_stride"},
        {"inheritDepthStencilState", @"inherit_depth_stencil_state"},
        {"inheritDepthBias", @"inherit_depth_bias"},
        {"inheritDepthClipMode", @"inherit_depth_clip_mode"},
        {"inheritCullMode", @"inherit_cull_mode"},
        {"inheritFrontFacingWinding", @"inherit_front_facing_winding"},
        {"inheritTriangleFillMode", @"inherit_triangle_fill_mode"},
        {"supportColorAttachmentMapping", @"support_color_attachment_mapping"},
    };
    const unsigned icbFlagCount = sizeof(icbFlags) / sizeof(icbFlags[0]);

    // One case: build a fresh descriptor, apply the perturbation, drive it, and
    // read every expectation back off the descriptor Metal kept rather than off
    // the value that was set -- Metal normalizes several of these. The ref is
    // the selector's own return value, which is what `addCreationCase` records.
    void (^icbCase)(NSString *, unsigned long, unsigned long, unsigned,
                    void (^)(MTLIndirectCommandBufferDescriptor *)) =
        ^(NSString *name, unsigned long commandCount, unsigned long options, unsigned seed,
          void (^configure)(MTLIndirectCommandBufferDescriptor *)) {
          __block struct __attribute__((packed)) {
            unsigned short s[4];
            unsigned int i[11];
          } layout;
          for (unsigned k = 0; k < 4; k++)
            layout.s[k] = (unsigned short)((seed << 8) + k);
          for (unsigned k = 0; k < 11; k++)
            layout.i[k] = ((seed | (seed << 8)) << 16) + k;

          MTLIndirectCommandBufferDescriptor *d =
              [[MTLIndirectCommandBufferDescriptor alloc] init];
          d.commandTypes = MTLIndirectCommandTypeDraw;
          d.maxVertexBufferBindCount = 4;
          d.maxFragmentBufferBindCount = 5;
          if (configure) configure(d);

          NSMutableDictionary *expect = [@{
            @"max_command_count" : @((unsigned long long)commandCount),
            @"options" : @((unsigned long long)options),
            @"command_types" : @((unsigned long long)d.commandTypes),
            @"max_vertex_buffer_bind_count" :
                @((unsigned long long)d.maxVertexBufferBindCount),
            @"max_fragment_buffer_bind_count" :
                @((unsigned long long)d.maxFragmentBufferBindCount),
            @"max_kernel_buffer_bind_count" :
                @((unsigned long long)d.maxKernelBufferBindCount),
            @"max_kernel_threadgroup_memory_bind_count" :
                @((unsigned long long)d.maxKernelThreadgroupMemoryBindCount),
            @"max_object_buffer_bind_count" :
                @((unsigned long long)d.maxObjectBufferBindCount),
            @"max_mesh_buffer_bind_count" :
                @((unsigned long long)d.maxMeshBufferBindCount),
            @"max_object_threadgroup_memory_bind_count" :
                @((unsigned long long)d.maxObjectThreadgroupMemoryBindCount),
          } mutableCopy];
          for (unsigned f = 0; f < icbFlagCount; f++) {
            SEL get = sel_getUid(icbFlags[f].selector);
            if (![d respondsToSelector:get]) continue;
            expect[icbFlags[f].key] =
                @(((char (*)(id, SEL))objc_msgSend)(d, get) ? 1 : 0);
          }
          for (unsigned k = 0; k < 4; k++) {
            expect[[NSString stringWithFormat:@"layout_s%u", k]] = @(layout.s[k]);
          }
          for (unsigned k = 0; k < 11; k++) {
            expect[[NSString stringWithFormat:@"layout_i%u", k]] = @(layout.i[k]);
          }

          addCreationCase(cases, name,
                          @"newIndirectCommandBufferWithDescriptor:layout:maxCommandCount:"
                          @"options:allocator:",
                          expect, ^unsigned {
                            return ((unsigned (*)(id, SEL, id, void *, unsigned long,
                                                  unsigned long, id))objc_msgSend)(
                                ser,
                                sel_getUid("newIndirectCommandBufferWithDescriptor:layout:"
                                           "maxCommandCount:options:allocator:"),
                                d, &layout, commandCount, options, cap);
                          });
        };

    icbCase(@"serializer_new_indirect_command_buffer", 0x33, 0, 0x11, nil);
    icbCase(@"serializer_new_indirect_command_buffer_command_types", 0x33, 0, 0x11,
            ^(MTLIndirectCommandBufferDescriptor *d) {
              // Four bits, none of them the baseline's, so a record that
              // carried the option set verbatim is separable from one that
              // carried a count of set bits or a remapped ordinal.
              d.commandTypes = MTLIndirectCommandTypeDrawIndexed |
                               MTLIndirectCommandTypeDrawPatches |
                               MTLIndirectCommandTypeConcurrentDispatch |
                               MTLIndirectCommandTypeConcurrentDispatchThreads;
            });
    icbCase(@"serializer_new_indirect_command_buffer_bind_counts", 0x33, 0, 0x11,
            ^(MTLIndirectCommandBufferDescriptor *d) {
              // Distinct counts, all different from each other and from the
              // baseline's, so a pair that swapped is visible on sight.
              d.maxVertexBufferBindCount = 0x11;
              d.maxFragmentBufferBindCount = 0x22;
              d.maxKernelBufferBindCount = 0x0e;
            });
    icbCase(@"serializer_new_indirect_command_buffer_stage_bind_counts", 0x33, 0, 0x11,
            ^(MTLIndirectCommandBufferDescriptor *d) {
              // The four counts the case above leaves at their defaults, and
              // two of those defaults are non-zero -- which is why they need
              // their own case rather than being read off the baseline.
              d.maxKernelThreadgroupMemoryBindCount = 0x0a;
              d.maxObjectBufferBindCount = 0x0b;
              d.maxMeshBufferBindCount = 0x0c;
              d.maxObjectThreadgroupMemoryBindCount = 0x0d;
            });

    // One case per BOOL, each inverted from the value the baseline read, so the
    // bit that moves names the property. Driving them as a block would clear
    // eleven bits at once and attribute none of them.
    for (unsigned f = 0; f < icbFlagCount; f++) {
      const char *setterName = icbFlags[f].selector;
      NSString *setter = [NSString
          stringWithFormat:@"set%c%s:", toupper(setterName[0]), setterName + 1];
      SEL set = sel_getUid(setter.UTF8String);
      SEL get = sel_getUid(setterName);
      MTLIndirectCommandBufferDescriptor *probe =
          [[MTLIndirectCommandBufferDescriptor alloc] init];
      if (![probe respondsToSelector:set] || ![probe respondsToSelector:get]) {
        fprintf(stderr, "note: MTLIndirectCommandBufferDescriptor has no %s\n", setterName);
        continue;
      }
      char baseline = ((char (*)(id, SEL))objc_msgSend)(probe, get);
      icbCase([NSString stringWithFormat:@"serializer_new_indirect_command_buffer_%s",
                                         icbFlags[f].key.UTF8String],
              0x33, 0, 0x11, ^(MTLIndirectCommandBufferDescriptor *d) {
                ((void (*)(id, SEL, char))objc_msgSend)(d, set, baseline ? 0 : 1);
              });
    }

    // The same record with the two *call* arguments moved instead of the
    // descriptor, and a second layout seed. Without this case the count and the
    // options are two words that happen to hold `0x33` and `0`, and nothing
    // says which is which -- nor that the layout is copied rather than a
    // constant the serializer keeps.
    icbCase(@"serializer_new_indirect_command_buffer_count_options", 0x5566,
            MTLResourceStorageModePrivate, 0x44, nil);
  }

  // The two rasterization-rate-map selectors, under the flag their family is
  // named for. `resetRasterizationRateMapWithDescriptor:existingID:allocator:`
  // takes the ref as an input, so its expectation is the ref passed in.
  withCapability(ser, @"RasterizationRateMap", ^{
    // Everything a rate map carries is a *count* of something else -- layers,
    // and per layer a horizontal and a vertical sample count -- and each of the
    // three sizes the sample arrays below.  So one descriptor cannot separate
    // them: a screen of 64x64 with one layer at 2x2 puts 64, 1 and 2 on the
    // wire and any of the three could be any of the fields. Each case moves one
    // of screen size, layer count, sample count and sample *values*, and the
    // asymmetric sizes are what keep width from being confused with height.
    //
    // The quality floats are set explicitly rather than left at whatever
    // `initWithSampleCount:` leaves, because at the default they read 0.0 --
    // and a field that is zero in every fixture cannot be told from one the
    // serializer never writes.
    NSDictionary *(^rateExpect)(MTLRasterizationRateMapDescriptor *) =
        ^(MTLRasterizationRateMapDescriptor *d) {
          NSMutableDictionary *e = [NSMutableDictionary dictionary];
          e[@"screen_width"] = @((unsigned long long)d.screenSize.width);
          e[@"screen_height"] = @((unsigned long long)d.screenSize.height);
          e[@"layer_count"] = @((unsigned long long)d.layerCount);
          for (NSUInteger i = 0; i < d.layerCount; i++) {
            MTLRasterizationRateLayerDescriptor *l = [d layerAtIndex:i];
            e[[NSString stringWithFormat:@"layer%lu_sample_width", (unsigned long)i]] =
                @((unsigned long long)l.sampleCount.width);
            e[[NSString stringWithFormat:@"layer%lu_sample_height", (unsigned long)i]] =
                @((unsigned long long)l.sampleCount.height);
            for (NSUInteger q = 0; q < l.sampleCount.width; q++) {
              e[[NSString stringWithFormat:@"layer%lu_horizontal%lu", (unsigned long)i,
                                           (unsigned long)q]] =
                  @(l.horizontalSampleStorage[q]);
            }
            for (NSUInteger q = 0; q < l.sampleCount.height; q++) {
              e[[NSString stringWithFormat:@"layer%lu_vertical%lu", (unsigned long)i,
                                           (unsigned long)q]] = @(l.verticalSampleStorage[q]);
            }
          }
          return (NSDictionary *)e;
        };

    // Distinct per position and per axis, so a horizontal quality read as a
    // vertical one is visible rather than plausible.
    static const float hq[4] = {0.5f, 0.25f, 0.125f, 0.0625f};
    static const float vq[4] = {0.75f, 0.375f, 0.1875f, 0.09375f};

    MTLRasterizationRateLayerDescriptor *(^layerOf)(unsigned, unsigned) =
        ^(unsigned w, unsigned h) {
          return [[MTLRasterizationRateLayerDescriptor alloc]
              initWithSampleCount:MTLSizeMake(w, h, 0)
                       horizontal:hq
                         vertical:vq];
        };

    void (^rateCase)(NSString *, MTLRasterizationRateMapDescriptor *) =
        ^(NSString *name, MTLRasterizationRateMapDescriptor *d) {
          addSerializerCase(cases, name, @"newRasterizationRateMapWithDescriptor:allocator:",
                            rateExpect(d), ^{
                              ((unsigned (*)(id, SEL, id, id))objc_msgSend)(
                                  ser,
                                  sel_getUid("newRasterizationRateMapWithDescriptor:"
                                             "allocator:"),
                                  d, cap);
                            });
        };

    MTLRasterizationRateMapDescriptor *rrd = [MTLRasterizationRateMapDescriptor
        rasterizationRateMapDescriptorWithScreenSize:MTLSizeMake(64, 64, 0)
                                               layer:layerOf(2, 2)];
    rateCase(@"serializer_new_rasterization_rate_map", rrd);

    rateCase(@"serializer_new_rasterization_rate_map_screen",
             [MTLRasterizationRateMapDescriptor
                 rasterizationRateMapDescriptorWithScreenSize:MTLSizeMake(0x140, 0xc8, 0)
                                                        layer:layerOf(2, 2)]);

    rateCase(@"serializer_new_rasterization_rate_map_samples",
             [MTLRasterizationRateMapDescriptor
                 rasterizationRateMapDescriptorWithScreenSize:MTLSizeMake(64, 64, 0)
                                                        layer:layerOf(4, 3)]);

    {
      // Two layers with different sample counts. This is the case that says
      // whether the record is variable-length and whether the per-layer blocks
      // are packed one after another or written at a fixed stride.
      MTLRasterizationRateLayerDescriptor *layers[2] = {layerOf(2, 2), layerOf(3, 4)};
      rateCase(@"serializer_new_rasterization_rate_map_two_layers",
               [MTLRasterizationRateMapDescriptor
                   rasterizationRateMapDescriptorWithScreenSize:MTLSizeMake(64, 64, 0)
                                                     layerCount:2
                                                         layers:layers]);
    }

    {
      // Every capture of this record declares sixteen more bytes than the
      // serializer writes, at every layer count and every sample count. `label`
      // is the one property of the descriptor that has no home in the written
      // part, so this case asks whether the tail is where it goes -- the answer
      // belongs in the view's doc either way, because a decoder that reads
      // those sixteen bytes is reading whatever the guest's ring last held.
      MTLRasterizationRateMapDescriptor *labelled = [MTLRasterizationRateMapDescriptor
          rasterizationRateMapDescriptorWithScreenSize:MTLSizeMake(64, 64, 0)
                                                 layer:layerOf(2, 2)];
      labelled.label = @"reimsRateMapLabel";
      rateCase(@"serializer_new_rasterization_rate_map_labelled", labelled);
    }

    // The reset form takes the ref as an input instead of allocating one, and
    // is otherwise the same record -- which is the claim the manifest makes by
    // giving both selectors one opcode, so it is driven off the same descriptor
    // as the baseline above.
    NSMutableDictionary *resetExpect = [rateExpect(rrd) mutableCopy];
    resetExpect[@"object_ref"] = @(STUB_RATE_MAP_REF);
    addSerializerCase(cases, @"serializer_reset_rasterization_rate_map",
                      @"resetRasterizationRateMapWithDescriptor:existingID:allocator:",
                      resetExpect, ^{
                        ((void (*)(id, SEL, id, unsigned, id))objc_msgSend)(
                            ser,
                            sel_getUid("resetRasterizationRateMapWithDescriptor:existingID:"
                                       "allocator:"),
                            rrd, STUB_RATE_MAP_REF, cap);
                      });
  });

  return cases;
}

// --- The serializer's object lifecycle -------------------------------------
//
// Three families of eleven, one per object kind, and the whole point of driving
// them is that their names do not say which of the three writes a record:
//
//   -newXRef            takes nothing and answers a fresh ref
//   -releaseXRef:       takes a ref
//   -deleteXRef:allocator:  takes a ref and an allocator
//
// A reader can guess that the allocator argument means "this one serializes",
// and the guess would be right — but a guess is not a derivation, and the two
// families that emit nothing are as much a finding as the one that does. The
// `silent` list is what records them, and
// `every_excluded_row_that_claims_silence_still_gets_it` re-checks it every
// capture, so a future build that starts emitting from `-releaseXRef:` fails
// the suite rather than losing the guest's release.
//
// The refs come from the serializer's own `-newXRef`, driven first, so a delete
// names an object this serializer really allocated rather than a number chosen
// here.
static NSArray *lifecycleCases(id ser, id cap) {
  NSMutableArray *cases = [NSMutableArray array];

  // Object kinds, in the spelling all three families share.
  NSArray *kinds = @[
    @"Buffer", @"ComputePipelineState", @"DepthStencilState", @"Fence", @"Function",
    @"Heap", @"IndirectCommandBuffer", @"RasterizationRateMap", @"RenderPipelineState",
    @"SamplerState", @"Texture"
  ];

  for (NSString *kind in kinds) {
    NSString *newSel = [NSString stringWithFormat:@"new%@Ref", kind];
    NSString *releaseSel = [NSString stringWithFormat:@"release%@Ref:", kind];
    NSString *deleteSel = [NSString stringWithFormat:@"delete%@Ref:allocator:", kind];
    NSString *slug = [kind lowercaseString];

    // `-newXRef` answers the ref the other two are then given, so the ref in a
    // delete record is one this serializer allocated rather than one invented
    // here. Captured as an `__block` so the expectation and the argument are the
    // same value by construction.
    __block unsigned ref = 0;
    addSerializerCase(cases, [NSString stringWithFormat:@"lifecycle_new_%@_ref", slug],
                      newSel, @{}, ^{
                        ref = ((unsigned (*)(id, SEL))objc_msgSend)(
                            ser, sel_getUid(newSel.UTF8String));
                      });

    addSerializerCase(cases, [NSString stringWithFormat:@"lifecycle_delete_%@_ref", slug],
                      deleteSel, @{@"object_ref" : @(ref)}, ^{
                        ((void (*)(id, SEL, unsigned, id))objc_msgSend)(
                            ser, sel_getUid(deleteSel.UTF8String), ref, cap);
                      });

    addSerializerCase(cases, [NSString stringWithFormat:@"lifecycle_release_%@_ref", slug],
                      releaseSel, @{@"object_ref" : @(ref)}, ^{
                        ((void (*)(id, SEL, unsigned))objc_msgSend)(
                            ser, sel_getUid(releaseSel.UTF8String), ref);
                      });
  }

  return cases;
}

// --- The capability flags --------------------------------------------------
//
// Sixteen `-setSupportsX:` / `-supportsX` pairs and two read-only flags. They
// look like accessors and almost certainly are, but "almost certainly" is what
// a manifest row must not rest on, and the cost of being wrong is a wire record
// nobody decodes.
//
// The setters are driven with the flag **inverted and then restored**, which is
// what makes this safe to run in the middle of a capture: a serializer left in
// a different capability state would change what every later case emits, and
// the failure would look like a layout error somewhere else entirely. Inverting
// rather than re-writing the current value is what makes it a perturbation
// instead of a no-op — a setter that serialized only on a real change would
// otherwise be recorded as silent.
/// The sixteen flags carrying both a `-supportsX` and a `-setSupportsX:`.
///
/// One home, because two readers need it: `capabilityCases` drives each pair,
/// and `recordCapabilityDefaults` publishes what each reads untouched. A second
/// copy would let the two disagree about which flags exist, and the defaults map
/// is what a reader uses to decide whether a `silent` row is trustworthy.
static NSArray *capabilityFlagNames(void) {
  return @[
    @"BlitEncoderSPI", @"CommandBufferJump", @"ComputePassDescriptorDispatchType",
    @"DefaultRasterSampleCount", @"DispatchThreadsIndirect", @"DynamicAttributeStride",
    @"ImageBlocks", @"InfoIndirect", @"InsertCompressedTextureReinterpretationFlush",
    @"ProgrammableSamplePositions", @"ProtectionOptionsEnvelope", @"RasterizationRateMap",
    @"SwizzledTextures", @"TextureDescriptor2", @"TileShaders", @"VertexAmplification"
  ];
}

static NSArray *capabilityCases(id ser) {
  NSMutableArray *cases = [NSMutableArray array];

  // The flags with both a getter and a setter.
  NSArray *pairs = capabilityFlagNames();
  // Read-only: no `-setSupportsX:` ships for these.
  NSArray *readOnly = @[ @"CorrectBaseVertex", @"OpenGL", @"SharedTextures" ];

  for (NSString *flag in pairs) {
    NSString *getter = [NSString stringWithFormat:@"supports%@", flag];
    NSString *setter = [NSString stringWithFormat:@"setSupports%@:", flag];
    NSString *slug = [flag lowercaseString];
    SEL getSel = sel_getUid(getter.UTF8String);
    SEL setSel = sel_getUid(setter.UTF8String);

    addSerializerCase(cases, [NSString stringWithFormat:@"capability_get_%@", slug],
                      getter, @{}, ^{
                        (void)((char (*)(id, SEL))objc_msgSend)(ser, getSel);
                      });

    char was = ((char (*)(id, SEL))objc_msgSend)(ser, getSel);
    addSerializerCase(cases, [NSString stringWithFormat:@"capability_set_%@", slug],
                      setter, @{}, ^{
                        ((void (*)(id, SEL, char))objc_msgSend)(ser, setSel, (char)!was);
                      });
    // Restore before the next case, whatever the setter did.
    ((void (*)(id, SEL, char))objc_msgSend)(ser, setSel, was);
  }

  for (NSString *flag in readOnly) {
    NSString *getter = [NSString stringWithFormat:@"supports%@", flag];
    addSerializerCase(cases, [NSString stringWithFormat:@"capability_get_%@",
                                                       [flag lowercaseString]],
                      getter, @{}, ^{
                        (void)((char (*)(id, SEL))objc_msgSend)(
                            ser, sel_getUid(getter.UTF8String));
                      });
  }

  return cases;
}

// Every selector Apple ships on the serializer classes, so the Rust manifest's
// notion of "all of them" comes from the runtime rather than from a list
// someone maintains.
//
// Each selector carries its Objective-C type encoding, which is the first of
// the three derivation sources AGENTS.md allows: it fixes every argument's
// width and order before a single byte is captured. `v@:QQQ` is three
// 64-bit arguments and `v@:fff` is three 32-bit floats, and the difference
// between those two is a whole class of wrong layout that no amount of staring
// at a hex dump settles.
static NSDictionary *inventory(void) {
  NSArray *classes = @[
    @"PGSerializer", @"PGSerializerRenderCommandEncoder",
    @"PGSerializerComputeCommandEncoder", @"PGSerializerBlitCommandEncoder",
    @"PGSerializerInfoCommandEncoder"
  ];
  NSMutableArray *out = [NSMutableArray array];
  for (NSString *cn in classes) {
    Class c = objc_getClass(cn.UTF8String);
    unsigned int n = 0;
    Method *ms = c ? class_copyMethodList(c, &n) : NULL;
    NSMutableArray *sels = [NSMutableArray arrayWithCapacity:n];
    for (unsigned i = 0; i < n; i++) {
      const char *enc = method_getTypeEncoding(ms[i]);
      [sels addObject:@{
        @"selector" : @(sel_getName(method_getName(ms[i]))),
        @"type_encoding" : enc ? @(enc) : [NSNull null],
      }];
    }
    free(ms);
    [sels sortUsingComparator:^NSComparisonResult(NSDictionary *a, NSDictionary *b) {
      return [a[@"selector"] compare:b[@"selector"]];
    }];
    [out addObject:@{@"class" : cn, @"instance_methods" : @(n), @"selectors" : sels}];
  }
  return @{@"schema" : @2, @"classes" : out};
}

static NSDictionary *provenance(void) {
  NSDictionary *plist = [NSDictionary dictionaryWithContentsOfFile:kBundlePlist];
  return @{
    @"bundle_version" : plist[@"CFBundleVersion"] ?: @"(unknown)",
    @"bundle_sha256" : sha256OfFile(kBundleBin),
    @"os_version" : [[NSProcessInfo processInfo] operatingSystemVersionString],
  };
}

static int writeJSON(NSDictionary *root, const char *path) {
  NSError *err = nil;
  NSData *json = [NSJSONSerialization
      dataWithJSONObject:root
                 options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                   error:&err];
  if (!json) {
    fprintf(stderr, "JSON encode failed: %s\n", err.localizedDescription.UTF8String);
    return 1;
  }
  if (![json writeToFile:@(path) atomically:YES]) {
    fprintf(stderr, "write failed: %s\n", path);
    return 1;
  }
  printf("wrote %s (%lu bytes)\n", path, (unsigned long)json.length);
  return 0;
}

/// Drive every case once, under one arena fill.
///
/// A fresh `PGSerializer` and a rewound `gNextRef` per pass, so the second pass
/// is a repeat rather than a continuation: the refs a case is handed, and the
/// serializer state a case starts from, must be the ones its twin saw. Anything
/// that differs between the passes for a reason other than the fill would be
/// read as an unwritten byte, which is the one mistake this instrument cannot
/// afford to make quietly.
/// What every capability flag reads on a serializer nobody has touched.
///
/// Recorded because a `silent` outcome is only true for the capability state it
/// was captured in, and several families emit nothing at all with their flag
/// off. `-supportsDynamicAttributeStride` reads false here, which is why four
/// real vertex binds looked like selectors Apple never emits until they were
/// driven through `withCapability`. Publishing the defaults makes every
/// `EMITS_NO_OPERATION` row auditable after the fact: a `false` in this map is
/// the list of families worth re-driving.
static NSMutableDictionary *gCapabilityDefaults = nil;

static void recordCapabilityDefaults(id ser) {
  if (gCapabilityDefaults) return; // the second poison pass sees the same serializer state
  gCapabilityDefaults = [NSMutableDictionary dictionary];
  for (NSString *flag in capabilityFlagNames()) {
    SEL getSel = sel_getUid([NSString stringWithFormat:@"supports%@", flag].UTF8String);
    if (![ser respondsToSelector:getSel]) continue;
    char v = ((char (*)(id, SEL))objc_msgSend)(ser, getSel);
    gCapabilityDefaults[flag] = v ? @YES : @NO;
  }
}

static NSArray *onePass(id<MTLDevice> dev, id cap, unsigned char poison, BOOL record) {
  gPoison = poison;
  gRecordOutcomes = record;
  gNextRef = 1;
  id ser = ((id (*)(id, SEL, id, id))objc_msgSend)(
      [objc_getClass("PGSerializer") alloc],
      sel_getUid("initWithDevice:objectRefAllocator:"), dev,
      [[RefAllocator alloc] init]);
  if (!ser) {
    fprintf(stderr, "PGSerializer init returned nil\n");
    return nil;
  }

  recordCapabilityDefaults(ser);

  // The sweep pass. Every flag on before the first case runs, so a selector
  // that emits only under some capability emits here -- whichever one it is.
  //
  // This exists because "which flag is this family gated on?" is a question
  // nobody can answer by reading, and getting it wrong writes a false
  // `EMITS_NO_OPERATION` row about Apple. Three families were found by guessing
  // the flag correctly; this pass finds the fourth without guessing.
  if (gForceAllCapabilities) {
    for (NSString *flag in capabilityFlagNames()) {
      NSString *setter = [NSString stringWithFormat:@"setSupports%@:", flag];
      ((void (*)(id, SEL, char))objc_msgSend)(ser, sel_getUid(setter.UTF8String), (char)1);
    }
  } else if (gForceOneCapability) {
    // An attribution pass. Exactly one flag on, so the selectors that stop
    // being silent are the ones this flag alone unlocks -- which is the
    // argument `withCapability` needs, measured rather than tried.
    NSString *setter = [NSString stringWithFormat:@"setSupports%@:", gForceOneCapability];
    ((void (*)(id, SEL, char))objc_msgSend)(ser, sel_getUid(setter.UTF8String), (char)1);
  }

  NSMutableArray *cases = [NSMutableArray array];
  [cases addObjectsFromArray:textureCases(ser, cap)];
  [cases addObjectsFromArray:creationCases(ser, cap)];
  [cases addObjectsFromArray:encoderCases(ser)];
  [cases addObjectsFromArray:blitCases(ser)];
  [cases addObjectsFromArray:computeCases(ser)];
  [cases addObjectsFromArray:infoCases(ser)];
  // Last, and in this order. `lifecycleCases` allocates object refs from the
  // serializer's own allocator, and `capabilityCases` inverts a flag before
  // restoring it -- neither is something to run ahead of a case whose record
  // is being read field by field.
  [cases addObjectsFromArray:lifecycleCases(ser, cap)];
  [cases addObjectsFromArray:capabilityCases(ser)];
  return cases;
}

/// Texture-only pass for comparing descriptor layouts across older serializer
/// versions whose selector surface cannot run the full modern inventory.
static NSArray *texturePass(id<MTLDevice> dev, id cap, unsigned char poison, BOOL record) {
  gPoison = poison;
  gRecordOutcomes = record;
  gNextRef = 1;
  id ser = ((id (*)(id, SEL, id, id))objc_msgSend)(
      [objc_getClass("PGSerializer") alloc],
      sel_getUid("initWithDevice:objectRefAllocator:"), dev,
      [[RefAllocator alloc] init]);
  if (!ser) return nil;
  recordCapabilityDefaults(ser);
  return textureCases(ser, cap);
}

/// Object-creation-only pass for older serializer versions that cannot run the
/// full modern encoder inventory.
static NSArray *creationPass(id<MTLDevice> dev, id cap, unsigned char poison, BOOL record) {
  gPoison = poison;
  gRecordOutcomes = record;
  gNextRef = 1;
  id ser = ((id (*)(id, SEL, id, id))objc_msgSend)(
      [objc_getClass("PGSerializer") alloc],
      sel_getUid("initWithDevice:objectRefAllocator:"), dev,
      [[RefAllocator alloc] init]);
  if (!ser) return nil;
  recordCapabilityDefaults(ser);
  return creationCases(ser, cap);
}

/// Blit-only pass, including segment framing, for older serializer versions
/// whose later encoder families cannot run the full inventory.
static NSArray *blitPass(id<MTLDevice> dev, id cap, unsigned char poison, BOOL record) {
  gPoison = poison;
  gRecordOutcomes = record;
  gNextRef = 1;
  id ser = ((id (*)(id, SEL, id, id))objc_msgSend)(
      [objc_getClass("PGSerializer") alloc],
      sel_getUid("initWithDevice:objectRefAllocator:"), dev,
      [[RefAllocator alloc] init]);
  if (!ser) return nil;
  recordCapabilityDefaults(ser);
  return blitCases(ser);
}

/// Attach each case's per-bit written mask, derived from its two passes.
///
/// `mask = ~(a ^ b)`: a bit the serializer wrote holds the same value under
/// both fills, and a bit it left alone holds the fill, which the two fills
/// disagree on by construction. So a set bit means written and a clear bit
/// means the guest's stale ring — which is what a decoder must not read.
///
/// The mask is per *bit* rather than per byte on purpose. Several records set a
/// bitfield inside a byte the serializer otherwise leaves alone, and a
/// byte-granular answer would have to call the whole byte one thing or the
/// other; both answers are wrong and one of them invites a decoder to read
/// noise.
static NSArray *mergeWrittenMasks(NSArray *first, NSArray *second,
                                  NSMutableArray *unmasked) {
  NSMutableDictionary *twin = [NSMutableDictionary dictionary];
  for (NSDictionary *c in second) twin[c[@"name"]] = c;

  NSMutableArray *out = [NSMutableArray arrayWithCapacity:first.count];
  for (NSDictionary *c in first) {
    NSDictionary *t = twin[c[@"name"]];
    NSString *why = nil;
    if (!t) {
      why = @"the second pass produced no case of this name";
    } else if (![t[@"selector"] isEqual:c[@"selector"]]) {
      why = @"the second pass recorded a different selector under this name";
    } else if (![t[@"allocated_len"] isEqual:c[@"allocated_len"]]) {
      why = @"the two passes allocated different lengths";
    }
    if (why) {
      [unmasked addObject:@{
        @"name" : c[@"name"],
        @"class" : c[@"class"],
        @"selector" : c[@"selector"],
        @"reason" : why,
      }];
      [out addObject:c];
      continue;
    }

    NSString *ha = c[@"buffer"], *hb = t[@"buffer"];
    NSMutableString *mask = [NSMutableString stringWithCapacity:ha.length];
    for (NSUInteger i = 0; i + 1 < ha.length; i += 2) {
      unsigned a = 0, b = 0;
      sscanf([ha substringWithRange:NSMakeRange(i, 2)].UTF8String, "%2x", &a);
      sscanf([hb substringWithRange:NSMakeRange(i, 2)].UTF8String, "%2x", &b);
      [mask appendFormat:@"%02x", (unsigned char)~(a ^ b)];
    }
    NSMutableDictionary *m = [c mutableCopy];
    m[@"written_mask"] = mask;
    [out addObject:m];
  }
  return out;
}

/// What a pass concluded about each case it did not turn into a record.
///
/// Keyed by case name so `diffAgainstDefault` can say *why* a case is missing
/// from one side. Without this an `absent` entry cannot be told apart from an
/// `extra` one read backwards: a selector that went silent, one that asserted,
/// and one that emitted two records where the case claimed one are three
/// different findings, and only the third is about record count.
static NSDictionary *outcomeIndex(NSArray *silent, NSArray *unsupported,
                                  NSArray *crashed, NSArray *multi) {
  NSMutableDictionary *m = [NSMutableDictionary dictionary];
  for (NSDictionary *e in silent) m[e[@"name"]] = @"silent";
  for (NSDictionary *e in unsupported) m[e[@"name"]] = @"unsupported";
  for (NSDictionary *e in crashed) m[e[@"name"]] = @"crashed";
  for (NSDictionary *e in multi)
    m[e[@"name"]] = [NSString stringWithFormat:@"multi (%@ operations, the case claimed %@)",
                                               e[@"operations"], e[@"expected"]];
  return m;
}

/// Every way one pass's record for a case can differ from the default pass's.
///
/// Separate kinds because they are separate claims. `bytes` is the one this
/// exists to find — the same selector, driven the same way, writing a different
/// record because a flag is on. `length` is the serializer allocating a
/// different extent, which is a stronger version of the same thing. `absent`
/// and `extra` are the suppress and unlock directions at *case* granularity.
///
/// Those last two are strictly stronger than the `capability_attribution` lists
/// beside them, and the difference is not cosmetic. Attribution diffs the two
/// passes' **silent** lists, so it sees only the selectors that returned without
/// writing. A selector that *asserts* at the default state is on `unsupported`
/// instead and is invisible to it, and a selector that emits a second record
/// under the flag lands on `multi` and is invisible to it too. Both happen.
/// This compares the records themselves, so it sees every case either pass
/// produced regardless of which outcome list the other pass filed it under —
/// which is why each entry carries the other side's outcome by name.
static void diffAgainstDefault(NSDictionary *base, NSArray *forced, NSString *flag,
                               NSDictionary *baseOutcomes, NSDictionary *forcedOutcomes,
                               NSMutableArray *out) {
  NSMutableSet *seen = [NSMutableSet set];
  for (NSDictionary *c in forced) {
    NSString *name = c[@"name"];
    [seen addObject:name];
    NSDictionary *b = base[name];
    if (!b) {
      [out addObject:@{
        @"flag" : flag,
        @"kind" : @"extra",
        @"name" : name,
        @"class" : c[@"class"],
        @"selector" : c[@"selector"],
        @"other_outcome" : baseOutcomes[name] ?: @"(no case of this name)",
        @"reason" : @"this case produced a record only with the flag forced on",
      }];
      continue;
    }
    if (![b[@"allocated_len"] isEqual:c[@"allocated_len"]]) {
      // Both records, not just the two lengths. A length delta's whole question
      // is *what* the extra bytes are, and answering it from the lengths alone
      // needs a throwaway probe every time -- which is a capture run to
      // rediscover something this pass already had in hand.
      [out addObject:@{
        @"flag" : flag,
        @"kind" : @"length",
        @"name" : name,
        @"class" : c[@"class"],
        @"selector" : c[@"selector"],
        @"default_len" : b[@"allocated_len"],
        @"forced_len" : c[@"allocated_len"],
        @"default_buffer" : b[@"buffer"],
        @"forced_buffer" : c[@"buffer"],
        @"reason" : @"the serializer allocated a different extent with the flag on",
      }];
      continue;
    }
    NSString *hb = b[@"buffer"], *hf = c[@"buffer"];
    if ([hb isEqual:hf]) continue;
    // Name the first differing byte and count the rest. A flag that moves one
    // field and a flag that re-lays the whole record are different findings,
    // and the offset is what a reader needs to open the view at.
    NSUInteger firstOff = 0, differing = 0;
    BOOL haveFirst = NO;
    unsigned db = 0, fb = 0;
    NSUInteger n = MIN(hb.length, hf.length);
    for (NSUInteger i = 0; i + 1 < n; i += 2) {
      unsigned a = 0, e = 0;
      sscanf([hb substringWithRange:NSMakeRange(i, 2)].UTF8String, "%2x", &a);
      sscanf([hf substringWithRange:NSMakeRange(i, 2)].UTF8String, "%2x", &e);
      if (a == e) continue;
      differing++;
      if (!haveFirst) {
        haveFirst = YES;
        firstOff = i / 2;
        db = a;
        fb = e;
      }
    }
    [out addObject:@{
      @"flag" : flag,
      @"kind" : @"bytes",
      @"name" : name,
      @"class" : c[@"class"],
      @"selector" : c[@"selector"],
      @"first_offset" : @(firstOff),
      @"default_byte" : @(db),
      @"forced_byte" : @(fb),
      @"differing_bytes" : @(differing),
      @"reason" : @"the same call wrote a different record with the flag on",
    }];
  }
  for (NSString *name in base) {
    if ([seen containsObject:name]) continue;
    NSDictionary *b = base[name];
    [out addObject:@{
      @"flag" : flag,
      @"kind" : @"absent",
      @"name" : name,
      @"class" : b[@"class"],
      @"selector" : b[@"selector"],
      @"other_outcome" : forcedOutcomes[name] ?: @"(no case of this name)",
      @"reason" : @"this case produced a record at the default state and none "
                  @"with the flag forced on",
    }];
  }
}

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc < 3) {
      fprintf(stderr, "usage: %s (fixtures|texture-fixtures|creation-fixtures|blit-fixtures|inventory) <out.json>\n",
              argv[0]);
      return 2;
    }
    if (!dlopen(kBundleBin, RTLD_NOW | RTLD_LOCAL)) {
      fprintf(stderr, "dlopen failed: %s\n", dlerror());
      fprintf(stderr, "(this must run as x86_64 under Rosetta; see wire-oracle.sh)\n");
      return 1;
    }
    gArena = malloc(ARENA_CAP);

    if (strcmp(argv[1], "inventory") == 0) {
      NSMutableDictionary *root = [inventory() mutableCopy];
      root[@"provenance"] = provenance();
      return writeJSON(root, argv[2]);
    }

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    gDevice = dev;
    if (!dev) {
      fprintf(stderr, "no Metal device\n");
      return 1;
    }
    id cap = [[CaptureAllocator alloc] init];
    gStagingBuffer = [[StubBuffer alloc] initWithRef:STUB_STAGING_REF];

    gUnsupported = [NSMutableArray array];
    gSilent = [NSMutableArray array];
    gCrashed = [NSMutableArray array];
    gMulti = [NSMutableArray array];
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = abortHandler;
    sigemptyset(&sa.sa_mask);
    for (int sig_i = 0; sig_i < 3; sig_i++) {
      int sig = (int[]){SIGABRT, SIGSEGV, SIGBUS}[sig_i];
      if (sigaction(sig, &sa, NULL) != 0) {
        fprintf(stderr, "could not install the signal-%d handler; one selector "
                        "would end the run and the JSON is written only at the "
                        "end, so every case would be lost\n",
                sig);
        return 1;
      }
    }

    if (strcmp(argv[1], "texture-fixtures") == 0) {
      NSArray *first = texturePass(dev, cap, ARENA_POISON, YES);
      NSArray *second = texturePass(dev, cap, ARENA_POISON_ALT, NO);
      if (!first || !second) return 1;
      NSMutableArray *unmasked = [NSMutableArray array];
      NSArray *cases = mergeWrittenMasks(first, second, unmasked);
      return writeJSON(@{
        @"schema" : @3,
        @"provenance" : provenance(),
        @"host_gpu" : dev.name ?: @"(unknown)",
        @"cases" : cases,
        @"poison" : @[ @(ARENA_POISON), @(ARENA_POISON_ALT) ],
        @"unmasked" : unmasked,
        @"unsupported" : gUnsupported,
        @"silent" : gSilent,
        @"crashed" : gCrashed,
        @"multi" : gMulti,
      }, argv[2]);
    }

    if (strcmp(argv[1], "creation-fixtures") == 0) {
      NSArray *first = creationPass(dev, cap, ARENA_POISON, YES);
      NSArray *second = creationPass(dev, cap, ARENA_POISON_ALT, NO);
      if (!first || !second) return 1;
      NSMutableArray *unmasked = [NSMutableArray array];
      NSArray *cases = mergeWrittenMasks(first, second, unmasked);
      return writeJSON(@{
        @"schema" : @3,
        @"provenance" : provenance(),
        @"host_gpu" : dev.name ?: @"(unknown)",
        @"cases" : cases,
        @"poison" : @[ @(ARENA_POISON), @(ARENA_POISON_ALT) ],
        @"unmasked" : unmasked,
        @"unsupported" : gUnsupported,
        @"silent" : gSilent,
        @"crashed" : gCrashed,
        @"multi" : gMulti,
      }, argv[2]);
    }

    if (strcmp(argv[1], "blit-fixtures") == 0) {
      NSArray *first = blitPass(dev, cap, ARENA_POISON, YES);
      NSArray *second = blitPass(dev, cap, ARENA_POISON_ALT, NO);
      if (!first || !second) return 1;
      NSMutableArray *unmasked = [NSMutableArray array];
      NSArray *cases = mergeWrittenMasks(first, second, unmasked);
      return writeJSON(@{
        @"schema" : @3,
        @"provenance" : provenance(),
        @"host_gpu" : dev.name ?: @"(unknown)",
        @"cases" : cases,
        @"poison" : @[ @(ARENA_POISON), @(ARENA_POISON_ALT) ],
        @"unmasked" : unmasked,
        @"unsupported" : gUnsupported,
        @"silent" : gSilent,
        @"crashed" : gCrashed,
        @"multi" : gMulti,
      }, argv[2]);
    }

    NSArray *first = onePass(dev, cap, ARENA_POISON, YES);
    if (!first) return 1;
    // The same run again over the complementary fill. A fresh serializer and a
    // rewound ref allocator make it the same run rather than a continuation:
    // every case must see the state its twin saw, or the bytes would differ for
    // a reason that has nothing to do with what the serializer wrote.
    NSArray *second = onePass(dev, cap, ARENA_POISON_ALT, NO);
    if (!second) return 1;

    NSMutableArray *unmasked = [NSMutableArray array];
    NSArray *cases = mergeWrittenMasks(first, second, unmasked);

    // A third pass with every capability forced on, for its `silent` list only.
    //
    // `-supportsDynamicAttributeStride`, `-supportsVertexAmplification` and
    // `-supportsTileShaders` each turned out to gate a family that emitted
    // nothing at the default state, and every one of those was found by
    // guessing which flag to try. Sixteen flags default off, so every `silent`
    // row in the manifest rests on the guess not having been needed. This
    // measures it instead: a selector silent here is silent under every
    // capability this serializer has.
    //
    // Its outcome arrays are swapped out and back, because the run that
    // produces the fixtures must be the one whose outcomes are reported -- a
    // record captured with a flag forced is not the record the manifest is
    // describing.
    NSMutableArray *savedUnsupported = gUnsupported;
    NSMutableArray *savedSilent = gSilent;
    NSMutableArray *savedCrashed = gCrashed;
    NSMutableArray *savedMulti = gMulti;

    // The default pass keyed by case name, so every later pass can be diffed
    // against it byte for byte. `first` and every capability pass run the same
    // fill, a fresh serializer and a rewound ref allocator, so a byte that
    // differs between them differs because of the flag and nothing else.
    NSMutableDictionary *baseByName = [NSMutableDictionary dictionary];
    for (NSDictionary *c in first) baseByName[c[@"name"]] = c;
    NSMutableArray *contentDeltas = [NSMutableArray array];

    gUnsupported = [NSMutableArray array];
    gSilent = [NSMutableArray array];
    gCrashed = [NSMutableArray array];
    gMulti = [NSMutableArray array];
    // The default pass keyed by case name, so every later pass can be diffed
    // against it byte for byte. `first` and every capability pass run the same
    // fill, a fresh serializer and a rewound ref allocator, so a byte that
    // differs between them differs because of the flag and nothing else.
    NSDictionary *baseOutcomes =
        outcomeIndex(savedSilent, savedUnsupported, savedCrashed, savedMulti);

    gForceAllCapabilities = YES;
    fprintf(stderr, "\n--- sweep pass: every capability forced on ---\n");
    NSArray *sweep = onePass(dev, cap, ARENA_POISON, YES);
    if (!sweep) return 1;
    gForceAllCapabilities = NO;
    NSArray *silentWithEveryCapability = gSilent;
    diffAgainstDefault(baseByName, sweep, @"(every flag)", baseOutcomes,
                       outcomeIndex(gSilent, gUnsupported, gCrashed, gMulti),
                       contentDeltas);

    // One more pass per flag, with that flag and nothing else on.
    //
    // The sweep says a selector is gated; this says on what. Without it the
    // next step -- wrapping the case in `withCapability(ser, @"Which?", ...)`
    // -- is a guess checked against a fast oracle, and three families were
    // added exactly that way before the sweep existed. Sixteen passes cost a
    // few seconds and turn the whole remaining queue into a lookup.
    //
    // Both directions are recorded. `unlocks` is the useful one. `suppresses`
    // should stay empty: a flag that stops a record being emitted would mean
    // the fixtures this crate pins are conditional on a capability being off,
    // which is a much larger claim than "some families need a flag".
    NSMutableArray *attribution = [NSMutableArray array];
    for (NSString *flag in capabilityFlagNames()) {
      gUnsupported = [NSMutableArray array];
      gSilent = [NSMutableArray array];
      gCrashed = [NSMutableArray array];
      gMulti = [NSMutableArray array];
      gForceOneCapability = flag;
      fprintf(stderr, "\n--- attribution pass: %s only ---\n", flag.UTF8String);
      NSArray *forced = onePass(dev, cap, ARENA_POISON, YES);
      if (!forced) return 1;
      gForceOneCapability = nil;
      diffAgainstDefault(baseByName, forced, flag, baseOutcomes,
                         outcomeIndex(gSilent, gUnsupported, gCrashed, gMulti),
                         contentDeltas);

      NSMutableSet *silentWithFlag = [NSMutableSet set];
      for (NSDictionary *e in gSilent)
        [silentWithFlag addObject:@[ e[@"class"], e[@"selector"] ]];
      NSMutableSet *silentAtDefault = [NSMutableSet set];
      for (NSDictionary *e in savedSilent)
        [silentAtDefault addObject:@[ e[@"class"], e[@"selector"] ]];

      NSMutableArray *unlocks = [NSMutableArray array];
      for (NSArray *k in silentAtDefault)
        if (![silentWithFlag containsObject:k])
          [unlocks addObject:@{@"class" : k[0], @"selector" : k[1]}];
      NSMutableArray *suppresses = [NSMutableArray array];
      for (NSArray *k in silentWithFlag)
        if (![silentAtDefault containsObject:k])
          [suppresses addObject:@{@"class" : k[0], @"selector" : k[1]}];

      NSArray *sortBy = @[
        [NSSortDescriptor sortDescriptorWithKey:@"class" ascending:YES],
        [NSSortDescriptor sortDescriptorWithKey:@"selector" ascending:YES],
      ];
      [attribution addObject:@{
        @"flag" : flag,
        @"unlocks" : [unlocks sortedArrayUsingDescriptors:sortBy],
        @"suppresses" : [suppresses sortedArrayUsingDescriptors:sortBy],
      }];
    }

    gUnsupported = savedUnsupported;
    gSilent = savedSilent;
    gCrashed = savedCrashed;
    gMulti = savedMulti;

    NSDictionary *root = @{
      @"schema" : @3,
      @"provenance" : provenance(),
      @"host_gpu" : dev.name ?: @"(unknown)",
      // What every capability flag reads on an untouched serializer. A `false`
      // here means any selector family gated on that flag emits nothing, so a
      // `silent` entry below is about this harness rather than about Apple.
      @"capability_defaults" : gCapabilityDefaults ?: @{},
      @"cases" : cases,
      // The two arena fills, so a reader can tell an unwritten byte's recorded
      // value from a written one without knowing this file.
      @"poison" : @[ @(ARENA_POISON), @(ARENA_POISON_ALT) ],
      // Cases whose two passes could not be compared, with the reason. Never
      // silent: a missing mask must not read as "nothing was written".
      @"unmasked" : unmasked,
      // Selectors driven this run that emitted nothing because the serializer
      // refused them. An empty list is a claim too: it says every case ran.
      @"unsupported" : gUnsupported,
      // Selectors driven this run that returned normally and wrote no record.
      @"silent" : gSilent,
      // The same, measured with all sixteen capabilities forced on. A selector
      // in `silent` but not here is one a capability unlocks, and its
      // `EMITS_NO_OPERATION` row would be a false claim about Apple.
      // `every_silent_selector_is_silent_under_every_capability` is the gate.
      @"silent_with_every_capability" : silentWithEveryCapability,
      // Which flag unlocks which selector, one pass per flag with that flag
      // alone on. The sweep above says a selector is gated; this says on what,
      // which is the argument `withCapability` takes. A gated selector absent
      // from every `unlocks` list needs more than one flag at once.
      @"capability_attribution" : attribution,
      // The third direction, and the one the two lists above cannot see: a flag
      // that changes what a record *contains*. Every capability pass's records
      // are diffed against the default pass's, case by case and byte by byte.
      //
      // This matters because nothing in `reims-vgpu` observes the guest
      // negotiating a capability. Every fixture this crate pins was captured at
      // the default state, so a flag that moved a field would make the pinned
      // layout wrong for exactly the guests that turned the flag on, and no
      // test here would say so. `no_capability_changes_what_a_record_contains`
      // is the gate; an entry is a layout that needs a capability in its key.
      @"capability_content_deltas" : contentDeltas,
      // Selectors that faulted. Evidence about this harness, not about Apple.
      @"crashed" : gCrashed,
      // Selectors whose record count no case claimed. Each is one or more wire
      // records with no fixture, which is why it is a list rather than a note.
      @"multi" : gMulti,
    };
    return writeJSON(root, argv[2]);
  }
}
