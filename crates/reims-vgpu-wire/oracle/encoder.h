// Encoder-record capture for the wire oracle.
//
// Object-creation records go through PGSerializerAllocator
// (-allocateOperationBytes:). Encoder records go through a different sink:
// PGSerializerCommandStream's -getCommandBufferBytes:, which a
// PGSerializer*CommandEncoder writes every command into. Both are "ask for
// exactly this many bytes, then write the record", so both give the record's
// true length alongside its content.
//
// Standing up an encoder needs three stubs, none of which touches IOKit:
//
//   * PGSerializerCommandStream -- 11 required methods, plus the handful
//     _MTLCommandEncoder's designated init asks its command buffer for
//     (`device`, `setCurrentCommandEncoder:`, `isStatEnabled`,
//     `protectionOptions`, `errorOptions`). The forwarding stub answers the
//     ones we do not model rather than crashing, and prints them so the set
//     stays visible instead of being rediscovered by crash.
//   * PGSerializerTexture -- four methods (`textureRef`, `pixelFormat`, and
//     PGSerializerResource's `serializerResourceRef`, `shouldSyncInBlit`).
//     Needed only so a render pass can carry a colour attachment.
//   * PGSerializerObjectRefAllocator -- one method.
#ifndef REIMS_WIRE_ORACLE_ENCODER_H
#define REIMS_WIRE_ORACLE_ENCODER_H

// Ref handed to the stub colour-attachment texture. Distinctive so it is
// recognisable wherever it lands in a 584-byte render-pass payload.
#define STUB_TEXTURE_REF 4242u

// Ref handed to the stub index buffer, likewise distinctive.
#define STUB_BUFFER_REF 5151u

// A blit names two resources of the same kind, so each stub takes its ref at
// init and the destination gets a different one. Without that, a record that
// wrote the source ref into both slots — or swapped them — would read back
// correct.
#define STUB_TEXTURE_DST_REF 4343u
#define STUB_BUFFER_DST_REF 5252u

// A third texture, differing from the other two in `pixelFormat` as well as in
// ref. `fillTexture:level:slice:region:color:` writes a format word its
// selector never mentions, and against a stub whose format never changes that
// word is indistinguishable from a constant.
#define STUB_TEXTURE_R8_REF 4444u

// An indexed indirect patch draw names *three* buffers — the patch index list,
// the control-point index list and the indirect arguments — so two refs cannot
// tell its three slots apart.
#define STUB_BUFFER_THIRD_REF 5353u

// The indirect-command-buffer selectors reach their argument through
// `indirectCommandBufferRef`, which is not one of the accessors a plain buffer
// answers — a StubBuffer passed to `resetCommandsInBuffer:withRange:` produced
// a record whose ref field read 0 rather than 5151, and 0 is a value a reader
// would take for "unbound" instead of for "the stub did not answer".
#define STUB_ICB_REF 7171u
#define STUB_ICB_DST_REF 7272u

// The buffer `-getBufferBytes:alignment:buffer:offset:` hands back for inline
// argument data. `setVertexBytes:length:atIndex:` and friends stage their bytes
// through it and then record the *staging buffer's* ref and offset, so a stream
// that answers `nil` at offset 0 produces a bind record naming buffer 0 — which
// reads as "unbound" rather than as "the stub declined".
#define STUB_STAGING_REF 8181u
#define STUB_STAGING_OFFSET 0x9999u

/// Handed back by `-getBufferBytes:...`; a `StubBuffer` at [`STUB_STAGING_REF`],
/// assigned in `main` once the class exists.
static id gStagingBuffer;

@interface CaptureCommandStream : NSObject {
  unsigned char *_continuationTarget;
}
- (void)setContinuationTarget:(void *)target;
@end

@implementation CaptureCommandStream
- (id<MTLDevice>)device { return gDevice; }
- (NSString *)label { return @"reims-wire-oracle"; }

- (void *)getCommandBufferBytes:(size_t)n {
  return arenaTake(n);
}

// Inline argument data (bind tables, index lists). Not a record, so it is not
// counted as one -- it must not consume an operation slot or the case would
// report two records where the encoder emitted one.
- (void *)getBufferBytes:(size_t)n
                alignment:(size_t)a
                   buffer:(id *)buf
                   offset:(size_t *)off {
  void *p = arenaTakeUncounted(n, a);
  if (buf) *buf = gStagingBuffer;
  if (off) *off = STUB_STAGING_OFFSET;
  return p;
}
- (void *)getBufferBytes:(size_t)n
                alignment:(size_t)a
                 poolType:(int)t
                   buffer:(id *)buf
                   offset:(size_t *)off {
  return [self getBufferBytes:n alignment:a buffer:buf offset:off];
}

- (char)addResourceReference:(id)r isWrite:(char)w { return 1; }
- (char)addHeapReference:(id)h { return 1; }
- (char)addStateReference:(id)s { return 1; }
- (char)addFenceReference:(id)f { return 1; }
- (char)addResourceMetadataReference:(id)r { return 1; }
- (char)addHeapMetadataReference:(id)h { return 1; }
- (void)endEncoding {}
- (void)setContinuationTarget:(void *)target {
  _continuationTarget = target;
}
- (void)beginContinuation {
  if (_continuationTarget) _continuationTarget[6] = 1;
}
- (void)merge:(id)o {}
- (unsigned long long)getNextTraceID {
  static unsigned long long t = 1;
  return t++;
}

- (NSMethodSignature *)methodSignatureForSelector:(SEL)sel {
  NSMethodSignature *s = [super methodSignatureForSelector:sel];
  if (s) return s;
  fprintf(stderr, "  note: stubbed -[CaptureCommandStream %s]\n", sel_getName(sel));
  return [NSMethodSignature signatureWithObjCTypes:"v@:@@@@"];
}
- (void)forwardInvocation:(NSInvocation *)inv { (void)inv; }
@end

@interface StubTexture : NSObject {
@public
  unsigned int _ref;
  unsigned long long _pixelFormat;
}
- (instancetype)initWithRef:(unsigned int)r;
- (instancetype)initWithRef:(unsigned int)r pixelFormat:(unsigned long long)f;
- (unsigned long long)pixelFormat;
@end

@implementation StubTexture
- (instancetype)initWithRef:(unsigned int)r pixelFormat:(unsigned long long)f {
  if ((self = [super init])) {
    _ref = r;
    _pixelFormat = f;
  }
  return self;
}
// The format is settable because one record reads it off the *texture* rather
// than off an argument: `fillTexture:level:slice:region:color:` writes a format
// word its selector never mentions. With a fixed stub that word reads
// BGRA8Unorm in every capture, which cannot tell "the serializer asked the
// texture" from "the field happens to hold 80". A second format settles it.
- (instancetype)initWithRef:(unsigned int)r {
  return [self initWithRef:r pixelFormat:MTLPixelFormatBGRA8Unorm];
}
- (instancetype)init { return [self initWithRef:STUB_TEXTURE_REF]; }
- (unsigned int)textureRef { return _ref; }
- (unsigned int)serializerResourceRef { return _ref; }
- (unsigned long long)pixelFormat { return _pixelFormat; }
- (char)shouldSyncInBlit { return 0; }
- (id<MTLDevice>)device { return gDevice; }
- (unsigned long long)width { return 640; }
- (unsigned long long)height { return 480; }
- (unsigned long long)depth { return 1; }
- (unsigned long long)textureType { return MTLTextureType2D; }
- (unsigned long long)mipmapLevelCount { return 1; }
- (unsigned long long)arrayLength { return 1; }
- (unsigned long long)sampleCount { return 1; }
- (unsigned long long)storageMode { return MTLStorageModePrivate; }
- (char)addToStream:(id)s isWrite:(char)w { return 1; }
- (NSMethodSignature *)methodSignatureForSelector:(SEL)sel {
  NSMethodSignature *s = [super methodSignatureForSelector:sel];
  if (s) return s;
  fprintf(stderr, "  note: stubbed -[StubTexture %s]\n", sel_getName(sel));
  return [NSMethodSignature signatureWithObjCTypes:"v@:@@@@"];
}
- (void)forwardInvocation:(NSInvocation *)inv { (void)inv; }
@end

// Stands in for the index buffer an indexed draw names. The record carries the
// buffer's serializer ref, so the only thing this has to be is a resource with
// a recognisable one -- no storage is ever read.
@interface StubBuffer : NSObject {
@public
  unsigned int _ref;
}
- (instancetype)initWithRef:(unsigned int)r;
@end

@implementation StubBuffer
- (instancetype)initWithRef:(unsigned int)r {
  if ((self = [super init])) _ref = r;
  return self;
}
- (instancetype)init { return [self initWithRef:STUB_BUFFER_REF]; }
- (unsigned int)bufferRef { return _ref; }
- (unsigned int)serializerResourceRef { return _ref; }
- (unsigned long long)length { return 1ull << 20; }
- (unsigned long long)gpuAddress { return 0; }
- (void *)contents { return NULL; }
- (char)shouldSyncInBlit { return 0; }
- (unsigned long long)storageMode { return MTLStorageModePrivate; }
- (id<MTLDevice>)device { return gDevice; }
// Registers the resource with the command stream. The indirect-command-buffer
// selectors call this and then *branch on the result* — forwardInvocation
// leaves it zero, which reads as "refused", and the encoder returns having
// written no record. That silence is indistinguishable from a selector that
// genuinely emits nothing, so it has to be answered rather than stubbed.
- (char)addToStream:(id)s isWrite:(char)w { return 1; }
- (NSMethodSignature *)methodSignatureForSelector:(SEL)sel {
  NSMethodSignature *s = [super methodSignatureForSelector:sel];
  if (s) return s;
  fprintf(stderr, "  note: stubbed -[StubBuffer %s]\n", sel_getName(sel));
  return [NSMethodSignature signatureWithObjCTypes:"v@:@@@@"];
}
- (void)forwardInvocation:(NSInvocation *)inv { (void)inv; }
@end

// Stands in for an indirect command buffer. Its ref accessor is its own, and it
// registers with the stream like any other resource.
@interface StubICB : NSObject {
@public
  unsigned int _ref;
}
- (instancetype)initWithRef:(unsigned int)r;
@end

@implementation StubICB
- (instancetype)initWithRef:(unsigned int)r {
  if ((self = [super init])) _ref = r;
  return self;
}
- (instancetype)init { return [self initWithRef:STUB_ICB_REF]; }
- (unsigned int)indirectCommandBufferRef { return _ref; }
- (unsigned int)serializerResourceRef { return _ref; }
- (unsigned long long)size { return 4096; }
- (char)addToStream:(id)s isWrite:(char)w { return 1; }
- (char)shouldSyncInBlit { return 0; }
- (id<MTLDevice>)device { return gDevice; }
- (NSMethodSignature *)methodSignatureForSelector:(SEL)sel {
  NSMethodSignature *s = [super methodSignatureForSelector:sel];
  if (s) return s;
  fprintf(stderr, "  note: stubbed -[StubICB %s]\n", sel_getName(sel));
  return [NSMethodSignature signatureWithObjCTypes:"v@:@@@@"];
}
- (void)forwardInvocation:(NSInvocation *)inv { (void)inv; }
@end

// Stand-ins for the state objects a bind record names. Each answers to every
// ref accessor the serializer might plausibly ask for, with a ref distinctive
// enough to recognise wherever it lands; anything it is asked and does not
// know prints itself, so the set stays visible rather than being rediscovered
// by a wrong value. The refs are far apart so a record that picks up the wrong
// object is obvious rather than off by one.
#define STUB_PIPELINE_REF 6161u
#define STUB_DEPTH_STENCIL_REF 6262u
#define STUB_SAMPLER_REF 6363u
#define STUB_FENCE_REF 6464u

#define STUB_STATE_CLASS(NAME, REF)                                            \
  @interface NAME : NSObject                                                   \
  @end                                                                         \
  @implementation NAME                                                         \
  -(unsigned int)serializerStateRef { return REF; }                            \
  -(unsigned int)stateRef { return REF; }                                      \
  -(unsigned int)serializerResourceRef { return REF; }                         \
  -(id<MTLDevice>)device { return gDevice; }                                   \
  -(NSMethodSignature *)methodSignatureForSelector : (SEL)sel {                \
    NSMethodSignature *s = [super methodSignatureForSelector:sel];             \
    if (s) return s;                                                           \
    fprintf(stderr, "  note: stubbed -[%s %s]\n", #NAME, sel_getName(sel));    \
    return [NSMethodSignature signatureWithObjCTypes:"v@:@@@@"];               \
  }                                                                            \
  -(void)forwardInvocation : (NSInvocation *)inv { (void)inv; }                \
  @end

STUB_STATE_CLASS(StubPipelineState, STUB_PIPELINE_REF)
STUB_STATE_CLASS(StubDepthStencilState, STUB_DEPTH_STENCIL_REF)
STUB_STATE_CLASS(StubSamplerState, STUB_SAMPLER_REF)
STUB_STATE_CLASS(StubFence, STUB_FENCE_REF)

// A heap, so `useHeap:stages:` can be driven. Its opcode is the question:
// `reims_vgpu::runtime::decode::render` records OP_USE_RESOURCE as 0x87 while
// the serializer writes 0x89 for `useResource:usage:stages:`, so 0x87 belongs
// to something else in this family and the residency selectors are where to
// look.
// The info encoder's coordinate mappers and rate-map queries reach their
// argument through `rasterizationRateMapRef`, which no other stub answers.
#define STUB_RATE_MAP_REF 6767u

#define STUB_HEAP_REF 6565u
#define STUB_HEAP2_REF 6666u
STUB_STATE_CLASS(StubHeap, STUB_HEAP_REF)
STUB_STATE_CLASS(StubHeap2, STUB_HEAP2_REF)
STUB_STATE_CLASS(StubRateMap, STUB_RATE_MAP_REF)

@interface StubRateMap (Refs)
@end
@implementation StubRateMap (Refs)
- (unsigned int)rasterizationRateMapRef { return STUB_RATE_MAP_REF; }
@end

@interface StubHeap2 (Refs)
@end
@implementation StubHeap2 (Refs)
- (unsigned int)heapRef { return STUB_HEAP2_REF; }
- (unsigned int)serializerHeapRef { return STUB_HEAP2_REF; }
@end

@interface StubHeap (Refs)
@end
@implementation StubHeap (Refs)
- (unsigned int)heapRef { return STUB_HEAP_REF; }
- (unsigned int)serializerHeapRef { return STUB_HEAP_REF; }
@end

@interface StubPipelineState (Refs)
@end
@implementation StubPipelineState (Refs)
- (unsigned int)pipelineRef { return STUB_PIPELINE_REF; }
- (unsigned int)renderPipelineStateRef { return STUB_PIPELINE_REF; }
- (unsigned int)pipelineStateRef { return STUB_PIPELINE_REF; }
@end

@interface StubDepthStencilState (Refs)
@end
@implementation StubDepthStencilState (Refs)
- (unsigned int)depthStencilRef { return STUB_DEPTH_STENCIL_REF; }
- (unsigned int)depthStencilStateRef { return STUB_DEPTH_STENCIL_REF; }
@end

@interface StubSamplerState (Refs)
@end
@implementation StubSamplerState (Refs)
- (unsigned int)samplerRef { return STUB_SAMPLER_REF; }
- (unsigned int)samplerStateRef { return STUB_SAMPLER_REF; }
@end

@interface StubFence (Refs)
@end
@implementation StubFence (Refs)
- (unsigned int)fenceRef { return STUB_FENCE_REF; }
- (unsigned int)serializerFenceRef { return STUB_FENCE_REF; }
@end

// The three object kinds only the tile and ray-tracing bind selectors name.
// Each is a distinct Metal protocol with its own ref accessor, so one shared
// stub would let a record that picked up the wrong object read back correct.
#define STUB_ACCEL_STRUCT_REF 6868u
#define STUB_VISIBLE_FN_TABLE_REF 6969u
#define STUB_INTERSECTION_FN_TABLE_REF 7070u

STUB_STATE_CLASS(StubAccelStruct, STUB_ACCEL_STRUCT_REF)
STUB_STATE_CLASS(StubVisibleFnTable, STUB_VISIBLE_FN_TABLE_REF)
STUB_STATE_CLASS(StubIntersectionFnTable, STUB_INTERSECTION_FN_TABLE_REF)

@interface StubAccelStruct (Refs)
@end
@implementation StubAccelStruct (Refs)
- (unsigned int)accelerationStructureRef { return STUB_ACCEL_STRUCT_REF; }
- (unsigned long long)size { return 4096; }
- (char)addToStream:(id)s isWrite:(char)w { return 1; }
- (char)shouldSyncInBlit { return 0; }
@end

@interface StubVisibleFnTable (Refs)
@end
@implementation StubVisibleFnTable (Refs)
- (unsigned int)visibleFunctionTableRef { return STUB_VISIBLE_FN_TABLE_REF; }
- (unsigned long long)size { return 4096; }
- (char)addToStream:(id)s isWrite:(char)w { return 1; }
- (char)shouldSyncInBlit { return 0; }
@end

@interface StubIntersectionFnTable (Refs)
@end
@implementation StubIntersectionFnTable (Refs)
- (unsigned int)intersectionFunctionTableRef { return STUB_INTERSECTION_FN_TABLE_REF; }
- (unsigned long long)size { return 4096; }
- (char)addToStream:(id)s isWrite:(char)w { return 1; }
- (char)shouldSyncInBlit { return 0; }
@end

#endif // REIMS_WIRE_ORACLE_ENCODER_H
