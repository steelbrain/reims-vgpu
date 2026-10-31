/*
 * RgfxArgumentEncoder: see RgfxArgumentEncoder.h. A documented stub. Our own ObjC++;
 * no Apple private SPI.
 */

#import "RgfxArgumentEncoder.h"
#import "RgfxUnsupported.h"

@interface RgfxArgumentEncoder ()
- (instancetype)initWithArguments:(NSArray<MTLArgumentDescriptor *> *)arguments
                           device:(nullable id<MTLDevice>)device
                       serializer:(nullable RgfxSerializer *)serializer;
@end

@implementation RgfxArgumentEncoder {
	NSUInteger _argumentCount;
}

+ (instancetype)argumentEncoderWithArguments:(NSArray<MTLArgumentDescriptor *> *)arguments
                                      device:(id<MTLDevice>)device
                                  serializer:(RgfxSerializer *)serializer
{
	return [[RgfxArgumentEncoder alloc] initWithArguments:arguments
	                                               device:device
	                                           serializer:serializer];
}

- (instancetype)initWithArguments:(NSArray<MTLArgumentDescriptor *> *)arguments
                           device:(id<MTLDevice>)device
                       serializer:(RgfxSerializer *)serializer
{
	// No host handle: an argument encoder is a guest-side layout object with no wire
	// create in this slice. RGFX_OBJ_NONE (0) is the base objId.
	self = [super initWithObjId:RGFX_OBJ_NONE device:device serializer:serializer];
	if (self) {
		_argumentCount = arguments.count;
	}
	return self;
}

- (NSUInteger)rgfxArgumentCount
{
	return _argumentCount;
}

// --- MTLArgumentEncoder: unsupported ----------------------------------------------
// encodedLength and the layout need the host platform's argument-buffer layout rules, and the
// writes need the host's view of the addresses and resource ids they encode: neither is on the
// wire. So the encoder lays out nothing: encodedLength is 0, the setters write nothing, and
// each member logs its first use (RgfxUnsupported.h). A nil or empty argument asks for nothing.

- (NSUInteger)encodedLength
{
	RGFX_UNSUPPORTED("argument encoders (no argument-buffer layout on the wire)");
	return 0;
}

- (NSUInteger)alignment
{
	return 256; // a safe argument-buffer alignment floor; the layout itself is not carried
}

- (void)setArgumentBuffer:(id<MTLBuffer>)argumentBuffer offset:(NSUInteger)offset
{
	if (argumentBuffer != nil)
		RGFX_UNSUPPORTED("argument encoders (no argument-buffer layout on the wire)");
}

- (void)setArgumentBuffer:(id<MTLBuffer>)argumentBuffer
              startOffset:(NSUInteger)startOffset
             arrayElement:(NSUInteger)arrayElement
{
	if (argumentBuffer != nil)
		RGFX_UNSUPPORTED("argument encoders (no argument-buffer layout on the wire)");
}

- (void)setBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index
{
	if (buffer != nil)
		RGFX_UNSUPPORTED("argument encoders (no argument-buffer layout on the wire)");
}

- (void)setBuffers:(const id<MTLBuffer> __nullable[__nonnull])buffers
           offsets:(const NSUInteger[__nonnull])offsets
         withRange:(NSRange)range
{
	if (range.length != 0)
		RGFX_UNSUPPORTED("argument encoders (no argument-buffer layout on the wire)");
}

- (void)setTexture:(id<MTLTexture>)texture atIndex:(NSUInteger)index
{
	if (texture != nil)
		RGFX_UNSUPPORTED("argument encoders (no argument-buffer layout on the wire)");
}

- (void)setTextures:(const id<MTLTexture> __nullable[__nonnull])textures withRange:(NSRange)range
{
	if (range.length != 0)
		RGFX_UNSUPPORTED("argument encoders (no argument-buffer layout on the wire)");
}

- (void)setSamplerState:(id<MTLSamplerState>)sampler atIndex:(NSUInteger)index
{
	if (sampler != nil)
		RGFX_UNSUPPORTED("argument encoders (no argument-buffer layout on the wire)");
}

- (void)setSamplerStates:(const id<MTLSamplerState> __nullable[__nonnull])samplers
               withRange:(NSRange)range
{
	if (range.length != 0)
		RGFX_UNSUPPORTED("argument encoders (no argument-buffer layout on the wire)");
}

- (void)setRenderPipelineState:(id<MTLRenderPipelineState>)pipeline atIndex:(NSUInteger)index
{
	if (pipeline != nil)
		RGFX_UNSUPPORTED("argument encoders (no argument-buffer layout on the wire)");
}

- (void)setRenderPipelineStates:(const id<MTLRenderPipelineState> __nullable[__nonnull])pipelines
                      withRange:(NSRange)range
{
	if (range.length != 0)
		RGFX_UNSUPPORTED("argument encoders (no argument-buffer layout on the wire)");
}

- (void)setComputePipelineState:(id<MTLComputePipelineState>)pipeline atIndex:(NSUInteger)index
{
	if (pipeline != nil)
		RGFX_UNSUPPORTED("argument encoders (no argument-buffer layout on the wire)");
}

- (void)setComputePipelineStates:(const id<MTLComputePipelineState> __nullable[__nonnull])pipelines
                       withRange:(NSRange)range
{
	if (range.length != 0)
		RGFX_UNSUPPORTED("argument encoders (no argument-buffer layout on the wire)");
}

- (void)setDepthStencilState:(id<MTLDepthStencilState>)depthStencilState atIndex:(NSUInteger)index
{
	if (depthStencilState != nil)
		RGFX_UNSUPPORTED("argument encoders (no argument-buffer layout on the wire)");
}

- (void)setDepthStencilStates:(const id<MTLDepthStencilState> __nullable[__nonnull])depthStencilStates
                    withRange:(NSRange)range
{
	if (range.length != 0)
		RGFX_UNSUPPORTED("argument encoders (no argument-buffer layout on the wire)");
}

// The device makes no indirect command buffer, acceleration structure or function table.
- (void)setIndirectCommandBuffer:(id<MTLIndirectCommandBuffer>)indirectCommandBuffer
                         atIndex:(NSUInteger)index
{
	if (indirectCommandBuffer != nil)
		RGFX_UNSUPPORTED("indirect command buffers");
}

- (void)setIndirectCommandBuffers:(const id<MTLIndirectCommandBuffer> __nullable[__nonnull])buffers
                        withRange:(NSRange)range
{
	if (range.length != 0)
		RGFX_UNSUPPORTED("indirect command buffers");
}

- (void)setAccelerationStructure:(id<MTLAccelerationStructure>)accelerationStructure
                         atIndex:(NSUInteger)index
{
	if (accelerationStructure != nil)
		RGFX_UNSUPPORTED("ray tracing");
}

- (void)setIntersectionFunctionTable:(id<MTLIntersectionFunctionTable>)intersectionFunctionTable
                             atIndex:(NSUInteger)index
{
	if (intersectionFunctionTable != nil)
		RGFX_UNSUPPORTED("ray tracing");
}

- (void)setIntersectionFunctionTables:
            (const id<MTLIntersectionFunctionTable> __nullable[__nonnull])intersectionFunctionTables
                            withRange:(NSRange)range
{
	if (range.length != 0)
		RGFX_UNSUPPORTED("ray tracing");
}

- (void)setVisibleFunctionTable:(id<MTLVisibleFunctionTable>)visibleFunctionTable
                        atIndex:(NSUInteger)index
{
	if (visibleFunctionTable != nil)
		RGFX_UNSUPPORTED("function pointers");
}

- (void)setVisibleFunctionTables:(const id<MTLVisibleFunctionTable> __nullable[__nonnull])visibleFunctionTables
                       withRange:(NSRange)range
{
	if (range.length != 0)
		RGFX_UNSUPPORTED("function pointers");
}

- (void *)constantDataAtIndex:(NSUInteger)index
{
	RGFX_UNSUPPORTED("argument encoders (no argument-buffer layout on the wire)");
	return nullptr; // no argument buffer is laid out
}

- (id<MTLArgumentEncoder>)newArgumentEncoderForBufferAtIndex:(NSUInteger)index
{
	RGFX_UNSUPPORTED("argument encoders (no argument-buffer layout on the wire)");
	return nil;
}

@end
