/*
 * RgfxArgumentEncoder: see RgfxArgumentEncoder.h. A documented stub. Our own ObjC++;
 * no Apple private SPI.
 */

#import "RgfxArgumentEncoder.h"

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

// --- MTLArgumentEncoder (DOCUMENTED STUB) ------------------------------------
// encodedLength/alignment need the platform's argument-buffer layout rules, which
// are not in this slice's wire; these return documented placeholders rather than
// fabricate a layout. The setters are no-ops until the argument-buffer transport
// lands. Callers that write argument buffers do not reach this path yet.

- (NSUInteger)encodedLength
{
	return 0; // placeholder until the argument-buffer layout transport lands
}

- (NSUInteger)alignment
{
	return 256; // a safe argument-buffer alignment floor; the real value is the
	            // platform's, carried later
}

- (void)setArgumentBuffer:(id<MTLBuffer>)argumentBuffer offset:(NSUInteger)offset
{
	(void)argumentBuffer;
	(void)offset;
}

- (void)setArgumentBuffer:(id<MTLBuffer>)argumentBuffer
              startOffset:(NSUInteger)startOffset
             arrayElement:(NSUInteger)arrayElement
{
	(void)argumentBuffer;
	(void)startOffset;
	(void)arrayElement;
}

- (void)setBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index
{
	(void)buffer;
	(void)offset;
	(void)index;
}

- (void)setTexture:(id<MTLTexture>)texture atIndex:(NSUInteger)index
{
	(void)texture;
	(void)index;
}

- (void)setSamplerState:(id<MTLSamplerState>)sampler atIndex:(NSUInteger)index
{
	(void)sampler;
	(void)index;
}

- (void *)constantDataAtIndex:(NSUInteger)index
{
	(void)index;
	return nullptr; // no argument-buffer backing in the stub
}

@end
