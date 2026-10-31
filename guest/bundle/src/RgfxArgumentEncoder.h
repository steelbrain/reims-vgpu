/*
 * RgfxArgumentEncoder: the reims-graphics bundle's MTLArgumentEncoder wrapper.
 *
 * An argument encoder lays out resource bindings into an argument buffer. There is
 * no argument-encoder create in this slice's wire (the serialiser has no such
 * message), and the host platform exposes no per-argument binding reflection (the
 * platform gap recorded with RgfxPipelineReflection), so this wrapper lays out
 * nothing: `encodedLength` is 0, its setters write nothing, and each member logs its
 * first use as unsupported (RgfxUnsupported.h). It carries the sentinel objId
 * RGFX_OBJ_NONE (no host handle). The real encoder — its layout computation and the
 * buffer writes — lands with the argument-buffer transport.
 *
 * Our own ObjC++: it conforms to MTLArgumentEncoder but subclasses only NSObject
 * (via RgfxResource), so it instantiates and is unit-tested on the Mac.
 */

#ifndef RGFX_ARGUMENT_ENCODER_H_
#define RGFX_ARGUMENT_ENCODER_H_

#import <Metal/Metal.h>

#import "RgfxResource.h"

#ifdef __cplusplus
class RgfxSerializer;
#endif

NS_ASSUME_NONNULL_BEGIN

@interface RgfxArgumentEncoder : RgfxResource <MTLArgumentEncoder>

/// A stub encoder over `arguments` (MTLArgumentDescriptor). `device`/`serializer`
/// are the owning device's (both may be nil in a unit test).
+ (instancetype)argumentEncoderWithArguments:(NSArray<MTLArgumentDescriptor *> *)arguments
                                      device:(nullable id<MTLDevice>)device
                                  serializer:(nullable RgfxSerializer *)serializer;

/// The number of argument descriptors the encoder was made with.
@property(nonatomic, readonly) NSUInteger rgfxArgumentCount;

@end

NS_ASSUME_NONNULL_END

#endif // RGFX_ARGUMENT_ENCODER_H_
