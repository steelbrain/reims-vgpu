/*
 * RgfxGLFunction: a function of GL IR, which the device answers macOS's OpenGL renderer over
 * Metal (AppleMetalOpenGLRenderer) for each shader it compiles
 * (newFunctionWithGLIR:, newFunctionWithGLCoreIR: and newFunctionWithGLESIR:, each with and
 * without an inputs description). The renderer makes each variant of the shader from it with
 * newFunctionWithPluginData:bitcodeType:, and its pipelines of the variants.
 *
 * Metal's own devices keep the IR and the plugin data in the function, and lower the two to AIR
 * when a pipeline is made, in their compiler service. The device has none: a variant is lowered
 * here, in the renderer's process (RgfxGLLowering.hpp), to a .metallib of one function, which the
 * device makes a library of (newLibraryWithData:, so it reaches the host as any library does, by
 * its content key), and the variant is that library's function. The generator's return data
 * stays with it (RgfxGLPluginReturnData), for the pipeline reflection to answer as Metal's does.
 *
 * Our own ObjC++ over public Metal API; it subclasses only NSObject, so it is unit-tested on the
 * Mac.
 */
#ifndef RGFX_GL_FUNCTION_H_
#define RGFX_GL_FUNCTION_H_

#import <Metal/Metal.h>

NS_ASSUME_NONNULL_BEGIN

@interface RgfxGLFunction : NSObject <MTLFunction>

/// A function of the GL IR at `ir` (a u32 length, then that many bytes, which are copied) of
/// stage `functionType`, on `device`. `inputsDescription` is kept as given. Nil where `ir` is
/// null.
- (nullable instancetype)initWithIR:(const void *_Nullable)ir
                       functionType:(MTLFunctionType)functionType
                  inputsDescription:(nullable id)inputsDescription
                             device:(id<MTLDevice>)device NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;

/// The variant of the shader that `pluginData` selects: the function of the device's library of
/// the variant's .metallib, holding the generator's return data. Nil, logged once per reason,
/// where the variant is not lowered or the device makes no library or function of it.
/// `bitcodeType` (1 for GL, 3 for GL ES) does not change the lowering.
- (nullable id<MTLFunction>)newFunctionWithPluginData:(NSData *)pluginData bitcodeType:(NSUInteger)bitcodeType;

/// The IR (its u32 length first), as the renderer gave it.
@property(nonatomic, readonly) NSData *rgfxIR;

/// The inputs description the renderer gave; nil where it gave none.
@property(nonatomic, readonly, nullable) id rgfxInputsDescription;

@end

/// The GL code generator's return data of a variant RgfxGLFunction made; nil for any other
/// function, and for a variant it gave none.
NSData *_Nullable RgfxGLPluginReturnData(id<MTLFunction> _Nullable function);

/// Keeps `data` as `function`'s return data (RgfxGLPluginReturnData answers it).
void RgfxGLSetPluginReturnData(id<MTLFunction> function, NSData *_Nullable data);

NS_ASSUME_NONNULL_END

#endif // RGFX_GL_FUNCTION_H_
