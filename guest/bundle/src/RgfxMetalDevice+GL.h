/*
 * RgfxMetalDevice (GL): the MTLDevice members macOS's OpenGL renderer over Metal
 * (AppleMetalOpenGLRenderer) makes its shaders with: newFunctionWithGLIR:, newFunctionWithGLCoreIR:
 * and newFunctionWithGLESIR:, each with and without an inputs description, answer a function of
 * the IR (RgfxGLFunction.h), whose variants are libraries of the device. The renderer reaches the
 * device by the accelerator's IOGLBundleName (guest/kext/src/Info.plist); Metal's own devices
 * answer these members from Metal's base device class.
 *
 * Like the rest of the device it is compiled into the bundle but not linked into the unit tests
 * (RgfxMetalDevice.h). Our own ObjC++.
 */

#ifndef RGFX_METAL_DEVICE_GL_H_
#define RGFX_METAL_DEVICE_GL_H_

#import "RgfxMetalDevice.h"

@interface RgfxMetalDevice (GL)
@end

#endif // RGFX_METAL_DEVICE_GL_H_
